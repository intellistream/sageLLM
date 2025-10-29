# 故障容错机制文档

本文档详细介绍 sageLLM Control Plane 的故障检测、恢复和容错机制，包括健康检查、自动重调度、配置指南和生产最佳实践。

## 目录

- [故障容错架构](#故障容错架构)
- [故障检测机制](#故障检测机制)
- [自动恢复策略](#自动恢复策略)
- [配置参数](#配置参数)
- [监控和告警](#监控和告警)
- [生产部署指南](#生产部署指南)
- [故障场景处理](#故障场景处理)

## 故障容错架构

### 核心组件

```
┌─────────────────────────────────────────────────┐
│           ControlPlaneManager                   │
│  ┌────────────────────────────────────────┐    │
│  │  on_instance_failure(instance_id)      │    │
│  │  - 标记实例不可用                        │    │
│  │  - 重调度运行中的请求                     │    │
│  │  - 触发告警                             │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
                    ▲
                    │ 回调通知
                    │
┌─────────────────────────────────────────────────┐
│              Executor                           │
│  ┌────────────────────────────────────────┐    │
│  │  health_check(instance_id)             │    │
│  │  - HTTP GET /health                    │    │
│  │  - 连续失败计数                          │    │
│  │  - 超过阈值触发回调                       │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
                    │
                    │ 定期健康检查 (30s)
                    ▼
┌─────────────────────────────────────────────────┐
│         vLLM Instance (HTTP API)                │
│  GET /health → 200 OK / 503 Error              │
└─────────────────────────────────────────────────┘
```

### 故障处理流程

```
1. 健康检查循环 (每30秒)
   └─> health_check(instance_id)
       ├─ 成功 → 重置 consecutive_failures = 0
       └─ 失败 → consecutive_failures++
           └─> 达到阈值 (3次) → 触发故障回调

2. 故障回调
   └─> Manager.on_instance_failure(instance_id)
       ├─ 标记实例: instance.available = False
       ├─ 获取运行中请求: running_requests[instance_id]
       └─ 重调度每个请求:
           ├─ 检查重试次数 < 3
           ├─ 记录失败实例: request.failed_instances.append(instance_id)
           ├─ 重试计数: request.retry_count++
           └─> 重新提交到队列: submit_request(request)

3. 自动恢复
   └─> 实例恢复后，health_check成功
       └─> 标记实例: instance.available = True
           └─> 可以接受新请求
```

## 故障检测机制

### 1. 主动健康检查

**实现**: `Executor.health_check()`

**检查频率**: 每30秒（由 `Manager._health_check_loop()` 调度）

**检查方法**:
```python
# HTTP GET /health 端点
async with self.http_session.get(
    f"http://{instance.host}:{instance.port}/health",
    timeout=aiohttp.ClientTimeout(total=5)  # 5秒超时
) as response:
    if response.status == 200:
        # 健康
        instance.metadata["consecutive_failures"] = 0
    else:
        # 不健康
        instance.metadata["consecutive_failures"] += 1
```

**vLLM健康端点**:
```bash
# vLLM 默认提供 /health 端点
curl http://localhost:8000/health
# 200 OK: 实例健康
# 503 Service Unavailable: 实例启动中或故障
```

### 2. 被动故障检测

**实现**: 请求执行失败时立即检测

```python
async def execute_request(self, request, instance):
    try:
        async with self.http_session.post(
            f"http://{instance.host}:{instance.port}/v1/completions",
            json={...},
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status != 200:
                # 立即标记为失败
                instance.metadata["consecutive_failures"] += 1
                await self._check_failure_threshold(instance)
    except Exception as e:
        # 网络错误、超时等
        instance.metadata["consecutive_failures"] += 1
        await self._check_failure_threshold(instance)
```

### 3. 连续失败阈值

**默认阈值**: 3次连续失败

**原因**: 避免网络抖动误报

**配置示例**:
```python
# 在 Executor 中配置阈值
class Executor:
    FAILURE_THRESHOLD = 3  # 修改为5次以提高容忍度
    
    async def _check_failure_threshold(self, instance):
        failures = instance.metadata.get("consecutive_failures", 0)
        if failures >= self.FAILURE_THRESHOLD:
            # 触发故障回调
            if self._on_instance_failure:
                await self._on_instance_failure(instance.instance_id)
```

### 4. 故障类型识别

不同故障类型的处理：

| 故障类型 | 检测方式 | 恢复时间 | 处理策略 |
|---------|---------|---------|---------|
| **进程崩溃** | HTTP连接失败 | 手动重启 | 立即标记不可用 |
| **GPU OOM** | 503错误 | 自动恢复 | 标记不可用，等待恢复 |
| **网络分区** | 连接超时 | 网络恢复 | 暂时不可用，自动恢复 |
| **慢响应** | 请求超时 | 负载降低 | 继续可用，但降低权重 |

## 自动恢复策略

### 1. 请求重调度

**触发条件**: 实例故障时，正在该实例上运行的请求

**重调度逻辑**:
```python
async def on_instance_failure(self, instance_id: str):
    """实例故障时的处理"""
    
    # 1. 标记实例不可用
    instance = self.executor.instances.get(instance_id)
    if instance:
        instance.available = False
        logger.error(f"Instance {instance_id} marked as unavailable")
    
    # 2. 获取该实例上运行的请求
    failed_requests = self.running_requests.get(instance_id, [])
    
    # 3. 重调度每个请求
    for request in failed_requests:
        # 检查重试次数
        if request.retry_count >= 3:
            logger.error(
                f"Request {request.request_id} exceeded max retries, dropping"
            )
            continue
        
        # 记录失败实例（避免再次调度到同一实例）
        request.failed_instances.append(instance_id)
        request.retry_count += 1
        
        logger.info(
            f"Rescheduling request {request.request_id} "
            f"(retry {request.retry_count}/3)"
        )
        
        # 重新提交
        await self.submit_request(request)
    
    # 4. 清除该实例的运行记录
    self.running_requests.pop(instance_id, None)
```

**重试限制**: 最多重试3次

**避免重复失败**: 记录 `failed_instances`，调度时排除

### 2. 实例自动恢复

**恢复检测**: 健康检查成功后自动恢复

```python
async def health_check(self, instance_id: str):
    """定期健康检查"""
    instance = self.instances.get(instance_id)
    if not instance:
        return
    
    try:
        async with self.http_session.get(
            f"http://{instance.host}:{instance.port}/health",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as response:
            if response.status == 200:
                # 健康 → 恢复
                instance.metadata["consecutive_failures"] = 0
                
                if not instance.available:
                    logger.info(f"Instance {instance_id} recovered")
                    instance.available = True  # 自动恢复
            else:
                # 不健康
                instance.metadata["consecutive_failures"] = \
                    instance.metadata.get("consecutive_failures", 0) + 1
    except Exception as e:
        instance.metadata["consecutive_failures"] = \
            instance.metadata.get("consecutive_failures", 0) + 1
```

**恢复条件**:
- 单次健康检查成功即可恢复
- 自动重新加入可用实例池

### 3. 失败实例黑名单

**目的**: 避免请求反复调度到已知失败的实例

**实现**:
```python
# RequestMetadata 字段
failed_instances: List[str] = field(default_factory=list)

# 在 Router 中过滤
class Router:
    def select_instance(self, request, instances):
        # 过滤掉失败过的实例
        available = [
            inst for inst in instances
            if inst.instance_id not in request.failed_instances
            and inst.available
        ]
        
        if not available:
            # 所有实例都失败过，清空黑名单重试
            request.failed_instances.clear()
            available = [inst for inst in instances if inst.available]
        
        if not available:
            raise NoAvailableInstanceError("No healthy instances")
        
        # 选择实例...
```

### 4. 降级策略

**场景**: 大量实例同时故障

**降级方案**:
```python
async def graceful_degradation(self):
    """优雅降级"""
    
    available_count = sum(1 for inst in self.executor.instances.values() if inst.available)
    total_count = len(self.executor.instances)
    
    availability_ratio = available_count / total_count
    
    if availability_ratio < 0.5:
        logger.critical(
            f"Only {availability_ratio*100:.1f}% instances available, "
            "entering degraded mode"
        )
        
        # 降级措施1: 拒绝低优先级请求
        self.reject_low_priority = True
        
        # 降级措施2: 延长超时时间
        self.request_timeout = 600  # 从300s增加到600s
        
        # 降级措施3: 触发告警
        await self.send_alert("CRITICAL: System degraded")
```

## 配置参数

### 健康检查配置

```python
class ControlPlaneManager:
    # 健康检查间隔（秒）
    HEALTH_CHECK_INTERVAL = 30
    
    # 健康检查超时（秒）
    HEALTH_CHECK_TIMEOUT = 5
    
    # 连续失败阈值
    FAILURE_THRESHOLD = 3
```

**调整建议**:

| 环境 | INTERVAL | TIMEOUT | THRESHOLD | 原因 |
|-----|----------|---------|-----------|------|
| **开发** | 10s | 3s | 2 | 快速检测故障 |
| **生产（稳定网络）** | 30s | 5s | 3 | 默认配置 |
| **生产（不稳定网络）** | 60s | 10s | 5 | 避免误报 |
| **高可用** | 15s | 3s | 2 | 快速故障切换 |

### 重试配置

```python
class RequestMetadata:
    retry_count: int = 0
    MAX_RETRIES: int = 3  # 最大重试次数
```

**调整建议**:
- **关键请求**: `MAX_RETRIES = 5`（多次重试）
- **普通请求**: `MAX_RETRIES = 3`（默认）
- **批处理**: `MAX_RETRIES = 1`（快速失败）

### 超时配置

```python
# 请求执行超时
class Executor:
    REQUEST_TIMEOUT = 300  # 5分钟
    
    async def execute_request(self, request, instance):
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        async with self.http_session.post(..., timeout=timeout):
            ...
```

**调整建议**:
- **短prompt**: 60s
- **长prompt**: 300s
- **超长生成**: 600s

## 监控和告警

### 1. 故障率监控

```python
async def monitor_failure_rate():
    """监控实例故障率"""
    
    while True:
        all_metrics = await manager.get_all_instance_metrics()
        
        for instance_id, metrics in all_metrics.items():
            # 计算故障率
            failure_rate = metrics.failed_requests / max(metrics.total_requests, 1)
            
            if failure_rate > 0.05:  # 5% 阈值
                logger.warning(
                    f"Instance {instance_id} failure rate high: "
                    f"{failure_rate*100:.2f}%"
                )
                
                # 触发告警
                await send_alert(
                    severity="warning",
                    title=f"High Failure Rate - {instance_id}",
                    message=f"Failure rate: {failure_rate*100:.2f}%"
                )
        
        await asyncio.sleep(60)  # 每分钟检查
```

### 2. 可用性监控

```python
async def monitor_availability():
    """监控集群可用性"""
    
    while True:
        total_instances = len(manager.executor.instances)
        available_instances = sum(
            1 for inst in manager.executor.instances.values()
            if inst.available
        )
        
        availability = available_instances / total_instances
        
        logger.info(
            f"Cluster availability: {availability*100:.1f}% "
            f"({available_instances}/{total_instances})"
        )
        
        # 可用性低于80%告警
        if availability < 0.8:
            await send_alert(
                severity="critical",
                title="Low Cluster Availability",
                message=f"Only {availability*100:.1f}% instances available"
            )
        
        await asyncio.sleep(30)
```

### 3. 重试率监控

```python
async def monitor_retry_rate():
    """监控请求重试率"""
    
    total_requests = 0
    retried_requests = 0
    
    # 在请求完成时记录
    async def on_request_complete(request):
        nonlocal total_requests, retried_requests
        total_requests += 1
        if request.retry_count > 0:
            retried_requests += 1
        
        retry_rate = retried_requests / total_requests
        if retry_rate > 0.1:  # 10% 阈值
            logger.warning(f"High retry rate: {retry_rate*100:.2f}%")
```

### 4. Prometheus 指标

```python
from prometheus_client import Counter, Gauge

# 定义指标
instance_failures = Counter(
    'sagellm_instance_failures_total',
    'Total instance failures',
    ['instance_id']
)

request_retries = Counter(
    'sagellm_request_retries_total',
    'Total request retries',
    ['reason']
)

cluster_availability = Gauge(
    'sagellm_cluster_availability_ratio',
    'Cluster availability ratio'
)

# 更新指标
async def update_prometheus_metrics():
    while True:
        available = sum(1 for i in manager.executor.instances.values() if i.available)
        total = len(manager.executor.instances)
        cluster_availability.set(available / total)
        
        await asyncio.sleep(10)
```

## 生产部署指南

### 1. 高可用配置

**最小实例数**: 3个（避免单点故障）

**跨机架部署**: 实例分布在多个机架

```python
# 注册实例时指定机架
instance_rack_a = ExecutionInstance(
    instance_id="inst-rack-a-1",
    rack_id="rack-a",
    ...
)

instance_rack_b = ExecutionInstance(
    instance_id="inst-rack-b-1",
    rack_id="rack-b",
    ...
)
```

**健康检查配置**:
```python
manager = ControlPlaneManager()
manager.HEALTH_CHECK_INTERVAL = 15  # 15秒检查（快速检测）
manager.FAILURE_THRESHOLD = 2       # 2次失败即切换（快速恢复）
```

### 2. 容灾预案

**自动扩容触发器**:
```python
async def auto_scale_trigger():
    """可用性低时触发自动扩容"""
    
    availability = get_cluster_availability()
    
    if availability < 0.7:
        logger.critical("Availability < 70%, triggering auto-scale")
        
        # 调用云平台 API 启动新实例
        await cloud_provider.launch_instances(count=2)
```

**手动故障转移**:
```bash
# 手动标记实例为不可用
curl -X POST http://control-plane:9000/admin/mark_unavailable \
  -d '{"instance_id": "inst-1"}'

# 手动触发重调度
curl -X POST http://control-plane:9000/admin/reschedule_instance \
  -d '{"instance_id": "inst-1"}'
```

### 3. 日志和审计

**故障日志**:
```python
import logging

# 配置专门的故障日志
failure_logger = logging.getLogger("sagellm.failures")
failure_logger.setLevel(logging.WARNING)

handler = logging.FileHandler("/var/log/sagellm/failures.log")
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
failure_logger.addHandler(handler)

# 记录故障
async def on_instance_failure(self, instance_id):
    failure_logger.error(
        f"Instance {instance_id} failed - consecutive failures exceeded threshold"
    )
```

**审计日志**:
```python
audit_logger = logging.getLogger("sagellm.audit")

async def on_request_rescheduled(request, old_instance, new_instance):
    audit_logger.info(
        f"Request {request.request_id} rescheduled: "
        f"{old_instance} → {new_instance} (retry {request.retry_count})"
    )
```

## 故障场景处理

### 场景1: 单个实例崩溃

**故障表现**:
- 健康检查连续失败3次
- 运行中的5个请求中断

**系统响应**:
1. 自动标记实例不可用
2. 重调度5个请求到其他实例
3. 记录故障日志
4. 发送告警通知

**手动操作**:
```bash
# 1. 检查实例状态
curl http://failed-instance:8000/health
# Connection refused

# 2. 检查进程
ps aux | grep vllm
# 进程不存在

# 3. 重启实例
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model llama-2-7b --port 8000

# 4. 等待健康检查恢复（30秒内）
```

### 场景2: GPU OOM

**故障表现**:
- vLLM返回503错误
- 实例仍然运行，但无法处理请求

**系统响应**:
1. 连续失败3次后标记不可用
2. 重调度运行中的请求
3. 实例会自动尝试GC和恢复

**手动操作**:
```bash
# 1. 检查GPU内存
nvidia-smi

# 2. 查看vLLM日志
tail -f /var/log/vllm/instance-0.log
# ERROR: CUDA out of memory

# 3. 降低并发度
# 修改 vLLM 启动参数
--max-num-seqs 128  # 从256降低到128

# 4. 重启实例
```

### 场景3: 网络分区

**故障表现**:
- 部分实例健康检查超时
- 跨机房延迟突增

**系统响应**:
1. 超时实例标记为不可用
2. 请求重调度到可达实例
3. 网络恢复后自动恢复

**预防措施**:
```python
# 使用拓扑感知路由，优先本地实例
manager = ControlPlaneManager(routing_strategy="topology_aware")

# 增加健康检查超时（避免网络抖动误报）
manager.HEALTH_CHECK_TIMEOUT = 10  # 从5s增加到10s
```

### 场景4: 全集群故障

**故障表现**:
- 所有实例同时不可用（如机房断电）
- 队列积压，无法处理请求

**系统响应**:
1. 拒绝新请求（返回503错误）
2. 保留队列中的请求
3. 实例恢复后自动处理积压

**恢复流程**:
```python
# 1. 检测全局故障
availability = get_cluster_availability()
if availability < 0.1:
    logger.critical("FULL CLUSTER OUTAGE")
    # 触发紧急告警
    await send_critical_alert("Full cluster outage detected")

# 2. 恢复后自动处理
# Control Plane 会自动检测实例恢复并处理队列
```

### 场景5: 慢请求累积

**故障表现**:
- 某个实例处理请求极慢
- 该实例负载持续增长
- 最终导致超时

**检测方法**:
```python
async def detect_slow_instances():
    """检测慢实例"""
    
    all_metrics = await manager.get_all_instance_metrics()
    
    for instance_id, metrics in all_metrics.items():
        # P95延迟超过2秒认为慢
        if metrics.p95_latency_ms > 2000:
            logger.warning(
                f"Slow instance detected: {instance_id} "
                f"(P95: {metrics.p95_latency_ms}ms)"
            )
            
            # 降低该实例的权重
            instance = manager.executor.instances[instance_id]
            instance.weight = 0.5  # 减少调度到该实例的概率
```

## 最佳实践总结

1. **健康检查配置**
   - 生产环境: 30秒间隔，3次失败阈值
   - 高可用: 15秒间隔，2次失败阈值

2. **重试策略**
   - 最多重试3次
   - 记录失败实例，避免重复失败

3. **监控告警**
   - 故障率 > 5% 告警
   - 可用性 < 80% 告警
   - 重试率 > 10% 告警

4. **日志审计**
   - 记录所有故障事件
   - 记录请求重调度历史

5. **容灾部署**
   - 至少3个实例
   - 跨机架分布
   - 拓扑感知路由

6. **故障恢复**
   - 自动恢复优先
   - 保留手动干预接口
   - 完善的故障预案

## 相关文档

- [监控指标文档](METRICS.md) - 如何监控故障率和可用性
- [拓扑感知配置](TOPOLOGY.md) - 跨机容灾部署
- [部署指南](DEPLOYMENT.md) - 生产环境部署最佳实践

## 总结

sageLLM 的故障容错机制提供了完善的检测、恢复和监控能力：

1. **自动检测**: 30秒健康检查 + 请求失败检测
2. **自动恢复**: 请求重调度 + 实例自动恢复
3. **容错配置**: 3次重试 + 失败黑名单
4. **监控告警**: 故障率 + 可用性 + Prometheus 集成

配置合理的故障容错参数，可以实现99.9%的服务可用性！
