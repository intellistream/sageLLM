# 监控指标文档

本文档详细介绍 sageLLM Control Plane 的监控指标系统，包括指标定义、收集方法、使用示例和最佳实践。

## 目录

- [架构概览](#%E6%9E%B6%E6%9E%84%E6%A6%82%E8%A7%88)
- [指标类型](#%E6%8C%87%E6%A0%87%E7%B1%BB%E5%9E%8B)
- [MetricsCollector API](#metricscollector-api)
- [调度指标](#%E8%B0%83%E5%BA%A6%E6%8C%87%E6%A0%87)
- [实例指标](#%E5%AE%9E%E4%BE%8B%E6%8C%87%E6%A0%87)
- [使用示例](#%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B)
- [可视化和告警](#%E5%8F%AF%E8%A7%86%E5%8C%96%E5%92%8C%E5%91%8A%E8%AD%A6)
- [性能优化](#%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96)

## 架构概览

### 指标收集流程

```
请求执行 → Manager记录 → MetricsCollector聚合 → API查询
                              ↓
                      滑动窗口存储 (deque)
                              ↓
                    实时计算统计指标 (avg, p50, p95, p99)
```

### 核心组件

1. **MetricsCollector** (`control_plane/monitoring.py`)

   - 指标收集和聚合
   - 滑动窗口管理（默认1000个样本）
   - 实时统计计算

1. **SchedulingMetrics** (`control_plane/types.py`)

   - 调度质量指标
   - SLO合规性
   - 队列动态

1. **InstanceMetrics** (`control_plane/types.py`)

   - 实例性能指标
   - 健康状态
   - 资源利用率

## 指标类型

### 1. 调度质量指标

衡量调度策略的效果：

| 指标名称                     | 类型   | 单位    | 说明                |
| ---------------------------- | ------ | ------- | ------------------- |
| `scheduling_latency_us`      | 延迟   | 微秒    | 调度决策耗时        |
| `prediction_accuracy_rate`   | 比率   | 0.0-1.0 | 延迟预测准确率      |
| `load_balance_variance`      | 方差   | -       | 实例间负载方差      |
| `slo_compliance_by_priority` | 字典   | %       | 各优先级的SLO合规率 |
| `avg_queue_length`           | 平均值 | 请求数  | 平均队列长度        |
| `max_queue_length`           | 最大值 | 请求数  | 峰值队列长度        |
| `total_scheduled_requests`   | 计数   | 请求数  | 总调度请求数        |

### 2. 实例性能指标

衡量单个vLLM实例的性能：

| 指标名称          | 类型   | 单位    | 说明             |
| ----------------- | ------ | ------- | ---------------- |
| `avg_latency_ms`  | 延迟   | 毫秒    | 平均响应延迟     |
| `p50_latency_ms`  | 分位数 | 毫秒    | 中位延迟         |
| `p95_latency_ms`  | 分位数 | 毫秒    | 95分位延迟       |
| `p99_latency_ms`  | 分位数 | 毫秒    | 99分位延迟       |
| `error_rate`      | 比率   | 0.0-1.0 | 错误率           |
| `throughput_rps`  | 吞吐量 | 请求/秒 | 每秒处理请求数   |
| `total_requests`  | 计数   | 请求数  | 总处理请求数     |
| `failed_requests` | 计数   | 请求数  | 失败请求数       |
| `current_load`    | 负载   | 请求数  | 当前运行中请求数 |
| `slo_violations`  | 计数   | 次数    | SLO违反次数      |

### 3. 队列动态指标

衡量请求队列的状态：

| 指标名称               | 类型     | 单位   | 说明                   |
| ---------------------- | -------- | ------ | ---------------------- |
| `queue_wait_time_ms`   | 延迟     | 毫秒   | 请求在队列中的等待时间 |
| `queue_length_samples` | 时间序列 | 请求数 | 队列长度历史记录       |

## MetricsCollector API

### 初始化

```python
from control_plane.monitoring import MetricsCollector

# 创建收集器（默认窗口大小1000）
collector = MetricsCollector(window_size=1000)

# 在 Manager 中自动创建
from control_plane.manager import ControlPlaneManager
manager = ControlPlaneManager()  # 内部自动创建 MetricsCollector
```

### 记录指标

#### 1. 记录请求完成

```python
await collector.record_request_completion(
    request=request,           # RequestMetadata对象
    instance_id="inst-1",      # 执行实例ID
    latency_ms=150.5,         # 实际延迟（毫秒）
    success=True,             # 是否成功
    error_message=None        # 错误信息（如果失败）
)
```

**参数说明**：

- `request`: 完成的请求元数据
- `instance_id`: 执行该请求的实例ID
- `latency_ms`: 端到端延迟（从提交到完成）
- `success`: 是否成功执行
- `error_message`: 失败时的错误信息

**自动计算指标**：

- 延迟分位数（p50, p95, p99）
- 吞吐量（基于时间窗口）
- 错误率
- SLO合规性（如果请求有slo_deadline_ms）

#### 2. 记录调度决策

```python
await collector.record_scheduling_decision(
    decision_latency_us=250,  # 调度耗时（微秒）
    queue_length=15           # 当前队列长度
)
```

**参数说明**：

- `decision_latency_us`: 策略的`get_next_request()`耗时
- `queue_length`: 调度时的队列长度

**自动计算指标**：

- 平均调度延迟
- 队列长度统计

#### 3. 更新实例指标

```python
await collector.update_instance_metrics(
    instance_id="inst-1",
    instance=execution_instance  # ExecutionInstance对象
)
```

**参数说明**：

- `instance_id`: 实例唯一标识
- `instance`: 实例对象（包含current_load等字段）

**同步字段**：

- `current_load`: 当前负载
- `avg_latency_ms`: 平均延迟（从实例对象）

### 查询指标

#### 1. 获取调度指标

```python
metrics: SchedulingMetrics = await collector.get_scheduling_metrics()

print(f"平均调度延迟: {metrics.scheduling_latency_us}μs")
print(f"平均队列长度: {metrics.avg_queue_length}")
print(f"最大队列长度: {metrics.max_queue_length}")
print(f"总调度请求: {metrics.total_scheduled_requests}")

# SLO合规率（按优先级）
for priority, compliance in metrics.slo_compliance_by_priority.items():
    print(f"{priority}: {compliance*100:.2f}% SLO合规")
```

**返回的 SchedulingMetrics 结构**：

```python
@dataclass
class SchedulingMetrics:
    scheduling_latency_us: float          # 平均调度延迟（微秒）
    prediction_accuracy_rate: float       # 预测准确率（0.0-1.0）
    load_balance_variance: float          # 负载方差
    slo_compliance_by_priority: Dict[str, float]  # 各优先级SLO合规率
    avg_queue_length: float               # 平均队列长度
    max_queue_length: int                 # 最大队列长度
    total_scheduled_requests: int         # 总调度请求数
```

#### 2. 获取实例指标（单个实例）

```python
metrics: InstanceMetrics = await collector.get_instance_metrics("inst-1")

print(f"平均延迟: {metrics.avg_latency_ms}ms")
print(f"P95延迟: {metrics.p95_latency_ms}ms")
print(f"P99延迟: {metrics.p99_latency_ms}ms")
print(f"错误率: {metrics.error_rate*100:.2f}%")
print(f"吞吐量: {metrics.throughput_rps} req/s")
print(f"当前负载: {metrics.current_load}")
```

**返回的 InstanceMetrics 结构**：

```python
@dataclass
class InstanceMetrics:
    instance_id: str                      # 实例ID
    avg_latency_ms: float                 # 平均延迟
    p50_latency_ms: float                 # 中位延迟
    p95_latency_ms: float                 # 95分位延迟
    p99_latency_ms: float                 # 99分位延迟
    error_rate: float                     # 错误率（0.0-1.0）
    throughput_rps: float                 # 吞吐量（请求/秒）
    total_requests: int                   # 总请求数
    failed_requests: int                  # 失败请求数
    current_load: int                     # 当前负载
    slo_violations: int                   # SLO违反次数
```

#### 3. 获取所有实例指标

```python
all_metrics: Dict[str, InstanceMetrics] = await collector.get_all_instance_metrics()

for instance_id, metrics in all_metrics.items():
    print(f"\n实例 {instance_id}:")
    print(f"  延迟: {metrics.avg_latency_ms}ms (P95: {metrics.p95_latency_ms}ms)")
    print(f"  吞吐: {metrics.throughput_rps} req/s")
    print(f"  错误率: {metrics.error_rate*100:.2f}%")
```

## 调度指标

### 调度延迟 (scheduling_latency_us)

**定义**: 调度策略做出决策的平均耗时（微秒）

**重要性**:

- 调度延迟直接影响系统响应性
- 目标: < 1000μs (1ms)

**优化建议**:

```python
# ❌ 避免复杂计算
def get_next_request(self, pending_queue, available_instances):
    for request in pending_queue:
        for instance in available_instances:
            # O(n²) 复杂度！
            if self.complex_calculation(request, instance):
                return request

# ✅ 优化为O(n)
def get_next_request(self, pending_queue, available_instances):
    if not pending_queue:
        return None
    # 预先过滤
    compatible = [i for i in available_instances if i.available]
    if compatible:
        return pending_queue[0]
```

**监控示例**:

```python
metrics = await collector.get_scheduling_metrics()
if metrics.scheduling_latency_us > 1000:
    logger.warning(f"Scheduling latency too high: {metrics.scheduling_latency_us}μs")
```

### SLO 合规率 (slo_compliance_by_priority)

**定义**: 按优先级统计的SLO满足率

**计算公式**:

```
SLO合规率 = (满足SLO的请求数) / (总请求数) × 100%

请求满足SLO条件:
  实际延迟 <= request.slo_deadline_ms
```

**使用示例**:

```python
metrics = await collector.get_scheduling_metrics()

# 检查高优先级请求的SLO合规性
high_priority_compliance = metrics.slo_compliance_by_priority.get("HIGH", 0.0)
if high_priority_compliance < 0.95:  # 目标95%
    logger.error(
        f"HIGH priority SLO compliance too low: {high_priority_compliance*100:.2f}%"
    )
    # 触发告警或策略调整
```

**优化策略**:

```python
# 使用 SLOAwarePolicy 提高合规率
from control_plane.strategies import SLOAwarePolicy
manager = ControlPlaneManager(scheduling_policy=SLOAwarePolicy())
```

### 负载均衡方差 (load_balance_variance)

**定义**: 实例间负载分布的方差

**计算公式**:

```
variance = Σ(instance.current_load - mean_load)² / N
```

**重要性**:

- 低方差 → 负载均衡良好
- 高方差 → 负载倾斜严重

**使用示例**:

```python
metrics = await collector.get_scheduling_metrics()

if metrics.load_balance_variance > 100:
    logger.warning("Load imbalance detected, consider using load_balanced router")
    # 切换到负载均衡路由
    manager.router.strategy = "load_balanced"
```

### 队列长度指标

**avg_queue_length**: 平均队列长度 **max_queue_length**: 峰值队列长度

**使用示例**:

```python
metrics = await collector.get_scheduling_metrics()

# 监控队列积压
if metrics.avg_queue_length > 50:
    logger.warning(f"Queue backlog: avg={metrics.avg_queue_length}")
    # 可能需要扩容实例

if metrics.max_queue_length > 100:
    logger.critical(f"Queue overflow risk: max={metrics.max_queue_length}")
    # 触发自动扩容
```

## 实例指标

### 延迟指标

**avg_latency_ms**: 平均延迟 **p50/p95/p99_latency_ms**: 延迟分位数

**使用示例**:

```python
instance_metrics = await collector.get_instance_metrics("inst-1")

# SLA监控：P95延迟应 < 1000ms
if instance_metrics.p95_latency_ms > 1000:
    logger.warning(
        f"Instance {instance_metrics.instance_id} P95 latency high: "
        f"{instance_metrics.p95_latency_ms}ms"
    )

# P99延迟用于检测异常值
if instance_metrics.p99_latency_ms > 2 * instance_metrics.p95_latency_ms:
    logger.info("Tail latency spike detected, possible GC or network issue")
```

**分位数解读**:

- **P50**: 中位数，50%请求的延迟 ≤ 此值
- **P95**: 95%请求的延迟 ≤ 此值（常用SLA指标）
- **P99**: 99%请求的延迟 ≤ 此值（长尾检测）

### 错误率 (error_rate)

**定义**: 失败请求占总请求的比例

**计算公式**:

```
error_rate = failed_requests / total_requests
```

**使用示例**:

```python
instance_metrics = await collector.get_instance_metrics("inst-1")

# 健康检查：错误率应 < 1%
if instance_metrics.error_rate > 0.01:
    logger.error(
        f"Instance {instance_metrics.instance_id} error rate high: "
        f"{instance_metrics.error_rate*100:.2f}%"
    )
    # 标记实例为不健康
    instance.available = False
```

### 吞吐量 (throughput_rps)

**定义**: 每秒处理的请求数

**计算方法**: 基于滑动窗口的时间范围

**使用示例**:

```python
instance_metrics = await collector.get_instance_metrics("inst-1")

# 容量规划：吞吐量接近上限需扩容
max_expected_rps = 100
if instance_metrics.throughput_rps > max_expected_rps * 0.8:
    logger.info(
        f"Instance {instance_metrics.instance_id} nearing capacity: "
        f"{instance_metrics.throughput_rps} req/s"
    )
    # 触发自动扩容或流量迁移
```

### SLO 违反次数 (slo_violations)

**定义**: 实际延迟超过SLO要求的请求数

**使用示例**:

```python
instance_metrics = await collector.get_instance_metrics("inst-1")

# 计算违反率
violation_rate = instance_metrics.slo_violations / instance_metrics.total_requests
if violation_rate > 0.05:  # 目标 < 5%
    logger.warning(
        f"Instance {instance_metrics.instance_id} SLO violation rate high: "
        f"{violation_rate*100:.2f}%"
    )
```

## 使用示例

### 完整监控示例

```python
import asyncio
import logging
from control_plane.manager import ControlPlaneManager
from control_plane.monitoring import MetricsCollector

async def monitor_control_plane(manager: ControlPlaneManager):
    """定期监控 Control Plane 指标"""
    logger = logging.getLogger(__name__)

    while True:
        try:
            # 获取调度指标
            sched_metrics = await manager.get_scheduling_metrics()
            logger.info(
                f"Scheduling: latency={sched_metrics.scheduling_latency_us}μs, "
                f"queue_avg={sched_metrics.avg_queue_length}, "
                f"queue_max={sched_metrics.max_queue_length}"
            )

            # 检查SLO合规性
            for priority, compliance in sched_metrics.slo_compliance_by_priority.items():
                if compliance < 0.95:
                    logger.warning(
                        f"SLO compliance low for {priority}: {compliance*100:.2f}%"
                    )

            # 获取所有实例指标
            all_metrics = await manager.get_all_instance_metrics()

            for instance_id, inst_metrics in all_metrics.items():
                logger.info(
                    f"Instance {instance_id}: "
                    f"latency={inst_metrics.avg_latency_ms}ms (P95: {inst_metrics.p95_latency_ms}ms), "
                    f"throughput={inst_metrics.throughput_rps} req/s, "
                    f"error_rate={inst_metrics.error_rate*100:.2f}%, "
                    f"load={inst_metrics.current_load}"
                )

                # 健康检查
                if inst_metrics.error_rate > 0.05:
                    logger.error(f"Instance {instance_id} unhealthy: high error rate")

                if inst_metrics.p95_latency_ms > 1000:
                    logger.warning(f"Instance {instance_id} slow: high P95 latency")

        except Exception as e:
            logger.error(f"Monitoring error: {e}")

        # 每30秒监控一次
        await asyncio.sleep(30)

# 启动监控
async def main():
    manager = ControlPlaneManager()

    # 注册实例...

    # 启动监控任务
    monitor_task = asyncio.create_task(monitor_control_plane(manager))

    # 运行 Control Plane...

    await monitor_task

asyncio.run(main())
```

### 自适应策略示例

基于指标动态调整策略：

```python
from control_plane.strategies import SchedulingPolicy, FIFOPolicy, SLOAwarePolicy

class MetricsAdaptivePolicy(SchedulingPolicy):
    """基于指标自适应切换策略"""

    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.fifo_policy = FIFOPolicy()
        self.slo_policy = SLOAwarePolicy()
        self.current_policy = self.fifo_policy
        self.check_interval = 100  # 每100次检查一次
        self.decision_count = 0

    async def get_next_request(self, pending_queue, available_instances):
        # 定期检查指标并切换策略
        if self.decision_count % self.check_interval == 0:
            await self._adapt_based_on_metrics()

        self.decision_count += 1
        return self.current_policy.get_next_request(pending_queue, available_instances)

    async def _adapt_based_on_metrics(self):
        """基于指标调整策略"""
        sched_metrics = await self.metrics.get_scheduling_metrics()

        # 计算总体SLO合规率
        total_compliance = sum(sched_metrics.slo_compliance_by_priority.values()) / max(
            len(sched_metrics.slo_compliance_by_priority), 1
        )

        # SLO合规率低 → 切换到SLO优先策略
        if total_compliance < 0.90:
            if not isinstance(self.current_policy, SLOAwarePolicy):
                logger.info("Switching to SLOAwarePolicy due to low compliance")
                self.current_policy = self.slo_policy
        else:
            # 合规率高 → 使用简单FIFO
            if not isinstance(self.current_policy, FIFOPolicy):
                logger.info("Switching to FIFOPolicy - SLO compliance good")
                self.current_policy = self.fifo_policy
```

### 性能调优示例

```python
async def tune_instance_based_on_metrics(manager, instance_id):
    """基于指标调整实例配置"""

    metrics = await manager.get_instance_metrics_detailed(instance_id)

    # 吞吐量不足 → 增加并发度
    if metrics.throughput_rps < 50 and metrics.current_load < metrics.max_concurrency * 0.5:
        logger.info(f"Instance {instance_id} underutilized, consider reducing TP size")
        # 建议：降低TP以增加可部署实例数

    # 延迟过高 → 增加TP
    if metrics.p95_latency_ms > 1000:
        logger.info(f"Instance {instance_id} slow, consider increasing TP size")
        # 建议：增加TP以提升单请求性能

    # 错误率高 → 检查实例健康
    if metrics.error_rate > 0.05:
        logger.error(f"Instance {instance_id} unhealthy, triggering health check")
        await manager.executor.health_check(instance_id)
```

## 可视化和告警

### Prometheus 导出

sageLLM 可以导出 Prometheus 格式的指标：

```python
from prometheus_client import Gauge, Counter, Histogram, start_http_server
from control_plane.manager import ControlPlaneManager

# 定义 Prometheus 指标
scheduling_latency = Histogram(
    'sagellm_scheduling_latency_seconds',
    'Scheduling decision latency',
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01]
)

instance_latency = Gauge(
    'sagellm_instance_latency_ms',
    'Instance average latency',
    ['instance_id', 'percentile']
)

slo_compliance = Gauge(
    'sagellm_slo_compliance_ratio',
    'SLO compliance rate',
    ['priority']
)

async def export_metrics_to_prometheus(manager):
    """定期导出指标到 Prometheus"""
    while True:
        # 获取调度指标
        sched_metrics = await manager.get_scheduling_metrics()
        scheduling_latency.observe(sched_metrics.scheduling_latency_us / 1_000_000)

        for priority, compliance in sched_metrics.slo_compliance_by_priority.items():
            slo_compliance.labels(priority=priority).set(compliance)

        # 获取实例指标
        all_metrics = await manager.get_all_instance_metrics()
        for instance_id, metrics in all_metrics.items():
            instance_latency.labels(instance_id=instance_id, percentile='p50').set(
                metrics.p50_latency_ms
            )
            instance_latency.labels(instance_id=instance_id, percentile='p95').set(
                metrics.p95_latency_ms
            )
            instance_latency.labels(instance_id=instance_id, percentile='p99').set(
                metrics.p99_latency_ms
            )

        await asyncio.sleep(10)  # 每10秒导出一次

# 启动 Prometheus HTTP 服务器
start_http_server(9090)
```

**Grafana 面板示例查询**:

```promql
# 平均调度延迟
rate(sagellm_scheduling_latency_seconds_sum[5m]) / rate(sagellm_scheduling_latency_seconds_count[5m])

# P95 延迟
sagellm_instance_latency_ms{percentile="p95"}

# SLO 合规率
sagellm_slo_compliance_ratio{priority="HIGH"}
```

### 告警规则

基于指标设置告警：

```python
async def setup_alerting(manager):
    """设置告警规则"""

    async def check_alerts():
        while True:
            # 获取指标
            sched_metrics = await manager.get_scheduling_metrics()
            all_metrics = await manager.get_all_instance_metrics()

            # 告警1: 队列积压
            if sched_metrics.avg_queue_length > 100:
                await send_alert(
                    severity="warning",
                    title="Queue Backlog",
                    message=f"Average queue length: {sched_metrics.avg_queue_length}"
                )

            # 告警2: SLO违反
            for priority, compliance in sched_metrics.slo_compliance_by_priority.items():
                if compliance < 0.90:
                    await send_alert(
                        severity="critical",
                        title=f"Low SLO Compliance - {priority}",
                        message=f"Compliance: {compliance*100:.2f}%"
                    )

            # 告警3: 实例异常
            for instance_id, metrics in all_metrics.items():
                if metrics.error_rate > 0.10:
                    await send_alert(
                        severity="critical",
                        title=f"High Error Rate - {instance_id}",
                        message=f"Error rate: {metrics.error_rate*100:.2f}%"
                    )

            await asyncio.sleep(60)  # 每分钟检查一次

    asyncio.create_task(check_alerts())

async def send_alert(severity, title, message):
    """发送告警（集成Slack/PagerDuty/Email等）"""
    logger.warning(f"[{severity.upper()}] {title}: {message}")
    # 实现具体告警通道...
```

## 性能优化

### 滑动窗口大小调优

```python
# 小窗口（100）：快速响应，但统计波动大
collector = MetricsCollector(window_size=100)

# 中窗口（1000）：平衡响应性和稳定性（默认）
collector = MetricsCollector(window_size=1000)

# 大窗口（10000）：稳定统计，但响应慢
collector = MetricsCollector(window_size=10000)
```

**选择建议**:

- 开发/测试环境: 100-500
- 生产环境: 1000-5000
- 高流量环境: 5000-10000

### 减少指标查询频率

```python
# ❌ 避免高频查询
async def bad_monitoring():
    while True:
        metrics = await collector.get_all_instance_metrics()  # 每次都计算
        await asyncio.sleep(1)  # 每秒查询

# ✅ 合理的查询频率
async def good_monitoring():
    while True:
        metrics = await collector.get_all_instance_metrics()
        await asyncio.sleep(30)  # 30秒查询一次
```

### 异步指标记录

所有指标记录方法都是 `async def`，避免阻塞：

```python
# 在请求完成回调中记录
async def on_request_done(request, instance_id, latency_ms, success):
    # 异步记录，不阻塞主流程
    await collector.record_request_completion(
        request=request,
        instance_id=instance_id,
        latency_ms=latency_ms,
        success=success
    )
```

## 最佳实践总结

1. **合理设置窗口大小**：根据流量选择1000-5000
1. **定期导出指标**：集成Prometheus/Grafana
1. **设置告警阈值**：
   - 调度延迟 < 1ms
   - SLO合规率 > 95%
   - 错误率 < 1%
   - P95延迟 < 1000ms
1. **基于指标优化**：
   - 高延迟 → 增加TP或增加实例
   - 低吞吐 → 降低TP以增加并发
   - SLO违反 → 切换SLO优先策略
1. **避免过度监控**：指标查询频率 ≥ 10秒

## 相关文档

- [自定义调度策略开发指南](CUSTOM_SCHEDULING.md) - 如何使用指标优化策略
- [拓扑感知配置](TOPOLOGY.md) - 拓扑信息如何影响性能指标
- [故障容错机制](FAULT_TOLERANCE.md) - 故障检测如何利用指标
