# Control Plane Quick Start Guide

## 5分钟快速开始

### 1. 导入所需模块

```python
import asyncio
from vllm.control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    RequestMetadata,
    RequestPriority,
)
```

### 2. 创建Control Plane实例

```python
# 使用自适应调度策略（推荐）
cp = ControlPlaneManager(
    scheduling_policy="adaptive",      # 自动选择最佳策略
    routing_strategy="load_balanced",  # 负载均衡路由
    enable_monitoring=True,            # 启用监控
)
```

### 3. 注册vLLM实例

```python
# 创建第一个实例（4 GPU张量并行）
instance1 = ExecutionInstance(
    instance_id="vllm-tp4",
    host="localhost",
    port=8000,
    model_name="llama-3-70b",
    tensor_parallel_size=4,
    gpu_count=4,
    gpu_memory_gb=80.0,
    max_concurrent_requests=100,
)
cp.register_instance(instance1)

# 创建第二个实例（混合并行）
instance2 = ExecutionInstance(
    instance_id="vllm-hybrid",
    host="localhost",
    port=8001,
    model_name="llama-3-70b",
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    gpu_count=4,
    gpu_memory_gb=80.0,
    max_concurrent_requests=80,
)
cp.register_instance(instance2)
```

### 4. 启动Control Plane

```python
async def main():
    # 启动后台任务
    await cp.start()
    print("✓ Control Plane started")
```

### 5. 提交推理请求

```python
    # 创建高优先级请求
    request = RequestMetadata(
        request_id="req-001",
        user_id="user-123",
        priority=RequestPriority.HIGH,
        slo_deadline_ms=1000,      # 1秒SLO
        max_tokens=100,
        model_name="llama-3-70b",
    )
    
    # 提交请求
    await cp.submit_request(request)
    print("✓ Request submitted")
```

### 6. 监控性能

```python
    # 等待处理
    await asyncio.sleep(2)
    
    # 获取指标
    metrics = cp.get_metrics()
    print(f"Completed: {metrics.completed_requests}")
    print(f"Failed: {metrics.failed_requests}")
    print(f"Avg Latency: {metrics.avg_latency_ms:.2f}ms")
    print(f"SLO Compliance: {metrics.slo_compliance_rate:.2%}")
    
    # 获取系统状态
    status = cp.get_status()
    print(f"Running: {status['running_requests']}")
    print(f"Queued: {status['pending_requests']}")
```

### 7. 停止Control Plane

```python
    # 停止所有后台任务
    await cp.stop()
    print("✓ Control Plane stopped")

# 运行
asyncio.run(main())
```

## 完整示例代码

```python
#!/usr/bin/env python3
import asyncio
from vllm.control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    RequestMetadata,
    RequestPriority,
)

async def main():
    # 1. 创建Control Plane
    cp = ControlPlaneManager(
        scheduling_policy="adaptive",
        routing_strategy="load_balanced",
    )
    
    # 2. 注册实例
    instance = ExecutionInstance(
        instance_id="vllm-1",
        host="localhost",
        port=8000,
        model_name="llama-3-70b",
        tensor_parallel_size=4,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=100,
    )
    cp.register_instance(instance)
    
    # 3. 启动
    await cp.start()
    
    # 4. 提交请求
    for i in range(5):
        request = RequestMetadata(
            request_id=f"req-{i:03d}",
            user_id=f"user-{i}",
            priority=RequestPriority.NORMAL if i % 2 else RequestPriority.HIGH,
            slo_deadline_ms=1000,
            max_tokens=100,
            model_name="llama-3-70b",
        )
        await cp.submit_request(request)
    
    # 5. 监控
    await asyncio.sleep(2)
    metrics = cp.get_metrics()
    print(f"Completed: {metrics.completed_requests}")
    print(f"SLO Compliance: {metrics.slo_compliance_rate:.2%}")
    
    # 6. 停止
    await cp.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 常见配置

### 配置1: SaaS应用（严格SLO）

```python
cp = ControlPlaneManager(
    scheduling_policy="slo_aware",     # SLO感知调度
    routing_strategy="affinity",        # 用户亲和路由
    enable_monitoring=True,
)

# 提交带SLO的请求
request = RequestMetadata(
    request_id="req-critical",
    priority=RequestPriority.HIGH,
    slo_deadline_ms=500,  # 严格：500ms
)
```

### 配置2: 批处理（成本优化）

```python
cp = ControlPlaneManager(
    scheduling_policy="cost_optimized",  # 成本优化
    routing_strategy="load_balanced",
)

# 提交带成本预算的请求
request = RequestMetadata(
    request_id="req-batch",
    priority=RequestPriority.LOW,
    cost_budget=0.05,  # $0.05预算
    max_tokens=500,
)
```

### 配置3: 混合场景（推荐）

```python
cp = ControlPlaneManager(
    scheduling_policy="adaptive",        # 自适应（推荐）
    routing_strategy="load_balanced",
    enable_monitoring=True,
)
```

### 配置4: 缓存优化

```python
cp = ControlPlaneManager(
    scheduling_policy="priority",
    routing_strategy="locality",  # 哈希路由提高缓存命中
)
```

## 动态策略切换

```python
# 初始策略
cp = ControlPlaneManager(scheduling_policy="fifo")

# 后期切换
cp.update_policy("adaptive")       # 切换到自适应
print(cp.get_status()['scheduling_policy'])  # 查看当前策略
```

## 查询请求状态

```python
# 提交请求
request_id = "req-001"
await cp.submit_request(request)

# 查询状态
status = await cp.get_request_status(request_id)
print(status)  # RequestStatus.QUEUED or RUNNING

# 取消请求（只能取消pending的）
await cp.cancel_request(request_id)
```

## 实例管理

```python
# 获取所有实例
instances = cp.get_instances()
for instance in instances:
    print(f"{instance.instance_id}: Load={instance.current_load:.2%}")

# 获取单个实例指标
metrics = cp.get_instance_metrics("vllm-1")
print(f"Active: {metrics['active_requests']}")
print(f"Latency: {metrics['avg_latency_ms']}ms")
print(f"Healthy: {metrics['is_healthy']}")

# 注销实例
cp.unregister_instance("vllm-1")
```

## 性能调优建议

### 1. 调度策略选择

```python
# 如果有严格SLO要求
cp.update_policy("slo_aware")

# 如果要最小化成本
cp.update_policy("cost_optimized")

# 不确定时用自适应
cp.update_policy("adaptive")
```

### 2. 并行策略优化

```python
# 通过parallelism_hint提示
request = RequestMetadata(
    request_id="req-001",
    parallelism_hint=ParallelismType.TENSOR_PARALLEL,  # 提示TP
)

# 或让系统自动选择（推荐）
request = RequestMetadata(
    request_id="req-002",
    # 不设置hint，系统自动优化
)
```

### 3. 路由策略优化

```python
# 需要缓存亲和性
cp.router.routing_strategy = "locality"

# 需要用户会话一致性
cp.router.routing_strategy = "affinity"

# 通用负载均衡
cp.router.routing_strategy = "load_balanced"
```

## 性能指标解读

```python
metrics = cp.get_metrics()

# 关键指标
print(f"吞吐量: {metrics.requests_per_second:.2f} req/s")
print(f"平均延迟: {metrics.avg_latency_ms:.2f}ms")
print(f"尾延迟(p95): {metrics.p95_latency_ms:.2f}ms")
print(f"尾延迟(p99): {metrics.p99_latency_ms:.2f}ms")
print(f"SLO遵守率: {metrics.slo_compliance_rate:.2%}")

# 如果SLO遵守率低
if metrics.slo_compliance_rate < 0.95:
    cp.update_policy("slo_aware")  # 切换到SLO感知
    
# 如果成本过高
if metrics.cost_per_token > 0.001:
    cp.update_policy("cost_optimized")  # 切换到成本优化
```

## 故障排查

### 问题1: 请求始终在队列中

```python
# 检查是否有可用实例
status = cp.get_status()
print(f"Available instances: {status['available_instances']}")

# 检查实例健康状态
for instance in cp.get_instances():
    print(f"{instance.instance_id}: healthy={instance.is_healthy}")
    
# 可能需要注册新实例或启动vLLM
```

### 问题2: 延迟过高

```python
# 检查SLO合规
metrics = cp.get_metrics()
print(f"SLO violations: {metrics.slo_violations}")

# 尝试优先级调度
cp.update_policy("priority")

# 或增加实例
new_instance = ExecutionInstance(...)
cp.register_instance(new_instance)
```

### 问题3: 某些实例负载不均衡

```python
# 检查各实例负载
for instance in cp.get_instances():
    inst_metrics = cp.get_instance_metrics(instance.instance_id)
    print(f"{instance.instance_id}: load={inst_metrics['current_load']:.2%}")

# 切换路由策略
cp.router.routing_strategy = "power_of_two"  # 更好的负载均衡

# 或使用weighted_round_robin
cp.load_balancer.select_instance(
    cp.get_instances(),
    algorithm="weighted_round_robin"
)
```

## API快速参考

### ControlPlaneManager 方法

```python
# 生命周期
await cp.start()
await cp.stop()

# 实例管理
cp.register_instance(instance)
cp.unregister_instance(instance_id)
cp.get_instances()
cp.get_instance_metrics(instance_id)

# 请求管理
await cp.submit_request(request)
await cp.get_request_status(request_id)
await cp.cancel_request(request_id)

# 监控
cp.get_metrics()
cp.get_status()

# 策略
cp.update_policy(policy_name)
```

### RequestMetadata 常用字段

```python
RequestMetadata(
    request_id="unique-id",              # 必需
    user_id="user-123",                  # 可选
    priority=RequestPriority.NORMAL,     # 优先级
    slo_deadline_ms=1000,                # SLO截止时间
    max_tokens=100,                      # 最大生成token数
    model_name="llama-3-70b",            # 模型名
    cost_budget=0.01,                    # 成本预算
    parallelism_hint=ParallelismType.TP, # 并行策略提示
)
```

## 更多信息

- 详细架构设计: 见 `DESIGN.md`
- 中文设计文档: 见 `DESIGN_SUMMARY_CN.md`
- 完整示例: 见 `example.py`
- API文档: 各模块源代码中的docstrings

祝使用愉快！🚀
