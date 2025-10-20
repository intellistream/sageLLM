# Control Plane 设计文档

## 概述

Control Plane 是 sageLLM 的智能请求调度和管理层，位于用户请求和 vLLM 执行实例之间，提供：

- **智能调度策略**：支持多种调度算法（FIFO、优先级、SLO感知、成本优化等）
- **动态并行策略**：自动选择最优的模型并行方案（TP、PP、DP、EP及混合策略）
- **负载均衡**：多种路由算法确保资源高效利用
- **性能监控**：实时监控和指标收集
- **弹性伸缩**：支持动态实例管理

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        Control Plane                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Control Plane Manager (核心管理器)              │   │
│  │  - 请求队列管理                                            │   │
│  │  - 调度循环                                                │   │
│  │  - 健康检查                                                │   │
│  │  - 性能监控                                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                  │                  │                │
│           ▼                  ▼                  ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Scheduling   │  │ Parallelism  │  │ Request      │          │
│  │ Policies     │  │ Optimizer    │  │ Router       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                  │                  │                │
│           └──────────────────┴──────────────────┘                │
│                              │                                   │
│                              ▼                                   │
│                  ┌──────────────────────┐                        │
│                  │ Execution Coordinator │                        │
│                  └──────────────────────┘                        │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│                      Execution Layer (vLLM)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ vLLM     │  │ vLLM     │  │ vLLM     │  │ vLLM     │        │
│  │ Instance │  │ Instance │  │ Instance │  │ Instance │        │
│  │    1     │  │    2     │  │    3     │  │    N     │        │
│  │ (TP=4)   │  │ (PP=2)   │  │ (Hybrid) │  │ (DP=2)   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 调度策略 (Scheduling Policies)

#### FIFOPolicy
- 先来先服务的基本调度策略
- 按到达时间排序请求

#### PriorityPolicy
- 基于优先级的调度
- 高优先级请求优先分配给性能更好的实例

#### SLOAwarePolicy
- SLO感知调度
- 考虑截止时间，紧急请求优先
- 计算紧迫度评分

#### CostOptimizedPolicy
- 成本优化调度
- 在满足SLO的前提下最小化成本
- 考虑GPU使用成本

#### AdaptivePolicy
- 自适应策略选择
- 根据系统负载和请求特征自动切换策略
- 高负载时优先SLO，低负载时优化成本

### 2. 并行策略 (Parallelism Strategies)

#### TensorParallelStrategy (TP)
- 将模型权重切分到多个GPU
- 适合单个模型太大无法放入单GPU
- 推荐GPU数：2, 4, 8, 16 (2的幂次)

#### PipelineParallelStrategy (PP)
- 将模型层切分到多个GPU
- 适合超大模型
- 有流水线气泡开销

#### DataParallelStrategy (DP)
- 复制模型到多个GPU
- 提高吞吐量
- 适合高并发场景

#### ExpertParallelStrategy (EP)
- 针对MoE模型的专家并行
- 将不同专家分配到不同GPU

#### HybridParallelStrategy
- 组合多种并行策略
- 自动优化配置：
  - 16+ GPU: TP=4, PP=2, DP=auto
  - 8-15 GPU: TP=4, DP=auto
  - 4-7 GPU: TP=4
  - <4 GPU: TP=min(gpu_count, 2)

### 3. 请求路由 (Request Router)

支持多种路由策略：
- **load_balanced**: 负载均衡，路由到负载最低的实例
- **round_robin**: 轮询
- **random**: 随机选择
- **affinity**: 用户亲和性，同一用户请求路由到同一实例
- **locality**: 基于哈希的局部性路由，提高缓存命中率

### 4. 执行协调器 (Execution Coordinator)

- 管理所有vLLM实例
- 协调请求执行
- 健康检查
- 指标收集

## 使用示例

### 基本使用

```python
import asyncio
from vllm.control_plane import (
    ControlPlaneManager,
    RequestMetadata,
    ExecutionInstance,
    RequestPriority,
)

async def main():
    # 创建Control Plane
    cp = ControlPlaneManager(
        scheduling_policy="adaptive",  # 自适应调度
        routing_strategy="load_balanced",  # 负载均衡路由
        enable_monitoring=True,
    )
    
    # 注册vLLM实例
    instance1 = ExecutionInstance(
        instance_id="vllm-1",
        host="localhost",
        port=8000,
        model_name="llama-3-70b",
        tensor_parallel_size=4,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=100,
    )
    cp.register_instance(instance1)
    
    instance2 = ExecutionInstance(
        instance_id="vllm-2",
        host="localhost",
        port=8001,
        model_name="llama-3-70b",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=50,
    )
    cp.register_instance(instance2)
    
    # 启动Control Plane
    await cp.start()
    
    # 提交推理请求
    request = RequestMetadata(
        request_id="req-001",
        user_id="user-123",
        priority=RequestPriority.HIGH,
        slo_deadline_ms=1000,  # 1秒SLO
        max_tokens=100,
        model_name="llama-3-70b",
    )
    
    request_id = await cp.submit_request(request)
    print(f"Request submitted: {request_id}")
    
    # 查询状态
    await asyncio.sleep(1)
    status = await cp.get_request_status(request_id)
    print(f"Request status: {status}")
    
    # 获取指标
    metrics = cp.get_metrics()
    print(f"Metrics: {metrics}")
    
    # 停止Control Plane
    await cp.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 高级配置

```python
# 使用特定调度策略
cp = ControlPlaneManager(
    scheduling_policy="slo_aware",  # SLO感知调度
    routing_strategy="affinity",     # 用户亲和性路由
)

# 提交带并行提示的请求
from vllm.control_plane.types import ParallelismType

request = RequestMetadata(
    request_id="req-002",
    priority=RequestPriority.CRITICAL,
    slo_deadline_ms=500,
    parallelism_hint=ParallelismType.HYBRID,  # 提示使用混合并行
    cost_budget=0.01,  # 成本预算
)
```

### 动态策略切换

```python
# 在运行时切换调度策略
cp.update_policy("cost_optimized")  # 切换到成本优化模式

# 获取Control Plane状态
status = cp.get_status()
print(f"Running: {status['running']}")
print(f"Policy: {status['scheduling_policy']}")
print(f"Pending: {status['pending_requests']}")
print(f"Running: {status['running_requests']}")
```

## 性能优化建议

### 1. 调度策略选择

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 生产环境 | adaptive | 自动适应不同场景 |
| 严格SLO | slo_aware | 优先保证延迟要求 |
| 成本敏感 | cost_optimized | 在满足要求下最小化成本 |
| 简单场景 | fifo | 低开销 |
| 混合优先级 | priority | 确保重要请求优先 |

### 2. 并行策略选择

| 模型大小 | GPU数量 | 推荐策略 |
|---------|--------|---------|
| <10B | 1-2 | TP=1 或 TP=2 |
| 10B-30B | 2-4 | TP=4 |
| 30B-70B | 4-8 | TP=4 或 TP=8 |
| 70B-175B | 8-16 | Hybrid (TP=4, PP=2) |
| >175B | 16+ | Hybrid (TP=8, PP=4) |
| MoE模型 | 8+ | EP + TP |

### 3. 路由策略选择

- **高吞吐场景**: load_balanced 或 power_of_two
- **需要缓存**: affinity 或 locality
- **简单场景**: round_robin
- **分布式推理**: locality

## 监控指标

Control Plane提供丰富的监控指标：

```python
metrics = cp.get_metrics()

# 请求指标
print(f"Total requests: {metrics.total_requests}")
print(f"Completed: {metrics.completed_requests}")
print(f"Failed: {metrics.failed_requests}")
print(f"Active: {metrics.active_requests}")
print(f"Queued: {metrics.queued_requests}")

# 延迟指标
print(f"Avg latency: {metrics.avg_latency_ms}ms")
print(f"P95 latency: {metrics.p95_latency_ms}ms")
print(f"P99 latency: {metrics.p99_latency_ms}ms")

# 吞吐指标
print(f"Tokens/sec: {metrics.tokens_per_second}")
print(f"Requests/sec: {metrics.requests_per_second}")

# SLO指标
print(f"SLO violations: {metrics.slo_violations}")
print(f"SLO compliance: {metrics.slo_compliance_rate:.2%}")

# 资源指标
print(f"GPU utilization: {metrics.avg_gpu_utilization:.2%}")
print(f"GPU memory used: {metrics.used_gpu_memory_gb}GB")
```

## 未来增强

1. **自动伸缩**: 根据负载自动扩缩容vLLM实例
2. **负载迁移**: 实时请求迁移以平衡负载
3. **智能缓存**: KV cache共享和管理
4. **多模型支持**: 同时管理多个不同模型
5. **成本预测**: 基于历史数据预测成本
6. **A/B测试**: 支持多策略对比测试
7. **故障恢复**: 自动故障检测和恢复
8. **配额管理**: 用户级别的配额和限流

## API参考

详见各模块文档：
- `types.py` - 数据类型定义
- `policies.py` - 调度策略
- `parallelism.py` - 并行策略
- `router.py` - 请求路由
- `executor.py` - 执行协调
- `manager.py` - 主管理器
