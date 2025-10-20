# sageLLM Control Plane 设计总结

## 📋 项目概述

为sageLLM设计了一个完整的**Control Plane组件**，作为用户请求和vLLM执行层之间的智能中间层。

## 🏗️ 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户请求入口                                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Request Submission API  │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│                          CONTROL PLANE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Control Plane Manager (中枢)                         │  │
│  │  职责:                                                            │  │
│  │  - 请求队列管理 (FIFO队列)                                       │  │
│  │  - 调度循环 (异步主循环)                                         │  │
│  │  - 健康检查 (周期性)                                             │  │
│  │  - 性能监控 (实时指标)                                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│           │                      │                       │              │
│           ▼                      ▼                       ▼              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│  │ Scheduling      │   │ Parallelism     │   │ Request Router  │   │
│  │ Policies        │   │ Optimizer       │   │ & Load Balancer │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              Execution Coordinator                              │  │
│  │  职责:                                                          │  │
│  │  - 实例管理 (注册、注销、状态跟踪)                             │  │
│  │  - 请求执行 (异步执行)                                         │  │
│  │  - 指标收集 (延迟、吞吐等)                                     │  │
│  │  - 健康监控 (实例可用性)                                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  vLLM Instance APIs      │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────▼──────────────────────────────────────┐
│                        EXECUTION LAYER (vLLM)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ vLLM     │  │ vLLM     │  │ vLLM     │  │ vLLM     │               │
│  │Instance  │  │Instance  │  │Instance  │  │Instance  │               │
│  │ TP=4     │  │ PP=2     │  │ Hybrid   │  │ DP=2     │               │
│  │ 4 GPUs   │  │ 4 GPUs   │  │ 8 GPUs   │  │ 2 GPUs   │               │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📦 核心组件

### 1️⃣ **types.py** - 数据类型和结构
定义了所有核心数据结构：
- `RequestMetadata` - 请求元数据（优先级、SLO、成本等）
- `ExecutionInstance` - vLLM实例表示
- `SchedulingDecision` - 调度决策
- `PerformanceMetrics` - 性能指标
- 枚举类型：优先级、状态、并行类型

```python
@dataclass
class RequestMetadata:
    request_id: str
    priority: RequestPriority
    slo_deadline_ms: Optional[float]  # SLO截止时间
    cost_budget: Optional[float]       # 成本预算
    parallelism_hint: Optional[ParallelismType]  # 并行建议
    # ... 其他字段
```

### 2️⃣ **policies.py** - 调度策略
5种调度策略支持不同场景：

#### **FIFOPolicy** - 先进先出
- 按到达时间排序
- 适合简单场景
- 低开销

#### **PriorityPolicy** - 优先级调度
- 高优先级请求优先
- 高优先级分配给性能更好的实例
- 适合混合优先级场景

#### **SLOAwarePolicy** - SLO感知
- 考虑截止时间
- 计算紧迫度评分
- 紧急请求优先分配快速实例
- **关键：** 保证时间敏感请求

#### **CostOptimizedPolicy** - 成本优化
- 在满足SLO前提下最小化成本
- 评估每个实例的成本
- 适合成本敏感场景

#### **AdaptivePolicy** - 自适应
- 根据系统状态自动切换策略
- 高负载→SLO感知
- 低负载→成本优化
- 有高优先级→优先级调度
- **推荐：** 生产环境使用

### 3️⃣ **parallelism.py** - 并行策略优化

#### 5种并行策略

| 策略 | 适用场景 | GPU分配 |
|------|---------|--------|
| **TensorParallel (TP)** | 单个模型太大 | 权重切分，推荐2,4,8,16 |
| **PipelineParallel (PP)** | 超大模型 | 层切分，有流水线气泡 |
| **DataParallel (DP)** | 高吞吐需求 | 模型复制，提高并发 |
| **ExpertParallel (EP)** | MoE模型 | 专家切分，针对MoE优化 |
| **HybridParallel** | 大规模部署 | TP+PP+DP组合，自动优化 |

#### 自动优化逻辑

```
GPU数量          推荐配置
─────────────────────────────────────
< 2             TP=1（单GPU）
2-3             TP=2
4-7             TP=4
8-15            TP=4, DP=auto
16+             TP=4, PP=2, DP=auto (混合)
MoE模型         EP=num_experts, TP=4
```

#### ParallelismOptimizer 自动选择

```python
strategy, config = optimizer.select_strategy(request, instance, gpu_count)
# 返回最优策略和配置
```

### 4️⃣ **router.py** - 请求路由和负载均衡

#### RequestRouter - 5种路由策略

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| **load_balanced** | 选择负载最低的实例 | 通用，推荐默认 |
| **round_robin** | 轮询 | 简单、快速 |
| **random** | 随机 | 分散负载 |
| **affinity** | 用户会话亲和性 | 需要保持连接状态 |
| **locality** | 基于哈希的局部性 | 提高缓存命中率 |

#### LoadBalancer - 多种负载均衡算法

- **weighted_round_robin** - 按容量权重选择
- **least_connections** - 最少活跃连接
- **least_response_time** - 最低响应时间
- **power_of_two** - 随机两选一（高性能）

### 5️⃣ **executor.py** - 执行协调器

职责：
- 💾 实例管理（注册、注销、查询）
- ⚡ 异步请求执行
- 📊 性能指标收集
- 🏥 健康检查
- 🔄 负载平衡

关键方法：
```python
# 执行请求
await coordinator.execute_request(request, instance, decision)

# 健康检查
await coordinator.health_check(instance_id)
await coordinator.health_check_all()

# 获取指标
metrics = coordinator.get_metrics()
```

### 6️⃣ **manager.py** - 主管理器

ControlPlaneManager 是整个系统的核心：

#### 初始化
```python
cp = ControlPlaneManager(
    scheduling_policy="adaptive",      # 调度策略
    routing_strategy="load_balanced",  # 路由策略
    enable_monitoring=True,            # 监控
)
```

#### 核心功能
1. **请求管理** - 提交、查询、取消请求
2. **调度循环** - 异步主循环，持续调度pending请求
3. **健康检查** - 周期性检查实例健康状态
4. **监控统计** - 收集性能指标

#### 后台任务
- `_scheduling_loop()` - 100ms调度一次
- `_health_check_loop()` - 10s检查一次
- `_monitoring_loop()` - 5s统计一次

## 🚀 使用流程

### 1. 初始化和启动
```python
import asyncio
from vllm.control_plane import ControlPlaneManager, RequestMetadata

cp = ControlPlaneManager(scheduling_policy="adaptive")
await cp.start()  # 启动后台任务
```

### 2. 注册vLLM实例
```python
from vllm.control_plane import ExecutionInstance

instance = ExecutionInstance(
    instance_id="vllm-1",
    host="localhost",
    port=8000,
    model_name="llama-3-70b",
    tensor_parallel_size=4,
    gpu_count=4,
    gpu_memory_gb=80.0,
)
cp.register_instance(instance)
```

### 3. 提交请求
```python
request = RequestMetadata(
    request_id="req-001",
    user_id="user-123",
    priority=RequestPriority.HIGH,
    slo_deadline_ms=1000,  # 1秒SLO
    max_tokens=100,
)
await cp.submit_request(request)
```

### 4. 监控和查询
```python
# 获取请求状态
status = await cp.get_request_status("req-001")

# 获取性能指标
metrics = cp.get_metrics()
print(f"Avg Latency: {metrics.avg_latency_ms}ms")
print(f"SLO Compliance: {metrics.slo_compliance_rate:.2%}")

# 获取系统状态
status = cp.get_status()
```

## 📊 性能指标

Control Plane收集丰富的指标：

```python
PerformanceMetrics:
├── 请求指标
│   ├── total_requests
│   ├── completed_requests
│   ├── failed_requests
│   ├── active_requests
│   └── queued_requests
├── 延迟指标
│   ├── avg_latency_ms
│   ├── p50_latency_ms
│   ├── p95_latency_ms
│   └── p99_latency_ms
├── 吞吐指标
│   ├── tokens_per_second
│   └── requests_per_second
├── 资源指标
│   ├── avg_gpu_utilization
│   ├── total_gpu_memory_gb
│   └── used_gpu_memory_gb
└── SLO指标
    ├── slo_violations
    └── slo_compliance_rate
```

## 🎯 应用场景和策略选择

### 场景1: 企业SaaS服务
```python
cp = ControlPlaneManager(
    scheduling_policy="slo_aware",     # 保证延迟
    routing_strategy="affinity",        # 用户亲和
)
# 表现: 严格的SLO遵守，用户体验一致
```

### 场景2: 成本敏感的批处理
```python
cp = ControlPlaneManager(
    scheduling_policy="cost_optimized", # 最小化成本
    routing_strategy="load_balanced",   # 均匀分布
)
# 表现: 低成本，吞吐量足够
```

### 场景3: 混合型生产系统
```python
cp = ControlPlaneManager(
    scheduling_policy="adaptive",       # 自适应切换
    routing_strategy="load_balanced",
)
# 表现: 自动适应不同负载，推荐生产
```

### 场景4: 实时交互系统
```python
cp = ControlPlaneManager(
    scheduling_policy="priority",       # 优先级调度
    routing_strategy="locality",        # 缓存局部性
)
# 表现: 优先处理关键请求，提高缓存命中
```

## 💡 关键设计亮点

### 1. **多层次调度**
- 全局调度策略选择（哪个实例）
- 实例级并行策略选择（如何并行）
- 路由算法选择（具体路由方式）

### 2. **异步架构**
- 非阻塞请求提交
- 异步调度和执行
- 后台监控循环

### 3. **性能感知**
- 实例性能自学习
- 基于历史数据的预测
- 动态调整策略

### 4. **高度可配置**
- 5种调度策略
- 5种并行策略
- 5种路由算法
- 可动态切换

### 5. **可观测性**
- 丰富的性能指标
- SLO遵守率监控
- 实例级健康检查

## 📁 文件结构

```
control_plane/
├── __init__.py           # 统一导入入口
├── types.py              # 数据类型定义
├── policies.py           # 5种调度策略
├── parallelism.py        # 5种并行策略 + 优化器
├── router.py             # 请求路由 + 负载均衡
├── executor.py           # 执行协调器
├── manager.py            # 主管理器
├── example.py            # 使用示例
└── README.md             # 详细文档
```

## 🔮 未来增强方向

1. **自动伸缩** - 根据负载自动扩缩容
2. **智能缓存** - KV cache共享和管理
3. **多模型支持** - 同时管理不同模型
4. **成本预测** - 基于历史数据预测成本
5. **故障恢复** - 自动故障检测和转移
6. **模型热迁移** - 无损模型更新
7. **用户配额** - 精细化资源控制
8. **A/B测试** - 支持多版本对比

## 📝 总结

这个Control Plane设计为sageLLM提供了：

✅ **智能调度** - 5种策略满足不同场景
✅ **动态并行** - 自动优化模型分割方案
✅ **高效路由** - 5种路由算法+5种负载均衡
✅ **性能监控** - 详细的性能指标和SLO跟踪
✅ **生产就绪** - 异步架构、错误处理、健康检查
✅ **高可配置** - 支持动态策略切换
✅ **可观测性** - 完整的监控和诊断能力

可以直接集成到现有的sageLLM中使用！
