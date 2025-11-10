# Autoscaling 指南

## 概述

sageLLM Control Plane 现已集成来自 Dynamo 的 SLA-based 自动扩缩容功能，能够根据负载动态调整 Prefill 和 Decode 实例数量，以满足 TTFT（Time To First Token）和 ITL（Inter-Token Latency）的 SLA 目标。

## 特性

### 核心能力

- ✅ **SLA 驱动扩缩容**：基于 TTFT/ITL 目标自动调整实例数
- ✅ **负载预测**：支持多种预测算法（常量、移动平均、ARIMA、Prophet）
- ✅ **性能插值**：基于预分析数据或在线学习估算性能
- ✅ **校正因子**：实时调整预测以适应实际性能偏差
- ✅ **GPU 预算管理**：在 GPU 预算约束下优化资源分配
- ✅ **优雅扩缩容**：平滑添加/移除实例，不中断现有请求

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                   ControlPlaneManager                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Scheduling │  │ PD Routing │  │  Executor  │            │
│  │  Policies  │  │  Strategy  │  │ Coordinator│            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         │                │               │                  │
│         └────────────────┴───────────────┘                  │
│                        │                                    │
│             ┌──────────▼──────────┐                        │
│             │    Autoscaler       │                        │
│             │  (SLA-based)        │                        │
│             └──────────┬──────────┘                        │
│                        │                                    │
│      ┌─────────────────┼─────────────────┐                │
│      │                 │                 │                 │
│  ┌───▼───┐      ┌──────▼──────┐   ┌─────▼─────┐          │
│  │ Load  │      │Performance  │   │  Metrics  │          │
│  │Predict│      │Interpolator │   │ Collector │          │
│  └───────┘      └─────────────┘   └───────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 基本配置

```python
from control_plane import (
    ControlPlaneManager,
    AutoscalerConfig,
    PDSeparationConfig,
)

# 配置 Autoscaler
autoscaler_config = AutoscalerConfig(
    # SLA 目标
    target_ttft_ms=200.0,      # 目标 TTFT: 200ms
    target_itl_ms=50.0,        # 目标 ITL: 50ms
    
    # 扩缩容参数
    adjustment_interval_sec=60,  # 每60秒调整一次
    min_prefill_instances=1,
    max_prefill_instances=5,
    min_decode_instances=2,
    max_decode_instances=10,
    
    # GPU 预算
    max_gpu_budget=24,
    prefill_gpus_per_instance=4,
    decode_gpus_per_instance=1,
    
    # 负载预测
    load_predictor_type="moving_average",
)

# 创建 Control Plane
manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    enable_auto_scaling=True,      # 启用自动扩缩容
    autoscaler_config=autoscaler_config,
    enable_pd_separation=True,
    pd_config=PDSeparationConfig(),
)
```

### 2. 启动并监控

```python
import asyncio

async def main():
    # 启动 Control Plane（自动启动 Autoscaler）
    await manager.start()
    
    # 注册初始实例
    manager.register_instance(prefill_instance)
    manager.register_instance(decode_instance_1)
    manager.register_instance(decode_instance_2)
    
    # 运行时监控
    while True:
        await asyncio.sleep(60)
        
        # 获取 Autoscaler 状态
        status = manager.get_autoscaler_status()
        print(f"Autoscaler running: {status['running']}")
        print(f"Correction factors: {status['correction_factors']}")
        
        # 获取实例数量
        instances = manager.get_instances()
        prefill_count = sum(1 for i in instances 
                          if i.instance_type == ExecutionInstanceType.PREFILLING)
        decode_count = sum(1 for i in instances 
                         if i.instance_type == ExecutionInstanceType.DECODING)
        print(f"Instances: Prefill={prefill_count}, Decode={decode_count}")

asyncio.run(main())
```

## 配置详解

### AutoscalerConfig 参数

#### SLA 目标

```python
target_ttft_ms: float = 200.0      # TTFT 目标（毫秒）
target_itl_ms: float = 50.0        # ITL 目标（毫秒）
```

#### 扩缩容参数

```python
adjustment_interval_sec: int = 60  # 调整间隔（秒）
min_prefill_instances: int = 1     # 最小 Prefill 实例数
max_prefill_instances: int = 10    # 最大 Prefill 实例数
min_decode_instances: int = 1      # 最小 Decode 实例数
max_decode_instances: int = 20     # 最大 Decode 实例数
```

#### GPU 预算

```python
max_gpu_budget: int = 32           # 总 GPU 预算
prefill_gpus_per_instance: int = 4 # 每个 Prefill 实例的 GPU 数
decode_gpus_per_instance: int = 1  # 每个 Decode 实例的 GPU 数
```

#### 负载预测

```python
load_predictor_type: str = "constant"  # 预测器类型
# 支持：
#   - "constant": 常量预测（假设下一周期与当前相同）
#   - "moving_average": 移动平均
#   - "exponential_smoothing": 指数平滑
#   - "arima": ARIMA 时间序列（需要 statsmodels）
#   - "prophet": Facebook Prophet（需要 prophet）

prediction_window_size: int = 10   # 历史窗口大小
```

#### 性能插值

```python
profile_data_dir: Optional[str] = None  # 预分析数据目录
enable_online_learning: bool = True     # 启用在线学习
```

#### 其他选项

```python
enable_correction: bool = True     # 启用校正因子
no_operation: bool = False         # Dry-run 模式（仅输出决策）
```

## 工作原理

### 扩缩容流程

```
每个调整周期（例如60秒）：

1. 收集指标
   ├─ 从 Prometheus 或直接从实例收集
   ├─ 请求数、ISL、OSL、TTFT、ITL
   └─ 当前实例数量

2. 更新负载预测器
   └─ 添加最新观测数据

3. 计算校正因子
   ├─ Prefill: actual_TTFT / expected_TTFT
   └─ Decode: actual_ITL / expected_ITL

4. 预测未来负载
   ├─ 下一周期的请求数
   ├─ 平均 ISL（输入序列长度）
   └─ 平均 OSL（输出序列长度）

5. 计算所需实例数
   ├─ Prefill: throughput_needed / throughput_per_instance
   └─ Decode: throughput_needed / throughput_per_instance
   
6. 应用约束
   ├─ 最小/最大实例数
   └─ GPU 预算限制

7. 执行扩缩容
   ├─ 扩容：创建新实例
   └─ 缩容：优雅移除实例
```

### 副本数计算

#### Prefill 副本数

```python
# 预测 Prefill 吞吐量需求
predicted_throughput = (
    next_num_req * next_isl / interval 
    * min(1.0, prefill_correction_factor)
)

# 每实例容量
capacity_per_instance = (
    throughput_per_gpu * gpus_per_instance
)

# 所需副本数
num_prefill = ceil(predicted_throughput / capacity_per_instance)
```

#### Decode 副本数

```python
# 调整 ITL 目标
corrected_itl = target_itl / decode_correction_factor

# 找到满足 ITL 的最大吞吐量
capacity_per_gpu = find_best_throughput(itl=corrected_itl, context_length)

# 预测 Decode 吞吐量需求
predicted_throughput = next_num_req * next_osl / interval

# 所需副本数
num_decode = ceil(predicted_throughput / capacity_per_gpu / gpus_per_instance)
```

## 高级用法

### 1. 使用 Prometheus 指标

```python
from control_plane import MetricsCollector

# 配置 Prometheus 数据源
metrics_collector = MetricsCollector(
    prometheus_url="http://localhost:9090",
    namespace="sagellm",
    model_name="llama-2-7b",
)

# Autoscaler 会自动从 Prometheus 收集指标
```

### 2. 自定义性能插值

```python
from control_plane import AutoscalerConfig

# 使用预分析的性能数据
config = AutoscalerConfig(
    profile_data_dir="/path/to/profile_results",
    enable_online_learning=False,  # 禁用在线学习，只用预分析数据
)
```

### 3. Dry-run 模式

```python
# 测试扩缩容决策而不实际执行
config = AutoscalerConfig(
    no_operation=True,  # 只输出决策，不实际扩缩容
)
```

### 4. 自定义负载预测器

```python
# 使用 ARIMA 预测器（需要安装 statsmodels）
config = AutoscalerConfig(
    load_predictor_type="arima",
    prediction_window_size=20,  # 更大的历史窗口
)

# 或使用 Prophet（需要安装 prophet）
config = AutoscalerConfig(
    load_predictor_type="prophet",
    prediction_window_size=30,
)
```

## 监控和调试

### 获取 Autoscaler 状态

```python
status = manager.get_autoscaler_status()
print(status)
# {
#     "enabled": True,
#     "running": True,
#     "config": {
#         "target_ttft_ms": 200.0,
#         "target_itl_ms": 50.0,
#         "adjustment_interval_sec": 60
#     },
#     "correction_factors": {
#         "prefill": 1.05,
#         "decode": 0.98
#     },
#     "last_adjustment_time": 1730198400.0
# }
```

### 查看日志

Autoscaler 会输出详细的日志：

```
INFO - Autoscaler initialized: TTFT target=200.0ms, ITL target=50.0ms
INFO - Starting autoscaling adjustment cycle
INFO - Observed metrics: requests=15.2, isl=1024.5, osl=256.3, ttft=185.3ms, itl=48.2ms
INFO - Predicted load: requests=16.1, isl=1050.2, osl=260.1
DEBUG - Correction factors: prefill=1.023, decode=0.982
DEBUG - Prefill calculation: throughput=275.2 / capacity=200.0 = 2
DEBUG - Decode calculation: throughput=69.8 / capacity=50.0 = 2
INFO - Scaling decision: prefill=2, decode=2
INFO - Autoscaling decision: prefill 2 -> 2 (delta=0), decode 2 -> 2 (delta=0)
```

## 最佳实践

### 1. 选择合适的调整间隔

- **短间隔（30-60秒）**：快速响应负载变化，但可能过于敏感
- **长间隔（2-5分钟）**：更稳定，但响应较慢
- **建议**：生产环境使用 60-120 秒

### 2. 设置合理的 SLA 目标

```python
# 交互式应用
target_ttft_ms=150.0,  # 快速首个 token
target_itl_ms=30.0,    # 低延迟生成

# 批处理应用
target_ttft_ms=500.0,  # 可接受较长 TTFT
target_itl_ms=100.0,   # 优化吞吐而非延迟
```

### 3. GPU 预算规划

```python
# 示例：24 GPU 预算
# - Prefill: 4 GPUs/instance, 最多 3 个实例 = 12 GPUs
# - Decode: 1 GPU/instance, 最多 12 个实例 = 12 GPUs

config = AutoscalerConfig(
    max_gpu_budget=24,
    prefill_gpus_per_instance=4,
    max_prefill_instances=3,
    decode_gpus_per_instance=1,
    max_decode_instances=12,
)
```

### 4. 选择负载预测器

- **Constant**：适合稳定负载
- **Moving Average**：适合平滑短期波动
- **Exponential Smoothing**：适合有缓慢趋势的负载
- **ARIMA**：适合有周期性模式的负载
- **Prophet**：适合复杂季节性模式

## 故障排查

### 问题：Autoscaler 不扩容

**可能原因：**
1. 已达到 `max_prefill_instances` 或 `max_decode_instances`
2. GPU 预算耗尽
3. 指标收集失败（检查 Prometheus 或实例连接）

**解决方法：**
```python
# 检查状态
status = manager.get_autoscaler_status()
print(status)

# 检查实例数量
instances = manager.get_instances()
print(f"Current: Prefill={len([i for i in instances if i.instance_type == PREFILLING])}")
```

### 问题：频繁扩缩容

**可能原因：**
1. 调整间隔太短
2. 负载预测器不稳定
3. 校正因子波动

**解决方法：**
```python
# 增加调整间隔
config = AutoscalerConfig(
    adjustment_interval_sec=120,  # 从 60 增加到 120
)

# 使用更稳定的预测器
config = AutoscalerConfig(
    load_predictor_type="moving_average",
    prediction_window_size=15,  # 增加窗口大小
)
```

## 与 Dynamo 的差异

| 特性 | Dynamo Planner | sageLLM Autoscaler |
|------|----------------|-------------------|
| 核心逻辑 | ✅ 相同 | ✅ 移植 |
| 负载预测 | ✅ ARIMA/Prophet | ✅ 相同 + 简单预测器 |
| 性能插值 | ✅ 预分析数据 | ✅ 预分析 + 在线学习 |
| 部署环境 | Kubernetes | 通用（需自定义） |
| 指标收集 | Prometheus | Prometheus + 直接收集 |
| 实例管理 | K8s API | 回调机制 |

## 参考资料

- [Dynamo SLA Planner 文档](../../dynamo/docs/planner/sla_planner.md)
- [集成方案](../../PD_SEPARATION_INTEGRATION_PLAN.md)
- [Control Plane 文档](./README.md)
- [PD 分离文档](./pd_separation.md)

## 示例代码

完整示例请参考：
- [`example_autoscaling.py`](./example_autoscaling.py) - 基本使用示例
- [`tests/control_plane/test_autoscaler.py`](../tests/control_plane/test_autoscaler.py) - 单元测试

---

**下一步：**
1. 尝试运行 `example_autoscaling.py`
2. 根据实际负载调整配置
3. 集成到生产环境
4. 监控和优化性能
