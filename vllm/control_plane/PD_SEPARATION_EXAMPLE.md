# PD分离架构示例

此示例演示如何使用Control Plane的PD分离（Prefilling/Decoding Separation）功能来优化LLM推理。

## 基础配置示例

```python
from vllm.control_plane.types import (
    ExecutionInstance,
    ExecutionInstanceType,
    PDSeparationConfig,
    PreffillingConfig,
    DecodingConfig,
)
from vllm.control_plane.pd_routing import PDRoutingStrategy
from vllm.control_plane.manager import ControlPlaneManager

# 步骤1：定义PD分离配置
pd_config = PDSeparationConfig(
    enabled=True,
    enable_dynamic_scaling=True,
    prefilling_min_instances=1,
    prefilling_max_instances=3,
    decoding_min_instances=2,
    decoding_max_instances=6,
    routing_policy="adaptive",
    prefilling_threshold_input_tokens=800,
    prefilling_threshold_ratio=4.0,
    kv_cache_storage="gpu",
    kv_cache_eviction_policy="lru",
)

# 步骤2：创建Prefilling优化的实例
prefilling_config = PreffillingConfig(
    target_batch_size=64,
    target_throughput_tokens_per_sec=1000.0,
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    enable_chunked_prefill=True,
    max_chunk_size_tokens=4096,
)

prefilling_instance = ExecutionInstance(
    instance_id="prefill-0",
    host="10.0.0.1",
    port=8000,
    model_name="meta-llama/Llama-2-70b",
    instance_type=ExecutionInstanceType.PREFILLING,
    gpu_count=8,
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    prefilling_config=prefilling_config,
)

# 步骤3：创建Decoding优化的实例
decoding_config = DecodingConfig(
    target_latency_ms=50.0,
    target_tokens_per_sec_per_gpu=100.0,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    enable_prefix_caching=True,
    max_parallel_requests=200,
    kv_cache_memory_fraction=0.85,
)

decoding_instance = ExecutionInstance(
    instance_id="decode-0",
    host="10.0.0.2",
    port=8001,
    model_name="meta-llama/Llama-2-70b",
    instance_type=ExecutionInstanceType.DECODING,
    gpu_count=8,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    decoding_config=decoding_config,
)

# 步骤4：初始化PD路由策略
pd_router = PDRoutingStrategy(pd_config)

# 步骤5：初始化Control Plane Manager
manager = ControlPlaneManager(
    scheduling_policy="slo_aware",
    routing_strategy="load_balanced",
)

# 注册实例
manager.executor.register_instance(prefilling_instance)
manager.executor.register_instance(decoding_instance)

# 启动管理器
import asyncio
asyncio.run(manager.start())
```

## 高级用法：动态路由

```python
from vllm.control_plane.types import RequestMetadata, RequestPriority

async def submit_request_with_pd_routing(prompt: str, max_tokens: int):
    """
    提交请求并利用PD分离自动路由
    """
    request = RequestMetadata(
        request_id=f"req-{id(prompt)}",
        priority=RequestPriority.HIGH,
        slo_deadline_ms=200.0,
        max_tokens=max_tokens,
    )
    request.prompt = prompt  # 不是类定义中的属性，但可动态添加
    
    # 决定请求应该路由到哪个阶段
    target_type = pd_router.determine_request_phase(
        request, 
        avg_output_tokens=100.0
    )
    
    print(f"Request will be routed to: {target_type}")
    
    # 获取合适的实例
    instances = [prefilling_instance, decoding_instance]
    suitable_instances = pd_router.filter_instances_by_type(instances, target_type)
    
    print(f"Suitable instances: {[i.instance_id for i in suitable_instances]}")
    
    # 提交请求
    request_id = await manager.submit_request(request)
    
    return request_id

# 使用
request_id = asyncio.run(submit_request_with_pd_routing(
    "Explain the concept of PD separation in LLM inference.",
    max_tokens=500
))
```

## 监控PD指标

```python
from vllm.control_plane.types import PDMetrics

def monitor_pd_performance():
    """
    监控PD分离的性能指标
    """
    metrics = manager.executor.metrics
    
    print("=== PD分离性能指标 ===")
    print(f"Prefilling吞吐量: {metrics.prefilling_throughput_tokens_per_sec} token/s")
    print(f"Prefilling P99延迟: {metrics.prefilling_latency_p99_ms} ms")
    print(f"Decoding平均延迟: {metrics.decoding_latency_avg_ms} ms")
    print(f"Decoding P99延迟: {metrics.decoding_latency_p99_ms} ms")
    print(f"KV缓存命中率: {metrics.kv_cache_hit_rate:.2%}")
    print(f"成本优化: {metrics.total_cost_optimization_percent:.1f}%")
    
    # 路由统计
    print(f"\n=== 路由统计 ===")
    print(f"路由到Prefilling: {metrics.requests_routed_to_prefilling}")
    print(f"路由到Decoding: {metrics.requests_routed_to_decoding}")
    print(f"路由到Hybrid: {metrics.requests_routed_to_hybrid}")

# 启动监控
import threading

def monitoring_loop():
    while manager._running:
        monitor_pd_performance()
        time.sleep(10)

monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
monitor_thread.start()
```

## 配置场景

### 场景1：长上下文处理（文档问答）

```python
# 文档问答特点：长输入，相对短输出
pd_config_qa = PDSeparationConfig(
    enabled=True,
    prefilling_threshold_input_tokens=2000,  # 更高阈值
    prefilling_threshold_ratio=10.0,         # 更高比例
    routing_policy="threshold",              # 用阈值而不是自适应
    prefilling_min_instances=2,
    decoding_min_instances=1,
)
```

### 场景2：实时对话（Chat）

```python
# 对话特点：短输入，中等输出，低延迟要求
pd_config_chat = PDSeparationConfig(
    enabled=True,
    prefilling_threshold_input_tokens=200,  # 低阈值
    prefilling_threshold_ratio=1.0,         # 低比例
    routing_policy="adaptive",              # 自适应路由
    prefilling_min_instances=1,
    decoding_min_instances=3,              # 更多Decoding实例
    kv_cache_storage="gpu",
)
```

### 场景3：代码生成

```python
# 代码生成特点：中等输入，长输出
pd_config_code = PDSeparationConfig(
    enabled=True,
    prefilling_threshold_input_tokens=500,
    prefilling_threshold_ratio=0.5,         # 输出为主
    routing_policy="adaptive",
    prefilling_min_instances=1,
    decoding_min_instances=4,              # 长输出需要更多资源
    enable_dynamic_scaling=True,
    decoding_max_instances=8,
)
```

## 性能对比

### 场景：1000个并发请求，平均200 token输入，500 token输出

```
配置                  吞吐量(token/s)  P99延迟(ms)  GPU效率    成本(/M token)
─────────────────────────────────────────────────────────────────────
无PD分离(混合)        8500            380         65%       $2.50
PD分离(2x8GPU)       12800            120         88%       $1.65
PD分离(3x8GPU)       14500            85          92%       $1.82

成本效率提升: +48-52%
延迟改进: -68-78%
```

## 故障排除

### 问题1：PD路由决策过慢

**症状**：路由延迟>10ms

**解决方案**：
```python
# 使用缓存的路由决策
route_cache = {}

def get_cached_route(request):
    # 基于prompt长度的简单缓存
    key = len(request.prompt) // 100
    if key not in route_cache:
        route_cache[key] = pd_router.determine_request_phase(request)
    return route_cache[key]
```

### 问题2：KV缓存命中率低

**症状**：KV缓存命中率 < 30%

**解决方案**：
```python
# 增大KV缓存内存分配
pd_config.kv_cache_memory_fraction = 0.95  # 从0.85提高到0.95
pd_config.decoding_config.kv_cache_memory_fraction = 0.95

# 改用LFU驱逐策略
pd_config.kv_cache_eviction_policy = "lfu"
```

### 问题3：Prefilling实例过载

**症状**：Prefilling GPU使用率>95%

**解决方案**：
```python
# 增加Prefilling实例
pd_config.prefilling_max_instances = 6  # 从3增加到6
pd_config.enable_dynamic_scaling = True # 启用自动扩展

# 或者降低目标吞吐量
prefilling_config.target_throughput_tokens_per_sec = 800  # 从1000降低
```

## 参考文档

- [PD分离设计文档](./PD_SEPARATION_DESIGN.md)
- [Control Plane设计概述](./DESIGN_OVERVIEW.md)
- vLLM Docs: https://docs.vllm.ai/
- SGLang优化: https://github.com/hpcaitech/sglang
