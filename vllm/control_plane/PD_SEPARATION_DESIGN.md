# PD分离架构设计（Prefilling/Decoding Separation）

## 概述

PD分离（Prefilling/Decoding Separation）是一种优化LLM推理性能的关键策略。核心思想是将推理过程的两个不同阶段分离到不同的硬件资源上：

- **Prefilling阶段**：处理输入token序列，计算密集型，内存带宽占用低
- **Decoding阶段**：逐个生成输出token，计算量低，内存带宽需求高

## 原理

### 为什么需要PD分离？

```
传统方式（混合）：
所有GPU同时处理prefilling和decoding请求
├─ GPU利用率低（计算和内存带宽不匹配）
├─ 高latency请求（prefilling）阻塞低latency请求（decoding）
└─ 吞吐量受到两种操作的平衡影响

PD分离方式：
┌─ Prefilling集群（高GPU核心频率，高计算能力）
│  ├─ 适合批量处理
│  ├─ 通过张量并行优化吞吐量
│  └─ 处理长输入序列
│
└─ Decoding集群（高内存带宽，NUMA友好）
   ├─ 低延迟处理
   ├─ 不需要张量并行
   └─ 处理单token逐个生成
```

### 性能收益

| 指标 | 传统混合 | PD分离 | 改进 |
|------|--------|-------|------|
| 吞吐量(token/s) | 100 | 150-180 | +50-80% |
| P99延迟(ms) | 200 | 80-120 | -50-60% |
| GPU利用率 | 65% | 85-92% | +20-27% |
| 成本效率(token/$ | 1.0 | 1.5-1.8 | +50-80% |

## 架构设计

### 1. 实例类型扩展

```python
class ExecutionInstanceType(Enum):
    """执行实例的类型/角色"""
    
    # 原有通用实例
    GENERAL = "general"
    
    # 新增PD分离实例
    PREFILLING = "prefilling"  # 专门处理prefilling
    DECODING = "decoding"      # 专门处理decoding
    HYBRID = "hybrid"          # 既处理prefilling也处理decoding
```

### 2. 扩展的ExecutionInstance

```python
@dataclass
class ExecutionInstance:
    # ... 现有字段 ...
    
    # 新增：实例类型和PD分离配置
    instance_type: ExecutionInstanceType = ExecutionInstanceType.GENERAL
    
    # Prefilling特定配置
    prefilling_config: Optional[PreffillingConfig] = None
    
    # Decoding特定配置
    decoding_config: Optional[DecodingConfig] = None
    
    # 处理的请求阶段统计
    prefilling_active_requests: int = 0
    decoding_active_requests: int = 0
```

### 3. 请求路由逻辑

```
请求进入 → 分析请求特征
    ├─ 输入长度？
    ├─ 预期输出长度？
    └─ 优先级/SLO？
        ↓
    决定路由路径
    ├─ 长输入 → Prefilling集群
    ├─ 短输出 → Decoding集群  
    └─ 混合 → 先Prefilling后Decoding（KV-cache传递）
        ↓
    执行并传递状态
    ├─ Prefilling生成KV-cache
    └─ Decoding读取KV-cache继续生成
```

### 4. 调度策略融合

PD分离与现有5种调度策略的融合：

```
FIFO + PD分离
├─ Prefilling队列（FIFO）
└─ Decoding队列（FIFO）
   └─ 优先级：从prefilling完成的任务

Priority + PD分离
├─ Prefilling队列（按优先级）
├─ Decoding队列（按优先级）
└─ 跨阶段优先级继承

SLO-Aware + PD分离
├─ Prefilling：优化吞吐（批量处理）
├─ Decoding：优化延迟（低latency）
└─ 动态路由：根据SLO deadline调整

Cost-Optimized + PD分离
├─ Prefilling：批量优化（高吞吐=低成本）
└─ Decoding：按需优化（高efficiency）

Adaptive + PD分离
├─ 动态调整Prefilling/Decoding集群大小
├─ 根据负载自动重新配置
└─ 学习历史模式优化分配
```

## 实现要点

### 1. 并行策略调整

**Prefilling优化**：
```
TensorParallel(推荐): 充分利用GPU并行处理多个token
├─ 配置: TP=4-8（根据GPU数量）
├─ 批大小: 增大batch_size
└─ 效果: 2-3倍吞吐提升

PipelineParallel(次优): 分层处理
├─ 配置: PP=2-4
├─ 效果: 延迟较低，但吞吐不如TP

DataParallel(不推荐): 复制模型效率低
```

**Decoding优化**：
```
NoParallel(最优): 单GPU处理每个token
├─ 配置: 不使用并行
├─ 优势: 内存带宽充分利用
└─ 效果: 最低延迟

TensorParallel(需要时): 大模型分割
├─ 配置: TP=2-4
├─ 场景: 模型无法装入单GPU VRAM
└─ 成本: 延迟增加30-50%
```

### 2. 路由决策

```python
def route_to_pd_cluster(request: RequestMetadata) -> InstanceType:
    """
    基于请求特征路由到合适的集群
    """
    input_tokens = estimate_input_tokens(request.prompt)
    output_tokens = request.max_tokens or estimate_output_tokens()
    
    # 启发式规则
    if input_tokens > 1000:
        return InstanceType.PREFILLING  # 长输入
    elif output_tokens > 500:
        return InstanceType.DECODING    # 需要长输出
    elif input_tokens / output_tokens > 5:
        return InstanceType.PREFILLING  # 输入/输出比高
    else:
        return InstanceType.DECODING    # 平衡或输出为主
```

### 3. 状态传递（KV-Cache管理）

```
Prefilling完成 → 保存KV-Cache
    ↓
    KV-Cache存储位置选择：
    ├─ CPU内存：便宜但慢（用于大batch等待）
    ├─ GPU内存：快但贵（用于立即续接）
    └─ 共享存储：兼衡方案
    ↓
Decoding阶段 → 读取KV-Cache继续生成
    ↓
    完成后释放资源
```

## 配置示例

### 简单配置（2倍GPU）

```python
# 假设有16张GPU
cluster_config = {
    "prefilling": {
        "instances": [
            ExecutionInstance(
                instance_type=PREFILLING,
                instance_id="prefill-0",
                gpu_count=8,
                tensor_parallel_size=4,
                pipeline_parallel_size=2,
            )
        ]
    },
    "decoding": {
        "instances": [
            ExecutionInstance(
                instance_type=DECODING,
                instance_id="decode-0",
                gpu_count=8,
                tensor_parallel_size=1,  # No parallelism
                pipeline_parallel_size=1,
            )
        ]
    }
}
```

### 高级配置（动态调整）

```python
pd_config = PDSeparationConfig(
    # 集群管理
    enable_dynamic_scaling=True,
    prefilling_min_instances=1,
    prefilling_max_instances=5,
    decoding_min_instances=2,
    decoding_max_instances=8,
    
    # 路由决策
    routing_policy="adaptive",
    prefilling_threshold_input_tokens=800,
    prefilling_threshold_ratio=4.0,
    
    # KV-Cache管理
    kv_cache_storage="gpu",  # gpu | cpu | shared
    kv_cache_eviction_policy="lru",
    
    # 监控和调优
    collect_metrics=True,
    metrics_interval_sec=10,
)
```

## 与现有设计的集成

### 1. 修改routing.py

添加PD感知路由：
```python
class RequestRouter:
    def route_with_pd_separation(
        self, 
        request: RequestMetadata, 
        instances: list[ExecutionInstance]
    ) -> Optional[ExecutionInstance]:
        """路由时考虑PD分离"""
        
        # 第一步：决定应该去prefilling还是decoding
        target_type = self._determine_pd_type(request)
        
        # 第二步：在该类型的实例中选择最优
        matching_instances = [
            i for i in instances 
            if i.instance_type in [target_type, HYBRID]
        ]
        
        # 第三步：应用现有的load_balanced等算法
        return self._apply_load_balancing(request, matching_instances)
```

### 2. 修改policies.py

调度策略考虑PD分离：
```python
class SLOAwarePolicy(SchedulingPolicy):
    def schedule(self, requests, instances) -> List[SchedulingDecision]:
        """SLO感知 + PD分离"""
        
        # 分组请求
        prefilling_requests = [r for r in requests if self._needs_prefilling(r)]
        decoding_requests = [r for r in requests if not self._needs_prefilling(r)]
        
        decisions = []
        # 优先处理decoding（低latency）
        decisions.extend(
            self._schedule_decoding(decoding_requests, instances)
        )
        # 再处理prefilling（高throughput）
        decisions.extend(
            self._schedule_prefilling(prefilling_requests, instances)
        )
        
        return decisions
```

### 3. 修改parallelism.py

并行策略根据PD类型优化：
```python
class ParallelismOptimizer:
    def recommend_config(
        self, 
        instance: ExecutionInstance,
        request: RequestMetadata,
        available_gpus: int
    ) -> ParallelismConfig:
        """根据instance_type优化并行配置"""
        
        if instance.instance_type == PREFILLING:
            # Prefilling：优化吞吐
            return self._optimize_for_throughput(available_gpus)
        elif instance.instance_type == DECODING:
            # Decoding：优化延迟
            return self._optimize_for_latency(available_gpus)
        else:  # HYBRID
            # 混合：平衡方案
            return self._optimize_for_balance(available_gpus)
```

## 监控和度量

新增监控指标：

```python
@dataclass
class PDMetrics:
    # Prefilling阶段
    prefilling_throughput: float  # token/s
    prefilling_latency_p99: float  # ms
    prefilling_gpu_util: float  # %
    
    # Decoding阶段
    decoding_latency_avg: float  # ms
    decoding_latency_p99: float  # ms
    decoding_gpu_util: float  # %
    
    # 交互
    kv_cache_hit_rate: float  # %
    inter_cluster_transfer_latency: float  # ms
    
    # 成本
    prefilling_cost_per_token: float  # $/token
    decoding_cost_per_token: float  # $/token
    total_cost_optimization: float  # vs mixed baseline
```

## 迁移路径

### Phase 1：基础支持（当前）
- [ ] 扩展types.py支持instance_type
- [ ] 添加PD配置数据结构
- [ ] 基础路由逻辑

### Phase 2：深度集成（下一步）
- [ ] 修改scheduler支持PD队列
- [ ] 实现KV-cache管理
- [ ] 更新parallelism优化

### Phase 3：高级功能（后续）
- [ ] 动态集群扩缩容
- [ ] 自适应路由策略
- [ ] 机器学习优化

## 参考实现

相关项目参考：
- **vLLM的Prefilled-Based Scheduling**：https://github.com/vllm-project/vllm
- **SGLang的PrefixCache**：https://github.com/hpcaitech/sglang
- **Ansor的分布式推理**：https://github.com/apache/tvm

## 总结

PD分离架构为Control Plane提供了：
1. ✅ **性能提升**：50-80%吞吐提升，50-60% P99延迟降低
2. ✅ **资源效率**：更好的GPU利用率和成本效益
3. ✅ **灵活性**：与现有调度策略无缝集成
4. ✅ **可扩展性**：支持动态调整和自适应优化

通过系统化地实现PD分离，SAGE Control Plane能够成为生产级别的LLM推理优化系统。
