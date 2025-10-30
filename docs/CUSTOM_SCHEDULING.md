# 自定义调度策略开发指南

本文档详细介绍如何在 sageLLM Control Plane 中开发自定义调度策略。

## 目录

- [调度策略架构](#调度策略架构)
- [快速开始](#快速开始)
- [策略接口规范](#策略接口规范)
- [内置策略详解](#内置策略详解)
- [高级特性](#高级特性)
- [最佳实践](#最佳实践)
- [调试和测试](#调试和测试)

## 调度策略架构

### 核心概念

sageLLM 的调度策略采用**策略模式**设计，所有策略都继承自抽象基类 `SchedulingPolicy`：

```python
from abc import ABC, abstractmethod
from typing import Optional, List
from control_plane.types import RequestMetadata, ExecutionInstance

class SchedulingPolicy(ABC):
    """调度策略抽象基类"""
    
    @abstractmethod
    def get_next_request(
        self,
        pending_queue: List[RequestMetadata],
        available_instances: List[ExecutionInstance]
    ) -> Optional[RequestMetadata]:
        """
        从待处理队列中选择下一个要执行的请求
        
        Args:
            pending_queue: 待处理请求队列（已按提交时间排序）
            available_instances: 当前可用的执行实例列表
            
        Returns:
            选中的请求，如果没有合适的请求则返回 None
        """
        pass
```

### 调度流程

```
用户请求 → Manager.submit_request() 
          ↓
    pending_queue (deque)
          ↓
    _scheduling_loop() 每100ms执行一次
          ↓
    policy.get_next_request(pending_queue, available_instances)
          ↓
    router.select_instance(request, instances)
          ↓
    executor.execute_request(request, instance)
          ↓
    running_requests (dict)
```

## 快速开始

### 创建最简单的策略

创建文件 `control_plane/strategies/my_policy.py`：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

from typing import Optional, List
from control_plane.strategies import SchedulingPolicy
from control_plane.types import RequestMetadata, ExecutionInstance

class MyCustomPolicy(SchedulingPolicy):
    """我的自定义调度策略"""
    
    def __init__(self):
        super().__init__()
        self.request_count = 0  # 示例：跟踪处理的请求数
    
    def get_next_request(
        self,
        pending_queue: List[RequestMetadata],
        available_instances: List[ExecutionInstance]
    ) -> Optional[RequestMetadata]:
        """选择队列中的第一个请求（FIFO）"""
        if not pending_queue:
            return None
        
        if not available_instances:
            return None  # 没有可用实例，等待
        
        # 简单策略：返回队列头部请求
        request = pending_queue[0]
        self.request_count += 1
        return request
```

### 使用自定义策略

```python
from control_plane.manager import ControlPlaneManager
from control_plane.strategies.my_policy import MyCustomPolicy

# 创建 Control Plane 并指定策略
manager = ControlPlaneManager(scheduling_policy=MyCustomPolicy())

# 注册实例并提交请求
await manager.register_instance(instance)
await manager.submit_request(request)
```

## 策略接口规范

### 必需实现的方法

#### `get_next_request()`

**签名**：
```python
def get_next_request(
    self,
    pending_queue: List[RequestMetadata],
    available_instances: List[ExecutionInstance]
) -> Optional[RequestMetadata]:
```

**职责**：
- 从 `pending_queue` 中选择一个请求
- 考虑 `available_instances` 的能力和负载
- 返回选中的请求，或 `None` 表示暂不调度

**约束**：
- **不要修改** `pending_queue` 和 `available_instances`（它们是只读的）
- Manager 会自动从队列中移除返回的请求
- 应尽量快速返回（调度循环是 100ms）

**可用的请求信息**（`RequestMetadata` 字段）：

```python
@dataclass
class RequestMetadata:
    request_id: str              # 唯一标识符
    prompt: str                  # 输入提示
    max_tokens: int              # 最大生成长度
    model_name: str              # 模型名称
    priority: RequestPriority    # 优先级（CRITICAL/HIGH/NORMAL/LOW/BACKGROUND）
    slo_deadline_ms: Optional[int]  # SLO 截止时间（毫秒）
    user_id: Optional[str]       # 用户ID（用于亲和性路由）
    submitted_at: float          # 提交时间戳
    retry_count: int = 0         # 重试次数
    failed_instances: List[str] = field(default_factory=list)  # 失败过的实例
```

**可用的实例信息**（`ExecutionInstance` 字段）：

```python
@dataclass
class ExecutionInstance:
    instance_id: str             # 实例唯一ID
    host: str                    # 主机地址
    port: int                    # 端口号
    model_name: str              # 部署的模型名称
    tensor_parallel_size: int    # TP 并行度
    pipeline_parallel_size: int  # PP 并行度
    gpu_count: int               # GPU 数量
    available: bool              # 是否可用
    
    # 拓扑信息（可选）
    machine_id: Optional[str]    # 机器ID
    gpu_bus_id: Optional[str]    # GPU 总线ID
    nvlink_peers: List[str]      # NVLINK 对等实例
    numa_node: Optional[int]     # NUMA 节点
    
    # 负载信息（由 executor 更新）
    current_load: int = 0        # 当前负载（运行中请求数）
    max_concurrency: int = 100   # 最大并发数
    avg_latency_ms: float = 0.0  # 平均延迟
```

### 可选实现的方法

策略可以覆写以下方法以支持高级功能：

#### `on_request_completed()`

请求完成时的回调：

```python
def on_request_completed(self, request: RequestMetadata, latency_ms: float):
    """请求完成时调用（可选）
    
    Args:
        request: 完成的请求
        latency_ms: 实际延迟
    """
    # 示例：更新统计信息
    self.total_requests += 1
    self.avg_latency = (self.avg_latency * (self.total_requests - 1) + latency_ms) / self.total_requests
```

#### `on_instance_registered()`

实例注册时的回调：

```python
def on_instance_registered(self, instance: ExecutionInstance):
    """新实例注册时调用（可选）"""
    # 示例：初始化实例相关的状态
    self.instance_stats[instance.instance_id] = {"requests": 0}
```

## 内置策略详解

### 1. FIFOPolicy（先进先出）

**适用场景**：简单公平调度，无特殊优先级需求

```python
class FIFOPolicy(SchedulingPolicy):
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        return pending_queue[0]  # 直接返回队首
```

**优点**：
- 实现简单
- 公平性好
- 延迟可预测

**缺点**：
- 忽略请求优先级
- 忽略 SLO 约束
- 无负载均衡

### 2. PriorityPolicy（优先级调度）

**适用场景**：有明确优先级分级的系统

```python
class PriorityPolicy(SchedulingPolicy):
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        # 按优先级排序（CRITICAL > HIGH > NORMAL > LOW > BACKGROUND）
        sorted_queue = sorted(
            pending_queue,
            key=lambda r: (r.priority.value, r.submitted_at)
        )
        return sorted_queue[0]
```

**优先级定义**（`RequestPriority` 枚举）：
```python
class RequestPriority(Enum):
    CRITICAL = 5    # 关键请求（如生产故障诊断）
    HIGH = 4        # 高优先级（如付费用户）
    NORMAL = 3      # 正常优先级（默认）
    LOW = 2         # 低优先级（如批处理）
    BACKGROUND = 1  # 后台任务（如预热缓存）
```

**优点**：
- 保障关键请求
- 简单直观

**缺点**：
- 低优先级请求可能饥饿
- 无 SLO 保障

### 3. SLOAwarePolicy（SLO 感知调度）

**适用场景**：有严格延迟要求的生产系统

```python
class SLOAwarePolicy(SchedulingPolicy):
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        now = time.time() * 1000  # 当前时间（毫秒）
        
        # 计算每个请求的紧迫度
        def urgency(request):
            if request.slo_deadline_ms is None:
                return float('inf')  # 无 SLO 的请求优先级最低
            
            time_remaining = request.slo_deadline_ms - (now - request.submitted_at)
            return time_remaining  # 剩余时间越少越紧迫
        
        sorted_queue = sorted(pending_queue, key=urgency)
        return sorted_queue[0]
```

**SLO 计算示例**：
```python
# 用户提交请求时指定 SLO
request = RequestMetadata(
    request_id="req-123",
    prompt="Translate to English: ...",
    max_tokens=100,
    slo_deadline_ms=5000,  # 5秒内必须完成
    priority=RequestPriority.HIGH
)
```

**优点**：
- 优先保障 SLO 即将违反的请求
- 减少 SLO 违反率

**缺点**：
- 可能牺牲无 SLO 请求
- 需要准确的延迟预测

### 4. CostOptimizedPolicy（成本优化调度）

**适用场景**：多实例类型环境，希望降低成本

```python
class CostOptimizedPolicy(SchedulingPolicy):
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        # 优先选择能满足 SLO 的最便宜实例
        for request in sorted(pending_queue, key=lambda r: r.priority.value, reverse=True):
            # 计算每个实例的成本（假设已配置）
            suitable_instances = [
                inst for inst in available_instances
                if self._can_meet_slo(request, inst)
            ]
            
            if suitable_instances:
                # 选择最便宜的实例（通过某种成本模型）
                cheapest = min(suitable_instances, key=lambda i: i.cost_per_token)
                return request
        
        return None
```

**优点**：
- 降低运营成本
- 满足 SLO 约束

**缺点**：
- 实现复杂
- 需要准确的成本模型

### 5. AdaptivePolicy（自适应调度）

**适用场景**：负载模式动态变化的系统

```python
class AdaptivePolicy(SchedulingPolicy):
    def __init__(self):
        self.policies = {
            "fifo": FIFOPolicy(),
            "priority": PriorityPolicy(),
            "slo": SLOAwarePolicy()
        }
        self.current_policy = "fifo"
        self.request_count = 0
    
    def get_next_request(self, pending_queue, available_instances):
        # 每1000个请求检查一次负载模式
        if self.request_count % 1000 == 0:
            self._adapt_policy(pending_queue)
        
        self.request_count += 1
        return self.policies[self.current_policy].get_next_request(
            pending_queue, available_instances
        )
    
    def _adapt_policy(self, pending_queue):
        """根据队列状态切换策略"""
        slo_requests = sum(1 for r in pending_queue if r.slo_deadline_ms)
        high_priority = sum(1 for r in pending_queue if r.priority.value >= 4)
        
        if slo_requests > len(pending_queue) * 0.5:
            self.current_policy = "slo"  # 大量 SLO 请求 → SLO 优先
        elif high_priority > len(pending_queue) * 0.3:
            self.current_policy = "priority"  # 大量高优先级 → 优先级调度
        else:
            self.current_policy = "fifo"  # 默认 FIFO
```

**优点**：
- 自动适应负载变化
- 综合多种策略优势

**缺点**：
- 复杂度高
- 策略切换可能不平滑

## 高级特性

### 1. 利用拓扑信息优化调度

访问实例的拓扑信息以优化跨机调度：

```python
class TopologyAwarePolicy(SchedulingPolicy):
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        request = pending_queue[0]
        
        # 优先选择同一机器上的实例（减少网络延迟）
        local_instances = [
            inst for inst in available_instances
            if inst.machine_id == self.preferred_machine
        ]
        
        if local_instances:
            # 进一步优先 NVLINK 连接的实例
            nvlink_instances = [
                inst for inst in local_instances
                if self.last_instance_id in inst.nvlink_peers
            ]
            if nvlink_instances:
                self.last_instance_id = nvlink_instances[0].instance_id
        
        return request
```

**可用拓扑字段**：
- `machine_id`: 机器标识符
- `rack_id`: 机架标识符
- `gpu_bus_id`: GPU 总线 ID
- `nvlink_peers`: NVLINK 连接的其他实例
- `numa_node`: NUMA 节点编号

### 2. 集成监控指标

利用 `MetricsCollector` 的数据做智能决策：

```python
class MetricsAwarePolicy(SchedulingPolicy):
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
    
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue:
            return None
        
        # 获取实例性能指标
        instance_metrics = self.metrics.get_all_instance_metrics()
        
        # 过滤高延迟实例
        healthy_instances = [
            inst for inst in available_instances
            if instance_metrics.get(inst.instance_id, {}).get("avg_latency_ms", 0) < 1000
        ]
        
        if not healthy_instances:
            return None
        
        return pending_queue[0]
```

**可用指标**（通过 `MetricsCollector`）：
- `avg_latency_ms`: 平均延迟
- `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`: 延迟分位数
- `error_rate`: 错误率
- `throughput_rps`: 吞吐量（请求/秒）
- `current_load`: 当前负载

### 3. 请求批处理优化

某些策略可能需要批量调度请求：

```python
class BatchAwarePolicy(SchedulingPolicy):
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.pending_batch = []
    
    def get_next_request(self, pending_queue, available_instances):
        if not available_instances:
            return None
        
        # 累积请求直到达到批次大小
        if len(self.pending_batch) < self.batch_size and pending_queue:
            self.pending_batch.extend(pending_queue[:self.batch_size])
        
        if self.pending_batch:
            request = self.pending_batch.pop(0)
            return request
        
        return None
```

**注意**：当前 vLLM 的 HTTP API 不支持原生批处理，但策略可以将相似请求分组到同一实例以提高缓存命中率。

### 4. 预测性调度

使用历史数据预测请求延迟：

```python
class PredictivePolicy(SchedulingPolicy):
    def __init__(self):
        self.latency_history = {}  # {model_name: {prompt_length: [latencies]}}
    
    def predict_latency(self, request, instance):
        """预测请求在特定实例上的延迟"""
        model = request.model_name
        prompt_len = len(request.prompt)
        
        # 查找类似长度请求的历史延迟
        if model in self.latency_history:
            similar_latencies = [
                lat for pl, lats in self.latency_history[model].items()
                if abs(pl - prompt_len) < 100  # 长度相近
                for lat in lats
            ]
            if similar_latencies:
                return sum(similar_latencies) / len(similar_latencies)
        
        # 默认估算：基于 TP 大小
        base_latency = 100  # ms
        return base_latency / instance.tensor_parallel_size
    
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        # 选择预测延迟最低的请求-实例组合
        best_request = None
        best_latency = float('inf')
        
        for request in pending_queue:
            for instance in available_instances:
                predicted = self.predict_latency(request, instance)
                if predicted < best_latency:
                    best_latency = predicted
                    best_request = request
        
        return best_request
    
    def on_request_completed(self, request, latency_ms):
        """更新历史数据"""
        model = request.model_name
        prompt_len = len(request.prompt)
        
        if model not in self.latency_history:
            self.latency_history[model] = {}
        if prompt_len not in self.latency_history[model]:
            self.latency_history[model][prompt_len] = []
        
        self.latency_history[model][prompt_len].append(latency_ms)
```

## 最佳实践

### 1. 性能优化

**避免 O(n²) 复杂度**：
```python
# ❌ 错误：嵌套循环
for request in pending_queue:
    for instance in available_instances:
        if self.is_compatible(request, instance):
            return request

# ✅ 正确：预先过滤
compatible_instances = [i for i in available_instances if i.model_name == request.model_name]
if compatible_instances:
    return pending_queue[0]
```

**缓存计算结果**：
```python
class CachedPolicy(SchedulingPolicy):
    def __init__(self):
        self._cache = {}
    
    def get_next_request(self, pending_queue, available_instances):
        cache_key = (len(pending_queue), len(available_instances))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._compute_next_request(pending_queue, available_instances)
        self._cache[cache_key] = result
        return result
```

### 2. 容错处理

**检查边界条件**：
```python
def get_next_request(self, pending_queue, available_instances):
    # 检查空队列
    if not pending_queue:
        return None
    
    # 检查无可用实例
    if not available_instances:
        return None
    
    # 检查模型兼容性
    request = pending_queue[0]
    compatible = [i for i in available_instances if i.model_name == request.model_name]
    if not compatible:
        return None  # 等待兼容实例
    
    return request
```

**处理异常数据**：
```python
def get_next_request(self, pending_queue, available_instances):
    # 过滤已失败多次的请求
    valid_requests = [
        r for r in pending_queue
        if r.retry_count < 3  # 最多重试3次
    ]
    
    if not valid_requests:
        return None
    
    return valid_requests[0]
```

### 3. 可观测性

**添加日志和指标**：
```python
import logging

class ObservablePolicy(SchedulingPolicy):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decisions_count = 0
        self.skipped_count = 0
    
    def get_next_request(self, pending_queue, available_instances):
        self.logger.debug(
            f"Scheduling decision: queue_size={len(pending_queue)}, "
            f"available_instances={len(available_instances)}"
        )
        
        if not pending_queue:
            self.skipped_count += 1
            return None
        
        request = pending_queue[0]
        self.decisions_count += 1
        
        self.logger.info(
            f"Selected request {request.request_id} "
            f"(priority={request.priority.name}, "
            f"slo={request.slo_deadline_ms}ms)"
        )
        
        return request
```

### 4. 配置化设计

**支持参数调整**：
```python
class ConfigurablePolicy(SchedulingPolicy):
    def __init__(
        self,
        priority_weight: float = 1.0,
        slo_weight: float = 1.0,
        load_weight: float = 0.5
    ):
        """
        Args:
            priority_weight: 优先级权重
            slo_weight: SLO 紧迫度权重
            load_weight: 负载均衡权重
        """
        self.priority_weight = priority_weight
        self.slo_weight = slo_weight
        self.load_weight = load_weight
    
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        # 计算综合得分
        scored_requests = [
            (self._compute_score(r, available_instances), r)
            for r in pending_queue
        ]
        
        best_score, best_request = max(scored_requests, key=lambda x: x[0])
        return best_request
    
    def _compute_score(self, request, instances):
        score = 0.0
        
        # 优先级分数
        score += request.priority.value * self.priority_weight
        
        # SLO 紧迫度分数
        if request.slo_deadline_ms:
            time_elapsed = time.time() * 1000 - request.submitted_at
            urgency = 1.0 - (time_elapsed / request.slo_deadline_ms)
            score += urgency * self.slo_weight
        
        # 负载均衡分数（实例负载越低分数越高）
        avg_load = sum(i.current_load for i in instances) / len(instances)
        score += (1.0 - avg_load / 100) * self.load_weight
        
        return score
```

## 调试和测试

### 单元测试

创建 `tests/control_plane/test_my_policy.py`：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

import pytest
from control_plane.strategies.my_policy import MyCustomPolicy
from control_plane.types import RequestMetadata, ExecutionInstance, RequestPriority

@pytest.fixture
def policy():
    return MyCustomPolicy()

@pytest.fixture
def sample_requests():
    return [
        RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            max_tokens=10,
            model_name="llama-2-7b",
            priority=RequestPriority.NORMAL,
            submitted_at=1000.0
        ),
        RequestMetadata(
            request_id="req-2",
            prompt="World",
            max_tokens=20,
            model_name="llama-2-7b",
            priority=RequestPriority.HIGH,
            submitted_at=1001.0
        )
    ]

@pytest.fixture
def sample_instances():
    return [
        ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-2-7b",
            tensor_parallel_size=1,
            gpu_count=1
        )
    ]

def test_get_next_request_fifo(policy, sample_requests, sample_instances):
    """测试 FIFO 行为"""
    next_req = policy.get_next_request(sample_requests, sample_instances)
    assert next_req.request_id == "req-1"  # 应返回第一个请求

def test_empty_queue(policy, sample_instances):
    """测试空队列"""
    next_req = policy.get_next_request([], sample_instances)
    assert next_req is None

def test_no_instances(policy, sample_requests):
    """测试无可用实例"""
    next_req = policy.get_next_request(sample_requests, [])
    assert next_req is None
```

运行测试：
```bash
pytest tests/control_plane/test_my_policy.py -v
```

### 集成测试

测试策略在真实 Manager 中的行为：

```python
@pytest.mark.asyncio
async def test_policy_integration(mock_aiohttp, mock_vllm_completion_response):
    """测试策略在 Manager 中的集成"""
    policy = MyCustomPolicy()
    manager = ControlPlaneManager(scheduling_policy=policy)
    
    # 注册实例
    instance = ExecutionInstance(
        instance_id="test-inst",
        host="localhost",
        port=8000,
        model_name="llama-2-7b",
        tensor_parallel_size=1,
        gpu_count=1
    )
    await manager.register_instance(instance)
    
    # Mock vLLM API
    mock_aiohttp.post(
        'http://localhost:8000/v1/completions',
        payload=mock_vllm_completion_response
    )
    
    # 提交请求
    request = RequestMetadata(
        request_id="test-req",
        prompt="Test prompt",
        max_tokens=10,
        model_name="llama-2-7b",
        priority=RequestPriority.NORMAL
    )
    
    await manager.submit_request(request)
    await asyncio.sleep(0.2)  # 等待调度
    
    # 验证请求已执行
    assert request.request_id in manager.running_requests
```

### 性能基准测试

对比不同策略的性能：

```python
import time
from control_plane.strategies.my_policy import MyCustomPolicy
from control_plane.strategies import FIFOPolicy, PriorityPolicy

def benchmark_policy(policy_class, num_requests=1000):
    """测试策略的调度性能"""
    policy = policy_class()
    
    # 生成测试数据
    requests = [
        RequestMetadata(
            request_id=f"req-{i}",
            prompt="test" * 10,
            max_tokens=10,
            model_name="llama-2-7b",
            priority=RequestPriority.NORMAL
        )
        for i in range(num_requests)
    ]
    
    instances = [
        ExecutionInstance(
            instance_id=f"inst-{i}",
            host="localhost",
            port=8000 + i,
            model_name="llama-2-7b",
            tensor_parallel_size=1,
            gpu_count=1
        )
        for i in range(10)
    ]
    
    # 计时
    start = time.time()
    for _ in range(num_requests):
        policy.get_next_request(requests, instances)
    end = time.time()
    
    print(f"{policy_class.__name__}: {end - start:.4f}s for {num_requests} decisions")
    print(f"  Avg latency: {(end - start) / num_requests * 1000:.2f}ms per decision")

# 运行基准测试
benchmark_policy(FIFOPolicy)
benchmark_policy(PriorityPolicy)
benchmark_policy(MyCustomPolicy)
```

### 调试技巧

**启用详细日志**：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("control_plane")
logger.setLevel(logging.DEBUG)
```

**使用断点调试**：
```python
def get_next_request(self, pending_queue, available_instances):
    import pdb; pdb.set_trace()  # 设置断点
    
    # 检查队列状态
    print(f"Queue size: {len(pending_queue)}")
    print(f"First request: {pending_queue[0] if pending_queue else None}")
    
    return pending_queue[0] if pending_queue else None
```

## 示例：完整的生产级策略

以下是一个结合多种特性的生产级策略示例：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

import time
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from control_plane.strategies import SchedulingPolicy
from control_plane.types import RequestMetadata, ExecutionInstance, RequestPriority

@dataclass
class SchedulingDecision:
    """调度决策记录"""
    request_id: str
    instance_id: str
    timestamp: float
    predicted_latency_ms: float
    actual_latency_ms: Optional[float] = None

class ProductionPolicy(SchedulingPolicy):
    """生产级调度策略
    
    特性：
    - SLO 优先：优先保障即将违反 SLO 的请求
    - 负载均衡：避免实例过载
    - 拓扑感知：优先本地实例
    - 自适应：根据历史数据动态调整
    - 可观测：详细日志和指标
    """
    
    def __init__(
        self,
        slo_weight: float = 2.0,
        load_weight: float = 1.0,
        topology_weight: float = 0.5,
        max_load_threshold: float = 0.8
    ):
        self.logger = logging.getLogger(__name__)
        
        # 权重配置
        self.slo_weight = slo_weight
        self.load_weight = load_weight
        self.topology_weight = topology_weight
        self.max_load_threshold = max_load_threshold
        
        # 统计信息
        self.decisions: List[SchedulingDecision] = []
        self.latency_predictions: Dict[str, List[float]] = {}
        
        self.logger.info(
            f"ProductionPolicy initialized with weights: "
            f"slo={slo_weight}, load={load_weight}, topology={topology_weight}"
        )
    
    def get_next_request(
        self,
        pending_queue: List[RequestMetadata],
        available_instances: List[ExecutionInstance]
    ) -> Optional[RequestMetadata]:
        """选择下一个要调度的请求"""
        
        if not pending_queue:
            return None
        
        if not available_instances:
            self.logger.warning("No available instances")
            return None
        
        # 过滤过载实例
        healthy_instances = self._filter_healthy_instances(available_instances)
        if not healthy_instances:
            self.logger.warning("All instances are overloaded")
            return None
        
        # 计算每个请求的优先级分数
        scored_requests = []
        for request in pending_queue:
            # 检查模型兼容性
            compatible_instances = [
                i for i in healthy_instances
                if i.model_name == request.model_name
            ]
            
            if not compatible_instances:
                continue  # 跳过不兼容的请求
            
            score = self._compute_priority_score(request, compatible_instances)
            scored_requests.append((score, request))
        
        if not scored_requests:
            return None
        
        # 选择分数最高的请求
        best_score, best_request = max(scored_requests, key=lambda x: x[0])
        
        self.logger.info(
            f"Selected request {best_request.request_id} "
            f"(priority={best_request.priority.name}, "
            f"slo={best_request.slo_deadline_ms}ms, score={best_score:.2f})"
        )
        
        return best_request
    
    def _filter_healthy_instances(
        self,
        instances: List[ExecutionInstance]
    ) -> List[ExecutionInstance]:
        """过滤健康的实例"""
        healthy = []
        for inst in instances:
            if not inst.available:
                continue
            
            # 检查负载
            load_ratio = inst.current_load / inst.max_concurrency
            if load_ratio > self.max_load_threshold:
                self.logger.debug(
                    f"Instance {inst.instance_id} overloaded: "
                    f"load={inst.current_load}/{inst.max_concurrency}"
                )
                continue
            
            healthy.append(inst)
        
        return healthy
    
    def _compute_priority_score(
        self,
        request: RequestMetadata,
        compatible_instances: List[ExecutionInstance]
    ) -> float:
        """计算请求的优先级分数（越高越优先）"""
        score = 0.0
        
        # 1. 基础优先级分数
        score += request.priority.value * 10
        
        # 2. SLO 紧迫度分数
        if request.slo_deadline_ms:
            now = time.time() * 1000
            time_elapsed = now - request.submitted_at
            time_remaining = request.slo_deadline_ms - time_elapsed
            
            if time_remaining < 0:
                # SLO 已违反，最高优先级
                urgency_score = 100.0
            else:
                # 剩余时间越少，紧迫度越高
                urgency_score = 50.0 * (1.0 - time_remaining / request.slo_deadline_ms)
            
            score += urgency_score * self.slo_weight
        
        # 3. 负载均衡分数（倾向于选择低负载实例的请求）
        avg_load = sum(i.current_load for i in compatible_instances) / len(compatible_instances)
        load_score = 10.0 * (1.0 - avg_load / 100)
        score += load_score * self.load_weight
        
        # 4. 拓扑亲和性分数
        if compatible_instances and hasattr(self, 'preferred_machine_id'):
            local_instances = [
                i for i in compatible_instances
                if i.machine_id == self.preferred_machine_id
            ]
            if local_instances:
                score += 5.0 * self.topology_weight
        
        return score
    
    def on_request_completed(self, request: RequestMetadata, latency_ms: float):
        """请求完成回调"""
        self.logger.debug(
            f"Request {request.request_id} completed in {latency_ms:.2f}ms"
        )
        
        # 记录延迟用于预测
        model = request.model_name
        if model not in self.latency_predictions:
            self.latency_predictions[model] = []
        self.latency_predictions[model].append(latency_ms)
        
        # 保留最近1000个样本
        if len(self.latency_predictions[model]) > 1000:
            self.latency_predictions[model] = self.latency_predictions[model][-1000:]
    
    def get_stats(self) -> Dict:
        """获取策略统计信息"""
        return {
            "total_decisions": len(self.decisions),
            "avg_latency_by_model": {
                model: sum(lats) / len(lats)
                for model, lats in self.latency_predictions.items()
                if lats
            }
        }
```

使用示例：

```python
from control_plane.manager import ControlPlaneManager
from control_plane.strategies.production_policy import ProductionPolicy

# 创建策略
policy = ProductionPolicy(
    slo_weight=2.0,      # SLO 权重加倍
    load_weight=1.0,     # 标准负载均衡
    topology_weight=0.5  # 中等拓扑偏好
)

# 创建 Manager
manager = ControlPlaneManager(scheduling_policy=policy)

# 注册实例
await manager.register_instance(instance1)
await manager.register_instance(instance2)

# 提交请求
await manager.submit_request(high_priority_request)
await manager.submit_request(normal_request)

# 查看策略统计
stats = policy.get_stats()
print(f"Total decisions: {stats['total_decisions']}")
print(f"Avg latency: {stats['avg_latency_by_model']}")
```

## 总结

本文档介绍了如何在 sageLLM 中开发自定义调度策略：

1. **核心接口**：继承 `SchedulingPolicy` 并实现 `get_next_request()`
2. **内置策略**：FIFO、Priority、SLOAware、CostOptimized、Adaptive
3. **高级特性**：拓扑感知、指标集成、预测性调度
4. **最佳实践**：性能优化、容错处理、可观测性、配置化
5. **测试调试**：单元测试、集成测试、性能基准

开始开发您的自定义策略吧！如有问题，请参考 `control_plane/policies.py` 中的内置策略实现。
