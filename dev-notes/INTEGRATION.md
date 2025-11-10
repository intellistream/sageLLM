# sageLLM Control Plane - 集成架构文档

## 概述

Control Plane 是 sageLLM 的核心协调层，位于用户应用和多个 vLLM 实例之间，提供智能的请求调度、路由和性能优化。

### 核心功能

1. **智能请求调度**: FIFO、Priority、SLO-Aware、Cost-Optimized、Adaptive 五种策略
1. **PD 分离优化**: Prefilling/Decoding 分离路由，提升 50-80% 吞吐，降低 50-60% 延迟
1. **多实例管理**: 统一管理多个 vLLM 实例，支持不同并行策略
1. **动态并行优化**: 自动选择最优的 TP/PP/DP/EP/Hybrid 并行方案
1. **负载均衡**: 多种路由算法（load_balanced、affinity、locality 等）
1. **性能监控**: 实时监控和指标收集

## 集成架构

### Control Plane 在 SAGE 中的位置

```
┌──────────────────────────────────────────────────────────────┐
│              SAGE Applications Layer                         │
│  • Chat Service (对话服务)                                   │
│  • Embedding Service (向量化服务)                            │
│  • Batch Processing (批处理服务)                             │
│  • Fine-tuning Service (微调服务)                            │
└────────────────────┬─────────────────────────────────────────┘
                     │ RequestMetadata
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              Control Plane Manager                           │
│  ┌────────────────────────────────────────────────────┐      │
│  │  Request Queue Management                          │      │
│  │  • pending_queue: deque[RequestMetadata]           │      │
│  │  • running_requests: dict[str, RequestMetadata]    │      │
│  └────────────────────────────────────────────────────┘      │
│                                                               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Scheduling │  │ PD Routing   │  │ Request      │         │
│  │ Policies   │  │ Strategy     │  │ Router       │         │
│  └────────────┘  └──────────────┘  └──────────────┘         │
│                                                               │
│  ┌────────────────────────────────────────────────────┐      │
│  │  Execution Coordinator                             │      │
│  │  • Instance Registry                               │      │
│  │  • Health Monitoring                               │      │
│  │  • HTTP Client Pool                                │      │
│  └────────────────────────────────────────────────────┘      │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP API (OpenAI-compatible)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              vLLM Instances (Multiple)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Instance │  │ Instance │  │ Instance │  │ Instance │    │
│  │    1     │  │    2     │  │    3     │  │    N     │    │
│  │ Prefill  │  │ Decode   │  │ Decode   │  │ General  │    │
│  │ TP=4     │  │ TP=1     │  │ DP=2     │  │ Hybrid   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 请求处理流程

```python
# ============================================================
# 完整的请求处理流程示例
# ============================================================

# 1. SAGE 应用层提交请求到 Control Plane
from control_plane import ControlPlaneManager, RequestMetadata, RequestPriority

request = RequestMetadata(
    request_id="req-123",
    user_id="user-456",
    priority=RequestPriority.HIGH,
    slo_deadline_ms=1000,  # 1秒 SLO
    max_tokens=512,
    prompt="Explain quantum computing in simple terms.",
    model_name="meta-llama/Llama-2-7b",
)

# 提交请求
request_id = await manager.submit_request(request)

# ============================================================
# 2. Control Plane Manager 接收请求
# ============================================================
# manager.py: submit_request()
# • 验证请求参数
# • 分配唯一 request_id
# • 添加到 pending_queue
# • 触发调度循环

# ============================================================
# 3. Scheduling Policy 确定优先级/顺序
# ============================================================
# policies.py: get_next_request()
# • FIFO: 按到达时间排序
# • Priority: 按优先级排序
# • SLO-Aware: 计算紧迫度 (deadline - current_time)
# • Cost-Optimized: 考虑成本预算
# • Adaptive: 动态选择策略

scheduling_decision = await manager.scheduling_policy.schedule(
    request, available_instances
)

# ============================================================
# 4. PD Router 确定请求阶段 (如果启用)
# ============================================================
# pd_routing.py: determine_request_phase()
if manager.enable_pd_separation:
    phase = manager.pd_router.determine_request_phase(request)
    # • PREFILLING: 长输入处理 (input_tokens > threshold)
    # • DECODING: 生成输出 (input_tokens <= threshold)

# ============================================================
# 5. Request Router 选择合适的实例
# ============================================================
# router.py: select_instance()
instance = await manager.router.select_instance(
    request=request,
    available_instances=manager.executor.get_healthy_instances(),
    phase=phase,  # 如果启用 PD 分离
)
# 路由策略:
# • load_balanced: 选择负载最低的实例
# • round_robin: 轮询
# • affinity: 用户亲和性 (user_id hash)
# • locality: 请求局部性 (request hash)

# ============================================================
# 6. Execution Coordinator 通过 HTTP API 执行
# ============================================================
# executor.py: execute_request()
result = await manager.executor.execute_request(
    request=request,
    instance=instance,
    decision=scheduling_decision,
)
# HTTP 调用:
# • POST {instance.host}:{instance.port}/v1/completions
# • 或 POST {instance.host}:{instance.port}/v1/chat/completions
# • 支持流式 (stream=True) 或批量响应

# ============================================================
# 7. vLLM Instance 处理请求
# ============================================================
# vLLM 内部执行:
# • AsyncLLMEngine 接收请求
# • KV Cache 管理
# • GPU 调度和批处理
# • 生成 tokens

# ============================================================
# 8. 响应返回到 Control Plane
# ============================================================
# • 收集生成的 tokens
# • 计算延迟指标
# • 更新请求状态

# ============================================================
# 9. 指标收集和更新
# ============================================================
# manager.py: _update_metrics()
metrics = manager.get_metrics()
# • total_requests, completed_requests, failed_requests
# • avg_latency_ms, p95_latency_ms, p99_latency_ms
# • tokens_per_second, requests_per_second
# • slo_violations, slo_compliance_rate
# • gpu_utilization

# ============================================================
# 10. 结果返回到 SAGE 应用
# ============================================================
status = await manager.get_request_status(request_id)
# RequestStatus.COMPLETED
# RequestStatus.FAILED
# RequestStatus.RUNNING
```

## 集成方式

### 方式 1: 直接使用 Control Plane API

这是推荐的集成方式，适用于大多数场景。

```python
import asyncio
from control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    RequestMetadata,
    RequestPriority,
)


async def integrate_with_sage_app():
    """SAGE 应用集成示例"""

    # 1. 初始化 Control Plane
    manager = ControlPlaneManager(
        scheduling_policy="adaptive",
        routing_strategy="load_balanced",
        enable_pd_separation=True,
        enable_monitoring=True,
    )

    # 2. 注册 vLLM 实例
    # 注意: vLLM 实例需要预先启动并监听 HTTP 端口

    # Prefilling 优化实例
    prefilling_instance = ExecutionInstance(
        instance_id="prefill-1",
        host="localhost",  # 或远程 IP
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        instance_type="prefilling",
        tensor_parallel_size=4,
        gpu_count=4,
    )
    manager.register_instance(prefilling_instance)

    # Decoding 优化实例
    decoding_instance = ExecutionInstance(
        instance_id="decode-1",
        host="localhost",
        port=8001,
        model_name="meta-llama/Llama-2-7b",
        instance_type="decoding",
        tensor_parallel_size=1,
        gpu_count=1,
    )
    manager.register_instance(decoding_instance)

    # 3. 启动 Control Plane
    await manager.start()

    # 4. 在 SAGE 应用中提交请求
    request = RequestMetadata(
        request_id="sage-req-001",
        user_id="user-123",
        priority=RequestPriority.HIGH,
        slo_deadline_ms=1000,
        max_tokens=512,
        prompt="用简单的语言解释量子计算。",
    )

    request_id = await manager.submit_request(request)

    # 5. 等待完成并获取结果
    while True:
        status = await manager.get_request_status(request_id)
        if status.state in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)

    # 6. 获取性能指标
    metrics = manager.get_metrics()
    print(f"Throughput: {metrics.requests_per_second:.2f} req/s")
    print(f"Latency: {metrics.avg_latency_ms:.2f} ms")

    # 7. 停止 Control Plane
    await manager.stop()

    return status


# 运行集成
asyncio.run(integrate_with_sage_app())
```

### 方式 2: 嵌入到 SAGE Service 中

将 Control Plane 作为 SAGE Service 的一部分：

```python
# sage/services/llm_service.py

from control_plane import ControlPlaneManager
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """SAGE LLM 服务，集成 Control Plane"""

    def __init__(self, config):
        self.config = config
        self.manager = None

    async def initialize(self):
        """初始化服务"""
        logger.info("Initializing LLM Service with Control Plane")

        # 创建 Control Plane
        self.manager = ControlPlaneManager(
            scheduling_policy=self.config.get("scheduling_policy", "adaptive"),
            enable_pd_separation=self.config.get("enable_pd_separation", True),
        )

        # 从配置加载 vLLM 实例
        for instance_config in self.config.get("vllm_instances", []):
            instance = ExecutionInstance(**instance_config)
            self.manager.register_instance(instance)

        # 启动
        await self.manager.start()
        logger.info("LLM Service initialized successfully")

    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        request = RequestMetadata(
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            priority=kwargs.get("priority", RequestPriority.NORMAL),
        )

        request_id = await self.manager.submit_request(request)

        # 等待完成
        status = await self._wait_for_completion(request_id)

        if status.state == "completed":
            return status.output
        else:
            raise RuntimeError(f"Request failed: {status.error}")

    async def _wait_for_completion(self, request_id: str, timeout: float = 30.0):
        """等待请求完成"""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.manager.get_request_status(request_id)
            if status.state in ["completed", "failed"]:
                return status
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Request {request_id} timed out")

    async def shutdown(self):
        """关闭服务"""
        if self.manager:
            await self.manager.stop()
        logger.info("LLM Service shutdown complete")


# 使用示例
async def main():
    config = {
        "scheduling_policy": "adaptive",
        "enable_pd_separation": True,
        "vllm_instances": [
            {
                "instance_id": "vllm-1",
                "host": "localhost",
                "port": 8000,
                "model_name": "meta-llama/Llama-2-7b",
                "tensor_parallel_size": 2,
            }
        ]
    }

    service = LLMService(config)
    await service.initialize()

    try:
        output = await service.generate("Hello, how are you?")
        print(f"Output: {output}")
    finally:
        await service.shutdown()

asyncio.run(main())
```

## 运行测试

### 本地开发测试

```bash
# 进入测试目录
cd tests/control_plane

# 运行所有测试
python -m pytest -v

# 运行特定测试模块
python -m pytest test_scheduling.py -v

# 运行特定测试函数
python -m pytest test_integration.py::test_control_plane_basic_flow -v

# 显示详细输出
python -m pytest -v --tb=short

# 生成覆盖率报告
python -m pytest --cov=control_plane --cov-report=html
```

### CI/CD 测试

GitHub Actions 工作流会自动运行：

```bash
cd tests/control_plane && python -m pytest -v --tb=short
```

**测试隔离配置：**

- 使用 `tests/control_plane/pytest.ini` 进行隔离配置
- 避免加载 `tests/conftest.py`（有重型 vLLM 依赖）
- 每个测试目录有独立的 `conftest.py`
- 不依赖编译的 vLLM C 扩展

### 测试覆盖范围

| 测试文件                | 测试数量 | 覆盖范围                                            |
| ----------------------- | -------- | --------------------------------------------------- |
| `test_scheduling.py`    | 5        | 调度策略验证（FIFO、Priority、SLO、Cost、Adaptive） |
| `test_pd_separation.py` | 5        | PD 路由和实例专业化                                 |
| `test_executor.py`      | 5        | 执行器生命周期和实例管理                            |
| `test_integration.py`   | 5        | 完整 SAGE ↔ Control Plane ↔ vLLM 流程               |

**总计：20 个测试，全部通过 ✅**

## 核心组件详解

### 1. Types (`types.py`)

定义了所有核心数据模型：

```python
# 请求元数据
@dataclass
class RequestMetadata:
    request_id: str
    user_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    slo_deadline_ms: Optional[float] = None
    max_tokens: int = 512
    prompt: str = ""
    model_name: Optional[str] = None
    cost_budget: Optional[float] = None
    parallelism_hint: Optional[ParallelismType] = None

# 执行实例
@dataclass
class ExecutionInstance:
    instance_id: str
    host: str = "localhost"
    port: int = 8000
    model_name: str = ""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    gpu_count: int = 1
    instance_type: str = "general"  # prefilling, decoding, general
    health_status: str = "healthy"

# 调度决策
@dataclass
class SchedulingDecision:
    instance_id: str
    request_id: str
    priority: float
    estimated_latency_ms: float
    parallelism_config: Optional[ParallelismConfig] = None
```

### 2. Manager (`manager.py`)

Control Plane 的核心协调器：

```python
class ControlPlaneManager:
    def __init__(
        self,
        scheduling_policy: str = "adaptive",
        routing_strategy: str = "load_balanced",
        enable_pd_separation: bool = True,
        enable_monitoring: bool = True,
    ):
        # 核心组件
        self.executor = ExecutionCoordinator()
        self.router = RequestRouter(routing_strategy)
        self.scheduling_policy = self._create_policy(scheduling_policy)
        self.pd_router = PDRoutingStrategy() if enable_pd_separation else None

        # 请求队列
        self.pending_queue: deque[RequestMetadata] = deque()
        self.running_requests: dict[str, RequestMetadata] = {}

    async def submit_request(self, request: RequestMetadata) -> str:
        """提交请求到队列"""

    async def start(self):
        """启动 Control Plane 后台任务"""

    async def stop(self):
        """停止 Control Plane"""

    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
```

### 3. Executor (`executor.py`)

管理所有 vLLM 实例：

```python
class ExecutionCoordinator:
    def __init__(self):
        self.instances: dict[str, ExecutionInstance] = {}
        self.http_clients: dict[str, httpx.AsyncClient] = {}

    def register_instance(self, instance: ExecutionInstance):
        """注册新实例"""

    async def execute_request(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> dict:
        """执行请求 (HTTP API 调用)"""

    async def health_check(self, instance_id: str) -> bool:
        """健康检查"""

    def get_healthy_instances(self) -> list[ExecutionInstance]:
        """获取健康的实例列表"""
```

### 4. Policies (`policies.py`)

五种调度策略：

```python
class SchedulingPolicy(ABC):
    @abstractmethod
    async def schedule(
        self,
        request: RequestMetadata,
        instances: list[ExecutionInstance],
    ) -> SchedulingDecision:
        """调度逻辑"""

# 具体实现
class FIFOPolicy(SchedulingPolicy): ...
class PriorityPolicy(SchedulingPolicy): ...
class SLOAwarePolicy(SchedulingPolicy): ...
class CostOptimizedPolicy(SchedulingPolicy): ...
class AdaptivePolicy(SchedulingPolicy): ...
```

### 5. PD Routing (`pd_routing.py`)

Prefilling/Decoding 分离路由：

```python
class PDRoutingStrategy:
    def determine_request_phase(self, request: RequestMetadata) -> str:
        """确定请求阶段"""
        if len(request.prompt) > self.config.prefilling_threshold_input_tokens:
            return "prefilling"
        return "decoding"

    def get_instance_specialization(self, instance: ExecutionInstance) -> float:
        """计算实例专业化得分"""
```

### 6. Router (`router.py`)

请求路由和负载均衡：

```python
class RequestRouter:
    def __init__(self, strategy: str = "load_balanced"):
        self.strategy = strategy

    async def select_instance(
        self,
        request: RequestMetadata,
        available_instances: list[ExecutionInstance],
        phase: Optional[str] = None,
    ) -> ExecutionInstance:
        """选择实例"""

class LoadBalancer:
    def get_least_loaded_instance(
        self, instances: list[ExecutionInstance]
    ) -> ExecutionInstance:
        """获取负载最低的实例"""
```

### 7. Parallelism (`parallelism.py`)

并行策略优化：

```python
class ParallelismOptimizer:
    def recommend_strategy(
        self,
        model_size_gb: float,
        gpu_count: int,
        gpu_memory_gb: float,
    ) -> ParallelismConfig:
        """推荐并行策略"""

    def optimize_hybrid_config(
        self, gpu_count: int
    ) -> tuple[int, int, int]:
        """优化混合并行配置 (TP, PP, DP)"""
```

## 配置示例

### 基础配置

```python
from control_plane import ControlPlaneManager, ExecutionInstance

# 最简配置
manager = ControlPlaneManager(
    scheduling_policy="fifo",
    enable_pd_separation=False
)

instance = ExecutionInstance(
    instance_id="vllm-1",
    host="localhost",
    port=8000,
    model_name="meta-llama/Llama-2-7b",
    tensor_parallel_size=1,
    gpu_count=1
)

manager.register_instance(instance)
```

### PD 分离配置

```python
from control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    ExecutionInstanceType,
    PDSeparationConfig,
    PrefillingConfig,
    DecodingConfig,
)

# 启用 PD 分离
pd_config = PDSeparationConfig(
    enabled=True,
    routing_policy="adaptive",
    prefilling_threshold_input_tokens=2048,  # 超过 2048 tokens 视为 prefilling
)

manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    enable_pd_separation=True,
    pd_config=pd_config,
)

# Prefilling 优化实例
prefilling_instance = ExecutionInstance(
    instance_id="prefilling-1",
    host="localhost",
    port=8000,
    model_name="meta-llama/Llama-2-70b",
    instance_type=ExecutionInstanceType.PREFILLING,
    tensor_parallel_size=8,  # 高 TP 以提高吞吐
    gpu_count=8,
    prefilling_config=PrefillingConfig(
        target_batch_size=64,
        tensor_parallel_size=8,
        enable_chunked_prefill=True,
    ),
)

# Decoding 优化实例
decoding_instance = ExecutionInstance(
    instance_id="decoding-1",
    host="localhost",
    port=8001,
    model_name="meta-llama/Llama-2-70b",
    instance_type=ExecutionInstanceType.DECODING,
    tensor_parallel_size=2,  # 低 TP 以降低延迟
    gpu_count=2,
    decoding_config=DecodingConfig(
        target_latency_ms=50,
        max_parallel_requests=200,
    ),
)

manager.register_instance(prefilling_instance)
manager.register_instance(decoding_instance)
```

### 生产环境配置

```python
# 生产环境完整配置
manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    routing_strategy="affinity",  # 用户亲和性，提高缓存命中率
    enable_pd_separation=True,
    enable_monitoring=True,
    enable_auto_scaling=False,  # 目前不支持，未来版本
)

# 多个实例，不同并行策略
instances = [
    # TP=4 实例 (高吞吐 prefilling)
    ExecutionInstance(
        instance_id="vllm-tp4",
        host="192.168.1.100",
        port=8000,
        model_name="llama-3-70b",
        tensor_parallel_size=4,
        gpu_count=4,
        instance_type="prefilling",
    ),
    # TP=2, PP=2 混合并行 (大模型)
    ExecutionInstance(
        instance_id="vllm-hybrid",
        host="192.168.1.101",
        port=8000,
        model_name="llama-3-70b",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        gpu_count=4,
        instance_type="general",
    ),
    # DP=2 数据并行 (高并发 decoding)
    ExecutionInstance(
        instance_id="vllm-dp2",
        host="192.168.1.102",
        port=8000,
        model_name="llama-3-70b",
        data_parallel_size=2,
        tensor_parallel_size=2,
        gpu_count=4,
        instance_type="decoding",
    ),
]

for instance in instances:
    manager.register_instance(instance)
```

## 常见问题 (FAQ)

**Q: Control Plane 与 vLLM 内置的请求队列有什么区别？**

A: Control Plane 在 vLLM 队列之上添加了高级路由和调度策略。它协调多个 vLLM 实例，应用特定领域的策略（如 PD 分离）。vLLM 负责单实例内的动态批处理，Control
Plane 负责跨实例的智能调度。

**Q: 是否支持动态批处理？**

A: 是的。Control Plane 与 vLLM 的动态批处理协同工作。调度器决定哪个实例处理请求，然后 vLLM 在该实例上进行批处理。

**Q: 可以用于推理还是训练？**

A: 当前版本专注于推理场景。架构支持扩展到微调工作流，但尚未实现。

**Q: 故障恢复如何工作？**

A: 实例具有健康状态监控。如果实例变得不健康，请求会故障转移到健康的实例。支持自动健康检查和实例标记。

**Q: 如何处理模型加载和缓存？**

A: vLLM 实例独立管理模型加载。Control Plane 不参与模型管理，只负责请求路由。建议预热所有实例。

**Q: 支持流式输出吗？**

A: 支持。通过 HTTP API 的 `stream=True` 参数可以启用流式输出。Control Plane 会将流式响应透传给客户端。

**Q: 如何监控性能？**

A: 使用 `manager.get_metrics()` 获取实时指标，包括：

- 请求统计（总数、完成、失败）
- 延迟分布（平均、P95、P99）
- 吞吐量（tokens/s、requests/s）
- SLO 合规率
- GPU 利用率

**Q: 是否支持多模型？**

A: 当前版本每个实例运行一个模型。可以注册运行不同模型的多个实例，Control Plane 会根据请求中的 `model_name` 进行路由。

**Q: 如何选择调度策略？**

A:

- **生产环境**: 使用 `adaptive`（自动适应）
- **严格 SLO**: 使用 `slo_aware`
- **成本敏感**: 使用 `cost_optimized`
- **简单场景**: 使用 `fifo`
- **多优先级**: 使用 `priority`

**Q: PD 分离适合所有场景吗？**

A: 不一定。PD 分离在以下场景最有效：

- ✅ 输入长度差异大（有些很长，有些很短）
- ✅ 有足够 GPU 资源运行多个实例
- ✅ 需要同时优化吞吐和延迟
- ❌ 输入长度均匀的场景可能不需要

## 下一步计划

### 已完成 ✅

- [x] 5 种调度策略
- [x] PD 分离路由
- [x] 多种并行策略支持
- [x] 负载均衡和路由
- [x] 健康检查
- [x] 性能监控
- [x] 完整的单元测试

### 规划中 🚀

1. **自动伸缩**: 根据负载自动扩缩容 vLLM 实例
1. **分布式协调**: 支持多节点部署的分布式协调
1. **请求追踪**: 添加分布式追踪（OpenTelemetry）
1. **指标导出**: Prometheus/CloudWatch 集成
1. **智能缓存**: KV cache 跨实例共享
1. **多模型支持**: 同时管理多个不同模型的高级特性
1. **成本预测**: 基于历史数据的成本预测
1. **A/B 测试**: 策略对比和性能测试框架

## 参考资源

- **[sageLLM 主 README](../README.md)** - 项目概述
- **[部署指南](./DEPLOYMENT.md)** - vLLM 实例部署
- **[vLLM 文档](https://docs.vllm.ai/)** - vLLM 官方文档
- **[示例代码](../control_plane/examples/)** - 完整使用示例
  - [HTTP 客户端模式](../control_plane/examples/example_http_client.py) - 实际部署场景
  - [完整演示](../control_plane/examples/demo_control_plane.py) - 功能演示
