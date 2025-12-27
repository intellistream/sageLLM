# Task 1: runtime 执行图模块开发

**状态**: ✅ 已完成  
**完成时间**: 2024-12-23  
**预计时间**: 4h  
**课题对应**: 4.1 面向国产互联的高性能推理通信与数据通路优化  
**可并行**: ✅ 是（与 Task 2-5 并行）

---

## 背景

课题 4.1 要求"重构 PD/AF 分离的推理执行图"，本任务负责设计和实现执行图 IR 及其调度器。

**核心目标**：
- 设计支持 Prefill/Decode 分离的执行图 IR
- 实现执行图构建器和优化器
- 实现 PD 分离调度器

---

## 工作目录

```
/home/shuhao/SAGE/packages/sage-common/src/sage/common/components/sage_llm/sageLLM/runtime/
├── __init__.py
├── execution_graph/
│   ├── __init__.py
│   ├── ir.py          # ✅ 已实现
│   ├── builder.py     # ✅ 已实现
│   └── optimizer.py   # ✅ 已实现
├── comm/              # Task 1 不涉及，由通信部分处理
└── scheduler/
    ├── __init__.py
    ├── base.py        # ✅ 已实现
    └── pd_scheduler.py # ✅ 已实现
```

---

## 参考资料

- vLLM scheduler: https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
- FlexGen 执行图: https://github.com/FMInference/FlexGen
- Sarathi PD 分离: https://arxiv.org/abs/2308.16369
- DistServe: https://arxiv.org/abs/2401.09670

---

## 任务清单

### 1. 设计执行图 IR (`execution_graph/ir.py`)

需要实现的核心类型：

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any

class OpType(Enum):
    """算子类型"""
    # 计算算子
    PREFILL_ATTN = auto()      # Prefill 阶段注意力
    DECODE_ATTN = auto()       # Decode 阶段注意力
    FFN = auto()               # 前馈网络
    EMBEDDING = auto()         # Embedding 查找
    LM_HEAD = auto()           # 语言模型头
    LAYERNORM = auto()         # LayerNorm
    
    # 通信算子
    COMM_ALLREDUCE = auto()    # AllReduce
    COMM_ALLGATHER = auto()    # AllGather
    COMM_SEND = auto()         # Point-to-point Send
    COMM_RECV = auto()         # Point-to-point Recv
    
    # KV Cache 算子
    KV_LOAD = auto()           # 从存储加载 KV
    KV_STORE = auto()          # 存储 KV 到缓存
    KV_MIGRATE = auto()        # KV 跨层迁移


@dataclass
class TensorRef:
    """张量引用"""
    name: str
    shape: tuple
    dtype: str
    device_id: int = 0


@dataclass
class ExecutionNode:
    """执行图节点"""
    node_id: str
    op_type: OpType
    inputs: List[TensorRef]
    outputs: List[TensorRef]
    
    # 依赖关系
    dependencies: Set[str] = field(default_factory=set)  # 依赖的节点 ID
    
    # 调度信息
    device_id: int = 0
    stream_id: int = 0
    
    # 性能估算
    estimated_time_us: float = 0.0
    memory_bytes: int = 0
    
    # 元数据
    layer_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionGraph:
    """执行图"""
    graph_id: str
    nodes: Dict[str, ExecutionNode] = field(default_factory=dict)
    
    # 子图划分
    prefill_nodes: Set[str] = field(default_factory=set)
    decode_nodes: Set[str] = field(default_factory=set)
    comm_nodes: Set[str] = field(default_factory=set)
    
    # 图属性
    num_layers: int = 0
    model_config: Optional[Dict[str, Any]] = None
    
    def add_node(self, node: ExecutionNode) -> None:
        """添加节点"""
        self.nodes[node.node_id] = node
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加边（依赖关系）"""
        if to_node in self.nodes:
            self.nodes[to_node].dependencies.add(from_node)
    
    def topological_sort(self) -> List[ExecutionNode]:
        """拓扑排序，返回执行顺序"""
        ...
    
    def get_critical_path(self) -> List[ExecutionNode]:
        """获取关键路径"""
        ...
    
    def estimate_total_time(self) -> float:
        """估算总执行时间（微秒）"""
        ...
    
    def get_subgraph(self, node_ids: Set[str]) -> "ExecutionGraph":
        """提取子图"""
        ...
```

### 2. 实现执行图构建器 (`execution_graph/builder.py`)

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from .ir import ExecutionGraph, ExecutionNode, OpType, TensorRef


@dataclass
class ModelConfig:
    """模型配置"""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_len: int = 4096


@dataclass
class ParallelConfig:
    """并行配置"""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1


@dataclass
class PrefillBatch:
    """Prefill 批次"""
    request_ids: List[str]
    input_lengths: List[int]
    total_tokens: int


@dataclass  
class DecodeBatch:
    """Decode 批次"""
    request_ids: List[str]
    context_lengths: List[int]
    batch_size: int


class ExecutionGraphBuilder:
    """执行图构建器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
    
    def build_prefill_graph(self, batch: PrefillBatch) -> ExecutionGraph:
        """构建 Prefill 阶段执行图
        
        Prefill 特点：
        - 处理完整的输入序列
        - 计算密集（矩阵乘法为主）
        - 生成所有 KV Cache
        """
        graph = ExecutionGraph(graph_id=f"prefill_{batch.total_tokens}")
        
        # 1. Embedding
        # 2. 每层: Attention + FFN + Comm
        # 3. LM Head
        
        ...
        return graph
    
    def build_decode_graph(self, batch: DecodeBatch) -> ExecutionGraph:
        """构建 Decode 阶段执行图
        
        Decode 特点：
        - 每次只处理一个 token
        - 内存带宽密集（KV Cache 访问）
        - 逐 token 自回归
        """
        graph = ExecutionGraph(graph_id=f"decode_{batch.batch_size}")
        
        # 1. Embedding (单 token)
        # 2. 每层: Attention (with KV Load) + FFN + Comm  
        # 3. LM Head + Sampling
        
        ...
        return graph
    
    def build_hybrid_graph(
        self,
        prefill_batch: PrefillBatch,
        decode_batch: DecodeBatch,
    ) -> ExecutionGraph:
        """构建混合执行图（Chunked Prefill）
        
        混合模式：
        - Prefill 分 chunk 与 Decode 交替执行
        - 平衡 TTFT 和吞吐
        """
        ...
    
    def _build_attention_node(
        self,
        layer_id: int,
        is_prefill: bool,
        batch_info: Dict[str, Any],
    ) -> ExecutionNode:
        """构建注意力算子节点"""
        ...
    
    def _build_ffn_node(
        self,
        layer_id: int,
        batch_info: Dict[str, Any],
    ) -> ExecutionNode:
        """构建 FFN 算子节点"""
        ...
    
    def _build_comm_node(
        self,
        layer_id: int,
        comm_type: OpType,
    ) -> ExecutionNode:
        """构建通信算子节点"""
        ...
```

### 3. 实现执行图优化器 (`execution_graph/optimizer.py`)

```python
from abc import ABC, abstractmethod
from typing import List
from .ir import ExecutionGraph, ExecutionNode, OpType


class OptimizationPass(ABC):
    """优化 Pass 基类"""
    
    @abstractmethod
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """应用优化"""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pass 名称"""
        ...


class CommunicationFusionPass(OptimizationPass):
    """通信融合优化
    
    将多个小的 AllReduce 合并为一个大的，减少通信次数。
    """
    
    @property
    def name(self) -> str:
        return "comm_fusion"
    
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """合并相邻的通信算子"""
        # 1. 找出所有通信节点
        # 2. 按依赖关系分组
        # 3. 合并同组的通信操作
        ...


class ComputeCommOverlapPass(OptimizationPass):
    """计算-通信重叠优化
    
    让通信操作与无依赖的计算操作并行执行。
    """
    
    @property
    def name(self) -> str:
        return "compute_comm_overlap"
    
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """调整节点的 stream 分配，实现重叠"""
        # 1. 识别可重叠的计算-通信对
        # 2. 分配不同的 stream
        # 3. 更新依赖关系
        ...


class KVPrefetchPass(OptimizationPass):
    """KV 预取优化
    
    在计算当前层时预取下一层的 KV Cache。
    """
    
    @property
    def name(self) -> str:
        return "kv_prefetch"
    
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """插入 KV 预取节点"""
        # 1. 找出 KV_LOAD 节点
        # 2. 在前一层计算节点后插入预取
        # 3. 更新依赖关系
        ...


class MemoryOptimizationPass(OptimizationPass):
    """内存优化
    
    优化中间张量的生命周期，减少峰值内存。
    """
    
    @property
    def name(self) -> str:
        return "memory_opt"
    
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """分析并优化内存使用"""
        # 1. 分析张量生命周期
        # 2. 复用已释放的内存
        # 3. 更新节点的内存分配
        ...


class GraphOptimizer:
    """执行图优化器"""
    
    def __init__(self, passes: List[OptimizationPass] = None):
        self.passes = passes or [
            CommunicationFusionPass(),
            ComputeCommOverlapPass(),
            KVPrefetchPass(),
            MemoryOptimizationPass(),
        ]
    
    def optimize(self, graph: ExecutionGraph) -> ExecutionGraph:
        """依次应用所有优化 Pass"""
        optimized = graph
        for pass_ in self.passes:
            optimized = pass_.apply(optimized)
        return optimized
    
    def add_pass(self, pass_: OptimizationPass) -> None:
        """添加自定义优化 Pass"""
        self.passes.append(pass_)
```

### 4. 实现调度器基类 (`scheduler/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any
import time


class RequestStatus(Enum):
    """请求状态"""
    WAITING = auto()      # 等待调度
    PREFILLING = auto()   # Prefill 中
    DECODING = auto()     # Decode 中
    PREEMPTED = auto()    # 被抢占
    FINISHED = auto()     # 已完成


@dataclass
class Request:
    """推理请求"""
    request_id: str
    prompt_token_ids: List[int]
    
    # 生成参数
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    
    # 状态
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    
    # 时间戳
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    
    # KV Cache 信息
    kv_block_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    
    # 优先级
    priority: int = 0
    
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)
    
    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)
    
    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens


@dataclass
class Batch:
    """请求批次"""
    batch_id: str
    requests: List[Request]
    
    @property
    def batch_size(self) -> int:
        return len(self.requests)
    
    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.requests)


@dataclass
class ScheduleOutput:
    """调度输出"""
    prefill_batch: Optional[Batch] = None
    decode_batch: Optional[Batch] = None
    preempted_requests: List[Request] = field(default_factory=list)
    
    # 资源分配
    kv_blocks_to_allocate: Dict[str, List[int]] = field(default_factory=dict)
    kv_blocks_to_free: List[int] = field(default_factory=list)


class BaseScheduler(ABC):
    """调度器基类"""
    
    def __init__(self, max_batch_size: int = 256, max_tokens: int = 8192):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        
        # 请求队列
        self.waiting_queue: List[Request] = []
        self.running_requests: Dict[str, Request] = {}
        self.preempted_requests: List[Request] = []
    
    def add_request(self, request: Request) -> None:
        """添加新请求"""
        self.waiting_queue.append(request)
    
    def abort_request(self, request_id: str) -> bool:
        """中止请求"""
        # 从等待队列移除
        for i, req in enumerate(self.waiting_queue):
            if req.request_id == request_id:
                self.waiting_queue.pop(i)
                return True
        
        # 从运行中移除
        if request_id in self.running_requests:
            del self.running_requests[request_id]
            return True
        
        return False
    
    @abstractmethod
    def schedule(self) -> ScheduleOutput:
        """执行调度，返回下一批要执行的请求"""
        ...
    
    @abstractmethod
    def update_after_step(self, finished_request_ids: List[str]) -> None:
        """步骤完成后更新状态"""
        ...
    
    def get_num_waiting(self) -> int:
        return len(self.waiting_queue)
    
    def get_num_running(self) -> int:
        return len(self.running_requests)
```

### 5. 实现 PD 分离调度器 (`scheduler/pd_scheduler.py`)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from .base import BaseScheduler, Request, Batch, ScheduleOutput, RequestStatus


@dataclass
class PDSchedulerConfig:
    """PD 分离调度器配置"""
    # 设备分配
    prefill_device_ids: List[int] = field(default_factory=lambda: [0])
    decode_device_ids: List[int] = field(default_factory=lambda: [0])
    
    # 调度模式
    mode: Literal["strict", "time_share", "hybrid"] = "time_share"
    
    # Prefill 配置
    max_prefill_batch_size: int = 32
    max_prefill_tokens: int = 4096
    prefill_chunk_size: int = 512  # for hybrid mode
    
    # Decode 配置
    max_decode_batch_size: int = 256
    max_decode_tokens: int = 8192
    
    # 抢占策略
    enable_preemption: bool = True
    preemption_mode: Literal["swap", "recompute"] = "recompute"


class PDScheduler(BaseScheduler):
    """Prefill-Decode 分离调度器
    
    支持三种模式：
    1. strict: Prefill 和 Decode 在不同 GPU 上执行，完全分离
    2. time_share: 同一 GPU 上时分复用，交替执行 Prefill 和 Decode
    3. hybrid: Chunked Prefill，Prefill 分块与 Decode 混合执行
    
    参考论文：
    - DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving
    - Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
    """
    
    def __init__(self, config: PDSchedulerConfig):
        super().__init__(
            max_batch_size=config.max_decode_batch_size,
            max_tokens=config.max_decode_tokens,
        )
        self.config = config
        
        # 分离的队列
        self.prefill_queue: List[Request] = []
        self.decode_queue: List[Request] = []
        
        # 调度统计
        self.stats = {
            "total_prefill_batches": 0,
            "total_decode_batches": 0,
            "total_preemptions": 0,
        }
    
    def add_request(self, request: Request) -> None:
        """添加新请求到 Prefill 队列"""
        request.status = RequestStatus.WAITING
        self.prefill_queue.append(request)
    
    def schedule(self) -> ScheduleOutput:
        """执行 PD 分离调度"""
        if self.config.mode == "strict":
            return self._schedule_strict()
        elif self.config.mode == "time_share":
            return self._schedule_time_share()
        else:
            return self._schedule_hybrid()
    
    def _schedule_strict(self) -> ScheduleOutput:
        """严格分离模式
        
        Prefill 和 Decode 在不同设备上同时执行。
        """
        output = ScheduleOutput()
        
        # 调度 Prefill
        if self.prefill_queue:
            prefill_requests = self._select_prefill_requests()
            if prefill_requests:
                output.prefill_batch = Batch(
                    batch_id=f"prefill_{self.stats['total_prefill_batches']}",
                    requests=prefill_requests,
                )
                self.stats["total_prefill_batches"] += 1
        
        # 调度 Decode
        if self.decode_queue:
            decode_requests = self._select_decode_requests()
            if decode_requests:
                output.decode_batch = Batch(
                    batch_id=f"decode_{self.stats['total_decode_batches']}",
                    requests=decode_requests,
                )
                self.stats["total_decode_batches"] += 1
        
        return output
    
    def _schedule_time_share(self) -> ScheduleOutput:
        """时分复用模式
        
        优先 Decode（低延迟），Prefill 在空闲时执行。
        """
        output = ScheduleOutput()
        
        # 优先调度 Decode
        if self.decode_queue:
            decode_requests = self._select_decode_requests()
            if decode_requests:
                output.decode_batch = Batch(
                    batch_id=f"decode_{self.stats['total_decode_batches']}",
                    requests=decode_requests,
                )
                self.stats["total_decode_batches"] += 1
                return output  # Decode 优先
        
        # 无 Decode 时调度 Prefill
        if self.prefill_queue:
            prefill_requests = self._select_prefill_requests()
            if prefill_requests:
                output.prefill_batch = Batch(
                    batch_id=f"prefill_{self.stats['total_prefill_batches']}",
                    requests=prefill_requests,
                )
                self.stats["total_prefill_batches"] += 1
        
        return output
    
    def _schedule_hybrid(self) -> ScheduleOutput:
        """混合模式（Chunked Prefill）
        
        Prefill 分 chunk 与 Decode 一起执行。
        """
        output = ScheduleOutput()
        
        # 选择 Decode 请求
        decode_requests = []
        if self.decode_queue:
            decode_requests = self._select_decode_requests()
        
        # 计算剩余 token 预算
        decode_tokens = len(decode_requests)  # Decode 每请求 1 token
        remaining_tokens = self.config.max_decode_tokens - decode_tokens
        
        # 用剩余预算做 Chunked Prefill
        prefill_requests = []
        if self.prefill_queue and remaining_tokens > 0:
            prefill_requests = self._select_chunked_prefill(remaining_tokens)
        
        # 合并为一个批次
        if decode_requests:
            output.decode_batch = Batch(
                batch_id=f"decode_{self.stats['total_decode_batches']}",
                requests=decode_requests,
            )
            self.stats["total_decode_batches"] += 1
        
        if prefill_requests:
            output.prefill_batch = Batch(
                batch_id=f"prefill_{self.stats['total_prefill_batches']}",
                requests=prefill_requests,
            )
            self.stats["total_prefill_batches"] += 1
        
        return output
    
    def _select_prefill_requests(self) -> List[Request]:
        """选择 Prefill 请求"""
        selected = []
        total_tokens = 0
        
        for req in self.prefill_queue[:]:
            req_tokens = req.num_prompt_tokens - req.num_computed_tokens
            if (len(selected) < self.config.max_prefill_batch_size and
                total_tokens + req_tokens <= self.config.max_prefill_tokens):
                selected.append(req)
                total_tokens += req_tokens
                self.prefill_queue.remove(req)
                req.status = RequestStatus.PREFILLING
        
        return selected
    
    def _select_decode_requests(self) -> List[Request]:
        """选择 Decode 请求"""
        selected = []
        
        for req in self.decode_queue[:]:
            if len(selected) < self.config.max_decode_batch_size:
                selected.append(req)
                # 不从队列移除，Decode 是持续的
        
        return selected
    
    def _select_chunked_prefill(self, token_budget: int) -> List[Request]:
        """选择 Chunked Prefill 请求"""
        selected = []
        remaining = token_budget
        chunk_size = self.config.prefill_chunk_size
        
        for req in self.prefill_queue[:]:
            if remaining <= 0:
                break
            
            # 计算这个请求还需要多少 token
            req_remaining = req.num_prompt_tokens - req.num_computed_tokens
            chunk = min(req_remaining, chunk_size, remaining)
            
            if chunk > 0:
                selected.append(req)
                remaining -= chunk
                req.num_computed_tokens += chunk
                
                # 如果 Prefill 完成，移到 Decode 队列
                if req.num_computed_tokens >= req.num_prompt_tokens:
                    self.prefill_queue.remove(req)
                    self.decode_queue.append(req)
                    req.status = RequestStatus.DECODING
        
        return selected
    
    def update_after_step(self, finished_request_ids: List[str]) -> None:
        """步骤完成后更新状态"""
        for req_id in finished_request_ids:
            # 从 Decode 队列移除
            for req in self.decode_queue[:]:
                if req.request_id == req_id:
                    self.decode_queue.remove(req)
                    req.status = RequestStatus.FINISHED
                    break
    
    def prefill_to_decode(self, request_ids: List[str]) -> None:
        """Prefill 完成后转移到 Decode 队列"""
        for req_id in request_ids:
            for req in self.prefill_queue[:]:
                if req.request_id == req_id:
                    self.prefill_queue.remove(req)
                    self.decode_queue.append(req)
                    req.status = RequestStatus.DECODING
                    break
    
    def get_stats(self) -> Dict:
        """获取调度统计"""
        return {
            **self.stats,
            "prefill_queue_size": len(self.prefill_queue),
            "decode_queue_size": len(self.decode_queue),
        }
```

---

## 单元测试要求

创建 `tests/unit/test_runtime_execution_graph.py`：

```python
import pytest
from sageLLM.runtime.execution_graph.ir import (
    ExecutionGraph, ExecutionNode, OpType, TensorRef
)
from sageLLM.runtime.execution_graph.builder import (
    ExecutionGraphBuilder, ModelConfig, ParallelConfig, PrefillBatch, DecodeBatch
)
from sageLLM.runtime.execution_graph.optimizer import GraphOptimizer
from sageLLM.runtime.scheduler.pd_scheduler import PDScheduler, PDSchedulerConfig
from sageLLM.runtime.scheduler.base import Request


class TestExecutionGraphIR:
    """执行图 IR 测试"""
    
    def test_create_graph(self):
        """测试创建执行图"""
        graph = ExecutionGraph(graph_id="test")
        assert graph.graph_id == "test"
        assert len(graph.nodes) == 0
    
    def test_add_node(self):
        """测试添加节点"""
        graph = ExecutionGraph(graph_id="test")
        node = ExecutionNode(
            node_id="attn_0",
            op_type=OpType.PREFILL_ATTN,
            inputs=[],
            outputs=[],
        )
        graph.add_node(node)
        assert "attn_0" in graph.nodes
    
    def test_add_edge(self):
        """测试添加边"""
        graph = ExecutionGraph(graph_id="test")
        node1 = ExecutionNode(node_id="n1", op_type=OpType.EMBEDDING, inputs=[], outputs=[])
        node2 = ExecutionNode(node_id="n2", op_type=OpType.PREFILL_ATTN, inputs=[], outputs=[])
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge("n1", "n2")
        assert "n1" in graph.nodes["n2"].dependencies
    
    def test_topological_sort(self):
        """测试拓扑排序"""
        # 构建简单的图: n1 -> n2 -> n3
        graph = ExecutionGraph(graph_id="test")
        # ... 添加节点和边
        sorted_nodes = graph.topological_sort()
        # 验证顺序正确
        ...


class TestGraphBuilder:
    """执行图构建器测试"""
    
    @pytest.fixture
    def builder(self):
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
        )
        parallel = ParallelConfig(tensor_parallel_size=1)
        return ExecutionGraphBuilder(config, parallel)
    
    def test_build_prefill_graph(self, builder):
        """测试构建 Prefill 图"""
        batch = PrefillBatch(
            request_ids=["req_1"],
            input_lengths=[128],
            total_tokens=128,
        )
        graph = builder.build_prefill_graph(batch)
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.prefill_nodes) > 0
    
    def test_build_decode_graph(self, builder):
        """测试构建 Decode 图"""
        batch = DecodeBatch(
            request_ids=["req_1"],
            context_lengths=[128],
            batch_size=1,
        )
        graph = builder.build_decode_graph(batch)
        assert graph is not None
        assert len(graph.decode_nodes) > 0


class TestPDScheduler:
    """PD 分离调度器测试"""
    
    def test_strict_mode(self):
        """测试严格分离模式"""
        config = PDSchedulerConfig(
            mode="strict",
            prefill_device_ids=[0],
            decode_device_ids=[1],
        )
        scheduler = PDScheduler(config)
        
        # 添加请求
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)
        
        # 调度
        output = scheduler.schedule()
        assert output.prefill_batch is not None
        assert output.prefill_batch.batch_size == 1
    
    def test_time_share_mode(self):
        """测试时分复用模式"""
        config = PDSchedulerConfig(mode="time_share")
        scheduler = PDScheduler(config)
        # ...
    
    def test_hybrid_mode(self):
        """测试混合模式"""
        config = PDSchedulerConfig(
            mode="hybrid",
            prefill_chunk_size=256,
        )
        scheduler = PDScheduler(config)
        # ...
```

---

## 接口约定

### 输入接口（从其他模块接收）

| 接口 | 来源 | 说明 |
|------|------|------|
| `ModelConfig` | 配置文件/用户 | 模型配置 |
| `ParallelConfig` | 配置文件/用户 | 并行配置 |
| `KVBudget` | `kv_runtime` | KV Cache 预算 |

### 输出接口（提供给其他模块）

| 接口 | 目标 | 说明 |
|------|------|------|
| `ExecutionGraph` | `backends` | 执行图供后端执行 |
| `ScheduleOutput` | `kv_runtime` | 调度结果供 KV 分配 |
| `Batch` | `benchmarks` | 批次信息供指标收集 |

---

## 验收标准

- [ ] IR 类型定义完整，有 docstring
- [ ] `ExecutionGraph` 支持拓扑排序和关键路径计算
- [ ] `Builder` 能构建 Prefill/Decode/Hybrid 三种图
- [ ] `Optimizer` 实现至少 2 个优化 Pass（通信融合、计算重叠）
- [ ] `PDScheduler` 支持 strict/time_share/hybrid 三种模式
- [ ] 单元测试覆盖率 > 80%
- [ ] 代码通过 `ruff check` 和 `mypy`

---

## 输出物清单

```
runtime/
├── __init__.py              # 更新导出
├── execution_graph/
│   ├── __init__.py          # 更新导出
│   ├── ir.py                # ✅ 完整实现
│   ├── builder.py           # ✅ 完整实现
│   └── optimizer.py         # ✅ 完整实现
└── scheduler/
    ├── __init__.py          # 更新导出
    ├── base.py              # ✅ 完整实现
    └── pd_scheduler.py      # ✅ 完整实现

tests/unit/
└── test_runtime_execution_graph.py  # ✅ 测试文件
```
