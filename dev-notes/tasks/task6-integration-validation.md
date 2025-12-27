# Task 6: 模块集成与端到端验证

**状态**: 🔲 待开始  
**预计时间**: 4h  
**依赖**: Task 1-5 全部完成  
**可并行**: ❌ 否（依赖所有前置任务）

---

## 背景

Task 1-5 分别实现了各个模块：
- Task 1: `runtime/` (execution_graph, scheduler)
- Task 2: `kv_runtime/` (多粒度 KV 管理)
- Task 3: `accel/` (量化、稀疏)
- Task 4: `backends/` (硬件抽象)
- Task 5: `benchmarks/` (评测框架)

本任务负责将这些模块集成起来，确保它们能够协同工作。

---

## 工作目录

```
/home/shuhao/SAGE/packages/sage-common/src/sage/common/components/sage_llm/sageLLM/
├── __init__.py              # 更新主入口
├── engine.py                # 🆕 推理引擎（集成层）
├── config.py                # 🆕 统一配置
└── examples/                # 🆕 示例
    ├── __init__.py
    └── basic_inference.py
```

---

## 任务清单

### 1. 统一配置 (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
from pathlib import Path

from .backends import BackendType
from .accel.quantize import QuantizationType
from .accel.sparsity.structured import SparsityPattern
from .kv_runtime.blocks.multi_granular import KVGranularity, StorageTier
from .runtime.scheduler.pd_scheduler import ScheduleMode


class InferenceMode(Enum):
    """推理模式"""
    STANDARD = auto()      # 标准推理
    PREFILL_ONLY = auto()  # 仅 prefill
    DECODE_ONLY = auto()   # 仅 decode
    PD_SEPARATE = auto()   # PD 分离


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    
    # 模型结构
    num_layers: int = 32
    num_heads: int = 32
    hidden_size: int = 4096
    vocab_size: int = 32000
    max_seq_len: int = 4096
    
    # 精度
    dtype: str = "float16"
    
    # 量化
    quantization: Optional[QuantizationType] = None
    quantization_config: Dict[str, Any] = field(default_factory=dict)
    
    # 稀疏
    sparsity_pattern: Optional[SparsityPattern] = None
    sparsity_ratio: float = 0.0


@dataclass
class KVCacheConfig:
    """KV 缓存配置"""
    # 容量
    max_tokens: int = 65536
    block_size: int = 16
    
    # 粒度
    granularity: KVGranularity = KVGranularity.BLOCK
    
    # 分层存储
    enable_tiering: bool = False
    hbm_ratio: float = 0.7      # HBM 占比
    ddr_ratio: float = 0.2      # DDR 占比
    nvme_ratio: float = 0.1     # NVMe 占比
    nvme_path: Optional[str] = None
    
    # 复用
    enable_prefix_caching: bool = True
    enable_cross_request_sharing: bool = True
    
    # 迁移
    enable_migration: bool = True
    hot_threshold: float = 1.0   # 热块访问频率阈值
    cold_timeout_s: float = 60.0  # 冷块超时


@dataclass
class SchedulerConfig:
    """调度器配置"""
    # 模式
    mode: ScheduleMode = ScheduleMode.HYBRID
    
    # PD 分离
    prefill_workers: int = 1
    decode_workers: int = 1
    
    # 批处理
    max_batch_size: int = 64
    max_prefill_batch: int = 8
    max_decode_batch: int = 64
    
    # 超时
    request_timeout_s: float = 60.0
    queue_timeout_s: float = 30.0


@dataclass
class BackendConfig:
    """后端配置"""
    # 硬件
    backend_type: Optional[BackendType] = None  # None = 自动检测
    device_ids: List[int] = field(default_factory=lambda: [0])
    
    # 分布式
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


@dataclass
class BenchmarkConfig:
    """评测配置"""
    # 启用
    enable_profiling: bool = False
    enable_metrics: bool = True
    
    # CI 门控
    enable_gates: bool = False
    min_throughput_tps: Optional[float] = None
    max_ttft_ms: Optional[float] = None
    max_tpot_ms: Optional[float] = None


@dataclass
class SageLLMConfig:
    """sageLLM 统一配置"""
    model: ModelConfig
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    # 推理模式
    inference_mode: InferenceMode = InferenceMode.STANDARD
    
    @classmethod
    def from_yaml(cls, path: str) -> "SageLLMConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict) -> "SageLLMConfig":
        """从字典创建配置"""
        model = ModelConfig(**data.get("model", {}))
        kv_cache = KVCacheConfig(**data.get("kv_cache", {}))
        scheduler = SchedulerConfig(**data.get("scheduler", {}))
        backend = BackendConfig(**data.get("backend", {}))
        benchmark = BenchmarkConfig(**data.get("benchmark", {}))
        
        return cls(
            model=model,
            kv_cache=kv_cache,
            scheduler=scheduler,
            backend=backend,
            benchmark=benchmark,
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)
```

### 2. 推理引擎集成 (`engine.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncIterator
import logging
import asyncio

from .config import SageLLMConfig, InferenceMode
from .backends import get_backend, HardwareBackend, DeviceInfo
from .runtime.execution_graph import ExecutionGraph, ExecutionGraphBuilder
from .runtime.scheduler import PDScheduler, ScheduleOutput
from .kv_runtime.blocks.multi_granular import MultiGranularKVPool, KVPoolConfig
from .kv_runtime.hierarchy.tiered_storage import TieredKVStorage, TierConfig
from .kv_runtime.reuse.cross_request import CrossRequestKVCache
from .accel.quantize import QuantizerRegistry
from .benchmarks.metrics import MetricRegistry
from .benchmarks.metrics.throughput import ThroughputMetric
from .benchmarks.metrics.latency import LatencyMetric

logger = logging.getLogger(__name__)


@dataclass
class GenerateRequest:
    """生成请求"""
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None


@dataclass
class GenerateOutput:
    """生成输出"""
    request_id: str
    output_tokens: List[int]
    finish_reason: str  # "length", "stop", "error"
    metrics: Optional[Dict[str, float]] = None


class SageLLMEngine:
    """sageLLM 推理引擎
    
    集成所有模块，提供统一的推理接口。
    
    Usage:
        config = SageLLMConfig(...)
        engine = SageLLMEngine(config)
        engine.initialize()
        
        request = GenerateRequest(
            request_id="req_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=100,
        )
        
        output = engine.generate(request)
    """
    
    def __init__(self, config: SageLLMConfig):
        self.config = config
        
        # 组件（延迟初始化）
        self._backend: Optional[HardwareBackend] = None
        self._scheduler: Optional[PDScheduler] = None
        self._kv_pool: Optional[MultiGranularKVPool] = None
        self._kv_cache: Optional[CrossRequestKVCache] = None
        self._kv_storage: Optional[TieredKVStorage] = None
        
        # 指标
        self._throughput_metric: Optional[ThroughputMetric] = None
        self._latency_metric: Optional[LatencyMetric] = None
        
        # 状态
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化引擎"""
        if self._initialized:
            logger.warning("Engine already initialized")
            return
        
        logger.info(f"Initializing sageLLM engine for {self.config.model.model_id}")
        
        # 1. 初始化硬件后端
        self._init_backend()
        
        # 2. 初始化 KV 缓存
        self._init_kv_cache()
        
        # 3. 初始化调度器
        self._init_scheduler()
        
        # 4. 初始化指标
        if self.config.benchmark.enable_metrics:
            self._init_metrics()
        
        self._initialized = True
        logger.info("Engine initialization complete")
    
    def _init_backend(self) -> None:
        """初始化硬件后端"""
        backend_type = self.config.backend.backend_type
        self._backend = get_backend(backend_type)
        
        device_info = self._backend.get_device_info()
        logger.info(f"Using backend: {device_info.name}")
        logger.info(f"  Memory: {device_info.total_memory_gb:.1f} GB")
        logger.info(f"  Capabilities: {self._backend.get_capabilities()}")
    
    def _init_kv_cache(self) -> None:
        """初始化 KV 缓存"""
        kv_config = self.config.kv_cache
        
        # 创建 KV 池
        pool_config = KVPoolConfig(
            block_size=kv_config.block_size,
            default_granularity=kv_config.granularity,
            enable_sharing=kv_config.enable_cross_request_sharing,
            enable_tiering=kv_config.enable_tiering,
        )
        self._kv_pool = MultiGranularKVPool(pool_config)
        
        # 创建跨请求缓存
        self._kv_cache = CrossRequestKVCache(
            pool=self._kv_pool,
            enable_tenant_isolation=False,
        )
        
        # 如果启用分层存储
        if kv_config.enable_tiering:
            from .kv_runtime.blocks.multi_granular import StorageTier
            device_info = self._backend.get_device_info()
            
            hbm_capacity = int(device_info.total_memory_gb * kv_config.hbm_ratio * 1024**3)
            ddr_capacity = int(64 * kv_config.ddr_ratio * 1024**3)  # 假设 64GB 主存
            
            self._kv_storage = TieredKVStorage(
                hbm_config=TierConfig(
                    tier=StorageTier.HBM,
                    capacity_bytes=hbm_capacity,
                    bandwidth_gbps=2000.0,
                    latency_us=1.0,
                    device_id=self.config.backend.device_ids[0],
                ),
                ddr_config=TierConfig(
                    tier=StorageTier.DDR,
                    capacity_bytes=ddr_capacity,
                    bandwidth_gbps=50.0,
                    latency_us=100.0,
                ),
            )
        
        logger.info(f"KV cache initialized: block_size={kv_config.block_size}")
    
    def _init_scheduler(self) -> None:
        """初始化调度器"""
        sched_config = self.config.scheduler
        
        from .runtime.scheduler.pd_scheduler import PDSchedulerConfig
        
        scheduler_config = PDSchedulerConfig(
            mode=sched_config.mode,
            prefill_workers=sched_config.prefill_workers,
            decode_workers=sched_config.decode_workers,
            max_batch_size=sched_config.max_batch_size,
        )
        
        self._scheduler = PDScheduler(scheduler_config)
        logger.info(f"Scheduler initialized: mode={sched_config.mode.name}")
    
    def _init_metrics(self) -> None:
        """初始化指标"""
        self._throughput_metric = ThroughputMetric()
        self._latency_metric = LatencyMetric()
        logger.info("Metrics initialized")
    
    def generate(self, request: GenerateRequest) -> GenerateOutput:
        """同步生成
        
        Args:
            request: 生成请求
            
        Returns:
            生成输出
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # 开始计时
        if self._latency_metric:
            self._latency_metric.request_start()
        
        if self._throughput_metric:
            self._throughput_metric.start()
        
        # 1. 尝试 KV 复用
        reuse_result = self._kv_cache.try_reuse(
            request_id=request.request_id,
            token_ids=request.prompt_tokens,
        )
        
        if reuse_result.reused:
            logger.debug(f"KV reuse: {reuse_result.matched_tokens}/{reuse_result.total_tokens} tokens")
            # 从复用点开始生成
            start_pos = reuse_result.matched_tokens
        else:
            start_pos = 0
        
        # 2. 构建执行图
        builder = ExecutionGraphBuilder(
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            hidden_size=self.config.model.hidden_size,
        )
        
        # Prefill 图（如果需要）
        if start_pos < len(request.prompt_tokens):
            prefill_graph = builder.build_prefill_graph(
                seq_len=len(request.prompt_tokens) - start_pos,
            )
        
        # 3. 调度执行（简化实现）
        output_tokens = []
        
        # Prefill 阶段
        if self._latency_metric:
            self._latency_metric.prefill_done()
            self._latency_metric.first_token()
        
        # Decode 阶段（模拟）
        for i in range(request.max_new_tokens):
            # 这里应该调用实际的模型推理
            # 简化为生成占位符
            new_token = i + 1000  # placeholder
            output_tokens.append(new_token)
            
            if self._latency_metric:
                self._latency_metric.token_generated()
        
        # 4. 计算指标
        metrics = None
        if self._throughput_metric:
            self._throughput_metric.record(
                tokens=len(output_tokens),
                requests=1,
            )
            throughput_result = self._throughput_metric.compute()
            
            latency_result = self._latency_metric.compute() if self._latency_metric else None
            
            metrics = {
                "throughput_tps": throughput_result.tokens_per_second,
                "ttft_ms": latency_result.ttft_ms if latency_result else 0,
                "tpot_ms": latency_result.tpot_ms if latency_result else 0,
            }
        
        # 5. 提交 KV 供复用
        if self.config.kv_cache.enable_prefix_caching:
            # 分配新的 KV 块
            new_blocks = self._kv_pool.allocate(
                sequence_id=hash(request.request_id),
                request_id=request.request_id,
                num_tokens=len(request.prompt_tokens),
                layer_ids=list(range(self.config.model.num_layers)),
            )
            self._kv_cache.commit(
                request_id=request.request_id,
                token_ids=request.prompt_tokens,
                blocks=new_blocks,
            )
        
        return GenerateOutput(
            request_id=request.request_id,
            output_tokens=output_tokens,
            finish_reason="length",
            metrics=metrics,
        )
    
    async def generate_async(
        self,
        request: GenerateRequest,
    ) -> GenerateOutput:
        """异步生成"""
        # 简化实现：包装同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)
    
    async def generate_stream(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[int]:
        """流式生成"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        
        # 简化实现：逐 token yield
        for i in range(request.max_new_tokens):
            yield i + 1000  # placeholder
            await asyncio.sleep(0.001)  # 模拟延迟
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "initialized": self._initialized,
            "backend": self._backend.backend_type.name if self._backend else None,
        }
        
        if self._kv_pool:
            stats["kv_pool"] = self._kv_pool.get_stats()
        
        if self._kv_cache:
            stats["kv_cache"] = self._kv_cache.get_stats()
        
        return stats
    
    def shutdown(self) -> None:
        """关闭引擎"""
        logger.info("Shutting down engine")
        
        # 清理资源
        if self._backend:
            self._backend.empty_cache()
        
        self._initialized = False
```

### 3. 示例代码 (`examples/basic_inference.py`)

```python
#!/usr/bin/env python3
"""sageLLM 基本推理示例

演示如何使用 sageLLM 引擎进行推理。
"""

import logging
from sageLLM.config import SageLLMConfig, ModelConfig, KVCacheConfig
from sageLLM.engine import SageLLMEngine, GenerateRequest

logging.basicConfig(level=logging.INFO)


def main():
    # 1. 创建配置
    config = SageLLMConfig(
        model=ModelConfig(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            num_layers=32,
            num_heads=32,
            hidden_size=4096,
        ),
        kv_cache=KVCacheConfig(
            max_tokens=65536,
            enable_prefix_caching=True,
        ),
    )
    
    # 2. 初始化引擎
    engine = SageLLMEngine(config)
    engine.initialize()
    
    print(f"Engine stats: {engine.get_stats()}")
    
    # 3. 发送请求
    request = GenerateRequest(
        request_id="test_001",
        prompt_tokens=[1, 2, 3, 4, 5],  # 实际应该是 tokenized 的输入
        max_new_tokens=50,
    )
    
    output = engine.generate(request)
    
    print(f"\nGeneration result:")
    print(f"  Request ID: {output.request_id}")
    print(f"  Output tokens: {len(output.output_tokens)}")
    print(f"  Finish reason: {output.finish_reason}")
    
    if output.metrics:
        print(f"  Throughput: {output.metrics['throughput_tps']:.1f} tokens/s")
        print(f"  TTFT: {output.metrics['ttft_ms']:.1f} ms")
        print(f"  TPOT: {output.metrics['tpot_ms']:.1f} ms")
    
    # 4. 测试 KV 复用
    print("\n--- Testing KV reuse ---")
    
    # 使用相同前缀的请求
    request2 = GenerateRequest(
        request_id="test_002",
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7],  # 包含相同前缀
        max_new_tokens=30,
    )
    
    output2 = engine.generate(request2)
    print(f"Second request completed with KV reuse")
    
    # 5. 查看统计
    print(f"\nFinal stats: {engine.get_stats()}")
    
    # 6. 关闭
    engine.shutdown()


if __name__ == "__main__":
    main()
```

### 4. 更新主入口 (`__init__.py`)

```python
"""sageLLM: SAGE 自研 LLM 推理运行时

sageLLM 提供高性能 LLM 推理能力，支持：
- PD 分离调度
- 多粒度 KV 缓存管理
- 模型量化与稀疏
- 国产芯片支持

Quick Start:
    from sageLLM import SageLLMEngine, SageLLMConfig, ModelConfig
    
    config = SageLLMConfig(
        model=ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct"),
    )
    
    engine = SageLLMEngine(config)
    engine.initialize()
    
    output = engine.generate(GenerateRequest(
        request_id="1",
        prompt_tokens=[1, 2, 3],
    ))
"""

__version__ = "0.1.0"

# 配置
from .config import (
    SageLLMConfig,
    ModelConfig,
    KVCacheConfig,
    SchedulerConfig,
    BackendConfig,
    BenchmarkConfig,
    InferenceMode,
)

# 引擎
from .engine import (
    SageLLMEngine,
    GenerateRequest,
    GenerateOutput,
)

# 子模块
from . import runtime
from . import kv_runtime
from . import accel
from . import backends
from . import benchmarks

__all__ = [
    # 版本
    "__version__",
    # 配置
    "SageLLMConfig",
    "ModelConfig",
    "KVCacheConfig",
    "SchedulerConfig",
    "BackendConfig",
    "BenchmarkConfig",
    "InferenceMode",
    # 引擎
    "SageLLMEngine",
    "GenerateRequest",
    "GenerateOutput",
    # 子模块
    "runtime",
    "kv_runtime",
    "accel",
    "backends",
    "benchmarks",
]
```

### 5. 集成测试

创建 `tests/integration/test_engine.py`：

```python
import pytest
from sageLLM import (
    SageLLMEngine, SageLLMConfig, ModelConfig,
    GenerateRequest, KVCacheConfig,
)


class TestEngineIntegration:
    """引擎集成测试"""
    
    @pytest.fixture
    def engine(self):
        """创建测试引擎"""
        config = SageLLMConfig(
            model=ModelConfig(
                model_id="test-model",
                num_layers=2,
                num_heads=2,
                hidden_size=64,
            ),
            kv_cache=KVCacheConfig(
                max_tokens=1024,
                enable_prefix_caching=True,
            ),
        )
        engine = SageLLMEngine(config)
        engine.initialize()
        yield engine
        engine.shutdown()
    
    def test_basic_generate(self, engine):
        """测试基本生成"""
        request = GenerateRequest(
            request_id="test_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        
        output = engine.generate(request)
        
        assert output.request_id == "test_1"
        assert len(output.output_tokens) == 10
        assert output.finish_reason == "length"
    
    def test_kv_reuse(self, engine):
        """测试 KV 复用"""
        # 第一个请求
        request1 = GenerateRequest(
            request_id="req_1",
            prompt_tokens=[1, 2, 3, 4, 5],
            max_new_tokens=5,
        )
        engine.generate(request1)
        
        # 第二个请求（相同前缀）
        request2 = GenerateRequest(
            request_id="req_2",
            prompt_tokens=[1, 2, 3, 4, 5, 6, 7],
            max_new_tokens=5,
        )
        output2 = engine.generate(request2)
        
        assert output2.finish_reason == "length"
        
        # 检查 KV 缓存命中
        stats = engine.get_stats()
        kv_stats = stats.get("kv_cache", {})
        assert kv_stats.get("cache_hits", 0) > 0
    
    def test_metrics(self, engine):
        """测试指标收集"""
        request = GenerateRequest(
            request_id="test_metrics",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=20,
        )
        
        output = engine.generate(request)
        
        assert output.metrics is not None
        assert "throughput_tps" in output.metrics
        assert "ttft_ms" in output.metrics
        assert output.metrics["throughput_tps"] > 0


@pytest.mark.asyncio
class TestAsyncEngine:
    """异步引擎测试"""
    
    @pytest.fixture
    def engine(self):
        config = SageLLMConfig(
            model=ModelConfig(
                model_id="test-model",
                num_layers=2,
                num_heads=2,
                hidden_size=64,
            ),
        )
        engine = SageLLMEngine(config)
        engine.initialize()
        yield engine
        engine.shutdown()
    
    async def test_async_generate(self, engine):
        """测试异步生成"""
        request = GenerateRequest(
            request_id="async_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        
        output = await engine.generate_async(request)
        
        assert output.request_id == "async_1"
        assert len(output.output_tokens) == 10
    
    async def test_streaming(self, engine):
        """测试流式生成"""
        request = GenerateRequest(
            request_id="stream_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )
        
        tokens = []
        async for token in engine.generate_stream(request):
            tokens.append(token)
        
        assert len(tokens) == 5
```

---

## 验收标准

- [ ] 引擎初始化成功，所有组件正确加载
- [ ] 基本生成功能正常
- [ ] KV 缓存复用正常工作
- [ ] 指标正确收集
- [ ] 异步和流式 API 正常
- [ ] 集成测试全部通过
- [ ] 示例代码可运行

---

## 输出物清单

```
sageLLM/
├── __init__.py              # ✅ 更新
├── config.py                # ✅ 统一配置
├── engine.py                # ✅ 推理引擎
└── examples/
    ├── __init__.py
    └── basic_inference.py   # ✅ 示例

tests/integration/
└── test_engine.py           # ✅ 集成测试
```

---

## 后续工作

完成 Task 6 后，整个 sageLLM 模块重构完成。后续工作：

1. **性能优化**: 实现真正的模型加载和推理
2. **分布式**: 添加 TP/PP 支持
3. **文档**: 完善 API 文档
4. **CI/CD**: 添加性能回归测试
