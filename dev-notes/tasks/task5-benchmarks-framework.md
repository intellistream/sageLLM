# Task 5: benchmarks/ 统一评测框架

**状态**: 🔲 待开始  
**预计时间**: 3h  
**课题对应**: 4.1-4.3 评测指标  
**可并行**: ✅ 是（与 Task 1-4 并行）

---

## 背景

课题 4.1-4.3 需要统一的评测指标：
- **4.1**: 通信效率、PD 分离收益
- **4.2**: KV 命中率、迁移流量
- **4.3**: 量化误差、稀疏加速比、MFU

本任务创建 `benchmarks/` 模块，提供统一的评测框架。

---

## 工作目录

```
/home/shuhao/SAGE/packages/sage-common/src/sage/common/components/sage_llm/sageLLM/benchmarks/
├── __init__.py
├── metrics/                 # 指标定义
│   ├── __init__.py
│   ├── throughput.py       # 吞吐量指标
│   ├── latency.py          # 延迟指标
│   ├── memory.py           # 内存指标
│   ├── kv_cache.py         # KV 缓存指标
│   └── mfu.py              # MFU 计算
├── profiler/               # 性能分析
│   ├── __init__.py
│   └── trace.py            # 执行追踪
├── ci/                     # CI 集成
│   ├── __init__.py
│   └── gates.py            # 性能门控
└── reporters/              # 报告生成
    ├── __init__.py
    ├── console.py          # 控制台输出
    └── json_reporter.py    # JSON 报告
```

---

## 参考资料

- vLLM Benchmarks: https://github.com/vllm-project/vllm/tree/main/benchmarks
- MLPerf Inference: https://github.com/mlcommons/inference
- LLMPerf: https://github.com/ray-project/llmperf
- SAGE benchmark_control_plane: `packages/sage-benchmark/`

---

## 任务清单

### 1. 基础指标定义 (`metrics/__init__.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Generic, TypeVar
from enum import Enum, auto
import time

T = TypeVar("T")


class MetricType(Enum):
    """指标类型"""
    THROUGHPUT = auto()      # 吞吐量
    LATENCY = auto()         # 延迟
    MEMORY = auto()          # 内存
    KV_CACHE = auto()        # KV 缓存
    COMPUTE = auto()         # 计算效率
    COMMUNICATION = auto()   # 通信效率


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f} {self.unit}"


@dataclass
class MetricSummary:
    """指标摘要（聚合多次测量）"""
    name: str
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p99: float
    count: int
    unit: str
    
    @classmethod
    def from_values(cls, name: str, values: List[float], unit: str) -> "MetricSummary":
        """从值列表创建摘要"""
        import numpy as np
        arr = np.array(values)
        return cls(
            name=name,
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            count=len(values),
            unit=unit,
        )


class Metric(ABC, Generic[T]):
    """指标基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称"""
        ...
    
    @property
    @abstractmethod
    def unit(self) -> str:
        """指标单位"""
        ...
    
    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """指标类型"""
        ...
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> T:
        """计算指标值"""
        ...
    
    def to_metric_value(self, value: float) -> MetricValue:
        """转换为 MetricValue"""
        return MetricValue(
            name=self.name,
            value=value,
            unit=self.unit,
        )


class MetricRegistry:
    """指标注册表"""
    
    _metrics: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册指标"""
        def decorator(metric_cls):
            cls._metrics[name] = metric_cls
            return metric_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Metric:
        """获取指标实例"""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return cls._metrics[name]()
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有指标"""
        return list(cls._metrics.keys())
```

### 2. 吞吐量指标 (`metrics/throughput.py`)

```python
from dataclasses import dataclass
from typing import Optional
import time

from . import Metric, MetricType, MetricRegistry, MetricValue


@dataclass
class ThroughputResult:
    """吞吐量结果"""
    tokens_per_second: float
    requests_per_second: float
    total_tokens: int
    total_requests: int
    duration_s: float


@MetricRegistry.register("throughput")
class ThroughputMetric(Metric[ThroughputResult]):
    """吞吐量指标
    
    测量：
    - Tokens/s (TPS)
    - Requests/s (QPS)
    """
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._total_tokens = 0
        self._total_requests = 0
    
    @property
    def name(self) -> str:
        return "throughput"
    
    @property
    def unit(self) -> str:
        return "tokens/s"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.THROUGHPUT
    
    def start(self) -> None:
        """开始计时"""
        self._start_time = time.perf_counter()
        self._total_tokens = 0
        self._total_requests = 0
    
    def record(self, tokens: int, requests: int = 1) -> None:
        """记录生成的 token 数"""
        self._total_tokens += tokens
        self._total_requests += requests
    
    def compute(self) -> ThroughputResult:
        """计算吞吐量"""
        if self._start_time is None:
            raise RuntimeError("Call start() first")
        
        duration = time.perf_counter() - self._start_time
        
        return ThroughputResult(
            tokens_per_second=self._total_tokens / duration if duration > 0 else 0,
            requests_per_second=self._total_requests / duration if duration > 0 else 0,
            total_tokens=self._total_tokens,
            total_requests=self._total_requests,
            duration_s=duration,
        )


@MetricRegistry.register("decode_throughput")
class DecodeThroughputMetric(Metric[float]):
    """Decode 阶段吞吐量
    
    单独测量 decode（自回归生成）阶段的吞吐量，
    排除 prefill 的影响。
    """
    
    @property
    def name(self) -> str:
        return "decode_throughput"
    
    @property
    def unit(self) -> str:
        return "tokens/s"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.THROUGHPUT
    
    def compute(
        self,
        decode_tokens: int,
        decode_time_s: float,
    ) -> float:
        """计算 decode 吞吐量"""
        if decode_time_s <= 0:
            return 0.0
        return decode_tokens / decode_time_s
```

### 3. 延迟指标 (`metrics/latency.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional
import time

from . import Metric, MetricType, MetricRegistry, MetricSummary


@dataclass
class LatencyResult:
    """延迟结果"""
    ttft_ms: float      # Time To First Token
    tpot_ms: float      # Time Per Output Token
    e2e_ms: float       # End-to-End latency
    prefill_ms: float   # Prefill 阶段延迟
    decode_ms: float    # Decode 阶段总延迟


@MetricRegistry.register("latency")
class LatencyMetric(Metric[LatencyResult]):
    """延迟指标
    
    测量：
    - TTFT (Time To First Token): 首 token 延迟
    - TPOT (Time Per Output Token): 平均每 token 延迟
    - E2E (End-to-End): 端到端延迟
    """
    
    def __init__(self):
        self._request_start: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._prefill_end: Optional[float] = None
        self._decode_token_count = 0
    
    @property
    def name(self) -> str:
        return "latency"
    
    @property
    def unit(self) -> str:
        return "ms"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.LATENCY
    
    def request_start(self) -> None:
        """请求开始"""
        self._request_start = time.perf_counter()
        self._first_token_time = None
        self._prefill_end = None
        self._decode_token_count = 0
    
    def prefill_done(self) -> None:
        """Prefill 完成"""
        self._prefill_end = time.perf_counter()
    
    def first_token(self) -> None:
        """首 token 生成"""
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
    
    def token_generated(self) -> None:
        """Token 生成"""
        self._decode_token_count += 1
    
    def compute(self) -> LatencyResult:
        """计算延迟指标"""
        now = time.perf_counter()
        
        if self._request_start is None:
            raise RuntimeError("Call request_start() first")
        
        # TTFT
        ttft = (self._first_token_time - self._request_start) * 1000 if self._first_token_time else 0
        
        # Prefill
        prefill = (self._prefill_end - self._request_start) * 1000 if self._prefill_end else 0
        
        # Decode
        decode_start = self._first_token_time or self._request_start
        decode = (now - decode_start) * 1000
        
        # TPOT
        tpot = decode / self._decode_token_count if self._decode_token_count > 0 else 0
        
        # E2E
        e2e = (now - self._request_start) * 1000
        
        return LatencyResult(
            ttft_ms=ttft,
            tpot_ms=tpot,
            e2e_ms=e2e,
            prefill_ms=prefill,
            decode_ms=decode,
        )


@MetricRegistry.register("latency_percentiles")
class LatencyPercentilesMetric(Metric[MetricSummary]):
    """延迟分位数指标
    
    聚合多次测量，计算 P50/P90/P99。
    """
    
    def __init__(self):
        self._values: List[float] = []
    
    @property
    def name(self) -> str:
        return "latency_percentiles"
    
    @property
    def unit(self) -> str:
        return "ms"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.LATENCY
    
    def record(self, latency_ms: float) -> None:
        """记录一次延迟"""
        self._values.append(latency_ms)
    
    def reset(self) -> None:
        """重置"""
        self._values.clear()
    
    def compute(self) -> MetricSummary:
        """计算分位数"""
        if not self._values:
            raise RuntimeError("No values recorded")
        
        return MetricSummary.from_values(
            name=self.name,
            values=self._values,
            unit=self.unit,
        )
```

### 4. KV 缓存指标 (`metrics/kv_cache.py`)

```python
from dataclasses import dataclass
from typing import Dict

from . import Metric, MetricType, MetricRegistry


@dataclass
class KVCacheResult:
    """KV 缓存指标结果"""
    # 命中率
    hit_rate: float              # 总体命中率
    prefix_hit_rate: float       # 前缀命中率
    
    # 内存使用
    hbm_used_gb: float
    ddr_used_gb: float
    nvme_used_gb: float
    total_used_gb: float
    
    # 迁移
    migration_count: int
    migration_bytes: int
    
    # 复用
    reused_tokens: int
    total_tokens: int
    reuse_ratio: float


@MetricRegistry.register("kv_cache")
class KVCacheMetric(Metric[KVCacheResult]):
    """KV 缓存指标
    
    测量：
    - 命中率
    - 内存使用（按层级）
    - 迁移流量
    - 复用率
    """
    
    def __init__(self):
        self._hits = 0
        self._misses = 0
        self._prefix_hits = 0
        self._prefix_lookups = 0
        self._migrations = 0
        self._migration_bytes = 0
        self._reused_tokens = 0
        self._total_tokens = 0
        self._tier_usage: Dict[str, float] = {}
    
    @property
    def name(self) -> str:
        return "kv_cache"
    
    @property
    def unit(self) -> str:
        return ""  # 多种单位
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.KV_CACHE
    
    def record_hit(self) -> None:
        """记录缓存命中"""
        self._hits += 1
    
    def record_miss(self) -> None:
        """记录缓存未命中"""
        self._misses += 1
    
    def record_prefix_lookup(self, hit: bool) -> None:
        """记录前缀查找"""
        self._prefix_lookups += 1
        if hit:
            self._prefix_hits += 1
    
    def record_migration(self, bytes_migrated: int) -> None:
        """记录迁移"""
        self._migrations += 1
        self._migration_bytes += bytes_migrated
    
    def record_reuse(self, reused: int, total: int) -> None:
        """记录 token 复用"""
        self._reused_tokens += reused
        self._total_tokens += total
    
    def update_tier_usage(self, tier: str, used_gb: float) -> None:
        """更新层级使用"""
        self._tier_usage[tier] = used_gb
    
    def compute(self) -> KVCacheResult:
        """计算 KV 缓存指标"""
        total_lookups = self._hits + self._misses
        hit_rate = self._hits / total_lookups if total_lookups > 0 else 0
        
        prefix_hit_rate = self._prefix_hits / self._prefix_lookups if self._prefix_lookups > 0 else 0
        
        reuse_ratio = self._reused_tokens / self._total_tokens if self._total_tokens > 0 else 0
        
        return KVCacheResult(
            hit_rate=hit_rate,
            prefix_hit_rate=prefix_hit_rate,
            hbm_used_gb=self._tier_usage.get("HBM", 0),
            ddr_used_gb=self._tier_usage.get("DDR", 0),
            nvme_used_gb=self._tier_usage.get("NVME", 0),
            total_used_gb=sum(self._tier_usage.values()),
            migration_count=self._migrations,
            migration_bytes=self._migration_bytes,
            reused_tokens=self._reused_tokens,
            total_tokens=self._total_tokens,
            reuse_ratio=reuse_ratio,
        )
```

### 5. MFU 计算 (`metrics/mfu.py`)

```python
from dataclasses import dataclass
from typing import Optional

from . import Metric, MetricType, MetricRegistry


@dataclass
class MFUResult:
    """MFU 结果"""
    mfu: float              # Model FLOPs Utilization (0-1)
    achieved_tflops: float  # 实际达到的 TFLOPS
    peak_tflops: float      # 峰值 TFLOPS
    model_flops: int        # 模型 FLOPs
    duration_s: float       # 测量时间


@MetricRegistry.register("mfu")
class MFUMetric(Metric[MFUResult]):
    """Model FLOPs Utilization (MFU) 指标
    
    MFU = 实际 FLOPs / 理论峰值 FLOPs
    
    对于 Transformer:
    - Forward FLOPs ≈ 2 * params * tokens
    - Backward FLOPs ≈ 4 * params * tokens
    - Attention FLOPs = 4 * n_layers * n_heads * d_head * seq_len^2
    """
    
    @property
    def name(self) -> str:
        return "mfu"
    
    @property
    def unit(self) -> str:
        return "%"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.COMPUTE
    
    def compute(
        self,
        model_params: int,
        tokens_processed: int,
        duration_s: float,
        peak_tflops: float,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_head: Optional[int] = None,
        seq_len: Optional[int] = None,
        include_attention: bool = True,
    ) -> MFUResult:
        """计算 MFU
        
        Args:
            model_params: 模型参数量
            tokens_processed: 处理的 token 数
            duration_s: 耗时（秒）
            peak_tflops: 硬件峰值 TFLOPS
            n_layers: 层数（用于注意力计算）
            n_heads: 注意力头数
            d_head: 头维度
            seq_len: 序列长度
            include_attention: 是否包含注意力 FLOPs
        """
        # 基础 FLOPs（线性层）
        # Forward: 2 * params * tokens
        linear_flops = 2 * model_params * tokens_processed
        
        # 注意力 FLOPs
        attention_flops = 0
        if include_attention and all([n_layers, n_heads, d_head, seq_len]):
            # QKV projection: 3 * 4 * n_heads * d_head * seq_len (per layer)
            # Attention: 2 * n_heads * seq_len^2 * d_head (per layer)
            # Output projection: 4 * n_heads * d_head * seq_len (per layer)
            attention_flops = n_layers * (
                3 * 4 * n_heads * d_head * seq_len +
                2 * n_heads * seq_len * seq_len * d_head +
                4 * n_heads * d_head * seq_len
            )
        
        total_flops = linear_flops + attention_flops
        
        # 计算 achieved TFLOPS
        achieved_tflops = (total_flops / duration_s) / 1e12 if duration_s > 0 else 0
        
        # 计算 MFU
        mfu = achieved_tflops / peak_tflops if peak_tflops > 0 else 0
        
        return MFUResult(
            mfu=mfu,
            achieved_tflops=achieved_tflops,
            peak_tflops=peak_tflops,
            model_flops=total_flops,
            duration_s=duration_s,
        )


@MetricRegistry.register("mbu")
class MBUMetric(Metric[float]):
    """Memory Bandwidth Utilization (MBU) 指标
    
    MBU = 实际内存带宽 / 峰值内存带宽
    
    对于推理（memory-bound）：
    - 读取所有权重
    - 读写 KV cache
    - 读写激活
    """
    
    @property
    def name(self) -> str:
        return "mbu"
    
    @property
    def unit(self) -> str:
        return "%"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.MEMORY
    
    def compute(
        self,
        bytes_accessed: int,
        duration_s: float,
        peak_bandwidth_gbps: float,
    ) -> float:
        """计算 MBU
        
        Args:
            bytes_accessed: 访问的总字节数
            duration_s: 耗时（秒）
            peak_bandwidth_gbps: 峰值带宽（GB/s）
        """
        achieved_bandwidth = (bytes_accessed / duration_s) / 1e9 if duration_s > 0 else 0
        mbu = achieved_bandwidth / peak_bandwidth_gbps if peak_bandwidth_gbps > 0 else 0
        return mbu
```

### 6. CI 性能门控 (`ci/gates.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto


class GateStatus(Enum):
    """门控状态"""
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class GateResult:
    """门控检查结果"""
    name: str
    status: GateStatus
    expected: float
    actual: float
    threshold: float
    message: str
    
    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASSED


@dataclass
class PerformanceGateConfig:
    """性能门控配置"""
    # 吞吐量门控
    min_throughput_tps: Optional[float] = None      # 最小吞吐量
    
    # 延迟门控
    max_ttft_ms: Optional[float] = None             # 最大 TTFT
    max_tpot_ms: Optional[float] = None             # 最大 TPOT
    max_p99_latency_ms: Optional[float] = None      # 最大 P99 延迟
    
    # 内存门控
    max_memory_gb: Optional[float] = None           # 最大内存使用
    
    # KV 缓存门控
    min_kv_hit_rate: Optional[float] = None         # 最小命中率
    
    # MFU 门控
    min_mfu: Optional[float] = None                 # 最小 MFU
    
    # 回归门控（与基准比较）
    max_regression_pct: float = 5.0                 # 最大允许性能下降 %
    
    # 元数据
    baseline_commit: Optional[str] = None           # 基准 commit
    tags: List[str] = field(default_factory=list)


class PerformanceGate:
    """性能门控
    
    用于 CI 中的性能检查，确保性能不发生回归。
    """
    
    def __init__(self, config: PerformanceGateConfig):
        self.config = config
        self._results: List[GateResult] = []
    
    def check_throughput(self, actual_tps: float) -> GateResult:
        """检查吞吐量"""
        if self.config.min_throughput_tps is None:
            return GateResult(
                name="throughput",
                status=GateStatus.SKIPPED,
                expected=0,
                actual=actual_tps,
                threshold=0,
                message="No throughput gate configured",
            )
        
        passed = actual_tps >= self.config.min_throughput_tps
        result = GateResult(
            name="throughput",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            expected=self.config.min_throughput_tps,
            actual=actual_tps,
            threshold=self.config.min_throughput_tps,
            message=f"Throughput {actual_tps:.1f} TPS {'>='}{'<'} {self.config.min_throughput_tps:.1f} TPS",
        )
        self._results.append(result)
        return result
    
    def check_latency(
        self,
        ttft_ms: Optional[float] = None,
        tpot_ms: Optional[float] = None,
        p99_ms: Optional[float] = None,
    ) -> List[GateResult]:
        """检查延迟"""
        results = []
        
        if ttft_ms is not None and self.config.max_ttft_ms is not None:
            passed = ttft_ms <= self.config.max_ttft_ms
            results.append(GateResult(
                name="ttft",
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                expected=self.config.max_ttft_ms,
                actual=ttft_ms,
                threshold=self.config.max_ttft_ms,
                message=f"TTFT {ttft_ms:.1f} ms {'<='}{'>'} {self.config.max_ttft_ms:.1f} ms",
            ))
        
        if tpot_ms is not None and self.config.max_tpot_ms is not None:
            passed = tpot_ms <= self.config.max_tpot_ms
            results.append(GateResult(
                name="tpot",
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                expected=self.config.max_tpot_ms,
                actual=tpot_ms,
                threshold=self.config.max_tpot_ms,
                message=f"TPOT {tpot_ms:.1f} ms {'<='}{'>'} {self.config.max_tpot_ms:.1f} ms",
            ))
        
        if p99_ms is not None and self.config.max_p99_latency_ms is not None:
            passed = p99_ms <= self.config.max_p99_latency_ms
            results.append(GateResult(
                name="p99_latency",
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                expected=self.config.max_p99_latency_ms,
                actual=p99_ms,
                threshold=self.config.max_p99_latency_ms,
                message=f"P99 {p99_ms:.1f} ms {'<='}{'>'} {self.config.max_p99_latency_ms:.1f} ms",
            ))
        
        self._results.extend(results)
        return results
    
    def check_regression(
        self,
        metric_name: str,
        baseline: float,
        current: float,
        higher_is_better: bool = True,
    ) -> GateResult:
        """检查性能回归
        
        Args:
            metric_name: 指标名
            baseline: 基准值
            current: 当前值
            higher_is_better: 值越大越好
        """
        if baseline == 0:
            return GateResult(
                name=f"regression_{metric_name}",
                status=GateStatus.SKIPPED,
                expected=baseline,
                actual=current,
                threshold=0,
                message="No baseline available",
            )
        
        if higher_is_better:
            regression_pct = ((baseline - current) / baseline) * 100
        else:
            regression_pct = ((current - baseline) / baseline) * 100
        
        passed = regression_pct <= self.config.max_regression_pct
        
        result = GateResult(
            name=f"regression_{metric_name}",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            expected=baseline,
            actual=current,
            threshold=self.config.max_regression_pct,
            message=f"{metric_name} regression {regression_pct:.1f}% {'<='}{'>'} {self.config.max_regression_pct:.1f}%",
        )
        self._results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        passed = sum(1 for r in self._results if r.status == GateStatus.PASSED)
        failed = sum(1 for r in self._results if r.status == GateStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == GateStatus.SKIPPED)
        
        return {
            "total_checks": len(self._results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "all_passed": failed == 0,
            "results": [
                {
                    "name": r.name,
                    "status": r.status.name,
                    "message": r.message,
                }
                for r in self._results
            ],
        }
    
    def assert_all_passed(self) -> None:
        """断言所有检查通过（用于 CI）"""
        summary = self.get_summary()
        if not summary["all_passed"]:
            failed_msgs = [
                r["message"]
                for r in summary["results"]
                if r["status"] == "FAILED"
            ]
            raise AssertionError(
                f"Performance gate failed:\n" + "\n".join(failed_msgs)
            )
```

### 7. 控制台报告 (`reporters/console.py`)

```python
from typing import Dict, List, Any
from dataclasses import dataclass

from ..metrics import MetricValue, MetricSummary


@dataclass
class ConsoleReporterConfig:
    """控制台报告配置"""
    use_color: bool = True
    show_percentiles: bool = True
    precision: int = 4


class ConsoleReporter:
    """控制台报告器"""
    
    COLORS = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    
    def __init__(self, config: ConsoleReporterConfig = None):
        self.config = config or ConsoleReporterConfig()
    
    def _color(self, text: str, color: str) -> str:
        if not self.config.use_color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def report_metric(self, metric: MetricValue) -> str:
        """报告单个指标"""
        return f"  {metric.name}: {metric.value:.{self.config.precision}f} {metric.unit}"
    
    def report_summary(self, summary: MetricSummary) -> str:
        """报告指标摘要"""
        lines = [
            f"  {summary.name}:",
            f"    mean: {summary.mean:.{self.config.precision}f} {summary.unit}",
            f"    std:  {summary.std:.{self.config.precision}f} {summary.unit}",
            f"    min:  {summary.min:.{self.config.precision}f} {summary.unit}",
            f"    max:  {summary.max:.{self.config.precision}f} {summary.unit}",
        ]
        
        if self.config.show_percentiles:
            lines.extend([
                f"    p50:  {summary.p50:.{self.config.precision}f} {summary.unit}",
                f"    p90:  {summary.p90:.{self.config.precision}f} {summary.unit}",
                f"    p99:  {summary.p99:.{self.config.precision}f} {summary.unit}",
            ])
        
        lines.append(f"    count: {summary.count}")
        
        return "\n".join(lines)
    
    def report_benchmark(
        self,
        name: str,
        metrics: Dict[str, Any],
        duration_s: float,
    ) -> str:
        """报告完整 benchmark 结果"""
        lines = [
            self._color(f"\n{'='*60}", "blue"),
            self._color(f"Benchmark: {name}", "blue"),
            self._color(f"{'='*60}", "blue"),
            f"Duration: {duration_s:.2f}s",
            "",
            "Metrics:",
        ]
        
        for key, value in metrics.items():
            if isinstance(value, MetricValue):
                lines.append(self.report_metric(value))
            elif isinstance(value, MetricSummary):
                lines.append(self.report_summary(value))
            elif isinstance(value, (int, float)):
                lines.append(f"  {key}: {value:.{self.config.precision}f}")
            else:
                lines.append(f"  {key}: {value}")
        
        lines.append(self._color(f"{'='*60}\n", "blue"))
        
        return "\n".join(lines)
    
    def report_gate_results(self, summary: Dict[str, Any]) -> str:
        """报告门控结果"""
        lines = [
            self._color("\nPerformance Gate Results:", "blue"),
            f"  Total: {summary['total_checks']}",
        ]
        
        if summary['passed'] > 0:
            lines.append(self._color(f"  Passed: {summary['passed']}", "green"))
        if summary['failed'] > 0:
            lines.append(self._color(f"  Failed: {summary['failed']}", "red"))
        if summary['skipped'] > 0:
            lines.append(self._color(f"  Skipped: {summary['skipped']}", "yellow"))
        
        lines.append("")
        
        for result in summary['results']:
            if result['status'] == 'PASSED':
                icon = self._color("✓", "green")
            elif result['status'] == 'FAILED':
                icon = self._color("✗", "red")
            else:
                icon = self._color("-", "yellow")
            
            lines.append(f"  {icon} {result['message']}")
        
        return "\n".join(lines)
```

---

## 单元测试要求

创建 `tests/unit/test_benchmarks.py`：

```python
import pytest
import time
from sageLLM.benchmarks.metrics import MetricRegistry, MetricSummary
from sageLLM.benchmarks.metrics.throughput import ThroughputMetric
from sageLLM.benchmarks.metrics.latency import LatencyMetric
from sageLLM.benchmarks.metrics.mfu import MFUMetric
from sageLLM.benchmarks.ci.gates import PerformanceGate, PerformanceGateConfig


class TestThroughputMetric:
    """吞吐量指标测试"""
    
    def test_basic_throughput(self):
        """测试基本吞吐量计算"""
        metric = ThroughputMetric()
        metric.start()
        metric.record(tokens=1000, requests=10)
        time.sleep(0.1)  # 模拟处理时间
        result = metric.compute()
        
        assert result.total_tokens == 1000
        assert result.total_requests == 10
        assert result.tokens_per_second > 0


class TestLatencyMetric:
    """延迟指标测试"""
    
    def test_ttft(self):
        """测试 TTFT 计算"""
        metric = LatencyMetric()
        metric.request_start()
        time.sleep(0.05)
        metric.first_token()
        metric.prefill_done()
        
        for _ in range(10):
            metric.token_generated()
            time.sleep(0.001)
        
        result = metric.compute()
        
        assert result.ttft_ms > 40  # 至少 50ms
        assert result.prefill_ms > 0


class TestMFUMetric:
    """MFU 指标测试"""
    
    def test_mfu_calculation(self):
        """测试 MFU 计算"""
        metric = MFUMetric()
        
        result = metric.compute(
            model_params=7_000_000_000,  # 7B
            tokens_processed=1024,
            duration_s=1.0,
            peak_tflops=312.0,  # A100
        )
        
        assert result.mfu > 0
        assert result.mfu <= 1.0
        assert result.achieved_tflops > 0


class TestPerformanceGate:
    """性能门控测试"""
    
    def test_throughput_gate_pass(self):
        """测试吞吐量门控通过"""
        config = PerformanceGateConfig(min_throughput_tps=100.0)
        gate = PerformanceGate(config)
        
        result = gate.check_throughput(actual_tps=150.0)
        
        assert result.passed
    
    def test_throughput_gate_fail(self):
        """测试吞吐量门控失败"""
        config = PerformanceGateConfig(min_throughput_tps=100.0)
        gate = PerformanceGate(config)
        
        result = gate.check_throughput(actual_tps=50.0)
        
        assert not result.passed
    
    def test_regression_check(self):
        """测试回归检查"""
        config = PerformanceGateConfig(max_regression_pct=5.0)
        gate = PerformanceGate(config)
        
        # 3% 回归，应该通过
        result = gate.check_regression(
            metric_name="throughput",
            baseline=100.0,
            current=97.0,
            higher_is_better=True,
        )
        assert result.passed
        
        # 10% 回归，应该失败
        result = gate.check_regression(
            metric_name="throughput",
            baseline=100.0,
            current=90.0,
            higher_is_better=True,
        )
        assert not result.passed


class TestMetricSummary:
    """指标摘要测试"""
    
    def test_from_values(self):
        """测试从值列表创建摘要"""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        summary = MetricSummary.from_values("test", values, "ms")
        
        assert summary.mean == 55.0
        assert summary.min == 10.0
        assert summary.max == 100.0
        assert summary.count == 10
```

---

## 接口约定

### 输入接口

| 接口 | 来源 | 说明 |
|------|------|------|
| 测量数据 | runtime/scheduler | 执行时间、token 数等 |
| KV 统计 | kv_runtime | 命中率、迁移等 |
| 硬件规格 | backends | 峰值性能 |

### 输出接口

| 接口 | 目标 | 说明 |
|------|------|------|
| `MetricValue` | reporters | 单次测量结果 |
| `MetricSummary` | reporters | 聚合结果 |
| `GateResult` | CI | 门控检查结果 |

---

## 验收标准

- [ ] 吞吐量指标：正确计算 TPS/QPS
- [ ] 延迟指标：正确计算 TTFT/TPOT/P99
- [ ] MFU 指标：计算误差 < 10%
- [ ] CI 门控：正确判断 pass/fail
- [ ] 报告生成：控制台 + JSON 格式
- [ ] 单元测试覆盖率 > 80%

---

## 输出物清单

```
benchmarks/
├── __init__.py
├── metrics/
│   ├── __init__.py           # ✅ 基础定义
│   ├── throughput.py         # ✅ 吞吐量
│   ├── latency.py            # ✅ 延迟
│   ├── memory.py             # （可选）
│   ├── kv_cache.py           # ✅ KV 缓存
│   └── mfu.py                # ✅ MFU/MBU
├── profiler/
│   ├── __init__.py
│   └── trace.py              # （后续添加）
├── ci/
│   ├── __init__.py
│   └── gates.py              # ✅ 性能门控
└── reporters/
    ├── __init__.py
    ├── console.py            # ✅ 控制台
    └── json_reporter.py      # （可选）

tests/unit/
└── test_benchmarks.py        # ✅ 测试文件
```
