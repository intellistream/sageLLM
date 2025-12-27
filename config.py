# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sageLLM unified configuration."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .backends.protocols import BackendType


class InferenceMode(Enum):
    """Inference mode."""

    STANDARD = auto()  # Standard inference
    PREFILL_ONLY = auto()  # Prefill only
    DECODE_ONLY = auto()  # Decode only
    PD_SEPARATE = auto()  # PD separation


@dataclass
class ModelConfig:
    """Model configuration."""

    model_id: str

    # Model structure
    num_layers: int = 32
    num_heads: int = 32
    hidden_size: int = 4096
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Precision
    dtype: str = "float16"

    # Quantization
    quantization: Optional[str] = None
    quantization_config: dict[str, Any] = field(default_factory=dict)

    # Sparsity
    sparsity_pattern: Optional[str] = None
    sparsity_ratio: float = 0.0


@dataclass
class KVCacheConfig:
    """KV cache configuration."""

    # Capacity
    max_tokens: int = 65536
    block_size: int = 16

    # Granularity
    granularity: str = "block"  # "block", "token", "sequence"

    # Tiered storage
    enable_tiering: bool = False
    hbm_ratio: float = 0.7  # HBM ratio
    ddr_ratio: float = 0.2  # DDR ratio
    nvme_ratio: float = 0.1  # NVMe ratio
    nvme_path: Optional[str] = None

    # Reuse
    enable_prefix_caching: bool = True
    enable_cross_request_sharing: bool = True

    # Migration
    enable_migration: bool = True
    hot_threshold: float = 1.0  # Hot block access frequency threshold
    cold_timeout_s: float = 60.0  # Cold block timeout


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    # Mode
    mode: str = "hybrid"  # "fifo", "priority", "pd_separate", "hybrid"

    # PD separation
    prefill_workers: int = 1
    decode_workers: int = 1

    # Batching
    max_batch_size: int = 64
    max_prefill_batch: int = 8
    max_decode_batch: int = 64

    # Timeout
    request_timeout_s: float = 60.0
    queue_timeout_s: float = 30.0


@dataclass
class BackendConfig:
    """Backend configuration."""

    # Hardware
    backend_type: Optional[BackendType] = None  # None = auto-detect
    device_ids: list[int] = field(default_factory=lambda: [0])

    # Distributed
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    # Enable
    enable_profiling: bool = False
    enable_metrics: bool = True

    # CI gates
    enable_gates: bool = False
    min_throughput_tps: Optional[float] = None
    max_ttft_ms: Optional[float] = None
    max_tpot_ms: Optional[float] = None


@dataclass
class SageLLMConfig:
    """sageLLM unified configuration."""

    model: ModelConfig
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Inference mode
    inference_mode: InferenceMode = InferenceMode.STANDARD

    @classmethod
    def from_yaml(cls, path: str) -> "SageLLMConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SageLLMConfig":
        """Create configuration from dictionary."""
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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        import dataclasses

        return dataclasses.asdict(self)
