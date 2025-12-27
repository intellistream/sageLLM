# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
sageLLM - Self-developed LLM Inference Runtime.

sageLLM is a hardware-agnostic inference runtime designed for:
- Prefill-Decode (PD) disaggregation
- Multi-accelerator support (NVIDIA, Ascend, Cambricon, Hygon)
- Efficient KV cache management
- High-throughput batch inference

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    sageLLM Architecture                     │
    ├─────────────────────────────────────────────────────────────┤
    │  control_plane/         Request scheduling & orchestration  │
    │  runtime/               Self-developed inference runtime    │
    │    ├── execution_graph/   PD-separated execution IR        │
    │    ├── comm/              Communication layer              │
    │    └── scheduler/         Request scheduling               │
    │  backends/              Hardware abstraction layer          │
    │    ├── cuda/              NVIDIA GPU support               │
    │    ├── ascend/            Huawei Ascend NPU support        │
    │    ├── cambricon/         Cambricon MLU support            │
    │    └── hygon/             Hygon DCU support                │
    │  third_party_engines/   Optional third-party backends       │
    │    ├── lmdeploy/          LMDeploy/TurboMind               │
    │    ├── vllm/              vLLM (PagedAttention)            │
    │    └── vllm_ascend/       vLLM for Ascend NPU              │
    │  kv_runtime/            KV cache runtime                    │
    │  kv_policy/             KV cache eviction policies          │
    │  accel/                 Accelerator utilities               │
    └─────────────────────────────────────────────────────────────┘

Research Topics:
    1. PD Disaggregation: Separate prefill and decode for optimal resource usage
    2. Domestic Accelerator Support: Unified interface for various hardware
    3. KV Cache Optimization: Efficient memory management and reuse
    4. Scheduling Algorithms: SLO-aware, priority-based scheduling

Example:
    >>> from sageLLM import backends, runtime, third_party_engines
    >>> backend = backends.get_backend("cuda")
    >>> scheduler = runtime.scheduler.PDScheduler(config)
    >>> # Or use third-party engine for comparison
    >>> engine = third_party_engines.get_engine("vllm", model="...")
"""

from __future__ import annotations

__version__ = "0.1.0"

# Configuration
# Core modules
from . import (
    accel,
    backends,
    benchmarks,
    control_plane,
    core,
    kv_policy,
    kv_runtime,
    prefix_reuse,
    runtime,
    third_party_engines,
)
from .config import (
    BackendConfig,
    BenchmarkConfig,
    InferenceMode,
    KVCacheConfig,
    ModelConfig,
    SageLLMConfig,
    SchedulerConfig,
)

# Engine
from .engine import GenerateOutput, GenerateRequest, SageLLMEngine

__all__ = [
    "__version__",
    # Configuration
    "SageLLMConfig",
    "ModelConfig",
    "KVCacheConfig",
    "SchedulerConfig",
    "BackendConfig",
    "BenchmarkConfig",
    "InferenceMode",
    # Engine
    "SageLLMEngine",
    "GenerateRequest",
    "GenerateOutput",
    # Core infrastructure
    "core",
    "control_plane",
    # Self-developed runtime
    "runtime",
    "backends",
    # KV cache management
    "kv_runtime",
    "kv_policy",
    "prefix_reuse",
    # Acceleration
    "accel",
    # Third-party engines (for comparison/fallback)
    "third_party_engines",
    # Benchmarks
    "benchmarks",
]
