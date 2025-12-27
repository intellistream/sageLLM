# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Memory metrics (weights, KV cache, activations)."""

from __future__ import annotations

from dataclasses import dataclass

from . import Metric, MetricRegistry, MetricType


@dataclass
class MemoryResult:
    """Memory usage measurement.

    Attributes:
        weight_memory_gb: Model weight memory (GB)
        kv_cache_memory_gb: KV cache memory (GB)
        activation_memory_gb: Peak activation memory (GB)
        total_memory_gb: Total GPU memory used (GB)
        memory_utilization: Memory utilization ratio (0-1)
    """

    weight_memory_gb: float
    kv_cache_memory_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    memory_utilization: float


@MetricRegistry.register("memory")
class MemoryMetric(Metric[MemoryResult]):
    """Measures GPU memory usage.

    Tracks:
    - Model weight memory
    - KV cache memory
    - Activation memory
    - Total memory usage

    Example:
        >>> metric = MemoryMetric()
        >>> result = metric.compute(
        ...     weight_bytes=7e9,
        ...     kv_cache_bytes=2e9,
        ...     activation_bytes=1e9,
        ...     total_memory_gb=80
        ... )
        >>> print(f"Total: {result.total_memory_gb:.2f} GB")
    """

    @property
    def name(self) -> str:
        return "memory"

    @property
    def unit(self) -> str:
        return "GB"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.MEMORY

    def compute(
        self,
        weight_bytes: int = 0,
        kv_cache_bytes: int = 0,
        activation_bytes: int = 0,
        total_memory_gb: float = 80.0,
    ) -> MemoryResult:
        """Compute memory metrics.

        Args:
            weight_bytes: Model weight size in bytes
            kv_cache_bytes: KV cache size in bytes
            activation_bytes: Peak activation size in bytes
            total_memory_gb: Total GPU memory (default A100 80GB)

        Returns:
            MemoryResult with all memory metrics
        """
        weight_gb = weight_bytes / 1e9
        kv_cache_gb = kv_cache_bytes / 1e9
        activation_gb = activation_bytes / 1e9
        total_used_gb = weight_gb + kv_cache_gb + activation_gb

        utilization = total_used_gb / total_memory_gb if total_memory_gb > 0 else 0.0

        return MemoryResult(
            weight_memory_gb=weight_gb,
            kv_cache_memory_gb=kv_cache_gb,
            activation_memory_gb=activation_gb,
            total_memory_gb=total_used_gb,
            memory_utilization=utilization,
        )

    @staticmethod
    def measure_cuda_memory() -> tuple[float, float]:
        """Measure actual CUDA memory usage.

        Returns:
            (allocated_gb, reserved_gb) tuple

        Raises:
            ImportError: If torch not available
            RuntimeError: If CUDA not available
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for CUDA memory measurement")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9

        return allocated, reserved
