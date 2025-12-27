# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Metrics base classes and registry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class MetricType(Enum):
    """Metric type categories."""

    THROUGHPUT = auto()  # Tokens/s, Requests/s
    LATENCY = auto()  # Prefill, Decode, E2E latency
    MEMORY = auto()  # Weight, KV cache, activation memory
    KV_CACHE = auto()  # Hit rate, migration traffic
    COMPUTE = auto()  # MFU, GPU utilization
    COMMUNICATION = auto()  # Network bandwidth, P/D separation overhead


@dataclass
class MetricValue:
    """Single metric measurement.

    Attributes:
        name: Metric name
        value: Measured value
        unit: Unit of measurement
        timestamp: When the measurement was taken
        metadata: Additional context
    """

    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f} {self.unit}"


@dataclass
class MetricSummary:
    """Aggregated metric statistics.

    Attributes:
        name: Metric name
        mean: Average value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        p50: 50th percentile (median)
        p90: 90th percentile
        p99: 99th percentile
        count: Number of measurements
        unit: Unit of measurement
    """

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
    def from_values(cls, name: str, values: list[float], unit: str) -> MetricSummary:
        """Create summary from list of values.

        Args:
            name: Metric name
            values: List of measurements
            unit: Unit of measurement

        Returns:
            MetricSummary with statistics
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for MetricSummary")

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

    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"mean={self.mean:.4f}, "
            f"std={self.std:.4f}, "
            f"p50={self.p50:.4f}, "
            f"p90={self.p90:.4f}, "
            f"p99={self.p99:.4f} {self.unit} "
            f"(n={self.count})"
        )


class Metric(ABC, Generic[T]):
    """Base class for all metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        ...

    @property
    @abstractmethod
    def unit(self) -> str:
        """Metric unit."""
        ...

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Metric type category."""
        ...

    @abstractmethod
    def compute(self, *args, **kwargs) -> T:
        """Compute metric value."""
        ...

    def to_metric_value(self, value: float, metadata: dict[str, Any] | None = None) -> MetricValue:
        """Convert value to MetricValue.

        Args:
            value: Metric value
            metadata: Additional metadata

        Returns:
            MetricValue instance
        """
        return MetricValue(
            name=self.name,
            value=value,
            unit=self.unit,
            metadata=metadata or {},
        )


class MetricRegistry:
    """Registry for all metrics.

    Example:
        >>> @MetricRegistry.register("my_metric")
        ... class MyMetric(Metric):
        ...     pass
        >>> metric = MetricRegistry.get("my_metric")
    """

    _metrics: dict[str, type[Metric]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a metric.

        Args:
            name: Metric identifier

        Returns:
            Decorator function
        """

        def decorator(metric_cls: type[Metric]) -> type[Metric]:
            cls._metrics[name] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Metric:
        """Get metric instance by name.

        Args:
            name: Metric identifier

        Returns:
            Metric instance

        Raises:
            ValueError: If metric not found
        """
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available: {cls.list_all()}")
        return cls._metrics[name]()

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered metrics.

        Returns:
            List of metric names
        """
        return list(cls._metrics.keys())


# Import concrete metrics after base classes are defined
from .kv_cache import KVCacheMetric  # noqa: E402
from .latency import LatencyMetric  # noqa: E402
from .memory import MemoryMetric  # noqa: E402
from .mfu import MFUMetric  # noqa: E402
from .throughput import ThroughputMetric  # noqa: E402

__all__ = [
    # Base classes
    "Metric",
    "MetricType",
    "MetricValue",
    "MetricSummary",
    "MetricRegistry",
    # Concrete metrics
    "ThroughputMetric",
    "LatencyMetric",
    "MemoryMetric",
    "KVCacheMetric",
    "MFUMetric",
]
