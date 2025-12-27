# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""KV cache efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass

from . import Metric, MetricRegistry, MetricType


@dataclass
class KVCacheResult:
    """KV cache efficiency metrics.

    Attributes:
        hit_rate: Cache hit rate (0-1)
        migration_bytes: Total bytes migrated across tiers
        migration_count: Number of migration operations
        reuse_ratio: KV reuse ratio (0-1)
        avg_prefix_length: Average reused prefix length
    """

    hit_rate: float
    migration_bytes: int
    migration_count: int
    reuse_ratio: float
    avg_prefix_length: float


@MetricRegistry.register("kv_cache")
class KVCacheMetric(Metric[KVCacheResult]):
    """Measures KV cache efficiency.

    Tracks:
    - Hit rate (cross-request KV reuse)
    - Migration traffic (hot/cold tier movement)
    - Reuse ratio (prefix matching efficiency)

    Example:
        >>> metric = KVCacheMetric()
        >>> metric.record_hit()
        >>> metric.record_miss()
        >>> metric.record_migration(bytes_moved=1024*1024)
        >>> result = metric.compute()
        >>> print(f"Hit rate: {result.hit_rate:.2%}")
    """

    def __init__(self):
        self._hits = 0
        self._misses = 0
        self._migration_bytes = 0
        self._migration_count = 0
        self._reused_tokens = 0
        self._total_tokens = 0
        self._prefix_lengths: list[int] = []

    @property
    def name(self) -> str:
        return "kv_cache"

    @property
    def unit(self) -> str:
        return "ratio"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.KV_CACHE

    def record_hit(self) -> None:
        """Record a cache hit."""
        self._hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1

    def record_migration(self, bytes_moved: int) -> None:
        """Record a KV block migration.

        Args:
            bytes_moved: Number of bytes migrated
        """
        self._migration_bytes += bytes_moved
        self._migration_count += 1

    def record_reuse(self, reused_tokens: int, total_tokens: int) -> None:
        """Record KV reuse for a request.

        Args:
            reused_tokens: Number of tokens reused from cache
            total_tokens: Total number of tokens in request
        """
        self._reused_tokens += reused_tokens
        self._total_tokens += total_tokens
        if reused_tokens > 0:
            self._prefix_lengths.append(reused_tokens)

    def compute(self) -> KVCacheResult:
        """Compute KV cache efficiency metrics.

        Returns:
            KVCacheResult with all efficiency metrics
        """
        total_accesses = self._hits + self._misses
        hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0

        reuse_ratio = self._reused_tokens / self._total_tokens if self._total_tokens > 0 else 0.0

        avg_prefix = (
            sum(self._prefix_lengths) / len(self._prefix_lengths) if self._prefix_lengths else 0.0
        )

        return KVCacheResult(
            hit_rate=hit_rate,
            migration_bytes=self._migration_bytes,
            migration_count=self._migration_count,
            reuse_ratio=reuse_ratio,
            avg_prefix_length=avg_prefix,
        )
