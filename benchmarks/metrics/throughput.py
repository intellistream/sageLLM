# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Throughput metrics (tokens/s, requests/s)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import Metric, MetricRegistry, MetricType


@dataclass
class ThroughputResult:
    """Throughput measurement result.

    Attributes:
        tokens_per_second: Tokens generated per second
        requests_per_second: Requests processed per second
        total_tokens: Total tokens generated
        total_requests: Total requests processed
        duration_s: Measurement duration in seconds
    """

    tokens_per_second: float
    requests_per_second: float
    total_tokens: int
    total_requests: int
    duration_s: float


@MetricRegistry.register("throughput")
class ThroughputMetric(Metric[ThroughputResult]):
    """Measures system throughput.

    Tracks:
    - Tokens per second (TPS)
    - Requests per second (QPS)

    Example:
        >>> metric = ThroughputMetric()
        >>> metric.start()
        >>> metric.record(tokens=100, requests=1)
        >>> result = metric.compute()
        >>> print(f"TPS: {result.tokens_per_second:.2f}")
    """

    def __init__(self):
        self._start_time: float | None = None
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
        """Start measurement."""
        self._start_time = time.perf_counter()
        self._total_tokens = 0
        self._total_requests = 0

    def record(self, tokens: int, requests: int = 1) -> None:
        """Record generated tokens.

        Args:
            tokens: Number of tokens generated
            requests: Number of requests processed
        """
        self._total_tokens += tokens
        self._total_requests += requests

    def compute(self) -> ThroughputResult:
        """Compute throughput metrics.

        Returns:
            ThroughputResult with TPS and QPS

        Raises:
            RuntimeError: If start() was not called
        """
        if self._start_time is None:
            raise RuntimeError("Call start() before compute()")

        duration = time.perf_counter() - self._start_time

        if duration == 0:
            return ThroughputResult(
                tokens_per_second=0.0,
                requests_per_second=0.0,
                total_tokens=self._total_tokens,
                total_requests=self._total_requests,
                duration_s=duration,
            )

        return ThroughputResult(
            tokens_per_second=self._total_tokens / duration,
            requests_per_second=self._total_requests / duration,
            total_tokens=self._total_tokens,
            total_requests=self._total_requests,
            duration_s=duration,
        )
