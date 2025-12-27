# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Latency metrics (prefill, decode, E2E)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import Metric, MetricRegistry, MetricType


@dataclass
class LatencyResult:
    """Latency measurement result.

    Attributes:
        prefill_latency_ms: Prefill latency (ms)
        decode_latency_ms: Decode latency per token (ms)
        e2e_latency_ms: End-to-end latency (ms)
        ttft_ms: Time to first token (ms)
        tpot_ms: Time per output token (ms)
    """

    prefill_latency_ms: float
    decode_latency_ms: float
    e2e_latency_ms: float
    ttft_ms: float  # Time To First Token
    tpot_ms: float  # Time Per Output Token


@MetricRegistry.register("latency")
class LatencyMetric(Metric[LatencyResult]):
    """Measures inference latency.

    Tracks:
    - Prefill latency (prompt processing)
    - Decode latency (per-token generation)
    - TTFT (Time To First Token)
    - TPOT (Time Per Output Token)

    Example:
        >>> metric = LatencyMetric()
        >>> metric.record_prefill_start()
        >>> # ... prefill phase ...
        >>> metric.record_prefill_end()
        >>> metric.record_decode_token()  # First token
        >>> metric.record_decode_token()  # Second token
        >>> result = metric.compute()
        >>> print(f"TTFT: {result.ttft_ms:.2f} ms")
    """

    def __init__(self):
        self._prefill_start: float | None = None
        self._prefill_end: float | None = None
        self._decode_starts: list[float] = []
        self._decode_ends: list[float] = []

    @property
    def name(self) -> str:
        return "latency"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.LATENCY

    def record_prefill_start(self) -> None:
        """Record prefill start time."""
        self._prefill_start = time.perf_counter()

    def record_prefill_end(self) -> None:
        """Record prefill end time."""
        self._prefill_end = time.perf_counter()

    def record_decode_token(self) -> None:
        """Record decode token generation time."""
        now = time.perf_counter()
        if len(self._decode_starts) == 0:
            # First token
            self._decode_starts.append(self._prefill_end or now)
        else:
            self._decode_starts.append(self._decode_ends[-1] if self._decode_ends else now)
        self._decode_ends.append(now)

    def compute(self) -> LatencyResult:
        """Compute latency metrics.

        Returns:
            LatencyResult with all latency metrics

        Raises:
            RuntimeError: If insufficient data recorded
        """
        if self._prefill_start is None or self._prefill_end is None:
            raise RuntimeError("Record prefill phase first")

        prefill_latency_ms = (self._prefill_end - self._prefill_start) * 1000

        if len(self._decode_ends) == 0:
            # No decode tokens yet
            return LatencyResult(
                prefill_latency_ms=prefill_latency_ms,
                decode_latency_ms=0.0,
                e2e_latency_ms=prefill_latency_ms,
                ttft_ms=prefill_latency_ms,
                tpot_ms=0.0,
            )

        # TTFT: prefill + first token
        ttft_ms = (self._decode_ends[0] - self._prefill_start) * 1000

        # TPOT: average decode time per token
        total_decode_time = sum(
            end - start for start, end in zip(self._decode_starts, self._decode_ends, strict=False)
        )
        tpot_ms = (total_decode_time / len(self._decode_ends)) * 1000 if self._decode_ends else 0.0

        # Decode latency: same as TPOT
        decode_latency_ms = tpot_ms

        # E2E: total time
        e2e_latency_ms = (self._decode_ends[-1] - self._prefill_start) * 1000

        return LatencyResult(
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
            e2e_latency_ms=e2e_latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
        )
