# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Benchmark types and data structures.

This module defines the core types for benchmarking:
- BenchmarkConfig: Benchmark configuration
- BenchmarkMetrics: Performance metrics
- BenchmarkResult: Benchmark results
- CIGateConfig: CI performance gate configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkConfig:
    """Benchmark configuration.

    Attributes:
        name: Benchmark name/identifier.
        model: Model to benchmark.
        backend: Engine backend (lmdeploy, vllm).
        batch_sizes: List of batch sizes to test.
        seq_lengths: List of sequence lengths to test.
        num_iterations: Number of iterations per configuration.
        warmup_iterations: Number of warmup iterations.
        output_dir: Directory for benchmark results.
        metadata: Additional benchmark metadata.
    """

    name: str = "default"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: str = "lmdeploy"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    seq_lengths: list[int] = field(default_factory=lambda: [128, 512, 2048])
    num_iterations: int = 10
    warmup_iterations: int = 2
    output_dir: str = "benchmark_results"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "backend": self.backend,
            "batch_sizes": self.batch_sizes,
            "seq_lengths": self.seq_lengths,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "output_dir": self.output_dir,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkMetrics:
    """Performance metrics from a benchmark run.

    Attributes:
        ttft_ms: Time to first token (milliseconds).
        tpot_ms: Time per output token (milliseconds).
        throughput_tokens_per_sec: Tokens generated per second.
        mfu: Model FLOPs utilization (0.0-1.0).
        kv_hit_rate: KV cache hit rate (0.0-1.0).
        comm_overhead_ratio: Communication overhead ratio.
        gpu_memory_used_gb: GPU memory used (GB).
        cost_per_token: Cost per token (arbitrary unit).
        latency_p50_ms: 50th percentile latency (ms).
        latency_p95_ms: 95th percentile latency (ms).
        latency_p99_ms: 99th percentile latency (ms).
    """

    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    mfu: float = 0.0
    kv_hit_rate: float = 0.0
    comm_overhead_ratio: float = 0.0
    gpu_memory_used_gb: float = 0.0
    cost_per_token: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary."""
        return {
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "mfu": self.mfu,
            "kv_hit_rate": self.kv_hit_rate,
            "comm_overhead_ratio": self.comm_overhead_ratio,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "cost_per_token": self.cost_per_token,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Attributes:
        config: Benchmark configuration used.
        metrics: Performance metrics.
        batch_size: Batch size for this result.
        seq_length: Sequence length for this result.
        timestamp: When the benchmark was run.
        duration_sec: Total benchmark duration.
        success: Whether benchmark completed successfully.
        error_message: Error message if failed.
        metadata: Additional result metadata.
    """

    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    batch_size: int = 1
    seq_length: int = 128
    timestamp: datetime = field(default_factory=datetime.now)
    duration_sec: float = 0.0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "timestamp": self.timestamp.isoformat(),
            "duration_sec": self.duration_sec,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class CIGateConfig:
    """CI performance gate configuration.

    Defines thresholds for performance regression detection.

    Attributes:
        mfu_min_threshold: Minimum MFU (baseline - delta).
        ttft_max_regression_pct: Maximum TTFT regression percentage.
        tpot_max_regression_pct: Maximum TPOT regression percentage.
        throughput_min_regression_pct: Maximum throughput regression percentage.
        kv_hit_rate_improvement: Required KV hit rate improvement.
        baseline_path: Path to baseline results file.
        fail_on_regression: Whether to fail on regression.
        metadata: Additional gate configuration.
    """

    mfu_min_threshold: float = 0.0  # Baseline - 1%
    ttft_max_regression_pct: float = 5.0
    tpot_max_regression_pct: float = 5.0
    throughput_min_regression_pct: float = 5.0
    kv_hit_rate_improvement: float = 0.0  # Scenario-dependent
    baseline_path: str = "baseline.json"
    fail_on_regression: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def check_regression(
        self,
        baseline: BenchmarkMetrics,
        current: BenchmarkMetrics,
    ) -> tuple[bool, list[str]]:
        """Check for performance regressions.

        Args:
            baseline: Baseline metrics.
            current: Current metrics.

        Returns:
            Tuple of (passed, list of violation messages).
        """
        violations: list[str] = []

        # Check MFU
        if current.mfu < baseline.mfu - 0.01:  # 1% threshold
            violations.append(
                f"MFU regression: {current.mfu:.4f} < {baseline.mfu - 0.01:.4f}"
            )

        # Check TTFT
        if baseline.ttft_ms > 0:
            ttft_pct = (current.ttft_ms - baseline.ttft_ms) / baseline.ttft_ms * 100
            if ttft_pct > self.ttft_max_regression_pct:
                violations.append(
                    f"TTFT regression: {ttft_pct:.2f}% > {self.ttft_max_regression_pct}%"
                )

        # Check TPOT
        if baseline.tpot_ms > 0:
            tpot_pct = (current.tpot_ms - baseline.tpot_ms) / baseline.tpot_ms * 100
            if tpot_pct > self.tpot_max_regression_pct:
                violations.append(
                    f"TPOT regression: {tpot_pct:.2f}% > {self.tpot_max_regression_pct}%"
                )

        # Check throughput
        if baseline.throughput_tokens_per_sec > 0:
            throughput_pct = (
                (baseline.throughput_tokens_per_sec - current.throughput_tokens_per_sec)
                / baseline.throughput_tokens_per_sec
                * 100
            )
            if throughput_pct > self.throughput_min_regression_pct:
                violations.append(
                    f"Throughput regression: {throughput_pct:.2f}% > "
                    f"{self.throughput_min_regression_pct}%"
                )

        passed = len(violations) == 0 or not self.fail_on_regression
        return passed, violations
