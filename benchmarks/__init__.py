# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Benchmarks module for sageLLM - performance testing and CI gates.

This module provides unified benchmarking capabilities:
- BenchmarkRunner: Execute benchmark configurations
- ci_gate: Performance regression detection
- Metrics: MFU, TTFT, TPOT, KV hit rate, communication overhead

Benchmark configurations are stored in benchmarks/configs/*.yaml

Example:
    >>> from sage.common.components.sage_llm.sageLLM.benchmarks import (
    ...     BenchmarkRunner,
    ...     BenchmarkConfig,
    ... )
    >>> runner = BenchmarkRunner()
    >>> results = runner.run(config=BenchmarkConfig(
    ...     model="Qwen/Qwen2.5-7B-Instruct",
    ...     batch_sizes=[1, 4, 8],
    ...     seq_lengths=[128, 512, 2048],
    ... ))

CI Gate usage:
    python -m sage.common.components.sage_llm.sageLLM.benchmarks.ci_gate \
        --baseline baseline.json --current output.json
"""

from .types import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    CIGateConfig,
)

__all__ = [
    # Types
    "BenchmarkConfig",
    "BenchmarkMetrics",
    "BenchmarkResult",
    "CIGateConfig",
]
