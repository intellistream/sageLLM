# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Console reporter for benchmark results."""

from __future__ import annotations

from typing import Any


class ConsoleReporter:
    """Reports benchmark results to console.

    Example:
        >>> reporter = ConsoleReporter()
        >>> reporter.report_metric("throughput", 1234.5, "tokens/s")
        >>> reporter.report_summary({"mfu": 0.65, "latency_ms": 12.3})
    """

    def __init__(self, verbose: bool = True):
        """Initialize reporter.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def report_metric(self, name: str, value: float, unit: str) -> None:
        """Report a single metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
        """
        print(f"{name:30s}: {value:10.4f} {unit}")

    def report_summary(self, metrics: dict[str, Any]) -> None:
        """Report benchmark summary.

        Args:
            metrics: Dictionary of metric name -> value
        """
        print("\n=== Benchmark Summary ===")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"{name:30s}: {value:10.4f}")
            else:
                print(f"{name:30s}: {value}")
        print()

    def report_comparison(self, baseline: dict[str, float], current: dict[str, float]) -> None:
        """Report comparison between baseline and current.

        Args:
            baseline: Baseline metrics
            current: Current metrics
        """
        print("\n=== Performance Comparison ===")
        print(f"{'Metric':<30s} {'Baseline':>12s} {'Current':>12s} {'Change':>12s}")
        print("-" * 70)

        for name in baseline:
            if name not in current:
                continue

            base_val = baseline[name]
            curr_val = current[name]

            # Calculate change
            if base_val != 0:
                change_pct = ((curr_val - base_val) / base_val) * 100
                change_str = f"{change_pct:+.2f}%"
            else:
                change_str = "N/A"

            print(f"{name:<30s} {base_val:12.4f} {curr_val:12.4f} {change_str:>12s}")

        print()
