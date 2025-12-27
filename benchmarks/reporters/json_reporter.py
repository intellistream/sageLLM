# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""JSON reporter for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONReporter:
    """Reports benchmark results to JSON file.

    Example:
        >>> reporter = JSONReporter("results.json")
        >>> reporter.add_metric("throughput", 1234.5, "tokens/s")
        >>> reporter.save()
    """

    def __init__(self, output_path: str | Path):
        """Initialize reporter.

        Args:
            output_path: Output JSON file path
        """
        self.output_path = Path(output_path)
        self.data: dict[str, Any] = {
            "metrics": {},
            "metadata": {},
        }

    def add_metric(self, name: str, value: float, unit: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a metric to the report.

        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            metadata: Additional metadata
        """
        self.data["metrics"][name] = {
            "value": value,
            "unit": unit,
            "metadata": metadata or {},
        }

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the report.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.data["metadata"][key] = value

    def save(self) -> None:
        """Save report to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=2)

        print(f"Report saved to {self.output_path}")

    @staticmethod
    def load(filepath: str | Path) -> dict[str, Any]:
        """Load report from JSON file.

        Args:
            filepath: Input JSON file path

        Returns:
            Report data dictionary
        """
        with open(filepath) as f:
            return json.load(f)

    @staticmethod
    def compare(baseline_path: str | Path, current_path: str | Path) -> dict[str, Any]:
        """Compare two benchmark reports.

        Args:
            baseline_path: Baseline report path
            current_path: Current report path

        Returns:
            Comparison results
        """
        baseline = JSONReporter.load(baseline_path)
        current = JSONReporter.load(current_path)

        comparison = {
            "baseline": baseline["metrics"],
            "current": current["metrics"],
            "changes": {},
        }

        for name, baseline_metric in baseline["metrics"].items():
            if name not in current["metrics"]:
                continue

            base_val = baseline_metric["value"]
            curr_val = current["metrics"][name]["value"]

            change_pct = (curr_val - base_val) / base_val * 100 if base_val != 0 else 0.0

            comparison["changes"][name] = {
                "baseline": base_val,
                "current": curr_val,
                "change_pct": change_pct,
                "improved": change_pct > 0,  # Assume higher is better
            }

        return comparison
