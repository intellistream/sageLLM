# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Execution tracing for performance profiling."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """Single trace event.

    Attributes:
        name: Event name
        start_time: Event start timestamp
        end_time: Event end timestamp
        duration_ms: Event duration in milliseconds
        metadata: Additional event metadata
    """

    name: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.duration_ms:.2f} ms"


class ExecutionTracer:
    """Traces execution for performance analysis.

    Example:
        >>> tracer = ExecutionTracer()
        >>> with tracer.trace("prefill"):
        ...     # prefill code
        ...     pass
        >>> with tracer.trace("decode"):
        ...     # decode code
        ...     pass
        >>> events = tracer.get_events()
        >>> tracer.print_summary()
    """

    def __init__(self):
        self._events: list[TraceEvent] = []
        self._active_traces: dict[str, float] = {}

    @contextmanager
    def trace(self, name: str, metadata: dict[str, Any] | None = None):
        """Context manager to trace a code block.

        Args:
            name: Event name
            metadata: Additional metadata

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            event = TraceEvent(
                name=name,
                start_time=start,
                end_time=end,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )
            self._events.append(event)

    def get_events(self) -> list[TraceEvent]:
        """Get all recorded events.

        Returns:
            List of TraceEvent objects
        """
        return self._events.copy()

    def get_events_by_name(self, name: str) -> list[TraceEvent]:
        """Get events by name.

        Args:
            name: Event name to filter

        Returns:
            List of matching TraceEvent objects
        """
        return [e for e in self._events if e.name == name]

    def get_total_time(self, name: str | None = None) -> float:
        """Get total time for events.

        Args:
            name: Event name to filter (None for all events)

        Returns:
            Total duration in milliseconds
        """
        if name is None:
            return sum(e.duration_ms for e in self._events)
        return sum(e.duration_ms for e in self.get_events_by_name(name))

    def get_avg_time(self, name: str) -> float:
        """Get average time for named events.

        Args:
            name: Event name

        Returns:
            Average duration in milliseconds
        """
        events = self.get_events_by_name(name)
        if not events:
            return 0.0
        return sum(e.duration_ms for e in events) / len(events)

    def print_summary(self) -> None:
        """Print execution summary."""
        if not self._events:
            print("No events recorded")
            return

        print("\n=== Execution Trace Summary ===")
        print(f"Total events: {len(self._events)}")
        print(f"Total time: {self.get_total_time():.2f} ms\n")

        # Group by name
        event_names = {e.name for e in self._events}
        for name in sorted(event_names):
            events = self.get_events_by_name(name)
            total_time = sum(e.duration_ms for e in events)
            avg_time = total_time / len(events)
            print(
                f"{name:20s}: {len(events):4d} calls, "
                f"total={total_time:8.2f} ms, "
                f"avg={avg_time:6.2f} ms"
            )

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
        self._active_traces.clear()

    def export_chrome_trace(self, filepath: str) -> None:
        """Export trace to Chrome Tracing format.

        Args:
            filepath: Output JSON file path
        """
        import json

        trace_events = []
        for event in self._events:
            trace_events.append(
                {
                    "name": event.name,
                    "ph": "X",  # Complete event
                    "ts": event.start_time * 1e6,  # microseconds
                    "dur": event.duration_ms * 1e3,  # microseconds
                    "pid": 0,
                    "tid": 0,
                    "args": event.metadata,
                }
            )

        with open(filepath, "w") as f:
            json.dump(trace_events, f, indent=2)

        print(f"Trace exported to {filepath}")
        print("Open in Chrome: chrome://tracing")
