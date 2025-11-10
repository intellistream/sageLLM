# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for monitoring and metrics collection."""

import pytest

from control_plane import (
    ExecutionInstance,
    InstanceMetrics,
    MetricsCollector,
    RequestMetadata,
    RequestPriority,
    SchedulingDecision,
    SchedulingMetrics,
)
from control_plane.types import ParallelismType


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(window_size=100)
        assert collector.window_size == 100
        assert collector.total_requests == 0
        assert collector.completed_requests == 0
        assert collector.failed_requests == 0

    def test_record_request_completion_success(self):
        """Test recording successful request completion."""
        collector = MetricsCollector()

        request = RequestMetadata(
            request_id="test-1",
            priority=RequestPriority.NORMAL,
            slo_deadline_ms=1000,
        )
        # Set queue times for queue_wait_ms calculation
        from datetime import datetime, timedelta

        request.queue_time = datetime.now() - timedelta(milliseconds=50)
        request.schedule_time = datetime.now()

        decision = SchedulingDecision(
            request_id="test-1",
            target_instance_id="instance-1",
            parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
            estimated_latency_ms=100.0,
            estimated_cost=0.01,
        )

        collector.record_request_completion(
            request, decision, actual_latency_ms=105.0, success=True
        )

        assert collector.total_requests == 1
        assert collector.completed_requests == 1
        assert collector.failed_requests == 0
        assert len(collector.latency_history) == 1
        assert len(collector.prediction_errors) == 1
        assert len(collector.queue_wait_times) == 1

    def test_record_request_completion_failure(self):
        """Test recording failed request."""
        collector = MetricsCollector()

        request = RequestMetadata(request_id="test-1")
        decision = SchedulingDecision(
            request_id="test-1",
            target_instance_id="instance-1",
            parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
            estimated_latency_ms=100.0,
            estimated_cost=0.01,
        )

        collector.record_request_completion(request, decision, actual_latency_ms=0.0, success=False)

        assert collector.total_requests == 1
        assert collector.completed_requests == 0
        assert collector.failed_requests == 1

    def test_record_scheduling_decision(self):
        """Test recording scheduling decision time."""
        collector = MetricsCollector()

        collector.record_scheduling_decision(1000.0)  # 1ms in microseconds
        collector.record_scheduling_decision(1500.0)

        assert len(collector.scheduling_times) == 2

    def test_record_queue_length(self):
        """Test recording queue length."""
        collector = MetricsCollector()

        collector.record_queue_length(10)
        collector.record_queue_length(15)
        collector.record_queue_length(5)

        assert len(collector.queue_lengths) == 3
        assert max(collector.queue_lengths) == 15

    def test_update_instance_metrics(self):
        """Test updating instance metrics."""
        collector = MetricsCollector()

        instance = ExecutionInstance(
            instance_id="test-instance",
            host="localhost",
            port=8000,
            model_name="test-model",
            active_requests=5,
            gpu_utilization=0.75,
            current_load=0.6,
            is_healthy=True,
        )

        collector.update_instance_metrics(instance)

        metrics = collector.get_instance_metrics("test-instance")
        assert metrics is not None
        assert metrics.active_requests == 5
        assert metrics.gpu_utilization == 0.75
        assert metrics.current_load == 0.6
        assert metrics.is_healthy is True

    def test_get_scheduling_metrics(self):
        """Test getting scheduling metrics."""
        collector = MetricsCollector()

        # Add some data
        collector.record_scheduling_decision(1000.0)
        collector.record_queue_length(10)

        metrics = collector.get_scheduling_metrics("test-policy")

        assert isinstance(metrics, SchedulingMetrics)
        assert metrics.policy_name == "test-policy"
        assert metrics.queue_length_max == 10

    def test_slo_compliance_tracking(self):
        """Test SLO compliance tracking by priority."""
        collector = MetricsCollector()

        # Met SLO
        request1 = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.CRITICAL,
            slo_deadline_ms=100,
        )
        decision1 = SchedulingDecision(
            request_id="req-1",
            target_instance_id="inst-1",
            parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
            estimated_latency_ms=50.0,
            estimated_cost=0.01,
        )
        collector.record_request_completion(request1, decision1, 80.0, success=True)

        # Missed SLO
        request2 = RequestMetadata(
            request_id="req-2",
            priority=RequestPriority.CRITICAL,
            slo_deadline_ms=100,
        )
        decision2 = SchedulingDecision(
            request_id="req-2",
            target_instance_id="inst-1",
            parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
            estimated_latency_ms=50.0,
            estimated_cost=0.01,
        )
        collector.record_request_completion(request2, decision2, 150.0, success=True)

        metrics = collector.get_scheduling_metrics("test")
        assert "CRITICAL" in metrics.slo_compliance_by_priority
        assert metrics.slo_compliance_by_priority["CRITICAL"] == 0.5  # 50% compliance

    def test_record_instance_failure(self):
        """Test recording instance failure."""
        collector = MetricsCollector()

        instance = ExecutionInstance(
            instance_id="failing-instance",
            host="localhost",
            port=8000,
            model_name="test",
        )

        collector.update_instance_metrics(instance)
        collector.record_instance_failure("failing-instance")

        metrics = collector.get_instance_metrics("failing-instance")
        assert metrics is not None
        assert metrics.is_healthy is False
        assert metrics.consecutive_failures == 1

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        # Add some data
        collector.record_scheduling_decision(1000.0)
        collector.record_queue_length(10)

        assert collector.total_requests == 0
        assert len(collector.scheduling_times) == 1

        # Reset
        collector.reset()

        assert collector.total_requests == 0
        assert len(collector.scheduling_times) == 0
        assert len(collector.queue_lengths) == 0

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        collector = MetricsCollector()

        # Add latencies: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        for i in range(1, 11):
            request = RequestMetadata(request_id=f"req-{i}")
            decision = SchedulingDecision(
                request_id=f"req-{i}",
                target_instance_id="inst-1",
                parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
                estimated_latency_ms=10.0,
                estimated_cost=0.01,
            )
            collector.record_request_completion(request, decision, i * 10.0, success=True)

        metrics = collector.get_scheduling_metrics("test")

        # Check that percentiles are reasonable
        assert metrics.latency_prediction_error_p95 >= 0
        assert metrics.queue_wait_time_p50 >= 0
