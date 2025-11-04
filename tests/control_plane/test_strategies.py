# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Control Plane scheduling strategies."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (  # noqa: E402
    ExecutionInstance,
    ParallelismType,
    RequestMetadata,
    RequestPriority,
)
from control_plane.strategies import (  # noqa: E402
    AdaptivePolicy,
    CostOptimizedPolicy,
    FIFOPolicy,
    PriorityPolicy,
    SLOAwarePolicy,
)


@pytest.fixture
def sample_instances():
    """Create sample execution instances."""
    return [
        ExecutionInstance(
            instance_id="fast-instance",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            tensor_parallel_size=2,
            gpu_count=2,
            current_load=0.3,
            avg_latency_ms=30.0,
        ),
        ExecutionInstance(
            instance_id="medium-instance",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.5,
            avg_latency_ms=50.0,
        ),
        ExecutionInstance(
            instance_id="slow-instance",
            host="localhost",
            port=8002,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.8,
            avg_latency_ms=100.0,
        ),
    ]


@pytest.fixture
def sample_requests():
    """Create sample requests with different priorities and arrival times."""
    base_time = datetime.now()
    return [
        RequestMetadata(
            request_id="req-normal-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            arrival_time=base_time,
        ),
        RequestMetadata(
            request_id="req-high-1",
            priority=RequestPriority.HIGH,
            max_tokens=512,
            arrival_time=base_time + timedelta(milliseconds=10),
        ),
        RequestMetadata(
            request_id="req-critical-1",
            priority=RequestPriority.CRITICAL,
            max_tokens=256,
            arrival_time=base_time + timedelta(milliseconds=20),
        ),
        RequestMetadata(
            request_id="req-low-1",
            priority=RequestPriority.LOW,
            max_tokens=1024,
            arrival_time=base_time + timedelta(milliseconds=5),
        ),
    ]


class TestFIFOPolicy:
    """Tests for FIFO scheduling policy."""

    def test_fifo_ordering(self, sample_instances):
        """Test requests are scheduled in FIFO order."""
        policy = FIFOPolicy()
        base_time = datetime.now()

        requests = [
            RequestMetadata(
                request_id="req-3",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                arrival_time=base_time + timedelta(milliseconds=30),
            ),
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                arrival_time=base_time,
            ),
            RequestMetadata(
                request_id="req-2",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                arrival_time=base_time + timedelta(milliseconds=10),
            ),
        ]

        decisions = policy.schedule(requests, sample_instances)

        # Should be ordered by arrival time
        assert len(decisions) == 3
        assert decisions[0].request_id == "req-1"
        assert decisions[1].request_id == "req-2"
        assert decisions[2].request_id == "req-3"

    def test_fifo_selects_least_loaded(self, sample_instances):
        """Test FIFO selects least loaded instance."""
        policy = FIFOPolicy()
        requests = [
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
            )
        ]

        decisions = policy.schedule(requests, sample_instances)

        assert len(decisions) == 1
        # Should select instance with lowest load (0.3)
        assert decisions[0].target_instance_id == "fast-instance"

    def test_fifo_prioritize(self):
        """Test FIFO prioritize method."""
        policy = FIFOPolicy()
        base_time = datetime.now()

        requests = [
            RequestMetadata(
                request_id="req-2",
                priority=RequestPriority.HIGH,  # Priority doesn't matter in FIFO
                max_tokens=512,
                arrival_time=base_time + timedelta(milliseconds=20),
            ),
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.LOW,
                max_tokens=512,
                arrival_time=base_time,
            ),
        ]

        prioritized = policy.prioritize(requests)

        # Should be ordered by arrival time only
        assert prioritized[0].request_id == "req-1"
        assert prioritized[1].request_id == "req-2"

    def test_fifo_no_available_instances(self):
        """Test FIFO when no instances are available."""
        policy = FIFOPolicy()
        requests = [
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
            )
        ]

        unavailable_instances = [
            ExecutionInstance(
                instance_id="inst-1",
                host="localhost",
                port=8000,
                model_name="llama-7b",
                is_available=False,  # Mark as unavailable
            )
        ]

        decisions = policy.schedule(requests, unavailable_instances)
        assert len(decisions) == 0


class TestPriorityPolicy:
    """Tests for Priority scheduling policy."""

    def test_priority_ordering(self, sample_instances, sample_requests):
        """Test requests are scheduled by priority."""
        policy = PriorityPolicy()

        decisions = policy.schedule(sample_requests, sample_instances)

        # Should be ordered by priority: CRITICAL, HIGH, NORMAL, LOW
        assert len(decisions) == 4
        assert decisions[0].request_id == "req-critical-1"
        assert decisions[1].request_id == "req-high-1"
        assert decisions[2].request_id == "req-normal-1"
        assert decisions[3].request_id == "req-low-1"

    def test_priority_selects_fastest_for_critical(self, sample_instances):
        """Test high priority requests get fastest instances."""
        policy = PriorityPolicy()
        requests = [
            RequestMetadata(
                request_id="req-critical",
                priority=RequestPriority.CRITICAL,
                max_tokens=256,
            )
        ]

        decisions = policy.schedule(requests, sample_instances)

        assert len(decisions) == 1
        # Should select instance with lowest latency
        assert decisions[0].target_instance_id == "fast-instance"

    def test_priority_selects_least_loaded_for_normal(self, sample_instances):
        """Test normal priority requests get least loaded instances."""
        policy = PriorityPolicy()
        requests = [
            RequestMetadata(
                request_id="req-normal",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
            )
        ]

        decisions = policy.schedule(requests, sample_instances)

        assert len(decisions) == 1
        # Should select instance with lowest load
        assert decisions[0].target_instance_id == "fast-instance"

    def test_priority_same_priority_uses_fifo(self, sample_instances):
        """Test requests with same priority are ordered by arrival time."""
        policy = PriorityPolicy()
        base_time = datetime.now()

        requests = [
            RequestMetadata(
                request_id="req-2",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                arrival_time=base_time + timedelta(milliseconds=10),
            ),
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                arrival_time=base_time,
            ),
        ]

        prioritized = policy.prioritize(requests)

        # Same priority, should use arrival time
        assert prioritized[0].request_id == "req-1"
        assert prioritized[1].request_id == "req-2"


class TestSLOAwarePolicy:
    """Tests for SLO-aware scheduling policy."""

    def test_slo_aware_prioritizes_tight_deadlines(self, sample_instances):
        """Test SLO-aware policy prioritizes tight deadlines."""
        policy = SLOAwarePolicy()
        base_time = datetime.now()

        requests = [
            RequestMetadata(
                request_id="req-tight",
                priority=RequestPriority.NORMAL,
                max_tokens=256,
                slo_deadline_ms=50.0,
                arrival_time=base_time,
            ),
            RequestMetadata(
                request_id="req-relaxed",
                priority=RequestPriority.NORMAL,
                max_tokens=256,
                slo_deadline_ms=500.0,
                arrival_time=base_time,
            ),
        ]

        prioritized = policy.prioritize(requests)

        # Tighter deadline should come first
        assert prioritized[0].request_id == "req-tight"
        assert prioritized[1].request_id == "req-relaxed"

    def test_slo_aware_selects_fast_instance_for_tight_slo(self, sample_instances):
        """Test SLO-aware selects fast instance for tight SLO."""
        policy = SLOAwarePolicy()

        requests = [
            RequestMetadata(
                request_id="req-tight-slo",
                priority=RequestPriority.NORMAL,
                max_tokens=256,
                slo_deadline_ms=40.0,  # Tight deadline
            )
        ]

        decisions = policy.schedule(requests, sample_instances)

        assert len(decisions) == 1
        # Should select fastest instance to meet tight SLO
        assert decisions[0].target_instance_id == "fast-instance"

    def test_slo_aware_handles_no_slo(self, sample_instances):
        """Test SLO-aware handles requests without SLO."""
        policy = SLOAwarePolicy()

        requests = [
            RequestMetadata(
                request_id="req-no-slo",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                slo_deadline_ms=None,
            )
        ]

        decisions = policy.schedule(requests, sample_instances)

        # Should still make a decision (fallback to load balancing)
        assert len(decisions) == 1


class TestCostOptimizedPolicy:
    """Tests for Cost-optimized scheduling policy."""

    def test_cost_optimized_prefers_cheaper_instances(self):
        """Test cost-optimized policy prefers cheaper instances."""
        policy = CostOptimizedPolicy()

        instances = [
            ExecutionInstance(
                instance_id="expensive",
                host="localhost",
                port=8000,
                model_name="llama-70b",
                tensor_parallel_size=8,
                gpu_count=8,
                current_load=0.3,
                metadata={"cost_per_token": 0.01},
            ),
            ExecutionInstance(
                instance_id="cheap",
                host="localhost",
                port=8001,
                model_name="llama-7b",
                tensor_parallel_size=1,
                gpu_count=1,
                current_load=0.5,
                metadata={"cost_per_token": 0.001},
            ),
        ]

        requests = [
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                cost_budget=10.0,  # Has budget, can use cheaper instance
            )
        ]

        decisions = policy.schedule(requests, instances)

        assert len(decisions) == 1
        # Should prefer cheaper instance when budget allows
        assert decisions[0].target_instance_id == "cheap"

    def test_cost_optimized_handles_no_cost_metadata(self, sample_instances):
        """Test cost-optimized handles instances without cost metadata."""
        policy = CostOptimizedPolicy()

        requests = [
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
            )
        ]

        decisions = policy.schedule(requests, sample_instances)

        # Should still make decisions (fallback to load balancing)
        assert len(decisions) == 1


class TestAdaptivePolicy:
    """Tests for Adaptive scheduling policy."""

    def test_adaptive_switches_based_on_conditions(self, sample_instances):
        """Test adaptive policy switches strategies based on workload."""
        policy = AdaptivePolicy()

        # Critical request should use priority-based scheduling
        critical_requests = [
            RequestMetadata(
                request_id="req-critical",
                priority=RequestPriority.CRITICAL,
                max_tokens=256,
            )
        ]

        decisions = policy.schedule(critical_requests, sample_instances)
        assert len(decisions) == 1
        assert "Priority" in decisions[0].reason or "Adaptive" in decisions[0].reason

    def test_adaptive_uses_slo_for_deadline_requests(self, sample_instances):
        """Test adaptive uses SLO-aware for requests with deadlines."""
        policy = AdaptivePolicy()

        slo_requests = [
            RequestMetadata(
                request_id="req-slo",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
                slo_deadline_ms=100.0,
            )
        ]

        decisions = policy.schedule(slo_requests, sample_instances)
        assert len(decisions) == 1

    def test_adaptive_falls_back_to_fifo(self, sample_instances):
        """Test adaptive falls back to FIFO for normal requests."""
        policy = AdaptivePolicy()

        normal_requests = [
            RequestMetadata(
                request_id="req-normal",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
            )
        ]

        decisions = policy.schedule(normal_requests, sample_instances)
        assert len(decisions) == 1


class TestSchedulingPolicyIntegration:
    """Integration tests for scheduling policies."""

    @pytest.mark.parametrize(
        "policy_class",
        [FIFOPolicy, PriorityPolicy, SLOAwarePolicy, CostOptimizedPolicy, AdaptivePolicy],
    )
    def test_all_policies_handle_empty_requests(self, policy_class, sample_instances):
        """Test all policies handle empty request list."""
        policy = policy_class()
        decisions = policy.schedule([], sample_instances)
        assert len(decisions) == 0

    @pytest.mark.parametrize(
        "policy_class",
        [FIFOPolicy, PriorityPolicy, SLOAwarePolicy, CostOptimizedPolicy, AdaptivePolicy],
    )
    def test_all_policies_handle_empty_instances(self, policy_class, sample_requests):
        """Test all policies handle empty instance list."""
        policy = policy_class()
        decisions = policy.schedule(sample_requests, [])
        # Should return empty or partial decisions
        assert isinstance(decisions, list)

    @pytest.mark.parametrize(
        "policy_class",
        [FIFOPolicy, PriorityPolicy, SLOAwarePolicy, CostOptimizedPolicy, AdaptivePolicy],
    )
    def test_all_policies_produce_valid_decisions(
        self, policy_class, sample_requests, sample_instances
    ):
        """Test all policies produce valid scheduling decisions."""
        policy = policy_class()
        decisions = policy.schedule(sample_requests, sample_instances)

        for decision in decisions:
            # Each decision should have required fields
            assert decision.request_id is not None
            assert decision.target_instance_id is not None
            assert isinstance(decision.parallelism_strategy, ParallelismType)
            assert decision.estimated_latency_ms >= 0
            assert decision.estimated_cost >= 0

    def test_policy_comparison(self, sample_instances):
        """Test comparing decisions from different policies."""
        base_time = datetime.now()
        requests = [
            RequestMetadata(
                request_id="req-1",
                priority=RequestPriority.CRITICAL,
                max_tokens=256,
                arrival_time=base_time,
                slo_deadline_ms=50.0,
            ),
            RequestMetadata(
                request_id="req-2",
                priority=RequestPriority.LOW,
                max_tokens=1024,
                arrival_time=base_time + timedelta(milliseconds=10),
            ),
        ]

        fifo_policy = FIFOPolicy()
        priority_policy = PriorityPolicy()
        slo_policy = SLOAwarePolicy()

        fifo_decisions = fifo_policy.schedule(requests, sample_instances)
        priority_decisions = priority_policy.schedule(requests, sample_instances)
        slo_decisions = slo_policy.schedule(requests, sample_instances)

        # FIFO should schedule by arrival time
        assert fifo_decisions[0].request_id == "req-1"

        # Priority should schedule by priority
        assert priority_decisions[0].request_id == "req-1"

        # SLO should consider deadline
        assert slo_decisions[0].request_id == "req-1"
