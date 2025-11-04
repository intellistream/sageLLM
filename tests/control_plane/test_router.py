# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Control Plane request routing."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import ExecutionInstance, RequestMetadata, RequestPriority  # noqa: E402
from control_plane.router import LoadBalancer, RequestRouter  # noqa: E402


@pytest.fixture
def sample_instances():
    """Create sample execution instances for testing."""
    return [
        ExecutionInstance(
            instance_id="instance-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.3,
            active_requests=5,
            avg_latency_ms=50.0,
            machine_id="machine-1",
            nvlink_peers=["instance-2"],
        ),
        ExecutionInstance(
            instance_id="instance-2",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.5,
            active_requests=10,
            avg_latency_ms=80.0,
            machine_id="machine-1",
            nvlink_peers=["instance-1"],
        ),
        ExecutionInstance(
            instance_id="instance-3",
            host="localhost",
            port=8002,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.8,
            active_requests=15,
            avg_latency_ms=120.0,
            machine_id="machine-2",
        ),
    ]


@pytest.fixture
def sample_request():
    """Create a sample request for testing."""
    return RequestMetadata(
        request_id="test-req-1",
        priority=RequestPriority.NORMAL,
        max_tokens=512,
        model_name="llama-7b",
        user_id="user-123",
    )


class TestRequestRouter:
    """Tests for RequestRouter."""

    def test_load_balanced_routing(self, sample_instances, sample_request):
        """Test load-balanced routing selects least loaded instance."""
        router = RequestRouter(routing_strategy="load_balanced")
        selected = router.route(sample_request, sample_instances)

        assert selected is not None
        assert selected.instance_id == "instance-1"
        assert selected.current_load == min(i.current_load for i in sample_instances)

    def test_round_robin_routing(self, sample_instances, sample_request):
        """Test round-robin routing cycles through instances."""
        router = RequestRouter(routing_strategy="round_robin")

        # First request should go to instance-1
        selected1 = router.route(sample_request, sample_instances)
        assert selected1 is not None
        assert selected1.instance_id == "instance-1"

        # Second request should go to instance-2
        selected2 = router.route(sample_request, sample_instances)
        assert selected2 is not None
        assert selected2.instance_id == "instance-2"

        # Third request should go to instance-3
        selected3 = router.route(sample_request, sample_instances)
        assert selected3 is not None
        assert selected3.instance_id == "instance-3"

        # Fourth request should cycle back to instance-1
        selected4 = router.route(sample_request, sample_instances)
        assert selected4 is not None
        assert selected4.instance_id == "instance-1"

    def test_random_routing(self, sample_instances, sample_request):
        """Test random routing selects from available instances."""
        router = RequestRouter(routing_strategy="random")

        # Test multiple times to ensure it's selecting valid instances
        for _ in range(10):
            selected = router.route(sample_request, sample_instances)
            assert selected is not None
            assert selected.instance_id in [i.instance_id for i in sample_instances]

    def test_affinity_routing(self, sample_instances):
        """Test affinity routing maintains user-instance mapping."""
        router = RequestRouter(routing_strategy="affinity")

        # Create requests with different user IDs
        request1 = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            user_id="user-1",
        )
        request2 = RequestMetadata(
            request_id="req-2",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            user_id="user-2",
        )

        # Route first request for user-1
        selected1 = router.route(request1, sample_instances)
        assert selected1 is not None
        instance_id_1 = selected1.instance_id

        # Route second request for user-1 - should go to same instance
        selected1_again = router.route(request1, sample_instances)
        assert selected1_again is not None
        assert selected1_again.instance_id == instance_id_1

        # Route request for user-2 - may go to different instance
        selected2 = router.route(request2, sample_instances)
        assert selected2 is not None
        instance_id_2 = selected2.instance_id

        # Route second request for user-2 - should go to same instance
        selected2_again = router.route(request2, sample_instances)
        assert selected2_again is not None
        assert selected2_again.instance_id == instance_id_2

    def test_affinity_routing_without_user_id(self, sample_instances):
        """Test affinity routing falls back to load balancing without user_id."""
        router = RequestRouter(routing_strategy="affinity")

        request = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
        )

        selected = router.route(request, sample_instances)
        assert selected is not None
        # Should select least loaded instance
        assert selected.instance_id == "instance-1"

    def test_locality_routing(self, sample_instances):
        """Test locality routing provides consistent routing for similar requests."""
        router = RequestRouter(routing_strategy="locality")

        request1 = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            model_name="llama-7b",
            user_id="user-123",
        )
        request2 = RequestMetadata(
            request_id="req-2",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            model_name="llama-7b",
            user_id="user-123",
        )

        # Requests with same model and user should go to same instance (for caching)
        selected1 = router.route(request1, sample_instances)
        selected2 = router.route(request2, sample_instances)

        assert selected1 is not None
        assert selected2 is not None
        assert selected1.instance_id == selected2.instance_id

    def test_topology_aware_routing_with_nvlink(self, sample_instances):
        """Test topology-aware routing prefers NVLINK-connected instances."""
        router = RequestRouter(routing_strategy="topology_aware")

        # Request with preferred instance that has NVLINK peer
        request = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            preferred_instance_id="instance-1",
            user_id="user-1",
        )

        # Filter to only instance-1 and instance-2 (NVLINK peers)
        nvlink_instances = sample_instances[:2]

        selected = router.route(request, nvlink_instances)

        # Should select instance-2 (NVLINK peer of preferred instance-1)
        # or instance-1 itself if it's available
        assert selected is not None
        assert selected.instance_id in ["instance-1", "instance-2"]
        assert selected.machine_id == "machine-1"

    def test_topology_aware_routing_same_machine(self, sample_instances):
        """Test topology-aware routing prefers same-machine instances."""
        router = RequestRouter(routing_strategy="topology_aware")

        request = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            preferred_instance_id="instance-1",
        )

        selected = router.route(request, sample_instances)

        # Should prefer instances on same machine as preferred instance
        assert selected is not None
        assert selected.machine_id == "machine-1"

    def test_routing_with_no_available_instances(self, sample_request):
        """Test routing when no instances are available."""
        router = RequestRouter(routing_strategy="load_balanced")

        # Create instances that cannot accept requests
        unavailable_instances = [
            ExecutionInstance(
                instance_id="instance-1",
                host="localhost",
                port=8000,
                model_name="llama-7b",
                tensor_parallel_size=1,
                gpu_count=1,
                is_available=False,  # Mark as unavailable
            )
        ]

        selected = router.route(sample_request, unavailable_instances)
        assert selected is None

    def test_update_affinity(self, sample_instances):
        """Test manual affinity updates."""
        router = RequestRouter(routing_strategy="affinity")

        # Manually set affinity
        router.update_affinity("user-123", "instance-2")

        request = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            user_id="user-123",
        )

        selected = router.route(request, sample_instances)
        assert selected is not None
        assert selected.instance_id == "instance-2"

    def test_clear_affinity(self, sample_instances):
        """Test clearing affinity mappings."""
        router = RequestRouter(routing_strategy="affinity")

        # Set affinity
        router.update_affinity("user-123", "instance-2")
        router.update_affinity("user-456", "instance-3")

        # Clear specific user
        router.clear_affinity("user-123")
        assert "user-123" not in router.affinity_map
        assert "user-456" in router.affinity_map

        # Clear all
        router.clear_affinity()
        assert len(router.affinity_map) == 0


class TestLoadBalancer:
    """Tests for LoadBalancer."""

    def test_least_connections(self, sample_instances):
        """Test least connections algorithm."""
        balancer = LoadBalancer()
        selected = balancer.select_instance(sample_instances, "least_connections")

        assert selected is not None
        assert selected.instance_id == "instance-1"
        assert selected.active_requests == min(i.active_requests for i in sample_instances)

    def test_least_response_time(self, sample_instances):
        """Test least response time algorithm."""
        balancer = LoadBalancer()
        selected = balancer.select_instance(sample_instances, "least_response_time")

        assert selected is not None
        assert selected.instance_id == "instance-1"
        assert selected.avg_latency_ms == min(i.avg_latency_ms for i in sample_instances)

    def test_weighted_round_robin(self, sample_instances):
        """Test weighted round-robin algorithm."""
        balancer = LoadBalancer()

        # Test multiple selections
        selections = []
        for _ in range(20):
            selected = balancer.select_instance(sample_instances, "weighted_round_robin")
            assert selected is not None
            selections.append(selected.instance_id)

        # Instance-1 should be selected more often (higher available_capacity)
        assert "instance-1" in selections
        assert selections.count("instance-1") > selections.count("instance-3")

    def test_power_of_two(self, sample_instances):
        """Test power of two choices algorithm."""
        balancer = LoadBalancer()

        # Test multiple selections
        for _ in range(10):
            selected = balancer.select_instance(sample_instances, "power_of_two")
            assert selected is not None
            assert selected.instance_id in [i.instance_id for i in sample_instances]

    def test_power_of_two_with_few_instances(self):
        """Test power of two with only one or two instances."""
        balancer = LoadBalancer()

        # Single instance
        single_instance = [
            ExecutionInstance(
                instance_id="instance-1",
                host="localhost",
                port=8000,
                model_name="llama-7b",
                tensor_parallel_size=1,
                gpu_count=1,
                current_load=0.5,
            )
        ]

        selected = balancer.select_instance(single_instance, "power_of_two")
        assert selected is not None
        assert selected.instance_id == "instance-1"

    def test_record_request(self):
        """Test recording requests to instances."""
        balancer = LoadBalancer()

        balancer.record_request("instance-1")
        balancer.record_request("instance-1")
        balancer.record_request("instance-2")

        assert balancer.request_counts["instance-1"] == 2
        assert balancer.request_counts["instance-2"] == 1

    def test_record_latency(self):
        """Test recording latency for instances."""
        balancer = LoadBalancer()

        balancer.record_latency("instance-1", 50.0)
        balancer.record_latency("instance-1", 60.0)
        balancer.record_latency("instance-1", 70.0)

        assert len(balancer.latency_history["instance-1"]) == 3
        assert balancer.latency_history["instance-1"] == [50.0, 60.0, 70.0]

    def test_latency_history_limit(self):
        """Test latency history maintains max 100 samples."""
        balancer = LoadBalancer()

        # Record 150 latencies
        for i in range(150):
            balancer.record_latency("instance-1", float(i))

        # Should keep only last 100
        assert len(balancer.latency_history["instance-1"]) == 100
        assert balancer.latency_history["instance-1"][0] == 50.0  # First of last 100
        assert balancer.latency_history["instance-1"][-1] == 149.0  # Last recorded

    def test_get_stats(self):
        """Test getting statistics for an instance."""
        balancer = LoadBalancer()

        balancer.record_request("instance-1")
        balancer.record_request("instance-1")
        balancer.record_latency("instance-1", 50.0)
        balancer.record_latency("instance-1", 70.0)

        stats = balancer.get_stats("instance-1")

        assert stats["request_count"] == 2
        assert stats["avg_latency_ms"] == 60.0

    def test_get_stats_no_data(self):
        """Test getting statistics for instance with no data."""
        balancer = LoadBalancer()

        stats = balancer.get_stats("instance-unknown")

        assert stats["request_count"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_no_available_instances(self):
        """Test load balancer when no instances available."""
        balancer = LoadBalancer()

        unavailable = [
            ExecutionInstance(
                instance_id="instance-1",
                host="localhost",
                port=8000,
                model_name="llama-7b",
                tensor_parallel_size=1,
                gpu_count=1,
                is_available=False,  # Mark as unavailable
            )
        ]

        selected = balancer.select_instance(unavailable, "least_connections")
        assert selected is None

    def test_weighted_round_robin_zero_capacity(self):
        """Test weighted round-robin with zero total capacity."""
        balancer = LoadBalancer()

        zero_capacity_instances = [
            ExecutionInstance(
                instance_id="instance-1",
                host="localhost",
                port=8000,
                model_name="llama-7b",
                tensor_parallel_size=1,
                gpu_count=1,
                current_load=1.0,  # Full load = 0 available capacity
            ),
            ExecutionInstance(
                instance_id="instance-2",
                host="localhost",
                port=8001,
                model_name="llama-7b",
                tensor_parallel_size=1,
                gpu_count=1,
                current_load=1.0,  # Full load = 0 available capacity
                is_available=False,  # Mark as unavailable
            ),
        ]

        selected = balancer.select_instance(zero_capacity_instances, "weighted_round_robin")
        # Should return None since no instances have available capacity
        assert selected is None
