# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Aegaeon scheduling policy."""

import pytest

from control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    ExecutionInstanceType,
    RequestMetadata,
    RequestPriority,
)


@pytest.mark.asyncio
async def test_aegaeon_basic_initialization():
    """Test basic Aegaeon policy initialization."""
    manager = ControlPlaneManager(
        scheduling_policy="aegaeon",
        enable_pd_separation=True,
        mode="local",
    )

    assert manager.scheduling_policy.name == "aegaeon"


@pytest.mark.asyncio
async def test_aegaeon_prefill_grouping():
    """Test prefill phase grouping with MAX_GPSIZE=8."""
    from control_plane.strategies.aegaeon import AegaeonPolicy

    policy = AegaeonPolicy(max_group_size=8)

    # Create prefill instances
    prefill_instances = [
        ExecutionInstance(
            instance_id="prefill-1",
            host="localhost",
            port=8000,
            model_name="meta-llama/Llama-2-7b",
            instance_type=ExecutionInstanceType.PREFILLING,
            tensor_parallel_size=4,
            gpu_count=4,
        )
    ]

    # Submit 10 requests for same model
    model_name = "meta-llama/Llama-2-7b"
    requests = [
        RequestMetadata(
            request_id=f"req-{i}",
            model_name=model_name,
            priority=RequestPriority.NORMAL,
            max_tokens=100,
        )
        for i in range(10)
    ]

    # Schedule requests
    for request in requests:
        policy._schedule_prefill_request(request, prefill_instances)

    # Verify grouping: first 8 in one group, next 2 in another
    job_queue = policy.prefill_job_queues["prefill-1"]

    assert len(job_queue) == 2  # Two groups
    assert job_queue[0].size == 8  # First group is full
    assert job_queue[1].size == 2  # Second group has 2 requests


@pytest.mark.asyncio
async def test_aegaeon_multi_model_grouping():
    """Test that different models are grouped separately."""
    from control_plane.strategies.aegaeon import AegaeonPolicy

    policy = AegaeonPolicy()

    # Create prefill instances
    prefill_instances = [
        ExecutionInstance(
            instance_id=f"prefill-{i}",
            host="localhost",
            port=8000 + i,
            model_name="",
            instance_type=ExecutionInstanceType.PREFILLING,
            tensor_parallel_size=2,
            gpu_count=2,
        )
        for i in range(2)
    ]

    # Submit requests for different models
    models = ["llama-7b", "qwen-7b", "llama-7b", "internlm-7b", "llama-7b"]

    for i, model in enumerate(models):
        request = RequestMetadata(
            request_id=f"req-{i}",
            model_name=model,
            priority=RequestPriority.NORMAL,
            max_tokens=100,
        )
        policy._schedule_prefill_request(request, prefill_instances)

    # Verify same models are grouped together
    all_groups = []
    for instance_id in policy.prefill_job_queues:
        all_groups.extend(policy.prefill_job_queues[instance_id])

    # Count requests per model
    model_counts = {}
    for group in all_groups:
        model_counts[group.model_name] = model_counts.get(group.model_name, 0) + group.size

    assert model_counts.get("llama-7b", 0) == 3
    assert model_counts.get("qwen-7b", 0) == 1
    assert model_counts.get("internlm-7b", 0) == 1


@pytest.mark.asyncio
async def test_aegaeon_decoding_quota_calculation():
    """Test decoding phase time quota calculation."""
    from control_plane.strategies.aegaeon import AegaeonPolicy, DecodingBatch

    policy = AegaeonPolicy(
        estimated_scaling_overhead=3.0,  # c = 3 seconds
        qmax=3.0,
        min_alpha=0.5,
    )

    # Create test batches
    batch1 = DecodingBatch(
        model_name="model-1",
        requests=[
            RequestMetadata(
                request_id="req-1",
                slo_deadline_ms=100.0,  # TBT = 100ms
            )
        ],
        decoding_step_time=0.025,  # 25ms
    )

    policy.decoding_work_lists["instance-1"] = [batch1]

    # Calculate schedule
    schedule = policy.schedule_decoding_round("instance-1")

    assert len(schedule) == 1
    batch, quota = schedule[0]

    # Verify n_i = d/t_i = 100/25 = 4
    assert batch.n_value == 4.0

    # Verify quota is calculated correctly
    assert quota > 0.1  # At least minimum quota


@pytest.mark.asyncio
async def test_aegaeon_load_balancing():
    """Test that requests are balanced across instances."""
    from control_plane.strategies.aegaeon import AegaeonPolicy

    policy = AegaeonPolicy()

    # Create multiple prefill instances
    prefill_instances = [
        ExecutionInstance(
            instance_id=f"prefill-{i}",
            host="localhost",
            port=8000 + i,
            model_name="",
            instance_type=ExecutionInstanceType.PREFILLING,
            tensor_parallel_size=2,
            gpu_count=2,
        )
        for i in range(3)
    ]

    # Submit 25 requests to test load balancing across 3 instances
    # Expected distribution: 8, 8, 8, 1 (greedy group filling)
    for i in range(25):
        request = RequestMetadata(
            request_id=f"req-{i}",
            model_name="test-model",
            priority=RequestPriority.NORMAL,
            max_tokens=100,
        )
        policy._schedule_prefill_request(request, prefill_instances)

    # Check distribution
    instance_loads = {}
    for instance_id in policy.prefill_job_queues:
        total_requests = sum(g.size for g in policy.prefill_job_queues[instance_id])
        instance_loads[instance_id] = total_requests

    # Verify all instances are used and total is correct
    loads = list(instance_loads.values())
    assert sum(loads) == 25
    assert all(load > 0 for load in loads)  # All instances should have some load
    assert max(loads) <= 16  # No instance should have more than 2 full groups


@pytest.mark.asyncio
async def test_aegaeon_fifo_within_groups():
    """Test FIFO ordering within groups."""
    from control_plane.strategies.aegaeon import AegaeonPolicy

    policy = AegaeonPolicy()

    # Manually create a group with requests
    from control_plane.strategies.aegaeon import PrefillGroup

    group = PrefillGroup(
        model_name="test-model",
        requests=[
            RequestMetadata(request_id=f"req-{i}", model_name="test-model")
            for i in range(3)
        ],
    )

    policy.prefill_job_queues["instance-1"] = [group]

    # Get requests in order
    req1 = policy.get_next_prefill_request("instance-1")
    req2 = policy.get_next_prefill_request("instance-1")
    req3 = policy.get_next_prefill_request("instance-1")

    assert req1.request_id == "req-0"
    assert req2.request_id == "req-1"
    assert req3.request_id == "req-2"

    # Group should be removed after all requests are popped
    assert len(policy.prefill_job_queues["instance-1"]) == 0
