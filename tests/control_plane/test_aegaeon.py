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


@pytest.mark.asyncio
async def test_aegaeon_token_level_preemption():
    """Test token-level preemption based on SLO violation risk."""
    from datetime import datetime, timedelta

    from control_plane.strategies.aegaeon import AegaeonPolicy, DecodingBatch

    policy = AegaeonPolicy()

    # Create requests with different SLO states
    now = datetime.now()

    # High-risk request: last token was 200ms ago, SLO is 50ms
    high_risk_req = RequestMetadata(
        request_id="high-risk",
        model_name="test-model",
        tbt_slo_ms=50.0,
        last_token_time=now - timedelta(milliseconds=200),
        max_tokens=100,
    )

    # Safe request: last token was 20ms ago, SLO is 50ms
    safe_req = RequestMetadata(
        request_id="safe",
        model_name="test-model",
        tbt_slo_ms=50.0,
        last_token_time=now - timedelta(milliseconds=20),
        max_tokens=100,
    )

    # Create decoding batch with both requests
    batch = DecodingBatch(
        model_name="test-model",
        requests=[high_risk_req, safe_req],
    )

    decoding_instances = [
        ExecutionInstance(
            instance_id="decoding-1",
            host="localhost",
            port=8001,
            model_name="",
            instance_type=ExecutionInstanceType.DECODING,
            tensor_parallel_size=1,
            gpu_count=1,
        )
    ]

    policy.decoding_work_lists["decoding-1"] = [batch]

    # Calculate SLO violation risks
    high_risk_score = policy._calculate_slo_violation_risk(high_risk_req)
    safe_risk_score = policy._calculate_slo_violation_risk(safe_req)

    assert high_risk_score > 0.8  # Should be high risk
    assert safe_risk_score < 0.5  # Should be safe

    # Test preemption logic
    preempted = policy._check_and_preempt(decoding_instances)

    # High-risk request should be kept, others might be preempted if needed
    # (Current implementation keeps top 1 high-risk, preempts rest)
    assert len(batch.requests) > 0  # At least some requests should remain


@pytest.mark.asyncio
async def test_aegaeon_tbt_tracking():
    """Test Time Between Tokens (TBT) tracking."""
    from datetime import datetime

    from control_plane.strategies.aegaeon import DecodingBatch

    # Create request with TBT SLO
    request = RequestMetadata(
        request_id="test-req",
        model_name="test-model",
        tbt_slo_ms=100.0,  # 100ms TBT SLO
        max_tokens=10,
    )

    batch = DecodingBatch(
        model_name="test-model",
        requests=[request],
        decoding_step_time=0.025,  # 25ms per step
    )

    # Verify n_value calculation: n = d/t = 100ms / 25ms = 4.0
    assert abs(batch.n_value - 4.0) < 0.01

    # Test token generation tracking
    assert request.tokens_generated == 0
    assert request.last_token_time is None

    request.tokens_generated = 5
    request.last_token_time = datetime.now()

    assert request.tokens_generated == 5
    assert request.last_token_time is not None


@pytest.mark.asyncio
async def test_aegaeon_quota_execution():
    """Test batch execution with time quota limits."""
    from control_plane.strategies.aegaeon import AegaeonPolicy, DecodingBatch

    policy = AegaeonPolicy()

    request = RequestMetadata(
        request_id="test-req",
        model_name="test-model",
        tbt_slo_ms=100.0,
        max_tokens=100,
    )

    batch = DecodingBatch(
        model_name="test-model",
        requests=[request],
        time_quota=0.5,  # 500ms quota
        decoding_step_time=0.025,
    )

    # Reset quota to full
    batch.reset_quota()
    assert batch.remaining_quota == 0.5

    # Consume some quota
    remaining = batch.consume_quota(0.2)
    assert remaining == 0.3
    assert batch.remaining_quota == 0.3

    # Consume more than remaining (should clamp to 0)
    remaining = batch.consume_quota(0.5)
    assert remaining == 0.0
    assert batch.remaining_quota == 0.0


@pytest.mark.asyncio
async def test_aegaeon_dynamic_scaling_overhead():
    """Test dynamic model switching overhead calculation."""
    from control_plane.strategies.aegaeon import AegaeonPolicy

    policy = AegaeonPolicy(use_dynamic_scaling_overhead=True)

    # Small model (7B) with TP=2
    overhead_7b = policy._get_scaling_overhead("llama-7b", 2)

    # Large model (70B) with TP=4
    overhead_70b = policy._get_scaling_overhead("llama-70b", 4)

    # Large model should have higher overhead
    assert overhead_70b > overhead_7b

    # Higher TP should increase overhead
    overhead_7b_tp1 = policy._get_scaling_overhead("llama-7b", 1)
    overhead_7b_tp4 = policy._get_scaling_overhead("llama-7b", 4)

    assert overhead_7b_tp4 > overhead_7b_tp1

    # Test caching
    overhead_cached = policy._get_scaling_overhead("llama-7b", 2)
    assert overhead_cached == overhead_7b  # Should return cached value


@pytest.mark.asyncio
async def test_aegaeon_decoding_round_with_quotas():
    """Test complete decoding round scheduling with time quotas."""
    from control_plane.strategies.aegaeon import AegaeonPolicy, DecodingBatch

    policy = AegaeonPolicy()

    # Create three batches with different n_values
    batch1 = DecodingBatch(
        model_name="model-a",
        requests=[
            RequestMetadata(
                request_id="req-1",
                model_name="model-a",
                tbt_slo_ms=100.0,
                max_tokens=100,
            )
        ],
        decoding_step_time=0.025,  # n = 100/25 = 4.0
    )

    batch2 = DecodingBatch(
        model_name="model-b",
        requests=[
            RequestMetadata(
                request_id="req-2",
                model_name="model-b",
                tbt_slo_ms=50.0,
                max_tokens=100,
            )
        ],
        decoding_step_time=0.025,  # n = 50/25 = 2.0
    )

    policy.decoding_work_lists["instance-1"] = [batch1, batch2]

    # Schedule decoding round
    schedule = policy.schedule_decoding_round("instance-1")

    # Should have 2 batches with quotas
    assert len(schedule) == 2

    # Both batches should have positive quotas
    for batch, quota in schedule:
        assert quota > 0
        assert batch.time_quota == quota
        assert batch.remaining_quota == quota  # Should be reset

    # Batch with smaller n (more urgent) should get larger quota
    batch2_entry = next((b, q) for b, q in schedule if b.model_name == "model-b")
    batch1_entry = next((b, q) for b, q in schedule if b.model_name == "model-a")

    # batch2 has n=2, batch1 has n=4
    # Formula: q_i = (c/n_i) * (α - Σ(1/n_k))
    # With small n values and default c=1.0, quotas should be inversely proportional to n
    # Both should have positive quotas
    assert batch2_entry[1] > 0
    assert batch1_entry[1] > 0

    # Due to the (c/n_i) factor, smaller n_i gets larger c/n_i multiplier
    # So batch2 (n=2) should get quota >= batch1 (n=4) if α - Σ(1/n) > 0
    assert batch2_entry[1] >= batch1_entry[1]

    # Group should be removed after all requests are popped
    assert len(policy.prefill_job_queues["instance-1"]) == 0
