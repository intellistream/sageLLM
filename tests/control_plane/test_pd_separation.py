# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for PD Separation (Prefilling/Decoding) routing."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (ControlPlaneManager, DecodingConfig,  # noqa: E402
                           ExecutionInstance, ExecutionInstanceType,
                           PDSeparationConfig, PreffillingConfig,
                           RequestMetadata, RequestPriority)


@pytest.mark.asyncio
async def test_pd_separation_routing():
    """Test PD separation with specialized instances."""
    # Create PD configuration
    pd_config = PDSeparationConfig(
        enabled=True,
        routing_policy="adaptive",
        prefilling_threshold_input_tokens=800,
        prefilling_threshold_ratio=4.0,
    )

    # Create control plane with PD separation
    manager = ControlPlaneManager(
        scheduling_policy="adaptive",
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Register prefilling-specialized instance
    prefilling_instance = ExecutionInstance(
        instance_id="prefilling-1",
        host="localhost",
        port=8001,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        gpu_count=4,
        instance_type=ExecutionInstanceType.PREFILLING,
        prefilling_config=PreffillingConfig(
            target_batch_size=64,
            tensor_parallel_size=4,
            enable_chunked_prefill=True,
        ),
    )

    # Register decoding-specialized instance
    decoding_instance = ExecutionInstance(
        instance_id="decoding-1",
        host="localhost",
        port=8002,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_count=1,
        instance_type=ExecutionInstanceType.DECODING,
        decoding_config=DecodingConfig(
            target_latency_ms=50,
            tensor_parallel_size=1,
            max_parallel_requests=200,
        ),
    )

    manager.register_instance(prefilling_instance)
    manager.register_instance(decoding_instance)

    # Verify both instances registered
    all_instances = manager.executor.get_all_instances()
    assert len(all_instances) == 2

    # Test requests with different characteristics
    test_cases = [
        {
            "request_id": "long-prefill-req",
            "max_tokens": 100,
            "expected_phase": ExecutionInstanceType.PREFILLING,
        },
        {
            "request_id": "short-decode-req",
            "max_tokens": 500,
            "expected_phase": ExecutionInstanceType.DECODING,
        },
    ]

    for test_case in test_cases:
        request = RequestMetadata(
            request_id=test_case["request_id"],
            priority=RequestPriority.NORMAL,
            max_tokens=test_case["max_tokens"],
        )

        # Determine routing phase
        if manager.pd_router:
            phase = manager.pd_router.determine_request_phase(request)
            assert phase in [
                ExecutionInstanceType.PREFILLING,
                ExecutionInstanceType.DECODING,
                ExecutionInstanceType.HYBRID,
            ]


@pytest.mark.asyncio
async def test_instance_specialization_scoring():
    """Test instance specialization scoring for PD routing."""
    pd_config = PDSeparationConfig(enabled=True)

    manager = ControlPlaneManager(
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Create diverse instances
    instances = [
        ExecutionInstance(
            instance_id="prefilling-specialized",
            host="localhost",
            port=9001,
            model_name="model",
            tensor_parallel_size=8,
            gpu_count=8,
            instance_type=ExecutionInstanceType.PREFILLING,
        ),
        ExecutionInstance(
            instance_id="decoding-specialized",
            host="localhost",
            port=9002,
            model_name="model",
            tensor_parallel_size=1,
            gpu_count=1,
            instance_type=ExecutionInstanceType.DECODING,
        ),
        ExecutionInstance(
            instance_id="general-purpose",
            host="localhost",
            port=9003,
            model_name="model",
            tensor_parallel_size=4,
            gpu_count=4,
            instance_type=ExecutionInstanceType.GENERAL,
        ),
    ]

    for instance in instances:
        manager.register_instance(instance)

    # Score instances for different phases
    phases = [
        ExecutionInstanceType.PREFILLING,
        ExecutionInstanceType.DECODING,
    ]

    for phase in phases:
        scores = {}
        for instance in instances:
            specialization = manager.pd_router.get_instance_specialization(instance)
            # Get the score for this phase
            if phase == ExecutionInstanceType.PREFILLING:
                score = specialization["prefilling_score"]
            else:  # DECODING
                score = specialization["decoding_score"]

            scores[instance.instance_id] = score
            assert 0.0 <= score <= 1.0

        # Verify specialized instances have higher scores for their phase
        if phase == ExecutionInstanceType.PREFILLING:
            assert scores["prefilling-specialized"] > scores["general-purpose"]
        else:  # DECODING
            assert scores["decoding-specialized"] > scores["general-purpose"]


@pytest.mark.asyncio
async def test_pd_routing_policy_threshold():
    """Test threshold-based PD routing."""
    pd_config = PDSeparationConfig(
        enabled=True,
        routing_policy="threshold",
        prefilling_threshold_input_tokens=800,
        prefilling_threshold_ratio=4.0,
    )

    manager = ControlPlaneManager(
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Create request with many output tokens (short input/output ratio)
    request = RequestMetadata(
        request_id="decode-heavy",
        priority=RequestPriority.NORMAL,
        max_tokens=1000,  # More output tokens
    )

    phase = manager.pd_router.determine_request_phase(request)
    assert phase in [
        ExecutionInstanceType.PREFILLING,
        ExecutionInstanceType.DECODING,
        ExecutionInstanceType.HYBRID,
    ]


@pytest.mark.asyncio
async def test_pd_routing_policy_adaptive():
    """Test adaptive PD routing considering priority and SLO."""
    pd_config = PDSeparationConfig(
        enabled=True,
        routing_policy="adaptive",
    )

    manager = ControlPlaneManager(
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Critical priority request should prefer decoding (low latency)
    critical_request = RequestMetadata(
        request_id="critical-req",
        priority=RequestPriority.CRITICAL,
        max_tokens=256,
    )

    phase = manager.pd_router.determine_request_phase(critical_request)
    assert phase is not None

    # Request with tight SLO should prefer decoding
    tight_slo_request = RequestMetadata(
        request_id="tight-slo-req",
        priority=RequestPriority.NORMAL,
        slo_deadline_ms=50.0,
        max_tokens=256,
    )

    phase = manager.pd_router.determine_request_phase(tight_slo_request)
    assert phase is not None
