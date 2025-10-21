# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Integration tests for Control Plane with SAGE and vLLM.

This test file demonstrates how SAGE components can communicate with
vLLM through the Control Plane for large language model inference.
"""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (  # noqa: E402
    ControlPlaneManager,
    DecodingConfig,
    ExecutionInstance,
    ExecutionInstanceType,
    PDSeparationConfig,
    PreffillingConfig,
    RequestMetadata,
    RequestPriority,
)


@pytest.mark.asyncio
async def test_sage_control_plane_vllm_integration():
    """
    Test the integration flow: SAGE -> Control Plane -> vLLM.

    This demonstrates how SAGE applications submit inference requests
    through the Control Plane, which then routes them to appropriate
    vLLM instances based on scheduling and PD separation policies.
    """
    # Step 1: Create Control Plane (central coordination)
    pd_config = PDSeparationConfig(
        enabled=True,
        routing_policy="adaptive",
    )

    control_plane = ControlPlaneManager(
        scheduling_policy="adaptive",
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Step 2: Register vLLM instances
    # In a real deployment, these would be actual vLLM servers
    vllm_instances = [
        ExecutionInstance(
            instance_id="vllm-prefilling-1",
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
            ),
        ),
        ExecutionInstance(
            instance_id="vllm-decoding-1",
            host="localhost",
            port=8002,
            model_name="meta-llama/Llama-2-7b",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_count=1,
            instance_type=ExecutionInstanceType.DECODING,
            decoding_config=DecodingConfig(
                target_latency_ms=50,
                max_parallel_requests=200,
            ),
        ),
    ]

    for instance in vllm_instances:
        control_plane.register_instance(instance)

    # Verify instances registered
    assert len(control_plane.executor.get_all_instances()) == 2

    # Step 3: SAGE applications submit inference requests
    # These could be from different SAGE modules/services
    sage_requests = [
        {
            "app": "document_analysis",
            "request": RequestMetadata(
                request_id="doc-analysis-001",
                priority=RequestPriority.NORMAL,
                max_tokens=512,
            ),
        },
        {
            "app": "chat_service",
            "request": RequestMetadata(
                request_id="chat-001",
                priority=RequestPriority.HIGH,
                max_tokens=256,
            ),
        },
        {
            "app": "code_review",
            "request": RequestMetadata(
                request_id="code-review-001",
                priority=RequestPriority.NORMAL,
                slo_deadline_ms=1000.0,
                max_tokens=1024,
            ),
        },
    ]

    # Step 4: Submit requests through Control Plane
    submitted_ids = []
    for sage_app in sage_requests:
        request_id = await control_plane.submit_request(sage_app["request"])
        submitted_ids.append(request_id)

    # Verify all requests queued
    assert len(control_plane.pending_queue) == 3

    # Step 5: Control Plane determines request phases
    # This demonstrates the PD separation decision logic
    for sage_app in sage_requests:
        request = sage_app["request"]
        if control_plane.pd_router:
            phase = control_plane.pd_router.determine_request_phase(request)
            assert phase is not None
            print(f"Request {request.request_id} routed to phase: {phase.name}")

    # Step 6: Verify Control Plane state
    assert control_plane.executor is not None
    assert len(control_plane.executor.instances) == 2

    # Step 7: Check instance health status
    for instance_id in ["vllm-prefilling-1", "vllm-decoding-1"]:
        instance = control_plane.executor.get_instance(instance_id)
        assert instance is not None
        assert instance.instance_id == instance_id


@pytest.mark.asyncio
async def test_control_plane_request_flow():
    """
    Test the complete request flow through Control Plane.

    Demonstrates: SAGE App -> Control Plane Scheduler ->
    Router -> Executor -> vLLM Instance
    """
    # Initialize Control Plane
    control_plane = ControlPlaneManager(
        scheduling_policy="priority",
        enable_pd_separation=True,
    )

    # Register vLLM instance
    instance = ExecutionInstance(
        instance_id="vllm-default",
        host="localhost",
        port=8000,
        model_name="llama-7b",
        tensor_parallel_size=2,
        gpu_count=2,
    )

    control_plane.register_instance(instance)

    # Simulate requests from different SAGE components
    request_sources = [
        ("sage-chat", RequestPriority.HIGH),
        ("sage-embedding", RequestPriority.NORMAL),
        ("sage-batch", RequestPriority.LOW),
    ]

    for source, priority in request_sources:
        request = RequestMetadata(
            request_id=f"{source}-req-001",
            priority=priority,
            max_tokens=256,
        )

        request_id = await control_plane.submit_request(request)
        assert request_id == f"{source}-req-001"

    # Verify request queue
    assert len(control_plane.pending_queue) == 3

    # Verify scheduler can process requests
    all_instances = control_plane.executor.get_all_instances()
    assert len(all_instances) == 1
    assert all_instances[0].instance_id == "vllm-default"


@pytest.mark.asyncio
async def test_multi_model_deployment():
    """
    Test Control Plane managing multiple vLLM models.

    Demonstrates how Control Plane can coordinate inference
    across different model versions or sizes.
    """
    control_plane = ControlPlaneManager(
        scheduling_policy="adaptive",
    )

    # Register multiple model instances
    models = [
        ("llama-7b", 8001, 1),
        ("llama-13b", 8002, 2),
        ("llama-70b", 8003, 8),
    ]

    for model_name, port, gpu_count in models:
        instance = ExecutionInstance(
            instance_id=f"vllm-{model_name}",
            host="localhost",
            port=port,
            model_name=model_name,
            tensor_parallel_size=gpu_count // 2 if gpu_count > 1 else 1,
            gpu_count=gpu_count,
        )
        control_plane.register_instance(instance)

    # Verify all models registered
    all_instances = control_plane.executor.get_all_instances()
    assert len(all_instances) == 3

    # Verify model instances accessible
    model_ids = {inst.model_name for inst in all_instances}
    assert "llama-7b" in model_ids
    assert "llama-13b" in model_ids
    assert "llama-70b" in model_ids


@pytest.mark.asyncio
async def test_control_plane_health_monitoring():
    """
    Test Control Plane's instance health monitoring.

    SAGE applications rely on Control Plane to maintain
    healthy vLLM instance pool.
    """
    control_plane = ControlPlaneManager()

    # Register instances
    instances = [
        ExecutionInstance(
            instance_id="vllm-healthy",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            is_healthy=True,
            is_available=True,
        ),
        ExecutionInstance(
            instance_id="vllm-degraded",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            tensor_parallel_size=1,
            gpu_count=1,
            is_healthy=False,
            is_available=True,
        ),
    ]

    for instance in instances:
        control_plane.register_instance(instance)

    # Get available instances (should exclude unhealthy ones)
    available = control_plane.executor.get_available_instances()

    # Verify health status is considered
    for instance in available:
        assert instance.is_healthy or not instance.is_available

    # Mark instance as recovered
    recovered_instance = control_plane.executor.get_instance("vllm-degraded")
    if recovered_instance:
        recovered_instance.is_healthy = True

    # Verify updated health status
    updated_available = control_plane.executor.get_available_instances()
    assert len(updated_available) >= 1
