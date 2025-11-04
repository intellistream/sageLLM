# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Execution Coordinator with HTTP client mode."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

# flake8: noqa: E402
from control_plane import (
    ExecutionInstance,
    ParallelismType,
    RequestMetadata,
    RequestPriority,
    SchedulingDecision,
)
from control_plane.executors.http_client import HttpExecutionCoordinator as ExecutionCoordinator


@pytest.mark.asyncio
async def test_executor_initialization():
    """Test ExecutionCoordinator initialization."""
    executor = ExecutionCoordinator()

    assert len(executor.instances) == 0
    assert len(executor.active_requests) == 0
    assert executor.http_session is None

    # Test initialization
    await executor.initialize()
    assert executor.http_session is not None

    # Cleanup
    await executor.cleanup()
    assert executor.http_session is None


@pytest.mark.asyncio
async def test_instance_registration():
    """Test instance registration with executor."""
    executor = ExecutionCoordinator()

    instance = ExecutionInstance(
        instance_id="test-instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )

    executor.register_instance(instance)

    # Verify instance registered
    assert "test-instance-1" in executor.instances
    assert executor.instances["test-instance-1"] == instance


@pytest.mark.asyncio
async def test_instance_unregistration():
    """Test instance unregistration."""
    executor = ExecutionCoordinator()

    instance = ExecutionInstance(
        instance_id="test-instance-1",
        host="localhost",
        port=8000,
        model_name="llama-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )

    executor.register_instance(instance)
    assert len(executor.instances) == 1

    executor.unregister_instance("test-instance-1")
    assert len(executor.instances) == 0


@pytest.mark.asyncio
async def test_get_available_instances():
    """Test getting available instances."""
    executor = ExecutionCoordinator()

    # Create instances with different states
    healthy_instance = ExecutionInstance(
        instance_id="healthy-1",
        host="localhost",
        port=8001,
        model_name="llama-7b",
        tensor_parallel_size=1,
        gpu_count=1,
        is_healthy=True,
        is_available=True,
    )

    overloaded_instance = ExecutionInstance(
        instance_id="overloaded-1",
        host="localhost",
        port=8002,
        model_name="llama-7b",
        tensor_parallel_size=1,
        gpu_count=1,
        is_healthy=True,
        is_available=True,
        current_load=0.98,  # Almost fully loaded
        active_requests=99,
        max_concurrent_requests=100,
    )

    executor.register_instance(healthy_instance)
    executor.register_instance(overloaded_instance)

    available = executor.get_available_instances()

    # Only healthy instance should be available
    assert len(available) <= 2
    assert healthy_instance in available


@pytest.mark.asyncio
async def test_metrics_collection():
    """Test performance metrics collection."""
    executor = ExecutionCoordinator()

    # Initial metrics should be zero
    metrics = executor.get_metrics()
    assert metrics.total_requests == 0
    assert metrics.completed_requests == 0
    assert metrics.failed_requests == 0

    # Register instances for metrics calculation
    instance = ExecutionInstance(
        instance_id="test-1",
        host="localhost",
        port=8000,
        model_name="llama-7b",
        tensor_parallel_size=1,
        gpu_count=1,
        gpu_utilization=0.8,
    )

    executor.register_instance(instance)

    metrics = executor.get_metrics()
    assert metrics.active_requests == 0

    # Verify total GPU memory calculation
    assert metrics.total_gpu_memory_gb >= 0


@pytest.mark.asyncio
async def test_execute_request_http(mock_aiohttp, mock_vllm_completion_response):
    """Test request execution via HTTP API."""
    executor = ExecutionCoordinator()
    await executor.initialize()

    # Register instance
    instance = ExecutionInstance(
        instance_id="test-instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )
    executor.register_instance(instance)

    # Create request
    request = RequestMetadata(
        request_id="test-req-1",
        prompt="What is the capital of France?",
        max_tokens=100,
        priority=RequestPriority.NORMAL,
    )

    # Mock HTTP response
    mock_aiohttp.post(
        "http://localhost:8000/v1/completions",
        payload=mock_vllm_completion_response,
        status=200,
    )

    # Create scheduling decision
    decision = SchedulingDecision(
        request_id=request.request_id,
        target_instance_id=instance.instance_id,
        parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
        estimated_latency_ms=100.0,
        estimated_cost=0.01,
        reason="Test execution",
    )

    # Execute request
    result = await executor.execute_request(request, instance, decision)

    # Verify result
    assert result["id"] == "cmpl-test-123"
    assert result["model"] == "meta-llama/Llama-2-7b"
    assert result["usage"]["total_tokens"] == 30

    # Verify metrics updated
    metrics = executor.get_metrics()
    assert metrics.total_requests == 1
    assert metrics.completed_requests == 1

    await executor.cleanup()


@pytest.mark.asyncio
async def test_execute_request_failure(mock_aiohttp):
    """Test request execution failure handling."""
    executor = ExecutionCoordinator()
    await executor.initialize()

    instance = ExecutionInstance(
        instance_id="test-instance-1",
        host="localhost",
        port=8000,
        model_name="llama-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )
    executor.register_instance(instance)

    request = RequestMetadata(
        request_id="test-req-fail",
        prompt="Test prompt",
        max_tokens=100,
        priority=RequestPriority.NORMAL,
    )

    # Mock HTTP error
    mock_aiohttp.post(
        "http://localhost:8000/v1/completions",
        status=500,
        body="Internal Server Error",
    )

    decision = SchedulingDecision(
        request_id=request.request_id,
        target_instance_id=instance.instance_id,
        parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
        estimated_latency_ms=100.0,
        estimated_cost=0.01,
        reason="Test failure",
    )

    # Expect RuntimeError
    with pytest.raises(RuntimeError, match="vLLM returned status 500"):
        await executor.execute_request(request, instance, decision)

    # Verify metrics
    metrics = executor.get_metrics()
    assert metrics.total_requests == 1
    assert metrics.failed_requests == 1

    await executor.cleanup()


@pytest.mark.asyncio
async def test_health_check_http(mock_aiohttp):
    """Test health check via HTTP."""
    executor = ExecutionCoordinator()
    await executor.initialize()

    instance = ExecutionInstance(
        instance_id="test-instance-1",
        host="localhost",
        port=8000,
        model_name="llama-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )

    # Mock healthy response
    mock_aiohttp.get("http://localhost:8000/health", status=200)

    is_healthy = await executor.health_check(instance)
    assert is_healthy is True

    # Mock unhealthy response
    mock_aiohttp.get("http://localhost:8000/health", status=503)

    is_healthy = await executor.health_check(instance)
    assert is_healthy is False

    await executor.cleanup()


@pytest.mark.asyncio
async def test_get_instance_info_http(mock_aiohttp):
    """Test getting instance info via HTTP."""
    executor = ExecutionCoordinator()
    await executor.initialize()

    instance = ExecutionInstance(
        instance_id="test-instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )

    # Mock models endpoint response
    models_response = {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-2-7b",
                "object": "model",
                "created": 1677652288,
                "owned_by": "vllm",
            }
        ],
    }

    mock_aiohttp.get(
        "http://localhost:8000/v1/models",
        payload=models_response,
        status=200,
    )

    info = await executor.get_instance_info(instance)
    assert info is not None
    assert info["object"] == "list"
    assert len(info["data"]) == 1
    assert info["data"][0]["id"] == "meta-llama/Llama-2-7b"

    await executor.cleanup()
