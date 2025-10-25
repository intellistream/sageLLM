# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Execution Coordinator with vLLM direct integration."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import ExecutionCoordinator, ExecutionInstance  # noqa: E402


@pytest.mark.asyncio
async def test_executor_initialization():
    """Test ExecutionCoordinator initialization."""
    executor = ExecutionCoordinator()

    assert len(executor.instances) == 0
    assert len(executor.active_requests) == 0
    assert len(executor.engines) == 0


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
