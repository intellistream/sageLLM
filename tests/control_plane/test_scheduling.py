# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Control Plane scheduling policies."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (ControlPlaneManager,  # noqa: E402
                           ExecutionInstance, RequestMetadata, RequestPriority)


@pytest.mark.asyncio
async def test_basic_scheduling():
    """Test basic request scheduling without PD separation."""
    # Create control plane without PD separation
    manager = ControlPlaneManager(
        scheduling_policy="fifo",
        enable_pd_separation=False,
    )

    # Register a vLLM instance
    instance = ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_count=1,
    )

    manager.register_instance(instance)
    assert instance.instance_id in [
        i.instance_id for i in manager.executor.get_all_instances()
    ]

    # Submit test requests
    for i in range(3):
        request = RequestMetadata(
            request_id=f"req-{i}",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
        )
        request_id = await manager.submit_request(request)
        assert request_id == f"req-{i}"
        assert len(manager.pending_queue) == i + 1


@pytest.mark.asyncio
async def test_priority_scheduling():
    """Test priority-based scheduling."""
    manager = ControlPlaneManager(
        scheduling_policy="priority",
        enable_pd_separation=False,
    )

    instance = ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        gpu_count=2,
    )

    manager.register_instance(instance)

    # Submit requests with different priorities
    priorities = [
        RequestPriority.CRITICAL,
        RequestPriority.HIGH,
        RequestPriority.NORMAL,
        RequestPriority.LOW,
    ]

    for i, priority in enumerate(priorities):
        request = RequestMetadata(
            request_id=f"req-{priority.name}",
            priority=priority,
            max_tokens=256,
        )
        await manager.submit_request(request)
        assert len(manager.pending_queue) == i + 1


@pytest.mark.asyncio
async def test_slo_aware_scheduling():
    """Test SLO-aware scheduling with deadline constraints."""
    manager = ControlPlaneManager(
        scheduling_policy="slo_aware",
        enable_pd_separation=False,
    )

    instance = ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )

    manager.register_instance(instance)

    # Submit requests with SLO deadlines
    slo_deadlines = [50.0, 200.0, 1000.0, None]

    for i, slo_ms in enumerate(slo_deadlines):
        request = RequestMetadata(
            request_id=f"req-slo-{i}",
            slo_deadline_ms=slo_ms,
            max_tokens=512,
        )
        await manager.submit_request(request)
        assert len(manager.pending_queue) == i + 1

        if slo_ms:
            assert request.slo_deadline_ms == slo_ms


@pytest.mark.asyncio
async def test_instance_registration():
    """Test instance registration and management."""
    manager = ControlPlaneManager()

    # Register multiple instances
    instances = [
        ExecutionInstance(
            instance_id=f"instance-{i}",
            host="localhost",
            port=8000 + i,
            model_name="llama-7b",
            tensor_parallel_size=i + 1,
            gpu_count=i + 1,
        )
        for i in range(3)
    ]

    for instance in instances:
        manager.register_instance(instance)

    # Verify all instances registered
    all_instances = manager.executor.get_all_instances()
    assert len(all_instances) == 3

    for instance in instances:
        assert instance.instance_id in [i.instance_id for i in all_instances]

    # Test unregistration
    manager.unregister_instance("instance-0")
    remaining_instances = manager.executor.get_all_instances()
    assert len(remaining_instances) == 2
