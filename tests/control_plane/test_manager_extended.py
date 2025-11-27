# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for ControlPlaneManager extensions for hybrid scheduling.

This module contains unit tests for the ControlPlaneManager extensions
that support mixed LLM and Embedding workloads scheduling (T2.3).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (  # noqa: E402  # type: ignore[import-not-found]
    ControlPlaneManager,
    ExecutionInstance,
    ExecutionInstanceType,
    RequestMetadata,
    RequestPriority,
    RequestStatus,
    RequestType,
)


# ============ Fixtures ============


@pytest.fixture
def mock_executor():
    """Create a mock executor coordinator."""
    executor = MagicMock()
    executor.instances = {}
    executor.register_instance = MagicMock()
    executor.unregister_instance = MagicMock()
    executor.get_available_instances = MagicMock(return_value=[])
    executor.get_all_instances = MagicMock(return_value=[])
    executor.get_instance = MagicMock(return_value=None)
    executor.execute_request = AsyncMock()
    executor.health_check_all = AsyncMock()
    executor.get_metrics = MagicMock(
        return_value=MagicMock(
            active_requests=0,
            completed_requests=0,
            failed_requests=0,
            avg_latency_ms=0.0,
            slo_compliance_rate=1.0,
        )
    )
    executor.set_manager_callback = MagicMock()
    return executor


@pytest.fixture
def llm_instance():
    """Create a LLM execution instance."""
    return ExecutionInstance(
        instance_id="llm-1",
        host="localhost",
        port=8000,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        instance_type=ExecutionInstanceType.GENERAL,
        tensor_parallel_size=1,
        gpu_count=1,
        current_load=0.3,
        active_requests=5,
        max_concurrent_requests=100,
    )


@pytest.fixture
def embedding_instance():
    """Create an embedding execution instance."""
    return ExecutionInstance(
        instance_id="embed-1",
        host="localhost",
        port=8090,
        model_name="BAAI/bge-m3",
        instance_type=ExecutionInstanceType.EMBEDDING,
        embedding_model_loaded="BAAI/bge-m3",
        embedding_max_batch_size=32,
        embedding_active_requests=0,
        current_load=0.2,
    )


@pytest.fixture
def mixed_instance():
    """Create a mixed LLM+Embedding execution instance."""
    return ExecutionInstance(
        instance_id="mixed-1",
        host="localhost",
        port=8001,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        instance_type=ExecutionInstanceType.LLM_EMBEDDING,
        embedding_model_loaded="BAAI/bge-m3",
        embedding_max_batch_size=32,
        embedding_active_requests=5,
        current_load=0.4,
    )


@pytest.fixture
def llm_request():
    """Create a LLM request."""
    return RequestMetadata(
        request_id="llm-req-1",
        prompt="Hello, how are you?",
        request_type=RequestType.LLM_CHAT,
        priority=RequestPriority.NORMAL,
        arrival_time=datetime.now(),
    )


@pytest.fixture
def embedding_request():
    """Create an embedding request."""
    return RequestMetadata(
        request_id="embed-req-1",
        request_type=RequestType.EMBEDDING,
        embedding_texts=["Hello world", "How are you"],
        embedding_model="BAAI/bge-m3",
        priority=RequestPriority.NORMAL,
        arrival_time=datetime.now(),
    )


# ============ Manager Initialization Tests ============


class TestManagerInit:
    """Tests for ControlPlaneManager initialization."""

    def test_default_init(self):
        """Test default initialization."""
        with patch("control_plane.manager.HttpExecutionCoordinator"):
            manager = ControlPlaneManager()

            assert manager.mode == "http"
            assert manager.scheduling_policy is not None
            assert manager.scheduling_policy.name == "Adaptive"

    def test_init_with_hybrid_policy(self):
        """Test initialization with hybrid scheduling policy."""
        with patch("control_plane.manager.HttpExecutionCoordinator"):
            manager = ControlPlaneManager(scheduling_policy="hybrid")

            # Note: hybrid policy may not be directly registered in _create_policy
            # This tests backward compatibility with existing policy names

    def test_init_with_local_mode(self):
        """Test initialization with local execution mode."""
        with patch("control_plane.manager.LocalAsyncExecutionCoordinator"):
            manager = ControlPlaneManager(mode="local")

            assert manager.mode == "local"

    def test_init_with_http_mode(self):
        """Test initialization with HTTP execution mode."""
        with patch("control_plane.manager.HttpExecutionCoordinator"):
            manager = ControlPlaneManager(mode="http")

            assert manager.mode == "http"


# ============ Manager Request Submission Tests ============


class TestManagerSubmitRequest:
    """Tests for request submission."""

    @pytest.mark.asyncio
    async def test_submit_llm_request(self, mock_executor, llm_request):
        """Test submitting LLM request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            request_id = await manager.submit_request(llm_request)

            assert request_id == llm_request.request_id
            assert len(manager.pending_queue) == 1
            assert manager.pending_queue[0].request_type == RequestType.LLM_CHAT

    @pytest.mark.asyncio
    async def test_submit_embedding_request(self, mock_executor, embedding_request):
        """Test submitting embedding request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            request_id = await manager.submit_request(embedding_request)

            assert request_id == embedding_request.request_id
            assert len(manager.pending_queue) == 1
            assert manager.pending_queue[0].request_type == RequestType.EMBEDDING

    @pytest.mark.asyncio
    async def test_submit_multiple_requests(
        self, mock_executor, llm_request, embedding_request
    ):
        """Test submitting multiple mixed requests."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.submit_request(llm_request)
            await manager.submit_request(embedding_request)

            assert len(manager.pending_queue) == 2

            # Check both request types are in queue
            types = [r.request_type for r in manager.pending_queue]
            assert RequestType.LLM_CHAT in types
            assert RequestType.EMBEDDING in types


# ============ Manager Request Status Tests ============


class TestManagerRequestStatus:
    """Tests for request status tracking."""

    @pytest.mark.asyncio
    async def test_get_request_status_queued(self, mock_executor, llm_request):
        """Test getting status of queued request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.submit_request(llm_request)

            status = await manager.get_request_status(llm_request.request_id)
            assert status == RequestStatus.QUEUED

    @pytest.mark.asyncio
    async def test_get_request_status_not_found(self, mock_executor):
        """Test getting status of non-existent request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            status = await manager.get_request_status("non-existent-id")
            assert status is None

    @pytest.mark.asyncio
    async def test_get_request_status_running(self, mock_executor, llm_request):
        """Test getting status of running request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            # Simulate running request
            manager.running_requests[llm_request.request_id] = llm_request

            status = await manager.get_request_status(llm_request.request_id)
            assert status == RequestStatus.RUNNING


# ============ Manager Cancel Request Tests ============


class TestManagerCancelRequest:
    """Tests for request cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_queued_request(self, mock_executor, llm_request):
        """Test cancelling a queued request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.submit_request(llm_request)
            result = await manager.cancel_request(llm_request.request_id)

            assert result is True
            assert len(manager.pending_queue) == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_request(self, mock_executor):
        """Test cancelling a non-existent request."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            result = await manager.cancel_request("non-existent-id")

            assert result is False


# ============ Manager Instance Registration Tests ============


class TestManagerInstanceRegistration:
    """Tests for instance registration."""

    def test_register_llm_instance(self, mock_executor, llm_instance):
        """Test registering LLM instance."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            manager.register_instance(llm_instance)

            mock_executor.register_instance.assert_called_once_with(llm_instance)

    def test_register_embedding_instance(self, mock_executor, embedding_instance):
        """Test registering embedding instance."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            manager.register_instance(embedding_instance)

            mock_executor.register_instance.assert_called_once_with(embedding_instance)

    def test_register_mixed_instance(self, mock_executor, mixed_instance):
        """Test registering mixed LLM+Embedding instance."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            manager.register_instance(mixed_instance)

            mock_executor.register_instance.assert_called_once_with(mixed_instance)

    def test_unregister_instance(self, mock_executor):
        """Test unregistering an instance."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            manager.unregister_instance("instance-1")

            mock_executor.unregister_instance.assert_called_once_with("instance-1")


# ============ Manager Policy Update Tests ============


class TestManagerPolicyUpdate:
    """Tests for scheduling policy updates."""

    def test_update_policy(self, mock_executor):
        """Test updating scheduling policy."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(scheduling_policy="fifo")

            assert manager.scheduling_policy.name == "FIFO"

            manager.update_policy("priority")

            assert manager.scheduling_policy.name == "Priority"

    def test_update_policy_adaptive(self, mock_executor):
        """Test updating to adaptive policy."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(scheduling_policy="fifo")

            manager.update_policy("adaptive")

            assert manager.scheduling_policy.name == "Adaptive"

    def test_update_policy_slo_aware(self, mock_executor):
        """Test updating to SLO-aware policy."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(scheduling_policy="fifo")

            manager.update_policy("slo_aware")

            assert manager.scheduling_policy.name == "SLO-Aware"


# ============ Manager Status Tests ============


class TestManagerStatus:
    """Tests for manager status reporting."""

    def test_get_status(self, mock_executor):
        """Test getting manager status."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            status = manager.get_status()

            assert "running" in status
            assert "scheduling_policy" in status
            assert "pending_requests" in status
            assert "running_requests" in status

    @pytest.mark.asyncio
    async def test_get_status_with_pending_requests(
        self, mock_executor, llm_request, embedding_request
    ):
        """Test status with pending requests."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.submit_request(llm_request)
            await manager.submit_request(embedding_request)

            status = manager.get_status()

            assert status["pending_requests"] == 2


# ============ Manager Metrics Tests ============


class TestManagerMetrics:
    """Tests for metrics collection."""

    def test_get_metrics(self, mock_executor):
        """Test getting performance metrics."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            metrics = manager.get_metrics()

            assert metrics is not None
            assert hasattr(metrics, "active_requests")

    def test_get_scheduling_metrics(self, mock_executor):
        """Test getting scheduling-specific metrics."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            metrics = manager.get_scheduling_metrics()

            assert metrics is not None


# ============ Manager Instance Failure Handling Tests ============


class TestManagerInstanceFailure:
    """Tests for instance failure handling."""

    @pytest.mark.asyncio
    async def test_on_instance_failure_reschedules_requests(
        self, mock_executor, llm_request
    ):
        """Test that instance failure reschedules affected requests."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            # Simulate running request
            failed_requests = [("llm-req-1", llm_request)]

            await manager.on_instance_failure("instance-1", failed_requests)

            # Request should be re-queued
            assert len(manager.pending_queue) == 1
            assert manager.pending_queue[0].request_id == "llm-req-1"
            assert manager.pending_queue[0].tags.get("retry_count") == 1

    @pytest.mark.asyncio
    async def test_on_instance_failure_retry_limit(self, mock_executor, llm_request):
        """Test that retry limit is enforced."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            # Set retry count to max
            llm_request.tags["retry_count"] = 3
            failed_requests = [("llm-req-1", llm_request)]

            await manager.on_instance_failure("instance-1", failed_requests)

            # Request should NOT be re-queued (exceeded retry limit)
            assert len(manager.pending_queue) == 0


# ============ Manager Lifecycle Tests ============


class TestManagerLifecycle:
    """Tests for manager start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start(self, mock_executor):
        """Test starting the manager."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.start()

            assert manager._running is True
            assert len(manager.background_tasks) > 0

            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop(self, mock_executor):
        """Test stopping the manager."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.start()
            await manager.stop()

            assert manager._running is False
            assert len(manager.background_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_idempotent(self, mock_executor):
        """Test starting manager multiple times is idempotent."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            await manager.start()
            task_count = len(manager.background_tasks)

            # Start again should not create more tasks
            await manager.start()
            assert len(manager.background_tasks) == task_count

            await manager.stop()


# ============ Manager Get Instances Tests ============


class TestManagerGetInstances:
    """Tests for getting registered instances."""

    def test_get_instances(self, mock_executor, llm_instance, embedding_instance):
        """Test getting all registered instances."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager()
            manager.executor = mock_executor

            mock_executor.get_all_instances.return_value = [llm_instance, embedding_instance]

            instances = manager.get_instances()

            assert len(instances) == 2
            instance_types = [i.instance_type for i in instances]
            assert ExecutionInstanceType.GENERAL in instance_types
            assert ExecutionInstanceType.EMBEDDING in instance_types


# ============ Manager PD Separation Tests ============


class TestManagerPDSeparation:
    """Tests for Prefilling/Decoding separation support."""

    def test_init_with_pd_separation_enabled(self, mock_executor):
        """Test initialization with PD separation enabled."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(enable_pd_separation=True)

            assert manager.enable_pd_separation is True
            assert manager.pd_router is not None

    def test_init_with_pd_separation_disabled(self, mock_executor):
        """Test initialization with PD separation disabled."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(enable_pd_separation=False)

            assert manager.enable_pd_separation is False
            assert manager.pd_router is None


# ============ Manager Autoscaler Tests ============


class TestManagerAutoscaler:
    """Tests for autoscaler functionality."""

    def test_get_autoscaler_status_disabled(self, mock_executor):
        """Test autoscaler status when disabled."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(enable_auto_scaling=False)

            status = manager.get_autoscaler_status()

            assert status["enabled"] is False

    def test_init_with_autoscaler_enabled(self, mock_executor):
        """Test initialization with autoscaler enabled."""
        with patch("control_plane.manager.HttpExecutionCoordinator", return_value=mock_executor):
            manager = ControlPlaneManager(enable_auto_scaling=True)

            assert manager.autoscaler is not None
            assert manager.autoscaler_metrics_collector is not None

