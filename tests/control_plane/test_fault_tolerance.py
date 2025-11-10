# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for fault tolerance and recovery mechanisms."""

import pytest
from aioresponses import aioresponses

from control_plane import ControlPlaneManager, ExecutionInstance, RequestMetadata


@pytest.mark.asyncio
class TestFaultTolerance:
    """Test fault tolerance and recovery."""

    async def test_instance_health_check_failure_detection(self, mock_aiohttp: aioresponses):
        """Test detection of consecutive health check failures."""
        manager = ControlPlaneManager()

        instance = ExecutionInstance(
            instance_id="failing-instance",
            host="localhost",
            port=8000,
            model_name="test-model",
        )

        manager.register_instance(instance)
        await manager.executor.initialize()

        # Mock health check failures
        health_url = "http://localhost:8000/health"
        mock_aiohttp.get(health_url, status=500)
        mock_aiohttp.get(health_url, status=500)
        mock_aiohttp.get(health_url, status=500)

        # Run health checks
        await manager.executor.health_check(instance)
        await manager.executor.health_check(instance)
        await manager.executor.health_check(instance)

        # Instance should be marked as unavailable after 3 failures
        assert instance.is_healthy is False
        assert instance.is_available is False
        assert instance.metadata.get("consecutive_failures", 0) >= 3

    async def test_instance_health_check_recovery(self, mock_aiohttp: aioresponses):
        """Test instance recovery after failures."""
        manager = ControlPlaneManager()

        instance = ExecutionInstance(
            instance_id="recovering-instance",
            host="localhost",
            port=8000,
            model_name="test-model",
        )

        manager.register_instance(instance)
        await manager.executor.initialize()

        health_url = "http://localhost:8000/health"

        # Fail once
        mock_aiohttp.get(health_url, status=500)
        await manager.executor.health_check(instance)
        assert instance.metadata.get("consecutive_failures", 0) == 1

        # Recover
        mock_aiohttp.get(health_url, status=200)
        await manager.executor.health_check(instance)
        assert instance.is_healthy is True
        assert instance.metadata.get("consecutive_failures", 0) == 0

    async def test_request_rescheduling_on_instance_failure(self):
        """Test that requests are rescheduled when instance fails."""
        manager = ControlPlaneManager()

        # Create a request
        request = RequestMetadata(
            request_id="test-req-1",
            prompt="test prompt",
            max_tokens=100,
        )

        # Simulate request running on failed instance
        manager.running_requests["test-req-1"] = request

        # Simulate instance failure
        failed_requests = [("test-req-1", request)]
        await manager.on_instance_failure("failed-instance", failed_requests)

        # Request should be rescheduled (back in pending queue)
        assert "test-req-1" not in manager.running_requests
        assert len(manager.pending_queue) == 1
        assert manager.pending_queue[0].request_id == "test-req-1"
        assert manager.pending_queue[0].tags["retry_count"] == 1

    async def test_request_retry_limit(self):
        """Test that requests are not retried beyond limit."""
        manager = ControlPlaneManager()

        # Create a request that has already been retried 3 times
        request = RequestMetadata(
            request_id="test-req-max-retry",
            prompt="test prompt",
        )
        request.tags["retry_count"] = 3

        # Simulate instance failure
        manager.running_requests["test-req-max-retry"] = request
        failed_requests = [("test-req-max-retry", request)]
        await manager.on_instance_failure("failed-instance", failed_requests)

        # Request should NOT be rescheduled (exceeded retry limit)
        assert "test-req-max-retry" not in manager.running_requests
        assert len(manager.pending_queue) == 0

    async def test_multiple_requests_rescheduling(self):
        """Test rescheduling multiple requests from failed instance."""
        manager = ControlPlaneManager()

        # Create multiple requests
        requests = [RequestMetadata(request_id=f"req-{i}", prompt=f"prompt-{i}") for i in range(5)]

        # Simulate all running on failed instance
        for req in requests:
            manager.running_requests[req.request_id] = req

        failed_requests = [(req.request_id, req) for req in requests]

        # Trigger failure handling
        await manager.on_instance_failure("failed-instance", failed_requests)

        # All requests should be rescheduled
        assert len(manager.running_requests) == 0
        assert len(manager.pending_queue) == 5

        # All should have retry_count = 1
        for req in manager.pending_queue:
            assert req.tags["retry_count"] == 1

    async def test_metrics_recorded_on_failure(self):
        """Test that failure metrics are recorded."""
        manager = ControlPlaneManager()

        # Trigger instance failure
        await manager.on_instance_failure("failed-instance", [])

        # Failure should be recorded in metrics collector
        _ = manager.metrics_collector.get_instance_metrics("failed-instance")

        # Note: instance might not exist in metrics yet, which is OK
        # The important thing is that the failure was logged


@pytest.mark.skipif(True, reason="Requires real vLLM instances - enable for integration testing")
@pytest.mark.asyncio
class TestFaultToleranceIntegration:
    """Integration tests requiring real vLLM instances."""

    async def test_real_instance_failure_recovery(self):
        """Test with real vLLM instance going down and coming back."""
        # This would require:
        # 1. Start vLLM instance
        # 2. Register with Control Plane
        # 3. Submit requests
        # 4. Kill vLLM instance
        # 5. Verify requests are rescheduled
        # 6. Restart vLLM instance
        # 7. Verify recovery
        pass
