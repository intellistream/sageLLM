# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Execution coordinator for managing vLLM instances."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from vllm.control_plane.types import (
    ExecutionInstance,
    PerformanceMetrics,
    RequestMetadata,
    SchedulingDecision,
)

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinates execution across multiple vLLM instances."""

    def __init__(self):
        self.instances: dict[str, ExecutionInstance] = {}
        self.active_requests: dict[str, RequestMetadata] = {}
        self.request_to_instance: dict[str, str] = {}  # request_id -> instance_id

        # Monitoring
        self.metrics = PerformanceMetrics()

    def register_instance(self, instance: ExecutionInstance):
        """Register a new vLLM execution instance."""
        self.instances[instance.instance_id] = instance
        logger.info("Registered instance: %s", instance.instance_id)

    def unregister_instance(self, instance_id: str):
        """Unregister an execution instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info("Unregistered instance: %s", instance_id)

    def get_instance(self, instance_id: str) -> Optional[ExecutionInstance]:
        """Get instance by ID."""
        return self.instances.get(instance_id)

    def get_available_instances(self) -> list[ExecutionInstance]:
        """Get all available instances."""
        return [
            instance
            for instance in self.instances.values()
            if instance.can_accept_request
        ]

    def get_all_instances(self) -> list[ExecutionInstance]:
        """Get all registered instances."""
        return list(self.instances.values())

    async def execute_request(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> dict[str, Any]:
        """
        Execute a request on a specific instance.

        Args:
            request: Request metadata
            instance: Target execution instance
            decision: Scheduling decision

        Returns:
            Execution result
        """

        # Mark request as active
        self.active_requests[request.request_id] = request
        self.request_to_instance[request.request_id] = instance.instance_id

        # Update instance state
        instance.active_requests += 1
        instance.current_load = min(
            1.0, instance.active_requests / instance.max_concurrent_requests
        )

        # Update request timing
        request.start_time = datetime.now()

        try:
            # Execute on instance (this would call vLLM API)
            result = await self._execute_on_instance(request, instance, decision)

            # Mark as completed
            request.end_time = datetime.now()

            # Update metrics
            self._update_metrics(request, instance, success=True)

            return result

        except Exception as e:
            logger.error("Execution failed for request %s: %s", request.request_id, e)
            request.end_time = datetime.now()
            self._update_metrics(request, instance, success=False)
            raise

        finally:
            # Clean up
            instance.active_requests -= 1
            instance.current_load = max(
                0.0, instance.active_requests / instance.max_concurrent_requests
            )

            self.active_requests.pop(request.request_id, None)
            self.request_to_instance.pop(request.request_id, None)

    async def _execute_on_instance(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> dict[str, Any]:
        """Execute request on vLLM instance (placeholder)."""

        # This is where you would integrate with actual vLLM API
        # For now, simulate execution

        logger.info(
            "Executing request %s on instance %s with TP=%d, PP=%d",
            request.request_id,
            instance.instance_id,
            decision.tensor_parallel_size,
            decision.pipeline_parallel_size,
        )

        # Simulate execution time
        await asyncio.sleep(0.1)

        return {
            "request_id": request.request_id,
            "instance_id": instance.instance_id,
            "status": "completed",
            "output": "Simulated output",
            "tokens_generated": request.max_tokens or 100,
            "latency_ms": decision.estimated_latency_ms,
        }

    def _update_metrics(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        success: bool,
    ):
        """Update performance metrics."""

        self.metrics.total_requests += 1

        if success:
            self.metrics.completed_requests += 1

            # Update latency
            if request.latency_ms:
                self.metrics.avg_latency_ms = (
                    self.metrics.avg_latency_ms * (self.metrics.completed_requests - 1)
                    + request.latency_ms
                ) / self.metrics.completed_requests
        else:
            self.metrics.failed_requests += 1

        # Check SLO compliance
        if (
            request.slo_deadline_ms
            and request.latency_ms
            and request.latency_ms > request.slo_deadline_ms
        ):
            self.metrics.slo_violations += 1

        # Update SLO compliance rate
        if self.metrics.completed_requests > 0:
            self.metrics.slo_compliance_rate = 1.0 - (
                self.metrics.slo_violations / self.metrics.completed_requests
            )

    async def health_check(self, instance_id: str) -> bool:
        """Perform health check on an instance."""

        instance = self.get_instance(instance_id)
        if not instance:
            return False

        try:
            # Perform health check (placeholder)
            # In real implementation, ping the vLLM instance
            await asyncio.sleep(0.01)

            instance.is_healthy = True
            return True

        except Exception as e:
            logger.error("Health check failed for %s: %s", instance_id, e)
            instance.is_healthy = False
            instance.is_available = False
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """Health check all instances."""

        results = {}
        for instance_id in self.instances:
            results[instance_id] = await self.health_check(instance_id)

        return results

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""

        # Update real-time metrics
        self.metrics.active_requests = len(self.active_requests)
        self.metrics.timestamp = datetime.now()

        # Calculate resource metrics
        if self.instances:
            self.metrics.avg_gpu_utilization = sum(
                i.gpu_utilization for i in self.instances.values()
            ) / len(self.instances)

            self.metrics.total_gpu_memory_gb = sum(
                i.gpu_memory_gb * i.gpu_count for i in self.instances.values()
            )

        return self.metrics

    def get_instance_metrics(self, instance_id: str) -> Optional[dict[str, Any]]:
        """Get metrics for a specific instance."""

        instance = self.get_instance(instance_id)
        if not instance:
            return None

        return {
            "instance_id": instance.instance_id,
            "is_healthy": instance.is_healthy,
            "is_available": instance.is_available,
            "current_load": instance.current_load,
            "active_requests": instance.active_requests,
            "avg_latency_ms": instance.avg_latency_ms,
            "throughput_tokens_per_sec": instance.throughput_tokens_per_sec,
            "gpu_utilization": instance.gpu_utilization,
            "gpu_count": instance.gpu_count,
        }

    async def rebalance_load(self):
        """Rebalance load across instances (future enhancement)."""

        # Placeholder for load rebalancing logic
        # Could migrate requests from overloaded to underutilized instances
        pass

    async def scale_instances(self, target_count: int):
        """Scale number of instances (future enhancement)."""

        # Placeholder for auto-scaling logic
        current_count = len(self.instances)

        if target_count > current_count:
            logger.info(
                "Scaling up from %d to %d instances", current_count, target_count
            )
        elif target_count < current_count:
            logger.info(
                "Scaling down from %d to %d instances", current_count, target_count
            )
