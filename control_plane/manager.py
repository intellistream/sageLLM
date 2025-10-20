# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Main Control Plane Manager."""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Optional

from .executor import ExecutionCoordinator
from .parallelism import ParallelismOptimizer
from .policies import (
    AdaptivePolicy,
    CostOptimizedPolicy,
    FIFOPolicy,
    PriorityPolicy,
    SchedulingPolicy,
    SLOAwarePolicy,
)
from .router import LoadBalancer, RequestRouter
from .types import (
    ExecutionInstance,
    PerformanceMetrics,
    RequestMetadata,
    RequestStatus,
    SchedulingDecision,
)

logger = logging.getLogger(__name__)


class ControlPlaneManager:
    """
    Main Control Plane Manager for sageLLM.

    Coordinates request scheduling, routing, and execution across multiple
    vLLM instances with intelligent parallelism strategies.
    """

    def __init__(
        self,
        scheduling_policy: str = "adaptive",
        routing_strategy: str = "load_balanced",
        enable_auto_scaling: bool = False,
        enable_monitoring: bool = True,
    ):
        """
        Initialize Control Plane Manager.

        Args:
            scheduling_policy: Scheduling policy to use
                - "fifo": First-In-First-Out
                - "priority": Priority-based
                - "slo_aware": SLO-aware scheduling
                - "cost_optimized": Cost-optimized
                - "adaptive": Adaptive policy selection
            routing_strategy: Request routing strategy
            enable_auto_scaling: Enable automatic instance scaling
            enable_monitoring: Enable performance monitoring
        """

        # Core components
        self.executor = ExecutionCoordinator()
        self.router = RequestRouter(routing_strategy)
        self.load_balancer = LoadBalancer()
        self.parallelism_optimizer = ParallelismOptimizer()

        # Scheduling policy
        self.scheduling_policy = self._create_policy(scheduling_policy)

        # Request queues
        self.pending_queue: deque[RequestMetadata] = deque()
        self.running_requests: dict[str, RequestMetadata] = {}

        # Configuration
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_monitoring = enable_monitoring

        # Background tasks
        self.background_tasks: list[asyncio.Task] = []
        self._running = False

        logger.info(
            "Control Plane initialized with policy=%s, routing=%s",
            scheduling_policy,
            routing_strategy,
        )

    def _create_policy(self, policy_name: str) -> SchedulingPolicy:
        """Create scheduling policy instance."""

        policies = {
            "fifo": FIFOPolicy(),
            "priority": PriorityPolicy(),
            "slo_aware": SLOAwarePolicy(),
            "cost_optimized": CostOptimizedPolicy(),
            "adaptive": AdaptivePolicy(),
        }

        return policies.get(policy_name, AdaptivePolicy())

    async def start(self):
        """Start the Control Plane background tasks."""

        if self._running:
            logger.warning("Control Plane already running")
            return

        self._running = True

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._scheduling_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

        if self.enable_monitoring:
            self.background_tasks.append(asyncio.create_task(self._monitoring_loop()))

        logger.info("Control Plane started")

    async def stop(self):
        """Stop the Control Plane."""

        self._running = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        logger.info("Control Plane stopped")

    def register_instance(self, instance: ExecutionInstance):
        """Register a new vLLM execution instance."""
        self.executor.register_instance(instance)

    def unregister_instance(self, instance_id: str):
        """Unregister an execution instance."""
        self.executor.unregister_instance(instance_id)

    async def submit_request(
        self,
        request: RequestMetadata,
    ) -> str:
        """
        Submit a new inference request.

        Args:
            request: Request metadata

        Returns:
            Request ID
        """

        # Add to pending queue
        request.queue_time = datetime.now()
        self.pending_queue.append(request)

        logger.info(
            "Request %s submitted (priority=%s, queue_size=%d)",
            request.request_id,
            request.priority.name,
            len(self.pending_queue),
        )

        return request.request_id

    async def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a request."""

        # Check if running
        if request_id in self.running_requests:
            return RequestStatus.RUNNING

        # Check if in queue
        for req in self.pending_queue:
            if req.request_id == request_id:
                return RequestStatus.QUEUED

        return None

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""

        # Remove from pending queue
        for i, req in enumerate(self.pending_queue):
            if req.request_id == request_id:
                self.pending_queue.remove(req)
                logger.info("Request %s cancelled", request_id)
                return True

        logger.warning("Request %s not found or already running", request_id)
        return False

    async def _scheduling_loop(self):
        """Main scheduling loop."""

        while self._running:
            try:
                await self._schedule_pending_requests()
                await asyncio.sleep(0.1)  # Schedule every 100ms
            except Exception as e:
                logger.error("Scheduling loop error: %s", e)

    async def _schedule_pending_requests(self):
        """Schedule pending requests to instances."""

        if not self.pending_queue:
            return

        # Get available instances
        instances = self.executor.get_available_instances()
        if not instances:
            return

        # Get requests to schedule (up to available capacity)
        requests_to_schedule = []
        max_schedule = sum(
            max(0, i.max_concurrent_requests - i.active_requests) for i in instances
        )

        for _ in range(min(len(self.pending_queue), max_schedule)):
            if self.pending_queue:
                requests_to_schedule.append(self.pending_queue.popleft())

        if not requests_to_schedule:
            return

        # Apply scheduling policy
        decisions = self.scheduling_policy.schedule(
            requests_to_schedule,
            instances,
        )

        # Execute scheduled requests
        for decision, request in zip(decisions, requests_to_schedule):
            await self._execute_scheduled_request(request, decision)

    async def _execute_scheduled_request(
        self,
        request: RequestMetadata,
        decision: SchedulingDecision,
    ):
        """Execute a scheduled request."""

        # Get target instance
        instance = self.executor.get_instance(decision.target_instance_id)
        if not instance or not instance.can_accept_request:
            # Re-queue if instance not available
            self.pending_queue.appendleft(request)
            return

        # Optimize parallelism strategy
        strategy, config = self.parallelism_optimizer.select_strategy(
            request,
            instance,
            instance.gpu_count,
        )

        # Update decision with optimized config
        decision.tensor_parallel_size = config.tensor_parallel_size
        decision.pipeline_parallel_size = config.pipeline_parallel_size
        decision.parallelism_strategy = strategy.strategy_type

        # Mark as running
        request.schedule_time = datetime.now()
        self.running_requests[request.request_id] = request

        logger.info(
            "Scheduling request %s to instance %s with %s (TP=%d, PP=%d)",
            request.request_id,
            instance.instance_id,
            strategy.name,
            config.tensor_parallel_size,
            config.pipeline_parallel_size,
        )

        # Execute asynchronously
        asyncio.create_task(self._execute_and_cleanup(request, instance, decision))

    async def _execute_and_cleanup(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ):
        """Execute request and cleanup."""

        try:
            await self.executor.execute_request(request, instance, decision)

            logger.info(
                "Request %s completed (latency=%.2fms)",
                request.request_id,
                request.latency_ms,
            )

        except Exception as e:
            logger.error("Request %s failed: %s", request.request_id, e)

        finally:
            # Cleanup
            self.running_requests.pop(request.request_id, None)

    async def _health_check_loop(self):
        """Periodic health check of instances."""

        while self._running:
            try:
                await self.executor.health_check_all()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error("Health check loop error: %s", e)

    async def _monitoring_loop(self):
        """Periodic monitoring and metrics collection."""

        while self._running:
            try:
                metrics = self.executor.get_metrics()

                # Log metrics
                logger.info(
                    "Metrics: active=%d, queued=%d, completed=%d, failed=%d, "
                    "avg_latency=%.2fms, slo_compliance=%.2f%%",
                    metrics.active_requests,
                    len(self.pending_queue),
                    metrics.completed_requests,
                    metrics.failed_requests,
                    metrics.avg_latency_ms,
                    metrics.slo_compliance_rate * 100,
                )

                await asyncio.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logger.error("Monitoring loop error: %s", e)

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        metrics = self.executor.get_metrics()
        metrics.queued_requests = len(self.pending_queue)
        return metrics

    def get_instances(self) -> list[ExecutionInstance]:
        """Get all registered instances."""
        return self.executor.get_all_instances()

    def get_instance_metrics(self, instance_id: str) -> Optional[dict[str, Any]]:
        """Get metrics for a specific instance."""
        return self.executor.get_instance_metrics(instance_id)

    def update_policy(self, policy_name: str):
        """Update scheduling policy."""
        self.scheduling_policy = self._create_policy(policy_name)
        logger.info("Scheduling policy updated to: %s", policy_name)

    def get_status(self) -> dict[str, Any]:
        """Get Control Plane status."""

        return {
            "running": self._running,
            "scheduling_policy": self.scheduling_policy.name,
            "routing_strategy": self.router.routing_strategy,
            "pending_requests": len(self.pending_queue),
            "running_requests": len(self.running_requests),
            "registered_instances": len(self.executor.instances),
            "available_instances": len(self.executor.get_available_instances()),
            "metrics": self.get_metrics(),
        }
