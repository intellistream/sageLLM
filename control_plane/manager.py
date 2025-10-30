# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Control Plane Manager - orchestrates scheduling, routing, and execution."""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Optional

from .autoscaler import Autoscaler, AutoscalerConfig
from .executor import ExecutionCoordinator
from .metrics_collector import MetricsCollector as AutoscalerMetricsCollector
from .monitoring import MetricsCollector
from .parallelism import ParallelismOptimizer
from .pd_routing import PDRoutingStrategy
from .strategies import (
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
    PDSeparationConfig,
    PerformanceMetrics,
    RequestMetadata,
    RequestStatus,
    SchedulingDecision,
    SchedulingMetrics,
    InstanceMetrics,
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
        enable_pd_separation: bool = True,
        pd_config: PDSeparationConfig | None = None,
        autoscaler_config: AutoscalerConfig | None = None,
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
            enable_pd_separation: Enable Prefilling/Decoding separation
            pd_config: PD separation configuration
            autoscaler_config: Autoscaler configuration
        """

        # Core components
        self.executor = ExecutionCoordinator()
        self.router = RequestRouter(routing_strategy)
        self.load_balancer = LoadBalancer()
        self.parallelism_optimizer = ParallelismOptimizer()

        # Set executor callback for failure handling
        self.executor.set_manager_callback(self)

        # PD Separation routing
        self.enable_pd_separation = enable_pd_separation
        self.pd_config = pd_config or PDSeparationConfig()
        self.pd_router: Optional[PDRoutingStrategy] = (
            PDRoutingStrategy(self.pd_config) if enable_pd_separation else None
        )

        # Scheduling policy
        self.scheduling_policy = self._create_policy(scheduling_policy)

        # Request queues
        self.pending_queue: deque[RequestMetadata] = deque()
        self.running_requests: dict[str, RequestMetadata] = {}

        # Configuration
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_monitoring = enable_monitoring

        # Monitoring (from original code for scheduling metrics)
        self.metrics_collector = MetricsCollector()

        # Autoscaler (SLA-based dynamic scaling)
        self.autoscaler: Optional[Autoscaler] = None
        self.autoscaler_metrics_collector: Optional[AutoscalerMetricsCollector] = None
        if enable_auto_scaling:
            self.autoscaler_metrics_collector = AutoscalerMetricsCollector(
                executor_coordinator=self.executor
            )
            self.autoscaler = Autoscaler(
                config=autoscaler_config or AutoscalerConfig(),
                metrics_collector=self.autoscaler_metrics_collector,
                executor_callback=self._handle_scaling_decision,
            )
            logger.info("Autoscaler enabled")

        # Background tasks
        self.background_tasks: list[asyncio.Task] = []
        self._running = False

        logger.info(
            "Control Plane initialized with policy=%s, routing=%s, pd_separation=%s, autoscaling=%s",
            scheduling_policy,
            routing_strategy,
            enable_pd_separation,
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
        """Start the control plane background tasks."""
        if self._running:
            logger.warning("Control plane already running")
            return

        self._running = True
        logger.info("Starting Control Plane...")

        # Start scheduling loop
        task = asyncio.create_task(self._scheduling_loop())
        self.background_tasks.append(task)

        # Start health check loop
        task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.append(task)

        # Start autoscaler if enabled
        if self.autoscaler:
            await self.autoscaler.start()
            logger.info("Autoscaler started")

        # Start monitoring loop if enabled
        if self.enable_monitoring:
            task = asyncio.create_task(self._monitoring_loop())
            self.background_tasks.append(task)

        logger.info("Control Plane started successfully")

    async def stop(self):
        """Stop the control plane."""
        if not self._running:
            return

        logger.info("Stopping Control Plane...")
        self._running = False

        # Stop autoscaler
        if self.autoscaler:
            await self.autoscaler.stop()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

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

    async def get_request_status(self, request_id: str) -> RequestStatus | None:
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
        for _i, req in enumerate(self.pending_queue):
            if req.request_id == request_id:
                self.pending_queue.remove(req)
                logger.info("Request %s cancelled", request_id)
                return True

        logger.warning("Request %s not found or already running", request_id)
        return False

    async def on_instance_failure(
        self,
        instance_id: str,
        failed_requests: list[tuple[str, RequestMetadata]],
    ):
        """
        Handle instance failure by rescheduling affected requests.

        Args:
            instance_id: ID of the failed instance
            failed_requests: List of (request_id, request) tuples that were running
        """
        logger.warning(
            "Handling instance failure for %s, rescheduling %d requests",
            instance_id,
            len(failed_requests),
        )

        # Record failure in metrics
        self.metrics_collector.record_instance_failure(instance_id)

        # Reschedule each failed request
        for req_id, request in failed_requests:
            # Track retry count
            request.tags["retry_count"] = request.tags.get("retry_count", 0) + 1

            # Check retry limit (max 3 retries)
            if request.tags["retry_count"] > 3:
                logger.error(
                    "Request %s exceeded retry limit, marking as failed", req_id
                )
                self.running_requests.pop(req_id, None)
                continue

            # Re-add to pending queue (at the front for priority)
            request.queue_time = datetime.now()
            self.pending_queue.appendleft(request)

            # Remove from running requests
            self.running_requests.pop(req_id, None)

            logger.info(
                "Request %s rescheduled (retry: %d/%d)",
                req_id,
                request.tags["retry_count"],
                3,
            )

        logger.info(
            "Instance %s failure handled, %d requests rescheduled",
            instance_id,
            len(failed_requests),
        )

    async def _scheduling_loop(self):
        """Main scheduling loop."""

        while self._running:
            try:
                # Record queue length for metrics
                self.metrics_collector.record_queue_length(len(self.pending_queue))

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
        for decision, request in zip(decisions, requests_to_schedule, strict=False):
            await self._execute_scheduled_request(request, decision)

    async def _execute_scheduled_request(
        self,
        request: RequestMetadata,
        decision: SchedulingDecision,
    ):
        """Execute a scheduled request with PD-aware routing if enabled."""

        # Get target instance (may be adjusted by PD router)
        instance = self.executor.get_instance(decision.target_instance_id)
        if not instance or not instance.can_accept_request:
            # Re-queue if instance not available
            self.pending_queue.appendleft(request)
            return

        # PD-aware routing: determine request phase and route to appropriate instance
        if self.enable_pd_separation and self.pd_router:
            pd_router = self.pd_router  # Local reference for type checker
            target_phase = pd_router.determine_request_phase(request)

            if target_phase:  # Only proceed if phase was determined
                logger.debug(
                    "PD routing: request %s routed to phase %s",
                    request.request_id,
                    target_phase.name,
                )

                # Filter instances by target phase
                compatible_instances = pd_router.filter_instances_by_type(
                    self.executor.get_all_instances(),
                    target_phase,
                )

                if compatible_instances:
                    # Select best instance based on PD specialization
                    best_instance = max(
                        compatible_instances,
                        key=lambda inst: pd_router.get_instance_specialization(inst).get(target_phase.name, 0),
                    )
                    instance = best_instance
                    logger.debug(
                        "Selected PD-specialized instance %s for request %s",
                        instance.instance_id,
                        request.request_id,
                    )

        # Recommend parallelism config based on instance type
        if self.enable_pd_separation and self.pd_router:
            parallelism_config = self.pd_router.recommend_parallelism_config(instance, request)
            if parallelism_config:
                decision.tensor_parallel_size = parallelism_config['tensor_parallel_size']
                decision.pipeline_parallel_size = (
                    parallelism_config['pipeline_parallel_size']
                )

        # Otherwise optimize parallelism strategy
        else:
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
            "Scheduling request %s to instance %s (TP=%d, PP=%d, type=%s)",
            request.request_id,
            instance.instance_id,
            decision.tensor_parallel_size,
            decision.pipeline_parallel_size,
            instance.instance_type.name
            if hasattr(instance, "instance_type")
            else "unknown",
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

            # Record metrics
            if request.latency_ms:
                self.metrics_collector.record_request_completion(
                    request, decision, request.latency_ms, success=True
                )

        except Exception as e:
            logger.error("Request %s failed: %s", request.request_id, e)

            # Record failure
            if request.latency_ms:
                self.metrics_collector.record_request_completion(
                    request, decision, request.latency_ms, success=False
                )

        finally:
            # Cleanup
            self.running_requests.pop(request.request_id, None)

            # Update instance metrics
            self.metrics_collector.update_instance_metrics(instance)

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

    async def _handle_scaling_decision(self, decision):
        """
        Handle autoscaling decision from Autoscaler.

        This callback is invoked by the Autoscaler when it determines
        that instances need to be scaled up or down.

        Args:
            decision: ScalingDecision with target instance counts
        """
        from .types import ExecutionInstanceType

        # Get current instance counts
        instances = self.executor.get_all_instances()
        current_prefill = sum(
            1 for i in instances if i.instance_type == ExecutionInstanceType.PREFILLING
        )
        current_decode = sum(
            1 for i in instances if i.instance_type == ExecutionInstanceType.DECODING
        )

        # Calculate deltas
        prefill_delta = decision.num_prefill_instances - current_prefill
        decode_delta = decision.num_decode_instances - current_decode

        logger.info(
            f"Autoscaling decision: prefill {current_prefill} -> "
            f"{decision.num_prefill_instances} (delta={prefill_delta}), "
            f"decode {current_decode} -> {decision.num_decode_instances} "
            f"(delta={decode_delta})"
        )

        # Apply scaling changes
        try:
            # Scale prefill instances
            if prefill_delta > 0:
                await self._scale_up_prefill(prefill_delta)
            elif prefill_delta < 0:
                await self._scale_down_prefill(abs(prefill_delta))

            # Scale decode instances
            if decode_delta > 0:
                await self._scale_up_decode(decode_delta)
            elif decode_delta < 0:
                await self._scale_down_decode(abs(decode_delta))

            logger.info("Autoscaling completed successfully")

        except Exception as e:
            logger.error(f"Failed to apply autoscaling decision: {e}", exc_info=True)

    async def _scale_up_prefill(self, count: int):
        """
        Scale up prefill instances.

        Args:
            count: Number of instances to add

        Note: This is a placeholder. Actual implementation depends on
        deployment environment (Kubernetes, cloud APIs, etc.)
        """
        logger.info(f"Scaling up {count} prefill instances")
        # TODO: Implement instance creation
        # - For Kubernetes: create new pods
        # - For cloud: launch new VMs/containers
        # - For local: start new processes
        #
        # Example Kubernetes approach:
        # await kubernetes_client.scale_deployment(
        #     deployment_name="prefill-worker",
        #     replicas=current + count
        # )

    async def _scale_down_prefill(self, count: int):
        """
        Scale down prefill instances.

        Args:
            count: Number of instances to remove

        Note: Should gracefully drain instances before removal.
        """
        logger.info(f"Scaling down {count} prefill instances")
        # TODO: Implement graceful instance removal
        # 1. Select instances to remove (prefer least loaded)
        # 2. Mark as unavailable (stop accepting new requests)
        # 3. Wait for active requests to complete
        # 4. Shutdown and remove
        from .types import ExecutionInstanceType

        instances = [
            i
            for i in self.executor.get_all_instances()
            if i.instance_type == ExecutionInstanceType.PREFILLING
        ]

        # Sort by load (ascending) to remove least loaded first
        instances.sort(key=lambda x: x.active_requests)

        for i in range(min(count, len(instances))):
            instance = instances[i]
            logger.info(f"Removing prefill instance {instance.instance_id}")
            await self.executor.remove_instance_gracefully(instance.instance_id)

    async def _scale_up_decode(self, count: int):
        """
        Scale up decode instances.

        Args:
            count: Number of instances to add
        """
        logger.info(f"Scaling up {count} decode instances")
        # TODO: Implement instance creation (similar to prefill)

    async def _scale_down_decode(self, count: int):
        """
        Scale down decode instances.

        Args:
            count: Number of instances to remove
        """
        logger.info(f"Scaling down {count} decode instances")
        from .types import ExecutionInstanceType

        instances = [
            i
            for i in self.executor.get_all_instances()
            if i.instance_type == ExecutionInstanceType.DECODING
        ]

        instances.sort(key=lambda x: x.active_requests)

        for i in range(min(count, len(instances))):
            instance = instances[i]
            logger.info(f"Removing decode instance {instance.instance_id}")
            await self.executor.remove_instance_gracefully(instance.instance_id)

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # TODO: Implement metrics aggregation
        return PerformanceMetrics()

    def get_autoscaler_status(self) -> dict:
        """Get autoscaler status."""
        if not self.autoscaler:
            return {"enabled": False}

        return {
            "enabled": True,
            **self.autoscaler.get_status(),
        }

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        metrics = self.executor.get_metrics()
        metrics.queued_requests = len(self.pending_queue)
        return metrics

    def get_scheduling_metrics(self) -> "SchedulingMetrics":
        """
        Get scheduling-specific metrics.

        Returns:
            SchedulingMetrics with policy performance data
        """
        return self.metrics_collector.get_scheduling_metrics(
            self.scheduling_policy.name
        )

    def get_instance_metrics_detailed(self, instance_id: str) -> "InstanceMetrics | None":
        """
        Get detailed metrics for a specific instance.

        Args:
            instance_id: Instance identifier

        Returns:
            InstanceMetrics or None if not found
        """
        return self.metrics_collector.get_instance_metrics(instance_id)

    def get_all_instance_metrics(self) -> dict[str, "InstanceMetrics"]:
        """
        Get detailed metrics for all instances.

        Returns:
            Dictionary mapping instance_id to InstanceMetrics
        """
        return self.metrics_collector.get_all_instance_metrics()

    def get_instances(self) -> list[ExecutionInstance]:
        """Get all registered instances."""
        return self.executor.get_all_instances()

    def get_instance_metrics(self, instance_id: str) -> dict[str, Any] | None:
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
