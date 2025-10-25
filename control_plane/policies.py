# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Scheduling policies for the Control Plane."""

from abc import ABC, abstractmethod
from datetime import datetime

from .types import (ExecutionInstance, ParallelismType, RequestMetadata,
                    RequestPriority, SchedulingDecision)


class SchedulingPolicy(ABC):
    """Base class for scheduling policies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """
        Schedule requests to instances.

        Args:
            requests: List of pending requests to schedule
            instances: List of available execution instances

        Returns:
            List of scheduling decisions
        """
        pass

    @abstractmethod
    def prioritize(self, requests: list[RequestMetadata]) -> list[RequestMetadata]:
        """
        Prioritize requests for scheduling.

        Args:
            requests: List of requests to prioritize

        Returns:
            Sorted list of requests by priority
        """
        pass


class FIFOPolicy(SchedulingPolicy):
    """First-In-First-Out scheduling policy."""

    def __init__(self):
        super().__init__("FIFO")

    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """Schedule requests in FIFO order to least loaded instances."""
        decisions = []

        # Sort requests by arrival time
        sorted_requests = sorted(requests, key=lambda r: r.arrival_time)

        for request in sorted_requests:
            # Find least loaded available instance
            available_instances = [i for i in instances if i.can_accept_request]
            if not available_instances:
                continue

            target = min(available_instances, key=lambda i: i.current_load)

            decision = SchedulingDecision(
                request_id=request.request_id,
                target_instance_id=target.instance_id,
                parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=target.tensor_parallel_size,
                pipeline_parallel_size=target.pipeline_parallel_size,
                estimated_latency_ms=target.avg_latency_ms,
                estimated_cost=0.0,
                reason="FIFO scheduling to least loaded instance",
            )
            decisions.append(decision)

        return decisions

    def prioritize(self, requests: list[RequestMetadata]) -> list[RequestMetadata]:
        """Prioritize by arrival time (FIFO)."""
        return sorted(requests, key=lambda r: r.arrival_time)


class PriorityPolicy(SchedulingPolicy):
    """Priority-based scheduling policy."""

    def __init__(self):
        super().__init__("Priority")

    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """Schedule high-priority requests first."""
        decisions = []

        # Sort by priority then arrival time
        sorted_requests = self.prioritize(requests)

        for request in sorted_requests:
            available_instances = [i for i in instances if i.can_accept_request]
            if not available_instances:
                continue

            # For high priority, prefer instances with better performance
            if request.priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
                target = min(available_instances, key=lambda i: i.avg_latency_ms)
            else:
                target = min(available_instances, key=lambda i: i.current_load)

            decision = SchedulingDecision(
                request_id=request.request_id,
                target_instance_id=target.instance_id,
                parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=target.tensor_parallel_size,
                pipeline_parallel_size=target.pipeline_parallel_size,
                estimated_latency_ms=target.avg_latency_ms,
                estimated_cost=0.0,
                reason=f"Priority {request.priority.name} scheduling",
            )
            decisions.append(decision)

        return decisions

    def prioritize(self, requests: list[RequestMetadata]) -> list[RequestMetadata]:
        """Prioritize by priority level then arrival time."""
        return sorted(requests, key=lambda r: (r.priority.value, r.arrival_time))


class SLOAwarePolicy(SchedulingPolicy):
    """SLO-aware scheduling policy with deadline consideration."""

    def __init__(self):
        super().__init__("SLO-Aware")

    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """Schedule requests considering SLO deadlines."""
        decisions = []

        sorted_requests = self.prioritize(requests)

        for request in sorted_requests:
            available_instances = [i for i in instances if i.can_accept_request]
            if not available_instances:
                continue

            # Calculate urgency based on deadline
            deadline_urgency = self._calculate_urgency(request)

            # For urgent requests, prioritize fast instances
            if deadline_urgency > 0.7:
                target = min(available_instances, key=lambda i: i.avg_latency_ms)
            else:
                target = min(available_instances, key=lambda i: i.current_load)

            decision = SchedulingDecision(
                request_id=request.request_id,
                target_instance_id=target.instance_id,
                parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=target.tensor_parallel_size,
                pipeline_parallel_size=target.pipeline_parallel_size,
                estimated_latency_ms=target.avg_latency_ms,
                estimated_cost=0.0,
                reason=f"SLO-aware scheduling (urgency: {deadline_urgency:.2f})",
                confidence=1.0 - deadline_urgency,
            )
            decisions.append(decision)

        return decisions

    def prioritize(self, requests: list[RequestMetadata]) -> list[RequestMetadata]:
        """Prioritize by deadline urgency."""
        return sorted(requests, key=lambda r: self._calculate_urgency(r), reverse=True)

    def _calculate_urgency(self, request: RequestMetadata) -> float:
        """Calculate urgency score (0.0 to 1.0)."""
        if not request.slo_deadline_ms:
            return 0.0

        elapsed_ms = (datetime.now() - request.arrival_time).total_seconds() * 1000
        remaining_ms = request.slo_deadline_ms - elapsed_ms

        if remaining_ms <= 0:
            return 1.0  # Already missed deadline

        # Urgency increases as deadline approaches
        return min(1.0, elapsed_ms / request.slo_deadline_ms)


class CostOptimizedPolicy(SchedulingPolicy):
    """Cost-optimized scheduling policy."""

    def __init__(self, price_per_gpu_hour: float = 1.0):
        super().__init__("CostOptimized")
        self.price_per_gpu_hour = price_per_gpu_hour

    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """Schedule to minimize cost while meeting requirements."""
        decisions = []

        sorted_requests = self.prioritize(requests)

        for request in sorted_requests:
            available_instances = [i for i in instances if i.can_accept_request]
            if not available_instances:
                continue

            # Calculate cost for each instance
            instance_costs = []
            for instance in available_instances:
                cost = self._estimate_cost(request, instance)
                instance_costs.append((cost, instance))

            # Choose lowest cost instance that can meet SLO
            instance_costs.sort(key=lambda x: x[0])

            target_cost, target = instance_costs[0]

            decision = SchedulingDecision(
                request_id=request.request_id,
                target_instance_id=target.instance_id,
                parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=target.tensor_parallel_size,
                pipeline_parallel_size=target.pipeline_parallel_size,
                estimated_latency_ms=target.avg_latency_ms,
                estimated_cost=target_cost,
                reason=f"Cost-optimized scheduling (cost: ${target_cost:.4f})",
            )
            decisions.append(decision)

        return decisions

    def prioritize(self, requests: list[RequestMetadata]) -> list[RequestMetadata]:
        """Prioritize by cost budget and priority."""
        return sorted(
            requests, key=lambda r: (r.priority.value, r.cost_budget or float("inf"))
        )

    def _estimate_cost(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
    ) -> float:
        """Estimate cost for running request on instance."""
        # Estimate tokens (use max_tokens or default)
        estimated_tokens = request.max_tokens or 100

        # Estimate time in seconds
        if instance.throughput_tokens_per_sec > 0:
            estimated_time_sec = estimated_tokens / instance.throughput_tokens_per_sec
        else:
            estimated_time_sec = 1.0

        # Calculate cost based on GPU usage
        gpu_cost_per_sec = self.price_per_gpu_hour * instance.gpu_count / 3600

        return gpu_cost_per_sec * estimated_time_sec


class AdaptivePolicy(SchedulingPolicy):
    """Adaptive policy that switches between strategies based on conditions."""

    def __init__(self):
        super().__init__("Adaptive")
        self.slo_policy = SLOAwarePolicy()
        self.cost_policy = CostOptimizedPolicy()
        self.priority_policy = PriorityPolicy()

        # Thresholds
        self.high_load_threshold = 0.8
        self.slo_violation_threshold = 0.1

    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """Adaptively choose scheduling strategy."""

        # Calculate system metrics
        avg_load = (
            sum(i.current_load for i in instances) / len(instances) if instances else 0
        )
        has_slo_requests = any(r.slo_deadline_ms for r in requests)
        has_high_priority = any(
            r.priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]
            for r in requests
        )

        # Choose policy based on conditions
        if has_high_priority:
            policy = self.priority_policy
            reason = "high priority requests detected"
        elif has_slo_requests and avg_load > self.high_load_threshold:
            policy = self.slo_policy
            reason = "SLO requests under high load"
        elif avg_load < 0.3:
            policy = self.cost_policy
            reason = "low load, optimizing for cost"
        else:
            policy = self.slo_policy
            reason = "default SLO-aware scheduling"

        decisions = policy.schedule(requests, instances)

        # Add adaptive metadata
        for decision in decisions:
            decision.metadata["adaptive_policy"] = policy.name
            decision.metadata["adaptive_reason"] = reason

        return decisions

    def prioritize(self, requests: list[RequestMetadata]) -> list[RequestMetadata]:
        """Use priority policy for prioritization."""
        return self.priority_policy.prioritize(requests)
