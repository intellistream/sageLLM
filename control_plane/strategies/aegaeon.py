# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Aegaeon scheduling policy for multi-model LLM serving.

Based on: Aegaeon: Effective GPU Pooling for Concurrent LLM Serving (SOSP'25)
Core algorithms:
- Algorithm 1: Grouped prefill-phase scheduling
- Algorithm 2: Batched decoding-phase scheduling
"""

from collections import defaultdict
from dataclasses import dataclass, field

from .base import SchedulingPolicy
from ..types import (
    ExecutionInstance,
    ExecutionInstanceType,
    ParallelismType,
    RequestMetadata,
    SchedulingDecision,
)


@dataclass
class PrefillGroup:
    """Group of requests for the same model in prefill phase."""

    model_name: str
    requests: list[RequestMetadata] = field(default_factory=list)
    max_size: int = 8  # MAX_GPSIZE from paper

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_size

    def can_add(self, request: RequestMetadata) -> bool:
        return not self.is_full and request.model_name == self.model_name


@dataclass
class DecodingBatch:
    """Batch of decoding requests for the same model."""

    model_name: str
    requests: list[RequestMetadata] = field(default_factory=list)
    time_quota: float = 0.0  # q_i in seconds
    decoding_step_time: float = 0.025  # t_i estimated (25ms)

    @property
    def n_value(self) -> float:
        """Calculate n_i = d / t_i."""
        tbt_deadline = self._get_tbt_deadline()
        return tbt_deadline / self.decoding_step_time if self.decoding_step_time > 0 else 1.0

    def _get_tbt_deadline(self) -> float:
        """Get TBT deadline in seconds."""
        if self.requests:
            # Use slo_deadline_ms as TBT deadline
            return (self.requests[0].slo_deadline_ms or 100.0) / 1000.0
        return 0.1  # Default 100ms


class AegaeonPolicy(SchedulingPolicy):
    """Aegaeon scheduling policy.

    Key features:
    - Prefill: Grouped scheduling to minimize auto-scaling overhead
    - Decoding: Weighted round-robin with SLO-aware time quota
    - Token-level preemptive scheduling
    """

    def __init__(
        self,
        max_group_size: int = 8,
        qmax: float = 4.0,
        min_alpha: float = 0.5,
        estimated_scaling_overhead: float = 1.0,
    ):
        """Initialize Aegaeon policy.

        Args:
            max_group_size: MAX_GPSIZE - max requests per group
            qmax: QMAX - maximum time quota in seconds
            min_alpha: Minimum α for SLO guarantee
            estimated_scaling_overhead: c - auto-scaling overhead in seconds
        """
        super().__init__(name="aegaeon")
        self.max_group_size = max_group_size
        self.qmax = qmax
        self.min_alpha = min_alpha
        self.estimated_scaling_overhead = estimated_scaling_overhead

        # Prefill state: instance_id -> list of PrefillGroup
        self.prefill_job_queues: dict[str, list[PrefillGroup]] = defaultdict(list)

        # Decoding state: instance_id -> list of DecodingBatch
        self.decoding_work_lists: dict[str, list[DecodingBatch]] = defaultdict(list)

    def schedule(
        self,
        requests: list[RequestMetadata],
        instances: list[ExecutionInstance],
    ) -> list[SchedulingDecision]:
        """Schedule requests to instances."""
        decisions = []

        # Separate prefill and decoding instances
        prefill_instances = [
            i for i in instances if i.instance_type == ExecutionInstanceType.PREFILLING
        ]
        decoding_instances = [
            i for i in instances if i.instance_type == ExecutionInstanceType.DECODING
        ]

        for request in requests:
            decision = self._schedule_prefill_request(request, prefill_instances)
            if decision:
                decisions.append(decision)

        return decisions

    def _schedule_prefill_request(
        self,
        request: RequestMetadata,
        prefill_instances: list[ExecutionInstance],
    ) -> SchedulingDecision | None:
        """Algorithm 1: Grouped Prefill-Phase Scheduling."""
        if not prefill_instances:
            return None

        # Step 1: Try to add to existing group with same model
        for instance in prefill_instances:
            job_queue = self.prefill_job_queues[instance.instance_id]
            for group in job_queue:
                if group.can_add(request):
                    group.requests.append(request)
                    return SchedulingDecision(
                        request_id=request.request_id,
                        target_instance_id=instance.instance_id,
                        parallelism_strategy=request.parallelism_hint or ParallelismType.TENSOR_PARALLEL,
                        estimated_latency_ms=self._estimate_prefill_wait_time(instance) * 1000,
                        estimated_cost=0.0,
                    )

        # Step 2: Create new group and add to least loaded instance
        min_load = float("inf")
        best_instance = None

        for instance in prefill_instances:
            load = self._calculate_prefill_load(instance)
            if load < min_load:
                min_load = load
                best_instance = instance

        if best_instance:
            new_group = PrefillGroup(
                model_name=request.model_name or "default",
                requests=[request],
                max_size=self.max_group_size,
            )
            self.prefill_job_queues[best_instance.instance_id].append(new_group)

            return SchedulingDecision(
                request_id=request.request_id,
                target_instance_id=best_instance.instance_id,
                parallelism_strategy=request.parallelism_hint or ParallelismType.TENSOR_PARALLEL,
                estimated_latency_ms=self._estimate_prefill_wait_time(best_instance) * 1000,
                estimated_cost=0.0,
            )

        return None

    def _calculate_prefill_load(self, instance: ExecutionInstance) -> float:
        """Calculate total time to execute all groups in instance."""
        job_queue = self.prefill_job_queues[instance.instance_id]

        total_time = 0.0
        prev_model = None

        for group in job_queue:
            # Add auto-scaling time if model changes
            if prev_model and prev_model != group.model_name:
                total_time += self.estimated_scaling_overhead

            # Add execution time (batch size = 1 per paper)
            total_time += len(group.requests) * 1.0  # Assume 1s per request

            prev_model = group.model_name

        return total_time

    def _estimate_prefill_wait_time(self, instance: ExecutionInstance) -> float:
        """Estimate wait time for new request."""
        return self._calculate_prefill_load(instance)

    def get_next_prefill_request(self, instance_id: str) -> RequestMetadata | None:
        """Get next request to execute from prefill job queue (FIFO within groups)."""
        job_queue = self.prefill_job_queues[instance_id]

        if not job_queue:
            return None

        # Get from front group
        front_group = job_queue[0]
        if front_group.requests:
            request = front_group.requests.pop(0)

            # Remove empty group
            if not front_group.requests:
                job_queue.pop(0)

            return request

        return None

    def schedule_decoding_round(
        self,
        instance_id: str,
    ) -> list[tuple[DecodingBatch, float]]:
        """Algorithm 2: Batched Decoding-Phase Scheduling.

        Returns:
            List of (batch, time_quota) tuples
        """
        work_list = self.decoding_work_lists[instance_id]

        if not work_list:
            return []

        # Calculate time quotas for all batches
        n_values = [batch.n_value for batch in work_list]
        sum_inv_n = sum(1.0 / n for n in n_values)
        min_n = min(n_values)

        # Calculate α: α = max(c/(min_n * QMAX) + Σ(1/n_k), 0.5)
        alpha = max(
            self.estimated_scaling_overhead / (min_n * self.qmax) + sum_inv_n,
            self.min_alpha,
        )

        # Assign quota to each batch: q_i = (c/n_i) * (α - Σ(1/n_k))
        schedule = []
        for i, batch in enumerate(work_list):
            n_i = n_values[i]
            q_i = (self.estimated_scaling_overhead / n_i) * (alpha - sum_inv_n)

            # Ensure non-negative quota
            q_i = max(0.1, q_i)

            batch.time_quota = q_i
            schedule.append((batch, q_i))

        # Reorder work list to group batches with same model
        self._reorder_work_list_by_model(instance_id)

        return schedule

    def _reorder_work_list_by_model(self, instance_id: str):
        """Reorder work list to group batches with same model."""
        work_list = self.decoding_work_lists[instance_id]

        # Group by model
        model_batches = defaultdict(list)
        for batch in work_list:
            model_batches[batch.model_name].append(batch)

        # Flatten back to list
        reordered = []
        for model_name in sorted(model_batches.keys()):
            reordered.extend(model_batches[model_name])

        self.decoding_work_lists[instance_id] = reordered

    def add_to_decoding_batch(
        self,
        request: RequestMetadata,
        instance_id: str,
    ):
        """Add a prefilled request to decoding work list."""
        work_list = self.decoding_work_lists[instance_id]
        model_name = request.model_name or "default"

        # Try to find existing batch with same model
        for batch in work_list:
            if batch.model_name == model_name:
                batch.requests.append(request)
                return

        # Create new batch
        new_batch = DecodingBatch(
            model_name=model_name,
            requests=[request],
        )
        work_list.append(new_batch)

    def prioritize(
        self,
        requests: list[RequestMetadata],
    ) -> list[RequestMetadata]:
        """Prioritize requests (FIFO within groups for Aegaeon)."""
        return sorted(requests, key=lambda r: r.arrival_time)
