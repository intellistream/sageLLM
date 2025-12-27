# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PD Scheduler - Prefill-Decode Separation Scheduler.

This scheduler implements PD disaggregation for LLM inference, supporting
three scheduling modes:

1. **strict**: Prefill and Decode run on different GPUs, fully separated
   - Best for: Multi-GPU clusters with dedicated resources
   - Pros: Maximum isolation, predictable latency
   - Cons: Requires more GPUs, KV migration overhead

2. **time_share**: Same GPU, time-division multiplexing
   - Best for: Single-GPU or resource-constrained environments
   - Pros: Simple, no KV migration
   - Cons: Prefill can block Decode, affecting TBT

3. **hybrid**: Chunked Prefill interleaved with Decode (Sarathi-style)
   - Best for: Balancing TTFT and throughput
   - Pros: Better GPU utilization, lower TTFT variance
   - Cons: More complex scheduling

Example:
    >>> from sageLLM.runtime.scheduler.pd_scheduler import PDScheduler, PDSchedulerConfig
    >>> config = PDSchedulerConfig(
    ...     mode="hybrid",
    ...     prefill_chunk_size=512,
    ...     max_decode_batch_size=256,
    ... )
    >>> scheduler = PDScheduler(config)
    >>> scheduler.add_request(Request(request_id="r1", prompt_token_ids=[1,2,3,4]))
    >>> output = scheduler.schedule()

References:
    - DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving
    - Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
    - Splitwise: Efficient generative LLM inference with phase splitting
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import (
    BaseScheduler,
    Batch,
    Request,
    RequestPhase,
    RequestStatus,
    ScheduleOutput,
    SchedulerConfig,
)


@dataclass
class PDSchedulerConfig(SchedulerConfig):
    """Configuration for PD-separated scheduling.

    Attributes:
        mode: Scheduling mode - "strict", "time_share", or "hybrid"

        Device allocation:
        - prefill_device_ids: GPUs for prefill operations
        - decode_device_ids: GPUs for decode operations

        Prefill settings:
        - max_prefill_batch_size: Max requests per prefill batch
        - max_prefill_tokens: Max tokens per prefill batch
        - prefill_chunk_size: Chunk size for hybrid mode

        Decode settings:
        - max_decode_batch_size: Max requests per decode batch
        - max_decode_tokens: Max tokens per decode iteration

        Preemption:
        - enable_preemption: Allow preempting running requests
        - preemption_mode: "swap" (to CPU) or "recompute" (from scratch)
    """

    # PD separation is the core feature
    enable_pd_separation: bool = True

    # Scheduling mode
    mode: Literal["strict", "time_share", "hybrid"] = "time_share"

    # Device allocation
    prefill_device_ids: list[int] = field(default_factory=lambda: [0])
    decode_device_ids: list[int] = field(default_factory=lambda: [0])

    # Prefill config
    max_prefill_batch_size: int = 32
    max_prefill_tokens: int = 4096
    prefill_chunk_size: int = 512  # For hybrid mode

    # Decode config
    max_decode_batch_size: int = 256
    max_decode_tokens: int = 8192

    # Preemption config
    enable_preemption: bool = True
    preemption_mode: Literal["swap", "recompute"] = "recompute"


@dataclass
class KVMigrationTask:
    """Represents a KV cache migration task.

    Used in strict mode when KV cache needs to move from
    prefill device to decode device.
    """

    request_id: str
    source_device: int
    target_device: int
    kv_size_bytes: int
    created_time: float = field(default_factory=time.time)
    completed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class PDScheduler(BaseScheduler):
    """Prefill-Decode Separation Scheduler.

    Implements three scheduling modes:

    1. **strict** mode:
       - Prefill and Decode run on different GPU pools
       - KV cache is migrated after prefill completes
       - Both phases can execute in parallel on different GPUs

    2. **time_share** mode:
       - Single GPU (or shared pool) for both phases
       - Prioritizes Decode for low TBT (Time-Between-Tokens)
       - Prefill executes when no Decode work is pending

    3. **hybrid** mode (Sarathi/Chunked Prefill):
       - Prefill is split into chunks
       - Each chunk is scheduled alongside Decode
       - Token budget is shared between phases

    State Management:
    - prefill_queue: Requests waiting for prefill
    - decode_queue: Requests in decode phase (continuous batching)
    - Requests transition: prefill_queue -> decode_queue -> completed
    """

    def __init__(self, config: PDSchedulerConfig) -> None:
        """Initialize the PD scheduler.

        Args:
            config: PDSchedulerConfig with mode and resource settings
        """
        super().__init__(
            max_batch_size=config.max_decode_batch_size,
            max_tokens=config.max_decode_tokens,
            config=config,
        )
        self.pd_config = config

        # Separate queues for PD separation
        self.prefill_queue: list[Request] = []
        self.decode_queue: list[Request] = []

        # Track in-flight requests
        self._prefilling_requests: dict[str, Request] = {}
        self._decoding_requests: dict[str, Request] = {}

        # KV migration queue (for strict mode)
        self._pending_migrations: list[KVMigrationTask] = []

        # Device load tracking
        self._prefill_device_load: dict[int, int] = dict.fromkeys(
            config.prefill_device_ids, 0
        )
        self._decode_device_load: dict[int, int] = dict.fromkeys(
            config.decode_device_ids, 0
        )

        # Statistics
        self.stats: dict[str, int] = {
            "total_prefill_batches": 0,
            "total_decode_batches": 0,
            "total_preemptions": 0,
            "total_kv_migrations": 0,
        }

    def add_request(self, request: Request) -> None:
        """Add a new request to the prefill queue.

        Args:
            request: Request to add
        """
        request.status = RequestStatus.WAITING
        request.arrival_time = time.time()
        self.prefill_queue.append(request)
        # Also add to parent's tracking
        self.waiting_queue.append(request)
        self._pending_requests.append(request)

    def schedule(self) -> ScheduleOutput:
        """Execute scheduling based on configured mode.

        Returns:
            ScheduleOutput with prefill and/or decode batches
        """
        if self.pd_config.mode == "strict":
            return self._schedule_strict()
        elif self.pd_config.mode == "time_share":
            return self._schedule_time_share()
        else:  # hybrid
            return self._schedule_hybrid()

    def _schedule_strict(self) -> ScheduleOutput:
        """Strict separation mode scheduling.

        Prefill and Decode execute on separate device pools simultaneously.
        """
        output = ScheduleOutput()

        # Schedule Prefill batch (on prefill devices)
        if self.prefill_queue:
            prefill_requests = self._select_prefill_requests()
            if prefill_requests:
                output.prefill_batch = Batch(
                    batch_id=f"prefill_{self.stats['total_prefill_batches']}",
                    requests=prefill_requests,
                    phase=RequestPhase.PREFILLING,
                    created_time=time.time(),
                )
                self.stats["total_prefill_batches"] += 1

                # Mark requests as prefilling
                for req in prefill_requests:
                    req.status = RequestStatus.PREFILLING
                    req.metadata["prefill_start_time"] = time.time()
                    self._prefilling_requests[req.request_id] = req

        # Schedule Decode batch (on decode devices)
        if self.decode_queue:
            decode_requests = self._select_decode_requests()
            if decode_requests:
                output.decode_batch = Batch(
                    batch_id=f"decode_{self.stats['total_decode_batches']}",
                    requests=decode_requests,
                    phase=RequestPhase.DECODING,
                    created_time=time.time(),
                )
                self.stats["total_decode_batches"] += 1

        return output

    def _schedule_time_share(self) -> ScheduleOutput:
        """Time-share mode scheduling.

        Prioritizes Decode for low TBT. Prefill runs when Decode is idle.
        """
        output = ScheduleOutput()

        # Priority 1: Schedule Decode (continuous batching)
        if self.decode_queue:
            decode_requests = self._select_decode_requests()
            if decode_requests:
                output.decode_batch = Batch(
                    batch_id=f"decode_{self.stats['total_decode_batches']}",
                    requests=decode_requests,
                    phase=RequestPhase.DECODING,
                    created_time=time.time(),
                )
                self.stats["total_decode_batches"] += 1
                return output  # Decode takes priority

        # Priority 2: Schedule Prefill when no Decode work
        if self.prefill_queue:
            prefill_requests = self._select_prefill_requests()
            if prefill_requests:
                output.prefill_batch = Batch(
                    batch_id=f"prefill_{self.stats['total_prefill_batches']}",
                    requests=prefill_requests,
                    phase=RequestPhase.PREFILLING,
                    created_time=time.time(),
                )
                self.stats["total_prefill_batches"] += 1

                for req in prefill_requests:
                    req.status = RequestStatus.PREFILLING
                    req.metadata["prefill_start_time"] = time.time()
                    self._prefilling_requests[req.request_id] = req

        return output

    def _schedule_hybrid(self) -> ScheduleOutput:
        """Hybrid mode scheduling (Chunked Prefill / Sarathi-style).

        Prefill is chunked and interleaved with Decode within a token budget.
        """
        output = ScheduleOutput()

        # Step 1: Select Decode requests
        decode_requests: list[Request] = []
        if self.decode_queue:
            decode_requests = self._select_decode_requests()

        # Decode tokens: 1 per request (single token generation)
        decode_tokens = len(decode_requests)

        # Step 2: Calculate remaining token budget for Chunked Prefill
        remaining_tokens = self.pd_config.max_decode_tokens - decode_tokens

        # Step 3: Select Chunked Prefill requests with remaining budget
        prefill_requests: list[Request] = []
        if self.prefill_queue and remaining_tokens > 0:
            prefill_requests = self._select_chunked_prefill(remaining_tokens)

        # Step 4: Create batches
        if decode_requests:
            output.decode_batch = Batch(
                batch_id=f"decode_{self.stats['total_decode_batches']}",
                requests=decode_requests,
                phase=RequestPhase.DECODING,
                created_time=time.time(),
            )
            self.stats["total_decode_batches"] += 1

        if prefill_requests:
            output.prefill_batch = Batch(
                batch_id=f"prefill_{self.stats['total_prefill_batches']}",
                requests=prefill_requests,
                phase=RequestPhase.PREFILLING,
                created_time=time.time(),
            )
            self.stats["total_prefill_batches"] += 1

        return output

    def _select_prefill_requests(self) -> list[Request]:
        """Select requests for a prefill batch.

        Selection criteria:
        - Respects max_prefill_batch_size
        - Respects max_prefill_tokens
        - Sorted by priority then arrival time
        """
        selected: list[Request] = []
        total_tokens = 0

        # Sort by priority (desc) and arrival time (asc)
        sorted_queue = sorted(
            self.prefill_queue,
            key=lambda r: (-r.priority, r.arrival_time),
        )

        for req in sorted_queue:
            if len(selected) >= self.pd_config.max_prefill_batch_size:
                break

            # Tokens needed for this request
            tokens_needed = req.num_prompt_tokens - req.num_computed_tokens

            if total_tokens + tokens_needed > self.pd_config.max_prefill_tokens:
                break

            selected.append(req)
            total_tokens += tokens_needed

        # Remove selected from prefill queue
        for req in selected:
            self.prefill_queue.remove(req)
            if req in self.waiting_queue:
                self.waiting_queue.remove(req)
            if req in self._pending_requests:
                self._pending_requests.remove(req)
            self.running_requests[req.request_id] = req

        return selected

    def _select_decode_requests(self) -> list[Request]:
        """Select requests for a decode batch.

        For continuous batching, all requests in decode_queue
        are candidates (up to batch size limit).
        """
        selected: list[Request] = []

        for req in self.decode_queue:
            if len(selected) >= self.pd_config.max_decode_batch_size:
                break
            selected.append(req)

        # Note: Don't remove from decode_queue - continuous batching
        # keeps requests there until they finish
        return selected

    def _select_chunked_prefill(self, token_budget: int) -> list[Request]:
        """Select requests for chunked prefill with a token budget.

        Args:
            token_budget: Maximum tokens to use for prefill

        Returns:
            List of requests with updated num_computed_tokens
        """
        selected: list[Request] = []
        remaining = token_budget
        chunk_size = self.pd_config.prefill_chunk_size

        # Sort by priority and arrival
        sorted_queue = sorted(
            self.prefill_queue,
            key=lambda r: (-r.priority, r.arrival_time),
        )

        for req in sorted_queue[:]:  # Copy for safe iteration
            if remaining <= 0:
                break

            # Tokens remaining for this request's prefill
            req_remaining = req.num_prompt_tokens - req.num_computed_tokens

            # Take a chunk or whatever fits
            chunk = min(req_remaining, chunk_size, remaining)

            if chunk > 0:
                selected.append(req)
                remaining -= chunk

                # Update computed tokens (chunked progress)
                req.num_computed_tokens += chunk

                # Mark as prefilling
                req.status = RequestStatus.PREFILLING
                if req.metadata.get("prefill_start_time") is None:
                    req.metadata["prefill_start_time"] = time.time()
                self._prefilling_requests[req.request_id] = req

                # If prefill complete, transition to decode
                if req.num_computed_tokens >= req.num_prompt_tokens:
                    self.prefill_queue.remove(req)
                    self._transition_to_decode(req)

        return selected

    def _transition_to_decode(self, request: Request) -> None:
        """Transition a request from prefill to decode phase.

        Args:
            request: Request that completed prefill
        """
        request.status = RequestStatus.DECODING
        request.metadata["decode_start_time"] = time.time()

        # Record TTFT (Time-To-First-Token)
        if request.first_token_time is None:
            request.first_token_time = time.time()

        # Move to decode queue
        if request not in self.decode_queue:
            self.decode_queue.append(request)

        # Update tracking
        self._prefilling_requests.pop(request.request_id, None)
        self._decoding_requests[request.request_id] = request
        self.running_requests[request.request_id] = request

        # For strict mode, schedule KV migration
        if self.pd_config.mode == "strict":
            self._schedule_kv_migration(request)

    def _schedule_kv_migration(self, request: Request) -> None:
        """Schedule KV cache migration for strict mode.

        Args:
            request: Request whose KV cache needs migration
        """
        if not self.pd_config.prefill_device_ids or not self.pd_config.decode_device_ids:
            return

        source_device = self.pd_config.prefill_device_ids[0]  # Simplified
        target_device = self._select_decode_device()

        # Estimate KV size: 2 * num_layers * seq_len * head_dim * num_kv_heads * dtype_size
        # Simplified estimate
        kv_size = request.num_prompt_tokens * 1024 * 2  # Rough estimate

        task = KVMigrationTask(
            request_id=request.request_id,
            source_device=source_device,
            target_device=target_device,
            kv_size_bytes=kv_size,
        )
        self._pending_migrations.append(task)
        self.stats["total_kv_migrations"] += 1

    def _select_decode_device(self) -> int:
        """Select the best decode device (least loaded)."""
        if not self.pd_config.decode_device_ids:
            return 0
        return min(
            self._decode_device_load,
            key=self._decode_device_load.get,  # type: ignore[arg-type]
            default=0,
        )

    def update_after_step(self, finished_request_ids: list[str]) -> None:
        """Update state after an execution step.

        Args:
            finished_request_ids: IDs of requests that finished generating
        """
        for req_id in finished_request_ids:
            # Remove from decode queue
            for req in self.decode_queue[:]:
                if req.request_id == req_id:
                    self.decode_queue.remove(req)
                    req.status = RequestStatus.FINISHED
                    req.finish_time = time.time()

                    # Clean up tracking
                    self._decoding_requests.pop(req_id, None)
                    self.running_requests.pop(req_id, None)
                    self._completed_requests[req_id] = req
                    break

    def prefill_to_decode(self, request_ids: list[str]) -> None:
        """Transition requests from prefill to decode after prefill completes.

        Called after prefill execution finishes for a batch.

        Args:
            request_ids: IDs of requests that completed prefill
        """
        for req_id in request_ids:
            req = self._prefilling_requests.get(req_id)
            if req:
                # Mark prefill as fully computed
                req.num_computed_tokens = req.num_prompt_tokens
                self._transition_to_decode(req)

    def abort_request(self, request_id: str) -> bool:
        """Abort a request at any stage.

        Args:
            request_id: ID of request to abort

        Returns:
            True if request was found and aborted
        """
        # Check prefill queue
        for i, req in enumerate(self.prefill_queue):
            if req.request_id == request_id:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                self.prefill_queue.pop(i)
                self._completed_requests[request_id] = req
                return True

        # Check prefilling
        if request_id in self._prefilling_requests:
            req = self._prefilling_requests.pop(request_id)
            req.status = RequestStatus.FINISHED
            req.finish_time = time.time()
            self._completed_requests[request_id] = req
            return True

        # Check decode queue
        for i, req in enumerate(self.decode_queue):
            if req.request_id == request_id:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                self.decode_queue.pop(i)
                self._decoding_requests.pop(request_id, None)
                self._completed_requests[request_id] = req
                return True

        return super().abort_request(request_id)

    def get_pending_migrations(self) -> list[KVMigrationTask]:
        """Get pending KV migration tasks.

        Returns:
            List of incomplete migration tasks
        """
        return [m for m in self._pending_migrations if not m.completed]

    def complete_migration(self, request_id: str) -> None:
        """Mark a KV migration as complete.

        Args:
            request_id: ID of request whose migration completed
        """
        for migration in self._pending_migrations:
            if migration.request_id == request_id:
                migration.completed = True
                break

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduling statistics
        """
        return {
            **self.stats,
            "prefill_queue_size": len(self.prefill_queue),
            "decode_queue_size": len(self.decode_queue),
            "prefilling_count": len(self._prefilling_requests),
            "decoding_count": len(self._decoding_requests),
            "pending_migrations": len(self.get_pending_migrations()),
        }

    # Legacy properties for backward compatibility
    @property
    def num_prefilling(self) -> int:
        """Number of requests currently prefilling."""
        return len(self._prefilling_requests)

    @property
    def num_decoding(self) -> int:
        """Number of requests currently decoding."""
        return len(self._decoding_requests)

    @property
    def prefill_queue_size(self) -> int:
        """Size of prefill queue."""
        return len(self.prefill_queue)

    @property
    def decode_queue_size(self) -> int:
        """Size of decode queue."""
        return len(self.decode_queue)

    # Legacy compatibility methods
    def get_prefill_batch(self) -> Batch | None:
        """Legacy method to get prefill batch only."""
        if not self.prefill_queue:
            return None
        prefill_requests = self._select_prefill_requests()
        if not prefill_requests:
            return None
        return Batch(
            batch_id=f"prefill_{self.stats['total_prefill_batches']}",
            requests=prefill_requests,
            phase=RequestPhase.PREFILLING,
            created_time=time.time(),
        )

    def get_decode_batch(self) -> Batch | None:
        """Legacy method to get decode batch only."""
        if not self.decode_queue:
            return None
        decode_requests = self._select_decode_requests()
        if not decode_requests:
            return None
        return Batch(
            batch_id=f"decode_{self.stats['total_decode_batches']}",
            requests=decode_requests,
            phase=RequestPhase.DECODING,
            created_time=time.time(),
        )

    def complete_prefill(
        self,
        request_id: str,
        source_device: int,
        kv_size_bytes: int,
    ) -> None:
        """Legacy method for marking prefill complete.

        Args:
            request_id: ID of completed request
            source_device: Device where KV cache resides
            kv_size_bytes: Size of KV cache
        """
        self.prefill_to_decode([request_id])


# Legacy enum for backward compatibility
class KVMigrationStrategy:
    """Legacy KV migration strategy constants."""

    EAGER = "eager"
    LAZY = "lazy"
    CHUNKED = "chunked"


__all__ = [
    "PDSchedulerConfig",
    "PDScheduler",
    "KVMigrationTask",
    "KVMigrationStrategy",
]
