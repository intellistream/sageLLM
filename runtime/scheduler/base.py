# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler Base - Base classes and interfaces for scheduling.

This module defines:
- RequestStatus: Request lifecycle states
- Request: Represents an inference request with full metadata
- Batch: A batch of requests for processing
- ScheduleOutput: Output of a scheduling decision
- BaseScheduler: Abstract base class for all schedulers

The scheduler is responsible for:
- Managing request queues (waiting, running, preempted)
- Batch formation for Prefill and Decode phases
- KV cache allocation coordination
- Request lifecycle management

Example:
    >>> from sageLLM.runtime.scheduler.base import BaseScheduler, Request
    >>> scheduler = MyScheduler(max_batch_size=256)
    >>> scheduler.add_request(Request(request_id="r1", prompt_token_ids=[1,2,3]))
    >>> output = scheduler.schedule()
    >>> if output.prefill_batch:
    ...     # Execute prefill
    ...     pass

References:
    - vLLM scheduler: https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
    - Orca: https://www.usenix.org/conference/osdi22/presentation/yu
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RequestStatus(Enum):
    """Request lifecycle status.

    State transitions:
    WAITING -> PREFILLING -> DECODING -> FINISHED
                         -> PREEMPTED -> WAITING (recompute)
                         -> PREEMPTED -> SWAPPED (swap)
    Any state -> FINISHED (abort or complete)
    """

    WAITING = auto()  # In waiting queue, not yet scheduled
    PREFILLING = auto()  # Currently executing prefill phase
    DECODING = auto()  # Currently executing decode phase
    PREEMPTED = auto()  # Preempted, waiting to resume
    SWAPPED = auto()  # KV cache swapped to CPU
    FINISHED = auto()  # Completed (success or abort)


# Legacy alias
class RequestPhase(Enum):
    """Legacy request phase enum - use RequestStatus instead."""

    PENDING = auto()
    PREFILLING = auto()
    DECODING = auto()
    COMPLETED = auto()
    FAILED = auto()


class RequestPriority(Enum):
    """Priority levels for requests."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Request:
    """Represents an inference request.

    Attributes:
        request_id: Unique identifier for the request
        prompt_token_ids: Input token IDs

        Generation parameters:
        - max_new_tokens: Maximum tokens to generate
        - temperature: Sampling temperature
        - top_p: Nucleus sampling threshold
        - top_k: Top-k sampling (-1 for disabled)

        State tracking:
        - status: Current request status
        - output_token_ids: Generated tokens so far
        - kv_block_ids: Allocated KV cache block IDs
        - num_computed_tokens: Tokens computed in prefill

        Timing:
        - arrival_time: When request was added
        - first_token_time: When first token was generated (TTFT)
        - finish_time: When request completed

        Priority:
        - priority: Scheduling priority (higher = more urgent)
    """

    request_id: str
    prompt_token_ids: list[int]

    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: list[int] = field(default_factory=list)

    # State
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)

    # KV cache tracking
    kv_block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    # Timing
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    finish_time: float | None = None

    # Priority
    priority: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility
    @property
    def phase(self) -> RequestPhase:
        """Legacy phase property."""
        mapping = {
            RequestStatus.WAITING: RequestPhase.PENDING,
            RequestStatus.PREFILLING: RequestPhase.PREFILLING,
            RequestStatus.DECODING: RequestPhase.DECODING,
            RequestStatus.FINISHED: RequestPhase.COMPLETED,
            RequestStatus.PREEMPTED: RequestPhase.PENDING,
            RequestStatus.SWAPPED: RequestPhase.PENDING,
        }
        return mapping.get(self.status, RequestPhase.PENDING)

    @phase.setter
    def phase(self, value: RequestPhase) -> None:
        """Legacy phase setter."""
        mapping = {
            RequestPhase.PENDING: RequestStatus.WAITING,
            RequestPhase.PREFILLING: RequestStatus.PREFILLING,
            RequestPhase.DECODING: RequestStatus.DECODING,
            RequestPhase.COMPLETED: RequestStatus.FINISHED,
            RequestPhase.FAILED: RequestStatus.FINISHED,
        }
        self.status = mapping.get(value, RequestStatus.WAITING)

    @property
    def generated_token_ids(self) -> list[int]:
        """Legacy alias for output_token_ids."""
        return self.output_token_ids

    @property
    def prefill_start_time(self) -> float | None:
        """Legacy compatibility."""
        return self.metadata.get("prefill_start_time")

    @prefill_start_time.setter
    def prefill_start_time(self, value: float | None) -> None:
        """Legacy compatibility."""
        self.metadata["prefill_start_time"] = value

    @property
    def decode_start_time(self) -> float | None:
        """Legacy compatibility."""
        return self.metadata.get("decode_start_time")

    @decode_start_time.setter
    def decode_start_time(self, value: float | None) -> None:
        """Legacy compatibility."""
        self.metadata["decode_start_time"] = value

    @property
    def completion_time(self) -> float | None:
        """Legacy alias for finish_time."""
        return self.finish_time

    @completion_time.setter
    def completion_time(self, value: float | None) -> None:
        """Legacy alias setter."""
        self.finish_time = value

    @property
    def num_prompt_tokens(self) -> int:
        """Number of prompt tokens."""
        return len(self.prompt_token_ids)

    @property
    def prompt_len(self) -> int:
        """Legacy alias for num_prompt_tokens."""
        return self.num_prompt_tokens

    @property
    def num_output_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.output_token_ids)

    @property
    def generated_len(self) -> int:
        """Legacy alias for num_output_tokens."""
        return self.num_output_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + generated)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def total_len(self) -> int:
        """Legacy alias for total_tokens."""
        return self.total_tokens

    @property
    def context_len(self) -> int:
        """Current context length (for decode attention)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_complete(self) -> bool:
        """Whether the request has finished."""
        return self.status == RequestStatus.FINISHED

    @property
    def is_prefill_complete(self) -> bool:
        """Whether prefill phase is complete."""
        return self.num_computed_tokens >= self.num_prompt_tokens

    def get_ttft(self) -> float | None:
        """Get Time-To-First-Token in seconds."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def get_total_time(self) -> float | None:
        """Get total request time in seconds."""
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time


@dataclass
class Batch:
    """A batch of requests for processing.

    Attributes:
        batch_id: Unique identifier for the batch
        requests: List of requests in the batch
        phase: Phase being processed (PREFILLING or DECODING)
        created_time: When the batch was created
    """

    batch_id: str
    requests: list[Request]
    phase: RequestPhase = RequestPhase.PREFILLING
    created_time: float = field(default_factory=time.time)

    @property
    def batch_size(self) -> int:
        """Number of requests in the batch."""
        return len(self.requests)

    @property
    def size(self) -> int:
        """Legacy alias for batch_size."""
        return self.batch_size

    @property
    def total_tokens(self) -> int:
        """Total tokens in the batch."""
        if self.phase == RequestPhase.PREFILLING:
            return sum(r.num_prompt_tokens - r.num_computed_tokens for r in self.requests)
        return len(self.requests)  # Decode: 1 token per request

    @property
    def request_ids(self) -> list[str]:
        """List of request IDs in the batch."""
        return [r.request_id for r in self.requests]


@dataclass
class ScheduleOutput:
    """Output of a scheduling decision.

    Contains:
    - prefill_batch: Batch of requests for prefill (if any)
    - decode_batch: Batch of requests for decode (if any)
    - preempted_requests: Requests that were preempted this round
    - kv_blocks_to_allocate: KV blocks to allocate per request
    - kv_blocks_to_free: KV block IDs to free
    """

    prefill_batch: Batch | None = None
    decode_batch: Batch | None = None
    preempted_requests: list[Request] = field(default_factory=list)

    # KV cache operations
    kv_blocks_to_allocate: dict[str, list[int]] = field(default_factory=dict)
    kv_blocks_to_free: list[int] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        """Whether there is work to do."""
        return self.prefill_batch is not None or self.decode_batch is not None

    @property
    def total_requests(self) -> int:
        """Total number of requests to process."""
        prefill_count = self.prefill_batch.batch_size if self.prefill_batch else 0
        decode_count = self.decode_batch.batch_size if self.decode_batch else 0
        return prefill_count + decode_count


# Legacy scheduler config for backward compatibility
@dataclass
class SchedulerConfig:
    """Configuration for schedulers.

    Legacy class - new code should use scheduler-specific configs.
    """

    max_batch_size: int = 32
    max_tokens_per_batch: int = 4096
    max_prefill_tokens: int = 2048

    enable_pd_separation: bool = False
    prefill_device_ids: list[int] = field(default_factory=list)
    decode_device_ids: list[int] = field(default_factory=list)

    enable_priority: bool = True
    preemption_enabled: bool = False

    enable_slo: bool = False
    default_deadline_ms: float = 10000.0

    gpu_memory_utilization: float = 0.9
    swap_space_bytes: int = 0

    chunked_prefill_size: int = 512
    speculative_decode_enabled: bool = False
    speculative_tokens: int = 5


class BaseScheduler(ABC):
    """Abstract base class for request schedulers.

    Schedulers are responsible for:
    1. Managing request queues (waiting, running, preempted)
    2. Forming batches for Prefill and Decode phases
    3. Coordinating with KV cache manager for block allocation
    4. Handling preemption when resources are constrained

    Subclasses must implement:
    - schedule(): Make scheduling decisions
    - update_after_step(): Update state after execution

    Example:
        >>> class MyScheduler(BaseScheduler):
        ...     def schedule(self) -> ScheduleOutput:
        ...         # Form batches from waiting/running requests
        ...         ...
        ...     def update_after_step(self, finished_ids: list[str]) -> None:
        ...         # Update state after execution
        ...         ...
    """

    def __init__(
        self,
        max_batch_size: int = 256,
        max_tokens: int = 8192,
        config: SchedulerConfig | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            max_batch_size: Maximum requests per batch
            max_tokens: Maximum tokens per batch
            config: Legacy SchedulerConfig (optional)
        """
        if config is not None:
            self.config = config
            self.max_batch_size = config.max_batch_size
            self.max_tokens = config.max_tokens_per_batch
        else:
            self.config = SchedulerConfig(
                max_batch_size=max_batch_size,
                max_tokens_per_batch=max_tokens,
            )
            self.max_batch_size = max_batch_size
            self.max_tokens = max_tokens

        # Request queues
        self.waiting_queue: list[Request] = []
        self.running_requests: dict[str, Request] = {}
        self.preempted_requests: list[Request] = []

        # Legacy: pending and completed tracking
        self._pending_requests: list[Request] = []
        self._running_requests: dict[str, Request] = {}
        self._completed_requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.

        Args:
            request: Request to add
        """
        request.status = RequestStatus.WAITING
        request.arrival_time = time.time()
        self.waiting_queue.append(request)
        self._pending_requests.append(request)

    def abort_request(self, request_id: str) -> bool:
        """Abort a request.

        Args:
            request_id: ID of the request to abort

        Returns:
            True if request was found and aborted
        """
        # Check waiting queue
        for i, req in enumerate(self.waiting_queue):
            if req.request_id == request_id:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                self.waiting_queue.pop(i)
                self._completed_requests[request_id] = req
                return True

        # Check running requests
        if request_id in self.running_requests:
            req = self.running_requests.pop(request_id)
            req.status = RequestStatus.FINISHED
            req.finish_time = time.time()
            self._completed_requests[request_id] = req
            return True

        # Check preempted
        for i, req in enumerate(self.preempted_requests):
            if req.request_id == request_id:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                self.preempted_requests.pop(i)
                self._completed_requests[request_id] = req
                return True

        return False

    @abstractmethod
    def schedule(self) -> ScheduleOutput:
        """Make scheduling decision.

        Returns:
            ScheduleOutput with batches to execute
        """

    @abstractmethod
    def update_after_step(self, finished_request_ids: list[str]) -> None:
        """Update state after an execution step.

        Args:
            finished_request_ids: IDs of requests that finished
        """

    # Legacy method aliases
    def get_next_batch(self) -> Batch | None:
        """Legacy method - returns prefill or decode batch."""
        output = self.schedule()
        return output.prefill_batch or output.decode_batch

    def update_request(
        self,
        request_id: str,
        generated_tokens: list[int],
        is_complete: bool = False,
    ) -> None:
        """Legacy method for updating request state."""
        request = self.running_requests.get(request_id)
        if not request:
            return

        request.output_token_ids.extend(generated_tokens)

        if is_complete:
            self.update_after_step([request_id])

    def get_request(self, request_id: str) -> Request | None:
        """Get a request by ID."""
        # Check running
        if request_id in self.running_requests:
            return self.running_requests[request_id]
        # Check waiting
        for req in self.waiting_queue:
            if req.request_id == request_id:
                return req
        # Check preempted
        for req in self.preempted_requests:
            if req.request_id == request_id:
                return req
        # Check completed
        return self._completed_requests.get(request_id)

    def get_num_waiting(self) -> int:
        """Number of waiting requests."""
        return len(self.waiting_queue)

    def get_num_running(self) -> int:
        """Number of running requests."""
        return len(self.running_requests)

    @property
    def num_pending(self) -> int:
        """Legacy property for waiting count."""
        return self.get_num_waiting()

    @property
    def num_running(self) -> int:
        """Legacy property for running count."""
        return self.get_num_running()

    @property
    def num_completed(self) -> int:
        """Number of completed requests."""
        return len(self._completed_requests)


class FIFOScheduler(BaseScheduler):
    """Simple FIFO scheduler for baseline comparison.

    Schedules requests in arrival order without any optimization.
    """

    def schedule(self) -> ScheduleOutput:
        """Schedule in FIFO order."""
        output = ScheduleOutput()

        # First, check if we have decode requests (continuous batching)
        decode_requests = [
            r for r in self.running_requests.values()
            if r.status == RequestStatus.DECODING
        ]
        if decode_requests:
            output.decode_batch = Batch(
                batch_id=f"decode_{int(time.time() * 1000)}",
                requests=decode_requests[:self.max_batch_size],
                phase=RequestPhase.DECODING,
                created_time=time.time(),
            )

        # Then schedule prefill for waiting requests
        if self.waiting_queue:
            prefill_requests: list[Request] = []
            total_tokens = 0

            for req in self.waiting_queue[:]:
                if len(prefill_requests) >= self.max_batch_size:
                    break
                tokens_needed = req.num_prompt_tokens - req.num_computed_tokens
                if total_tokens + tokens_needed > self.max_tokens:
                    break

                prefill_requests.append(req)
                total_tokens += tokens_needed
                self.waiting_queue.remove(req)
                req.status = RequestStatus.PREFILLING
                req.metadata["prefill_start_time"] = time.time()
                self.running_requests[req.request_id] = req

            if prefill_requests:
                output.prefill_batch = Batch(
                    batch_id=f"prefill_{int(time.time() * 1000)}",
                    requests=prefill_requests,
                    phase=RequestPhase.PREFILLING,
                    created_time=time.time(),
                )

        return output

    def update_after_step(self, finished_request_ids: list[str]) -> None:
        """Update state after execution."""
        for req_id in finished_request_ids:
            req = self.running_requests.pop(req_id, None)
            if req:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                self._completed_requests[req_id] = req

    def prefill_complete(self, request_id: str) -> None:
        """Mark prefill as complete, transition to decode."""
        req = self.running_requests.get(request_id)
        if req and req.status == RequestStatus.PREFILLING:
            req.status = RequestStatus.DECODING
            req.num_computed_tokens = req.num_prompt_tokens
            req.metadata["decode_start_time"] = time.time()
            if req.first_token_time is None:
                req.first_token_time = time.time()


__all__ = [
    # Status enums
    "RequestStatus",
    "RequestPhase",
    "RequestPriority",
    # Data classes
    "Request",
    "Batch",
    "ScheduleOutput",
    "SchedulerConfig",
    # Scheduler classes
    "BaseScheduler",
    "FIFOScheduler",
]
