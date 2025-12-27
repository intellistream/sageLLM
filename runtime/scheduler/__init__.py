# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler Module - Request scheduling for inference runtime.

Components:
- base: Scheduler base class and common interfaces
- pd_scheduler: Prefill-Decode separation scheduler

The scheduler is responsible for:
- Batching requests for efficient execution
- Managing prefill-decode separation (PD disaggregation)
- Priority-based scheduling
- SLO-aware resource allocation

Example:
    >>> from sageLLM.runtime.scheduler import PDScheduler, PDSchedulerConfig, Request
    >>> config = PDSchedulerConfig(
    ...     mode="hybrid",
    ...     prefill_chunk_size=512,
    ...     max_decode_batch_size=256,
    ... )
    >>> scheduler = PDScheduler(config)
    >>> scheduler.add_request(Request(request_id="r1", prompt_token_ids=[1,2,3,4]))
    >>> output = scheduler.schedule()
    >>> if output.prefill_batch:
    ...     print(f"Prefill batch: {output.prefill_batch.batch_size} requests")
"""

from __future__ import annotations

# Re-export submodules
from . import base, pd_scheduler

# Re-export base types
from .base import (
    BaseScheduler,
    Batch,
    FIFOScheduler,
    Request,
    RequestPhase,
    RequestPriority,
    RequestStatus,
    ScheduleOutput,
    SchedulerConfig,
)

# Re-export PD scheduler types
from .pd_scheduler import (
    KVMigrationTask,
    PDScheduler,
    PDSchedulerConfig,
)

__all__ = [
    # Submodules
    "base",
    "pd_scheduler",
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
    # PD scheduler
    "PDSchedulerConfig",
    "PDScheduler",
    "KVMigrationTask",
]
