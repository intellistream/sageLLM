# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
sageLLM Runtime - Self-developed inference runtime.

This module provides the core inference runtime for sageLLM, including:
- execution_graph: PD/AF separated execution graph IR and optimization
- comm: Communication layer for distributed inference
- scheduler: Request scheduling and PD separation

Architecture:
    The runtime is designed to support hardware-agnostic inference with:
    - Execution Graph IR for representing inference computation
    - Communication primitives for multi-node/multi-device coordination
    - Schedulers for efficient request batching and PD separation

Example:
    >>> from sageLLM.runtime.execution_graph import (
    ...     ExecutionGraphBuilder, ModelConfig, ParallelConfig, PrefillBatch
    ... )
    >>> from sageLLM.runtime.scheduler import PDScheduler, PDSchedulerConfig
    >>>
    >>> # Build execution graph
    >>> config = ModelConfig(num_layers=32, hidden_size=4096, ...)
    >>> builder = ExecutionGraphBuilder(config, ParallelConfig())
    >>> graph = builder.build_prefill_graph(PrefillBatch(...))
    >>>
    >>> # Schedule requests
    >>> scheduler = PDScheduler(PDSchedulerConfig(mode="hybrid"))
    >>> scheduler.add_request(Request(...))
    >>> output = scheduler.schedule()
"""

from __future__ import annotations

# Re-export submodules
from . import comm, execution_graph, scheduler

# Re-export key classes for convenience
from .execution_graph import (
    ExecutionGraph,
    ExecutionGraphBuilder,
    ExecutionNode,
    GraphOptimizer,
    ModelConfig,
    OpType,
    ParallelConfig,
    TensorRef,
)
from .scheduler import (
    BaseScheduler,
    Batch,
    FIFOScheduler,
    PDScheduler,
    PDSchedulerConfig,
    Request,
    RequestStatus,
    ScheduleOutput,
)

__all__ = [
    # Submodules
    "execution_graph",
    "comm",
    "scheduler",
    # Execution Graph
    "OpType",
    "TensorRef",
    "ExecutionNode",
    "ExecutionGraph",
    "ModelConfig",
    "ParallelConfig",
    "ExecutionGraphBuilder",
    "GraphOptimizer",
    # Scheduler
    "RequestStatus",
    "Request",
    "Batch",
    "ScheduleOutput",
    "BaseScheduler",
    "FIFOScheduler",
    "PDSchedulerConfig",
    "PDScheduler",
]
