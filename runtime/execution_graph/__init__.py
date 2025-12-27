# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Execution Graph Module - IR definition, building, and optimization for PD/AF separated inference.

Components:
- ir: Execution graph intermediate representation (IR) definition
- builder: Graph construction utilities
- optimizer: Graph optimization passes (fusion, memory planning, etc.)

The execution graph represents the inference computation as a DAG of operations,
enabling hardware-agnostic scheduling and optimization.

Example:
    >>> from sageLLM.runtime.execution_graph import (
    ...     ExecutionGraphBuilder, ModelConfig, ParallelConfig, PrefillBatch
    ... )
    >>> config = ModelConfig(num_layers=32, hidden_size=4096, ...)
    >>> builder = ExecutionGraphBuilder(config, ParallelConfig())
    >>> batch = PrefillBatch(request_ids=["r1"], input_lengths=[128], total_tokens=128)
    >>> graph = builder.build_prefill_graph(batch)
"""

from __future__ import annotations

from .builder import (
    DecodeBatch,
    ExecutionGraphBuilder,
    GraphBuilder,
    ModelConfig,
    ParallelConfig,
    PrefillBatch,
    PrefillDecodeBuilder,
)
from .ir import (
    EdgeType,
    ExecutionEdge,
    ExecutionGraph,
    ExecutionNode,
    NodeType,
    OpType,
    TensorRef,
)
from .optimizer import (
    CommunicationFusionPass,
    ComputeCommOverlapPass,
    GraphOptimizer,
    KVPrefetchPass,
    MemoryOptimizationPass,
    OptimizationPass,
)

__all__ = [
    # IR types
    "OpType",
    "NodeType",
    "EdgeType",
    "TensorRef",
    "ExecutionNode",
    "ExecutionEdge",
    "ExecutionGraph",
    # Builder types
    "ModelConfig",
    "ParallelConfig",
    "PrefillBatch",
    "DecodeBatch",
    "ExecutionGraphBuilder",
    "GraphBuilder",
    "PrefillDecodeBuilder",
    # Optimizer types
    "OptimizationPass",
    "CommunicationFusionPass",
    "ComputeCommOverlapPass",
    "KVPrefetchPass",
    "MemoryOptimizationPass",
    "GraphOptimizer",
]
