# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Execution Graph Optimizer - Optimization passes for execution graphs.

This module provides optimization passes that transform execution graphs:
- CommunicationFusionPass: Fuse multiple small AllReduce operations
- ComputeCommOverlapPass: Overlap computation and communication
- KVPrefetchPass: Prefetch KV cache for next layer
- MemoryOptimizationPass: Optimize memory allocation and reuse

Example:
    >>> from sageLLM.runtime.execution_graph.optimizer import GraphOptimizer
    >>> optimizer = GraphOptimizer()
    >>> optimized_graph = optimizer.optimize(graph)

References:
    - ZeRO: Memory Optimization Towards Training Trillion Parameter Models
    - Megatron-LM: Training Multi-Billion Parameter Language Models
    - FlashAttention: Fast and Memory-Efficient Exact Attention
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .ir import ExecutionGraph, ExecutionNode, OpType

if TYPE_CHECKING:
    pass


class OptimizationPass(ABC):
    """Base class for optimization passes.

    An optimization pass transforms an execution graph to improve
    performance, memory usage, or both.

    Subclasses must implement:
    - name: Return the name of the pass
    - apply: Apply the optimization to a graph
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this pass."""

    @abstractmethod
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Apply the optimization pass on a graph.

        Args:
            graph: The input execution graph

        Returns:
            The optimized graph (may be modified in place or copied)
        """

    # Legacy alias
    def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Legacy alias for apply()."""
        return self.apply(graph)


class CommunicationFusionPass(OptimizationPass):
    """Fuse multiple communication operations to reduce overhead.

    This pass identifies sequences of AllReduce/AllGather operations
    that can be fused into a single larger operation, reducing:
    - Kernel launch overhead
    - Synchronization overhead
    - Network protocol overhead

    Fusion rules:
    - Same communication type (e.g., AllReduce + AllReduce)
    - No data dependency between them
    - Total fused size below threshold
    """

    def __init__(self, max_fused_size_mb: float = 256.0) -> None:
        """Initialize the pass.

        Args:
            max_fused_size_mb: Maximum size of fused communication in MB
        """
        self.max_fused_size_bytes = int(max_fused_size_mb * 1024 * 1024)

    @property
    def name(self) -> str:
        return "comm_fusion"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Fuse compatible adjacent communication operations."""
        if len(graph.comm_nodes) < 2:
            return graph

        # Find fusable comm node groups
        fusable_groups = self._find_fusable_groups(graph)

        if not fusable_groups:
            return graph

        # Create a copy to modify
        optimized = self._copy_graph(graph)

        for group in fusable_groups:
            if len(group) < 2:
                continue
            self._fuse_comm_nodes(optimized, group)

        return optimized

    def _find_fusable_groups(
        self,
        graph: ExecutionGraph,
    ) -> list[list[str]]:
        """Find groups of communication nodes that can be fused."""
        groups: list[list[str]] = []
        visited: set[str] = set()

        # Group by comm type and layer
        comm_by_type: dict[OpType, list[str]] = {}
        for node_id in graph.comm_nodes:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            comm_type = node.op_type
            if comm_type not in comm_by_type:
                comm_by_type[comm_type] = []
            comm_by_type[comm_type].append(node_id)

        # For each type, find independent nodes that can be fused
        for comm_type, node_ids in comm_by_type.items():
            current_group: list[str] = []
            current_size = 0

            for node_id in node_ids:
                if node_id in visited:
                    continue

                node = graph.nodes[node_id]
                node_size = node.memory_bytes or 0

                # Check if can add to current group
                if (current_size + node_size <= self.max_fused_size_bytes
                    and self._can_fuse_with_group(graph, node_id, current_group)):
                    current_group.append(node_id)
                    current_size += node_size
                    visited.add(node_id)
                else:
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [node_id]
                    current_size = node_size
                    visited.add(node_id)

            if len(current_group) >= 2:
                groups.append(current_group)

        return groups

    def _can_fuse_with_group(
        self,
        graph: ExecutionGraph,
        node_id: str,
        group: list[str],
    ) -> bool:
        """Check if a node can be fused with an existing group."""
        if not group:
            return True

        node = graph.nodes[node_id]

        # Check no data dependencies within the group
        for other_id in group:
            if node_id in graph.nodes[other_id].dependencies:
                return False
            if other_id in node.dependencies:
                return False

        return True

    def _fuse_comm_nodes(
        self,
        graph: ExecutionGraph,
        node_ids: list[str],
    ) -> None:
        """Fuse a group of communication nodes into one."""
        if len(node_ids) < 2:
            return

        # Use the first node as the base
        base_node = graph.nodes[node_ids[0]]
        fused_id = f"fused_comm_{node_ids[0]}"

        # Merge dependencies and metadata
        all_deps: set[str] = set()
        total_size = 0
        total_time = 0.0

        for node_id in node_ids:
            node = graph.nodes[node_id]
            all_deps.update(node.dependencies)
            total_size += node.memory_bytes or 0
            total_time = max(total_time, node.estimated_time_us)

        # Remove internal dependencies
        all_deps -= set(node_ids)

        # Create fused node
        fused_node = ExecutionNode(
            node_id=fused_id,
            op_type=base_node.op_type,
            dependencies=all_deps,
            estimated_time_us=total_time * 1.1,  # Slight overhead for fusion
            memory_bytes=total_size,
            metadata={
                "fused_from": node_ids,
                "fused_count": len(node_ids),
            },
        )

        # Update graph
        graph.nodes[fused_id] = fused_node
        graph.comm_nodes.add(fused_id)

        # Update successors to depend on fused node
        for node_id in node_ids:
            for succ_id in graph.get_successors(node_id):
                if succ_id in graph.nodes:
                    graph.nodes[succ_id].dependencies.discard(node_id)
                    graph.nodes[succ_id].dependencies.add(fused_id)

        # Remove original nodes
        for node_id in node_ids:
            if node_id in graph.nodes:
                del graph.nodes[node_id]
            graph.comm_nodes.discard(node_id)

    def _copy_graph(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Create a deep copy of the graph."""
        return copy.deepcopy(graph)


class ComputeCommOverlapPass(OptimizationPass):
    """Overlap computation and communication operations.

    This pass identifies opportunities to run communication operations
    in parallel with independent computation operations by:
    - Assigning them to different CUDA streams
    - Adjusting dependencies to enable overlap

    Benefits:
    - Hide communication latency behind computation
    - Improve GPU utilization
    """

    @property
    def name(self) -> str:
        return "compute_comm_overlap"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Adjust stream assignments to enable overlap."""
        if not graph.comm_nodes:
            return graph

        optimized = copy.deepcopy(graph)

        # Find overlappable compute-comm pairs
        pairs = self._find_overlappable_pairs(optimized)

        for compute_id, comm_id in pairs:
            self._enable_overlap(optimized, compute_id, comm_id)

        return optimized

    def _find_overlappable_pairs(
        self,
        graph: ExecutionGraph,
    ) -> list[tuple[str, str]]:
        """Find pairs of (compute, comm) nodes that can overlap."""
        pairs: list[tuple[str, str]] = []

        for comm_id in graph.comm_nodes:
            comm_node = graph.nodes.get(comm_id)
            if comm_node is None:
                continue

            # Find compute nodes that can run in parallel
            for node_id, node in graph.nodes.items():
                if node_id == comm_id:
                    continue
                if node_id in graph.comm_nodes:
                    continue

                # Check if they are independent
                if self._are_independent(graph, node_id, comm_id):
                    # Check if compute node has significant runtime
                    if node.estimated_time_us >= comm_node.estimated_time_us * 0.5:
                        pairs.append((node_id, comm_id))
                        break  # One pair per comm node

        return pairs

    def _are_independent(
        self,
        graph: ExecutionGraph,
        node_a: str,
        node_b: str,
    ) -> bool:
        """Check if two nodes are independent (no path between them)."""
        # Check direct dependencies
        node_a_obj = graph.nodes.get(node_a)
        node_b_obj = graph.nodes.get(node_b)

        if node_a_obj is None or node_b_obj is None:
            return False

        if node_b in node_a_obj.dependencies or node_a in node_b_obj.dependencies:
            return False

        # Check transitive dependencies (simplified: only direct check)
        return True

    def _enable_overlap(
        self,
        graph: ExecutionGraph,
        compute_id: str,
        comm_id: str,
    ) -> None:
        """Enable overlap between compute and comm nodes."""
        compute_node = graph.nodes.get(compute_id)
        comm_node = graph.nodes.get(comm_id)

        if compute_node is None or comm_node is None:
            return

        # Assign different streams
        compute_node.stream_id = 0  # Default compute stream
        comm_node.stream_id = 1  # Communication stream

        # Update metadata
        compute_node.metadata["overlapped_with"] = comm_id
        comm_node.metadata["overlapped_with"] = compute_id


class KVPrefetchPass(OptimizationPass):
    """Prefetch KV cache for the next layer.

    This pass inserts KV cache prefetch operations that run in parallel
    with the current layer's computation, hiding memory latency.

    Strategy:
    - While computing layer N, prefetch KV for layer N+1
    - Use separate memory stream for prefetch
    """

    @property
    def name(self) -> str:
        return "kv_prefetch"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Insert KV prefetch operations."""
        if not graph.kv_nodes:
            return graph

        optimized = copy.deepcopy(graph)

        # Group KV load nodes by layer
        kv_loads_by_layer: dict[int, list[str]] = {}
        for node_id in graph.kv_nodes:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if node.op_type == OpType.KV_LOAD and node.layer_id is not None:
                layer = node.layer_id
                if layer not in kv_loads_by_layer:
                    kv_loads_by_layer[layer] = []
                kv_loads_by_layer[layer].append(node_id)

        # For each layer (except last), create prefetch for next layer
        sorted_layers = sorted(kv_loads_by_layer.keys())
        for i, layer_id in enumerate(sorted_layers[:-1]):
            next_layer = sorted_layers[i + 1]
            self._insert_prefetch(
                optimized,
                layer_id,
                next_layer,
                kv_loads_by_layer.get(next_layer, []),
            )

        return optimized

    def _insert_prefetch(
        self,
        graph: ExecutionGraph,
        current_layer: int,
        next_layer: int,
        kv_load_ids: list[str],
    ) -> None:
        """Insert prefetch for next layer KV loads."""
        if not kv_load_ids:
            return

        # Find the attention node for current layer
        current_attn_id = None
        for node_id, node in graph.nodes.items():
            if (node.layer_id == current_layer and
                node.op_type in {OpType.PREFILL_ATTN, OpType.DECODE_ATTN}):
                current_attn_id = node_id
                break

        if current_attn_id is None:
            return

        # Mark KV loads as prefetchable
        for kv_id in kv_load_ids:
            kv_node = graph.nodes.get(kv_id)
            if kv_node is None:
                continue

            # Add prefetch metadata
            kv_node.metadata["prefetch_after"] = current_attn_id
            kv_node.stream_id = 2  # Prefetch stream

            # Update dependencies to allow earlier start
            # Only depend on the previous layer's attention, not FFN
            original_deps = set(kv_node.dependencies)
            for dep_id in original_deps:
                dep_node = graph.nodes.get(dep_id)
                if dep_node and dep_node.op_type == OpType.FFN:
                    # Remove FFN dependency to allow prefetch
                    kv_node.dependencies.discard(dep_id)
                    kv_node.metadata["relaxed_dep"] = dep_id


class MemoryOptimizationPass(OptimizationPass):
    """Optimize memory allocation and tensor reuse.

    This pass analyzes tensor lifetimes and plans memory allocation
    to minimize peak memory usage through:
    - Tensor reuse (reuse memory for non-overlapping tensors)
    - Activation checkpointing hints
    - Memory pool planning
    """

    @property
    def name(self) -> str:
        return "memory_opt"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Analyze and optimize memory usage."""
        optimized = copy.deepcopy(graph)

        # 1. Compute tensor lifetimes
        lifetimes = self._compute_lifetimes(optimized)

        # 2. Plan memory reuse
        reuse_plan = self._plan_memory_reuse(lifetimes)

        # 3. Annotate nodes with memory plan
        self._annotate_memory_plan(optimized, reuse_plan)

        return optimized

    def _compute_lifetimes(
        self,
        graph: ExecutionGraph,
    ) -> dict[str, tuple[int, int]]:
        """Compute lifetime (first_use, last_use) for each tensor."""
        lifetimes: dict[str, tuple[int, int]] = {}

        try:
            sorted_nodes = graph.topological_sort()
        except ValueError:
            # Graph has cycles, return empty lifetimes
            return lifetimes

        for order, node in enumerate(sorted_nodes):
            # Mark outputs as created
            for output in node.outputs:
                if output.name not in lifetimes:
                    lifetimes[output.name] = (order, order)

            # Mark inputs as used
            for input_tensor in node.inputs:
                if input_tensor.name in lifetimes:
                    first_use, _ = lifetimes[input_tensor.name]
                    lifetimes[input_tensor.name] = (first_use, order)

        return lifetimes

    def _plan_memory_reuse(
        self,
        lifetimes: dict[str, tuple[int, int]],
    ) -> dict[str, str]:
        """Plan which tensors can reuse memory from other tensors."""
        reuse_plan: dict[str, str] = {}
        sorted_tensors = sorted(lifetimes.items(), key=lambda x: x[1][0])

        for i, (tensor_name, (start, end)) in enumerate(sorted_tensors):
            # Find a tensor that ended before this one started
            for prev_name, (prev_start, prev_end) in sorted_tensors[:i]:
                if prev_end < start and prev_name not in reuse_plan.values():
                    reuse_plan[tensor_name] = prev_name
                    break

        return reuse_plan

    def _annotate_memory_plan(
        self,
        graph: ExecutionGraph,
        reuse_plan: dict[str, str],
    ) -> None:
        """Annotate nodes with memory reuse information."""
        for node in graph.nodes.values():
            for output in node.outputs:
                if output.name in reuse_plan:
                    node.metadata["memory_reuse"] = {
                        "output": output.name,
                        "reuse_from": reuse_plan[output.name],
                    }


# ======================== Legacy Pass Classes ========================
# These are kept for backward compatibility with existing code


class FusionPass(OptimizationPass):
    """Legacy fusion pass - delegates to CommunicationFusionPass."""

    @property
    def name(self) -> str:
        return "fusion"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Apply communication fusion."""
        return CommunicationFusionPass().apply(graph)


class MemoryPlanningPass(OptimizationPass):
    """Legacy memory planning pass - delegates to MemoryOptimizationPass."""

    @property
    def name(self) -> str:
        return "memory_planning"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Apply memory optimization."""
        return MemoryOptimizationPass().apply(graph)


class DevicePlacementPass(OptimizationPass):
    """Device placement pass - assigns nodes to devices."""

    @property
    def name(self) -> str:
        return "device_placement"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Assign device placement for each operation.

        Simple strategy:
        - Prefill nodes to prefill devices
        - Decode nodes to decode devices
        - Comm nodes follow their dependent compute nodes
        """
        # For now, this is a placeholder that respects existing device hints
        return graph


class SchedulingPass(OptimizationPass):
    """Scheduling pass - reorders operations for parallelism."""

    @property
    def name(self) -> str:
        return "scheduling"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Reorder operations within topological constraints.

        Delegates to ComputeCommOverlapPass for overlap optimization.
        """
        return ComputeCommOverlapPass().apply(graph)


# ======================== Main Optimizer Class ========================


@dataclass
class GraphOptimizer:
    """Optimizer that runs a sequence of optimization passes on a graph.

    Example:
        >>> optimizer = GraphOptimizer()
        >>> optimized = optimizer.optimize(graph)

        >>> # Custom passes
        >>> optimizer = GraphOptimizer(passes=[
        ...     CommunicationFusionPass(),
        ...     ComputeCommOverlapPass(),
        ... ])
        >>> optimized = optimizer.optimize(graph)
    """

    passes: list[OptimizationPass] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        """Initialize with default passes if none provided."""
        if not self.passes:
            self.passes = [
                CommunicationFusionPass(),
                ComputeCommOverlapPass(),
                KVPrefetchPass(),
                MemoryOptimizationPass(),
            ]

    def add_pass(self, opt_pass: OptimizationPass) -> None:
        """Add an optimization pass to the pipeline."""
        self.passes.append(opt_pass)

    def remove_pass(self, pass_name: str) -> bool:
        """Remove a pass by name.

        Args:
            pass_name: Name of the pass to remove

        Returns:
            True if a pass was removed, False otherwise
        """
        for i, p in enumerate(self.passes):
            if p.name == pass_name:
                self.passes.pop(i)
                return True
        return False

    def optimize(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Run all optimization passes on the graph.

        Args:
            graph: Input execution graph

        Returns:
            Optimized execution graph
        """
        current_graph = graph
        for opt_pass in self.passes:
            if self.verbose:
                print(f"Running optimization pass: {opt_pass.name}")
            current_graph = opt_pass.apply(current_graph)
        return current_graph

    @classmethod
    def default(cls) -> GraphOptimizer:
        """Create an optimizer with default passes."""
        return cls(
            passes=[
                CommunicationFusionPass(),
                ComputeCommOverlapPass(),
                KVPrefetchPass(),
                MemoryOptimizationPass(),
            ]
        )

    @classmethod
    def minimal(cls) -> GraphOptimizer:
        """Create an optimizer with minimal passes (fast but less optimized)."""
        return cls(
            passes=[
                CommunicationFusionPass(),
            ]
        )

    @classmethod
    def aggressive(cls) -> GraphOptimizer:
        """Create an optimizer with all available passes."""
        return cls(
            passes=[
                CommunicationFusionPass(max_fused_size_mb=512.0),
                ComputeCommOverlapPass(),
                KVPrefetchPass(),
                MemoryOptimizationPass(),
            ]
        )


__all__ = [
    # Base class
    "OptimizationPass",
    # Core passes
    "CommunicationFusionPass",
    "ComputeCommOverlapPass",
    "KVPrefetchPass",
    "MemoryOptimizationPass",
    # Legacy passes
    "FusionPass",
    "MemoryPlanningPass",
    "DevicePlacementPass",
    "SchedulingPass",
    # Optimizer
    "GraphOptimizer",
]
