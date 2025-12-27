# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Execution Graph IR - Intermediate Representation for inference computation.

This module defines the core IR types for representing inference workloads:
- OpType: Operation types (attention, FFN, communication, KV cache)
- TensorRef: Reference to a tensor with shape and dtype
- ExecutionNode: A single operation in the graph
- ExecutionGraph: The complete computation graph with subgraph support

The IR is designed to be:
- Hardware-agnostic: No assumptions about specific accelerator
- Optimizable: Supports graph transformations and fusion
- PD-aware: Native support for Prefill/Decode separation
- Serializable: Can be saved/loaded for caching and distribution

Example:
    >>> from sageLLM.runtime.execution_graph.ir import ExecutionGraph, OpType, TensorRef
    >>> graph = ExecutionGraph(graph_id="prefill_batch_1")
    >>> hidden = TensorRef("hidden", (batch, seq, hidden_size), "float16")
    >>> node = ExecutionNode("attn_0", OpType.PREFILL_ATTN, inputs=[hidden], outputs=[hidden])
    >>> graph.add_node(node)
    >>> graph.prefill_nodes.add("attn_0")

References:
    - vLLM scheduler: https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
    - FlexGen: https://github.com/FMInference/FlexGen
    - DistServe: https://arxiv.org/abs/2401.09670
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class OpType(Enum):
    """Operation types in the execution graph.

    Categorized into:
    - Compute ops: Attention, FFN, embedding, etc.
    - Communication ops: AllReduce, AllGather, Send, Recv
    - KV Cache ops: Load, Store, Migrate
    """

    # ==================== Compute Operations ====================
    # Attention operations
    PREFILL_ATTN = auto()  # Prefill phase attention (compute-intensive)
    DECODE_ATTN = auto()  # Decode phase attention (memory-intensive)
    FLASH_ATTENTION = auto()  # FlashAttention variant
    PAGED_ATTENTION = auto()  # PagedAttention for KV cache

    # FFN operations
    FFN = auto()  # Feed-forward network
    MOE_FFN = auto()  # Mixture of experts FFN

    # Other compute operations
    EMBEDDING = auto()  # Token embedding lookup
    LM_HEAD = auto()  # Language model head (vocab projection)
    LAYERNORM = auto()  # Layer normalization
    ROTARY_EMB = auto()  # Rotary position embedding
    SAMPLING = auto()  # Token sampling

    # ==================== Communication Operations ====================
    COMM_ALLREDUCE = auto()  # AllReduce for tensor parallel
    COMM_ALLGATHER = auto()  # AllGather for tensor parallel
    COMM_REDUCESCATTER = auto()  # ReduceScatter for tensor parallel
    COMM_SEND = auto()  # Point-to-point send
    COMM_RECV = auto()  # Point-to-point receive
    COMM_BROADCAST = auto()  # Broadcast

    # ==================== KV Cache Operations ====================
    KV_LOAD = auto()  # Load KV cache from storage/memory
    KV_STORE = auto()  # Store KV cache to memory
    KV_MIGRATE = auto()  # Migrate KV cache between devices
    KV_SWAP_IN = auto()  # Swap KV cache from CPU to GPU
    KV_SWAP_OUT = auto()  # Swap KV cache from GPU to CPU

    # ==================== Control Operations ====================
    BARRIER = auto()  # Synchronization barrier
    NOP = auto()  # No-op placeholder


# Legacy aliases for backward compatibility
class NodeType(Enum):
    """Legacy node types - use OpType instead."""

    PREFILL = auto()
    CHUNKED_PREFILL = auto()
    DECODE = auto()
    SPECULATIVE_DECODE = auto()
    ATTENTION = auto()
    FLASH_ATTENTION = auto()
    PAGED_ATTENTION = auto()
    FFN = auto()
    MOE_FFN = auto()
    KV_CACHE_LOAD = auto()
    KV_CACHE_STORE = auto()
    KV_CACHE_MIGRATE = auto()
    ALL_REDUCE = auto()
    ALL_GATHER = auto()
    SEND = auto()
    RECV = auto()


class EdgeType(Enum):
    """Types of edges (dependencies) in the execution graph."""

    DATA = auto()  # Data dependency (tensor flow)
    KV_CACHE = auto()  # KV cache dependency
    CONTROL = auto()  # Control dependency (ordering only)
    MEMORY = auto()  # Memory allocation dependency
    COMM = auto()  # Communication dependency


@dataclass
class TensorRef:
    """Reference to a tensor in the execution graph.

    Attributes:
        name: Unique name for the tensor
        shape: Tensor shape as tuple (can contain symbolic dims)
        dtype: Data type string (e.g., "float16", "bfloat16", "int8")
        device_id: Device where the tensor resides
        is_contiguous: Whether the tensor is contiguous in memory
    """

    name: str
    shape: tuple[int | str, ...]
    dtype: str
    device_id: int = 0
    is_contiguous: bool = True

    @property
    def numel(self) -> int | None:
        """Number of elements (None if shape contains symbolic dims)."""
        try:
            result = 1
            for dim in self.shape:
                if isinstance(dim, str):
                    return None
                result *= dim
            return result
        except (TypeError, ValueError):
            return None

    @property
    def size_bytes(self) -> int | None:
        """Size in bytes (None if cannot be computed)."""
        numel = self.numel
        if numel is None:
            return None
        dtype_sizes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int8": 1,
            "int32": 4,
            "int64": 8,
        }
        return numel * dtype_sizes.get(self.dtype, 4)


@dataclass
class ExecutionNode:
    """A node in the execution graph representing a single operation.

    Attributes:
        node_id: Unique identifier for the node
        op_type: Type of operation (from OpType enum)
        inputs: List of input tensor references
        outputs: List of output tensor references
        dependencies: Set of node IDs this node depends on
        device_id: Device for execution
        stream_id: CUDA stream for execution
        estimated_time_us: Estimated execution time in microseconds
        memory_bytes: Memory footprint in bytes
        layer_id: Layer index (for transformer layers)
        metadata: Additional metadata
    """

    node_id: str
    op_type: OpType
    inputs: list[TensorRef] = field(default_factory=list)
    outputs: list[TensorRef] = field(default_factory=list)

    # Dependency tracking
    dependencies: set[str] = field(default_factory=set)

    # Device and stream assignment
    device_id: int = 0
    stream_id: int = 0

    # Performance estimates
    estimated_time_us: float = 0.0
    memory_bytes: int = 0

    # Layer information
    layer_id: int | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility
    @property
    def id(self) -> str:
        """Legacy alias for node_id."""
        return self.node_id

    @property
    def node_type(self) -> OpType:
        """Legacy alias for op_type."""
        return self.op_type

    @property
    def config(self) -> dict[str, Any]:
        """Legacy alias for metadata."""
        return self.metadata

    @property
    def device_hint(self) -> str | None:
        """Legacy compatibility property."""
        return str(self.device_id) if self.device_id is not None else None

    @property
    def priority(self) -> int:
        """Legacy compatibility property."""
        return self.metadata.get("priority", 0)

    @property
    def estimated_latency_ms(self) -> float | None:
        """Legacy compatibility property."""
        return self.estimated_time_us / 1000 if self.estimated_time_us else None

    @property
    def memory_footprint_bytes(self) -> int | None:
        """Legacy compatibility property."""
        return self.memory_bytes if self.memory_bytes else None


@dataclass
class ExecutionEdge:
    """An edge in the execution graph representing a dependency.

    Legacy class for backward compatibility. Prefer using ExecutionNode.dependencies.
    """

    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DATA
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionGraph:
    """The complete execution graph for an inference workload.

    The graph is a DAG where nodes are operations and edges represent
    data/control dependencies. Native support for Prefill/Decode separation.

    Attributes:
        graph_id: Unique identifier for the graph
        nodes: Dictionary of nodes by node_id
        prefill_nodes: Set of node IDs belonging to prefill phase
        decode_nodes: Set of node IDs belonging to decode phase
        comm_nodes: Set of node IDs that are communication operations
        num_layers: Number of transformer layers
        model_config: Model configuration dictionary
        metadata: Additional graph metadata
    """

    graph_id: str
    nodes: dict[str, ExecutionNode] = field(default_factory=dict)

    # Subgraph classification for PD separation
    prefill_nodes: set[str] = field(default_factory=set)
    decode_nodes: set[str] = field(default_factory=set)
    comm_nodes: set[str] = field(default_factory=set)
    kv_nodes: set[str] = field(default_factory=set)

    # Graph properties
    num_layers: int = 0
    model_config: dict[str, Any] | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy: Edge list for backward compatibility
    edges: list[ExecutionEdge] = field(default_factory=list)

    def add_node(
        self,
        node: ExecutionNode | None = None,
        *,
        node_id: str | None = None,
        node_type: OpType | NodeType | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExecutionNode:
        """Add a node to the graph.

        Can be called in two ways:
        1. add_node(ExecutionNode(...)) - pass an ExecutionNode directly
        2. add_node(node_id=..., node_type=...) - legacy API

        Args:
            node: ExecutionNode instance to add
            node_id: Node ID (legacy API)
            node_type: Node type (legacy API)
            config: Configuration dict (legacy API)
            **kwargs: Additional node attributes

        Returns:
            The added ExecutionNode
        """
        if node is not None:
            # New API: add ExecutionNode directly
            self.nodes[node.node_id] = node
            # Auto-classify based on op_type
            self._classify_node(node)
            return node

        # Legacy API
        if node_id is None or node_type is None:
            msg = "Either 'node' or both 'node_id' and 'node_type' must be provided"
            raise ValueError(msg)

        # Convert legacy NodeType to OpType if needed
        op_type = self._convert_node_type(node_type)

        new_node = ExecutionNode(
            node_id=node_id,
            op_type=op_type,
            metadata=config or {},
            **kwargs,
        )
        self.nodes[node_id] = new_node
        self._classify_node(new_node)
        return new_node

    def _convert_node_type(self, node_type: OpType | NodeType) -> OpType:
        """Convert legacy NodeType to OpType."""
        if isinstance(node_type, OpType):
            return node_type
        # Map legacy types
        mapping = {
            NodeType.PREFILL: OpType.PREFILL_ATTN,
            NodeType.CHUNKED_PREFILL: OpType.PREFILL_ATTN,
            NodeType.DECODE: OpType.DECODE_ATTN,
            NodeType.ATTENTION: OpType.PREFILL_ATTN,
            NodeType.FLASH_ATTENTION: OpType.FLASH_ATTENTION,
            NodeType.PAGED_ATTENTION: OpType.PAGED_ATTENTION,
            NodeType.FFN: OpType.FFN,
            NodeType.MOE_FFN: OpType.MOE_FFN,
            NodeType.KV_CACHE_LOAD: OpType.KV_LOAD,
            NodeType.KV_CACHE_STORE: OpType.KV_STORE,
            NodeType.KV_CACHE_MIGRATE: OpType.KV_MIGRATE,
            NodeType.ALL_REDUCE: OpType.COMM_ALLREDUCE,
            NodeType.ALL_GATHER: OpType.COMM_ALLGATHER,
            NodeType.SEND: OpType.COMM_SEND,
            NodeType.RECV: OpType.COMM_RECV,
        }
        return mapping.get(node_type, OpType.NOP)

    def _classify_node(self, node: ExecutionNode) -> None:
        """Auto-classify node into subgraphs based on op_type."""
        op = node.op_type
        if op in {OpType.PREFILL_ATTN, OpType.FLASH_ATTENTION}:
            self.prefill_nodes.add(node.node_id)
        elif op in {OpType.DECODE_ATTN, OpType.PAGED_ATTENTION}:
            self.decode_nodes.add(node.node_id)
        elif op in {
            OpType.COMM_ALLREDUCE,
            OpType.COMM_ALLGATHER,
            OpType.COMM_REDUCESCATTER,
            OpType.COMM_SEND,
            OpType.COMM_RECV,
            OpType.COMM_BROADCAST,
        }:
            self.comm_nodes.add(node.node_id)
        elif op in {
            OpType.KV_LOAD,
            OpType.KV_STORE,
            OpType.KV_MIGRATE,
            OpType.KV_SWAP_IN,
            OpType.KV_SWAP_OUT,
        }:
            self.kv_nodes.add(node.node_id)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType = EdgeType.DATA,
        **metadata: Any,
    ) -> ExecutionEdge | None:
        """Add an edge (dependency) between two nodes.

        Updates the dependencies set of the target node.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            edge_type: Type of dependency
            **metadata: Additional edge metadata

        Returns:
            ExecutionEdge for legacy compatibility, or None if target not found
        """
        if to_node in self.nodes:
            self.nodes[to_node].dependencies.add(from_node)

        # Create edge for legacy compatibility
        edge = ExecutionEdge(
            source_id=from_node,
            target_id=to_node,
            edge_type=edge_type,
            metadata=metadata,
        )
        self.edges.append(edge)
        return edge

    def get_predecessors(self, node_id: str) -> list[str]:
        """Get all predecessor nodes of a given node."""
        if node_id in self.nodes:
            return list(self.nodes[node_id].dependencies)
        return [e.source_id for e in self.edges if e.target_id == node_id]

    def get_successors(self, node_id: str) -> list[str]:
        """Get all successor nodes of a given node."""
        successors = []
        for nid, node in self.nodes.items():
            if node_id in node.dependencies:
                successors.append(nid)
        return successors

    def topological_sort(self) -> list[ExecutionNode]:
        """Return nodes in topological execution order.

        Uses Kahn's algorithm for topological sorting.

        Returns:
            List of ExecutionNode in topological order

        Raises:
            ValueError: If the graph contains cycles
        """
        # Build in-degree map
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in self.nodes:
                    # dep -> node, so node has one more incoming edge
                    pass
            # Count how many nodes depend on each node
        for node_id, node in self.nodes.items():
            in_degree[node_id] = len(
                [d for d in node.dependencies if d in self.nodes]
            )

        # Start with nodes that have no dependencies
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[ExecutionNode] = []

        while queue:
            # Sort by priority for deterministic ordering
            queue.sort(key=lambda x: self.nodes[x].metadata.get("priority", 0),
                       reverse=True)
            node_id = queue.pop(0)
            result.append(self.nodes[node_id])

            # Reduce in-degree for successors
            for succ_id in self.get_successors(node_id):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        if len(result) != len(self.nodes):
            msg = "Graph contains cycles"
            raise ValueError(msg)

        return result

    def get_critical_path(self) -> list[ExecutionNode]:
        """Compute the critical path of the graph.

        The critical path is the longest path through the graph,
        determined by node execution times.

        Returns:
            List of ExecutionNode on the critical path
        """
        if not self.nodes:
            return []

        # Compute longest path to each node using dynamic programming
        sorted_nodes = self.topological_sort()
        longest_dist: dict[str, float] = {n.node_id: 0.0 for n in sorted_nodes}
        predecessor: dict[str, str | None] = {n.node_id: None for n in sorted_nodes}

        for node in sorted_nodes:
            node_time = node.estimated_time_us
            for dep_id in node.dependencies:
                if dep_id in longest_dist:
                    new_dist = longest_dist[dep_id] + node_time
                    if new_dist > longest_dist[node.node_id]:
                        longest_dist[node.node_id] = new_dist
                        predecessor[node.node_id] = dep_id
            # If no dependencies, the path length is just the node time
            if not node.dependencies:
                longest_dist[node.node_id] = node_time

        # Find the node with the longest path
        end_node_id = max(longest_dist, key=lambda x: longest_dist[x])

        # Backtrack to construct the path
        path: list[ExecutionNode] = []
        current: str | None = end_node_id
        while current is not None:
            path.append(self.nodes[current])
            current = predecessor[current]

        return list(reversed(path))

    def estimate_total_time(self) -> float:
        """Estimate total execution time in microseconds.

        Uses critical path length as the estimate (assumes perfect parallelism
        for non-critical path operations).

        Returns:
            Estimated execution time in microseconds
        """
        critical_path = self.get_critical_path()
        return sum(node.estimated_time_us for node in critical_path)

    def get_subgraph(self, node_ids: set[str]) -> ExecutionGraph:
        """Extract a subgraph containing only the specified nodes.

        Args:
            node_ids: Set of node IDs to include

        Returns:
            New ExecutionGraph containing only the specified nodes
        """
        subgraph = ExecutionGraph(graph_id=f"{self.graph_id}_sub")
        subgraph.num_layers = self.num_layers
        subgraph.model_config = self.model_config

        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Create a copy with filtered dependencies
                new_node = ExecutionNode(
                    node_id=node.node_id,
                    op_type=node.op_type,
                    inputs=list(node.inputs),
                    outputs=list(node.outputs),
                    dependencies={d for d in node.dependencies if d in node_ids},
                    device_id=node.device_id,
                    stream_id=node.stream_id,
                    estimated_time_us=node.estimated_time_us,
                    memory_bytes=node.memory_bytes,
                    layer_id=node.layer_id,
                    metadata=dict(node.metadata),
                )
                subgraph.nodes[node_id] = new_node
                subgraph._classify_node(new_node)

        return subgraph

    def get_prefill_subgraph(self) -> ExecutionGraph:
        """Get subgraph containing only prefill nodes."""
        return self.get_subgraph(self.prefill_nodes)

    def get_decode_subgraph(self) -> ExecutionGraph:
        """Get subgraph containing only decode nodes."""
        return self.get_subgraph(self.decode_nodes)

    def get_comm_subgraph(self) -> ExecutionGraph:
        """Get subgraph containing only communication nodes."""
        return self.get_subgraph(self.comm_nodes)

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes


__all__ = [
    # Core types
    "OpType",
    "TensorRef",
    "ExecutionNode",
    "ExecutionEdge",
    "ExecutionGraph",
    # Legacy compatibility
    "NodeType",
    "EdgeType",
]
