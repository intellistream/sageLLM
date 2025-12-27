# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Execution Graph Builder - Utilities for constructing execution graphs.

This module provides builder patterns for creating execution graphs:
- ModelConfig: Model architecture configuration
- ParallelConfig: Parallelism configuration (TP, PP, DP)
- PrefillBatch/DecodeBatch: Batch information
- ExecutionGraphBuilder: Main builder for constructing graphs

The builder supports:
- Prefill graph construction (compute-intensive)
- Decode graph construction (memory-intensive)
- Hybrid graph construction (chunked prefill)

Example:
    >>> from sageLLM.runtime.execution_graph.builder import (
    ...     ExecutionGraphBuilder, ModelConfig, ParallelConfig, PrefillBatch
    ... )
    >>> config = ModelConfig(num_layers=32, hidden_size=4096, ...)
    >>> builder = ExecutionGraphBuilder(config, ParallelConfig())
    >>> batch = PrefillBatch(request_ids=["r1"], input_lengths=[128], total_tokens=128)
    >>> graph = builder.build_prefill_graph(batch)

References:
    - vLLM: https://github.com/vllm-project/vllm
    - FlexGen: https://github.com/FMInference/FlexGen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import (
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    NodeType,
    OpType,
    TensorRef,
)


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA/MQA)
        intermediate_size: FFN intermediate dimension
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        head_dim: Per-head dimension (computed if not provided)
        dtype: Data type for computation
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_len: int = 4096
    head_dim: int | None = None
    dtype: str = "float16"

    def __post_init__(self) -> None:
        """Compute derived attributes."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class ParallelConfig:
    """Parallelism configuration.

    Attributes:
        tensor_parallel_size: Number of tensor parallel ranks
        pipeline_parallel_size: Number of pipeline parallel stages
        data_parallel_size: Number of data parallel replicas
        world_size: Total number of processes
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        """Total number of processes."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
        )

    @property
    def needs_communication(self) -> bool:
        """Whether communication ops are needed."""
        return self.tensor_parallel_size > 1 or self.pipeline_parallel_size > 1


@dataclass
class PrefillBatch:
    """Prefill batch information.

    Attributes:
        request_ids: List of request identifiers
        input_lengths: Input sequence lengths per request
        total_tokens: Total number of tokens in the batch
    """

    request_ids: list[str]
    input_lengths: list[int]
    total_tokens: int

    @property
    def batch_size(self) -> int:
        """Number of requests in the batch."""
        return len(self.request_ids)


@dataclass
class DecodeBatch:
    """Decode batch information.

    Attributes:
        request_ids: List of request identifiers
        context_lengths: Current context lengths per request
        batch_size: Number of requests in the batch
    """

    request_ids: list[str]
    context_lengths: list[int]
    batch_size: int


class ExecutionGraphBuilder:
    """Builder for constructing execution graphs.

    Supports building:
    - Prefill graphs: Compute-intensive, full attention over input
    - Decode graphs: Memory-intensive, incremental KV cache access
    - Hybrid graphs: Chunked prefill interleaved with decode

    Performance estimation is based on:
    - Prefill: ~O(seq_len^2) for attention
    - Decode: ~O(context_len) for KV cache access
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        """Initialize the builder.

        Args:
            model_config: Model architecture configuration
            parallel_config: Parallelism configuration
        """
        self.model_config = model_config
        self.parallel_config = parallel_config
        self._node_counter = 0

    def _next_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def build_prefill_graph(self, batch: PrefillBatch) -> ExecutionGraph:
        """Build execution graph for prefill phase.

        Prefill characteristics:
        - Processes complete input sequences
        - Compute-intensive (matrix multiplications)
        - Generates all KV cache entries at once

        Graph structure per layer:
        1. LayerNorm
        2. Self-Attention (Q, K, V projection + attention + output projection)
        3. AllReduce (if TP > 1)
        4. LayerNorm
        5. FFN (up projection + activation + down projection)
        6. AllReduce (if TP > 1)

        Args:
            batch: Prefill batch information

        Returns:
            ExecutionGraph for prefill execution
        """
        graph = ExecutionGraph(
            graph_id=f"prefill_{batch.total_tokens}",
            num_layers=self.model_config.num_layers,
            model_config={
                "hidden_size": self.model_config.hidden_size,
                "num_attention_heads": self.model_config.num_attention_heads,
                "num_kv_heads": self.model_config.num_kv_heads,
            },
        )

        # Input embedding
        embed_id = self._next_id("embed")
        embed_node = self._build_embedding_node(embed_id, batch.total_tokens)
        graph.add_node(embed_node)
        prev_node_id = embed_id

        # Build layers
        for layer_idx in range(self.model_config.num_layers):
            layer_nodes = self._build_prefill_layer(
                layer_idx,
                batch,
                prev_node_id,
            )
            for node in layer_nodes:
                graph.add_node(node)
                if prev_node_id:
                    graph.add_edge(prev_node_id, node.node_id, EdgeType.DATA)
                prev_node_id = node.node_id

        # Output head
        lm_head_id = self._next_id("lm_head")
        lm_head_node = self._build_lm_head_node(lm_head_id, batch.total_tokens)
        graph.add_node(lm_head_node)
        graph.add_edge(prev_node_id, lm_head_id, EdgeType.DATA)

        return graph

    def build_decode_graph(self, batch: DecodeBatch) -> ExecutionGraph:
        """Build execution graph for decode phase.

        Decode characteristics:
        - Processes one token per request
        - Memory-bandwidth intensive (KV cache access)
        - Incremental token generation

        Graph structure per layer:
        1. LayerNorm
        2. Self-Attention with KV cache load
        3. KV cache store
        4. AllReduce (if TP > 1)
        5. LayerNorm
        6. FFN
        7. AllReduce (if TP > 1)

        Args:
            batch: Decode batch information

        Returns:
            ExecutionGraph for decode execution
        """
        graph = ExecutionGraph(
            graph_id=f"decode_{batch.batch_size}",
            num_layers=self.model_config.num_layers,
            model_config={
                "hidden_size": self.model_config.hidden_size,
                "num_attention_heads": self.model_config.num_attention_heads,
                "num_kv_heads": self.model_config.num_kv_heads,
            },
        )

        # Input embedding (single token per request)
        embed_id = self._next_id("embed")
        embed_node = self._build_embedding_node(embed_id, batch.batch_size)
        graph.add_node(embed_node)
        prev_node_id = embed_id

        # Build layers
        for layer_idx in range(self.model_config.num_layers):
            layer_nodes = self._build_decode_layer(
                layer_idx,
                batch,
                prev_node_id,
            )
            for node in layer_nodes:
                graph.add_node(node)
                if prev_node_id:
                    graph.add_edge(prev_node_id, node.node_id, EdgeType.DATA)
                prev_node_id = node.node_id

        # Output head
        lm_head_id = self._next_id("lm_head")
        lm_head_node = self._build_lm_head_node(lm_head_id, batch.batch_size)
        graph.add_node(lm_head_node)
        graph.add_edge(prev_node_id, lm_head_id, EdgeType.DATA)

        # Sampling
        sample_id = self._next_id("sample")
        sample_node = ExecutionNode(
            node_id=sample_id,
            op_type=OpType.SAMPLING,
            metadata={"batch_size": batch.batch_size},
            estimated_time_us=10.0,  # Sampling is fast
        )
        graph.add_node(sample_node)
        graph.add_edge(lm_head_id, sample_id, EdgeType.DATA)

        return graph

    def build_hybrid_graph(
        self,
        prefill_batch: PrefillBatch,
        decode_batch: DecodeBatch,
    ) -> ExecutionGraph:
        """Build hybrid graph for chunked prefill + decode.

        Hybrid mode (Sarathi-style chunked prefill):
        - Prefill is split into chunks
        - Chunks are interleaved with decode iterations
        - Balances TTFT and throughput

        Args:
            prefill_batch: Prefill batch information
            decode_batch: Decode batch information

        Returns:
            ExecutionGraph for hybrid execution
        """
        graph = ExecutionGraph(
            graph_id=f"hybrid_p{prefill_batch.total_tokens}_d{decode_batch.batch_size}",
            num_layers=self.model_config.num_layers,
        )

        # Combined embedding
        total_tokens = prefill_batch.total_tokens + decode_batch.batch_size
        embed_id = self._next_id("embed")
        embed_node = self._build_embedding_node(embed_id, total_tokens)
        graph.add_node(embed_node)
        prev_node_id = embed_id

        # Build layers with mixed attention
        for layer_idx in range(self.model_config.num_layers):
            # Prefill attention
            prefill_attn_id = self._next_id(f"prefill_attn_L{layer_idx}")
            prefill_attn_node = self._build_attention_node(
                prefill_attn_id,
                layer_idx,
                is_prefill=True,
                batch_info={
                    "batch_size": prefill_batch.batch_size,
                    "seq_len": prefill_batch.total_tokens // prefill_batch.batch_size,
                },
            )
            graph.add_node(prefill_attn_node)
            graph.add_edge(prev_node_id, prefill_attn_id, EdgeType.DATA)
            graph.prefill_nodes.add(prefill_attn_id)

            # Decode attention (can run in parallel on different requests)
            decode_attn_id = self._next_id(f"decode_attn_L{layer_idx}")
            decode_attn_node = self._build_attention_node(
                decode_attn_id,
                layer_idx,
                is_prefill=False,
                batch_info={
                    "batch_size": decode_batch.batch_size,
                    "context_lengths": decode_batch.context_lengths,
                },
            )
            graph.add_node(decode_attn_node)
            graph.add_edge(prev_node_id, decode_attn_id, EdgeType.DATA)
            graph.decode_nodes.add(decode_attn_id)

            # FFN (processes both)
            ffn_id = self._next_id(f"ffn_L{layer_idx}")
            ffn_node = self._build_ffn_node(
                ffn_id,
                layer_idx,
                {"batch_size": prefill_batch.batch_size + decode_batch.batch_size},
            )
            graph.add_node(ffn_node)
            graph.add_edge(prefill_attn_id, ffn_id, EdgeType.DATA)
            graph.add_edge(decode_attn_id, ffn_id, EdgeType.DATA)

            # Communication if needed
            if self.parallel_config.needs_communication:
                comm_id = self._next_id(f"comm_L{layer_idx}")
                comm_node = self._build_comm_node(
                    comm_id,
                    layer_idx,
                    OpType.COMM_ALLREDUCE,
                )
                graph.add_node(comm_node)
                graph.add_edge(ffn_id, comm_id, EdgeType.DATA)
                prev_node_id = comm_id
            else:
                prev_node_id = ffn_id

        # Output head
        lm_head_id = self._next_id("lm_head")
        lm_head_node = self._build_lm_head_node(lm_head_id, total_tokens)
        graph.add_node(lm_head_node)
        graph.add_edge(prev_node_id, lm_head_id, EdgeType.DATA)

        return graph

    def _build_prefill_layer(
        self,
        layer_idx: int,
        batch: PrefillBatch,
        prev_node_id: str,
    ) -> list[ExecutionNode]:
        """Build nodes for a single prefill layer."""
        nodes: list[ExecutionNode] = []
        batch_info = {
            "batch_size": batch.batch_size,
            "seq_len": batch.total_tokens // max(batch.batch_size, 1),
            "total_tokens": batch.total_tokens,
        }

        # LayerNorm
        ln1_id = self._next_id(f"ln1_L{layer_idx}")
        ln1_node = ExecutionNode(
            node_id=ln1_id,
            op_type=OpType.LAYERNORM,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_layernorm_time(batch.total_tokens),
        )
        nodes.append(ln1_node)

        # Attention
        attn_id = self._next_id(f"attn_L{layer_idx}")
        attn_node = self._build_attention_node(
            attn_id,
            layer_idx,
            is_prefill=True,
            batch_info=batch_info,
        )
        nodes.append(attn_node)

        # KV Store
        kv_store_id = self._next_id(f"kv_store_L{layer_idx}")
        kv_store_node = ExecutionNode(
            node_id=kv_store_id,
            op_type=OpType.KV_STORE,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_kv_time(batch.total_tokens, is_store=True),
        )
        nodes.append(kv_store_node)

        # AllReduce for attention output (if TP > 1)
        if self.parallel_config.tensor_parallel_size > 1:
            comm_id = self._next_id(f"comm_attn_L{layer_idx}")
            comm_node = self._build_comm_node(comm_id, layer_idx, OpType.COMM_ALLREDUCE)
            nodes.append(comm_node)

        # LayerNorm
        ln2_id = self._next_id(f"ln2_L{layer_idx}")
        ln2_node = ExecutionNode(
            node_id=ln2_id,
            op_type=OpType.LAYERNORM,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_layernorm_time(batch.total_tokens),
        )
        nodes.append(ln2_node)

        # FFN
        ffn_id = self._next_id(f"ffn_L{layer_idx}")
        ffn_node = self._build_ffn_node(ffn_id, layer_idx, batch_info)
        nodes.append(ffn_node)

        # AllReduce for FFN output (if TP > 1)
        if self.parallel_config.tensor_parallel_size > 1:
            comm_id = self._next_id(f"comm_ffn_L{layer_idx}")
            comm_node = self._build_comm_node(comm_id, layer_idx, OpType.COMM_ALLREDUCE)
            nodes.append(comm_node)

        return nodes

    def _build_decode_layer(
        self,
        layer_idx: int,
        batch: DecodeBatch,
        prev_node_id: str,
    ) -> list[ExecutionNode]:
        """Build nodes for a single decode layer."""
        nodes: list[ExecutionNode] = []
        avg_context_len = (
            sum(batch.context_lengths) // max(len(batch.context_lengths), 1)
        )
        batch_info = {
            "batch_size": batch.batch_size,
            "context_lengths": batch.context_lengths,
            "avg_context_len": avg_context_len,
        }

        # LayerNorm
        ln1_id = self._next_id(f"ln1_L{layer_idx}")
        ln1_node = ExecutionNode(
            node_id=ln1_id,
            op_type=OpType.LAYERNORM,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_layernorm_time(batch.batch_size),
        )
        nodes.append(ln1_node)

        # KV Load (before attention)
        kv_load_id = self._next_id(f"kv_load_L{layer_idx}")
        kv_load_node = ExecutionNode(
            node_id=kv_load_id,
            op_type=OpType.KV_LOAD,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_kv_time(
                batch.batch_size * avg_context_len,
                is_store=False,
            ),
        )
        nodes.append(kv_load_node)

        # Attention
        attn_id = self._next_id(f"attn_L{layer_idx}")
        attn_node = self._build_attention_node(
            attn_id,
            layer_idx,
            is_prefill=False,
            batch_info=batch_info,
        )
        nodes.append(attn_node)

        # KV Store (append new KV)
        kv_store_id = self._next_id(f"kv_store_L{layer_idx}")
        kv_store_node = ExecutionNode(
            node_id=kv_store_id,
            op_type=OpType.KV_STORE,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_kv_time(batch.batch_size, is_store=True),
        )
        nodes.append(kv_store_node)

        # AllReduce for attention output (if TP > 1)
        if self.parallel_config.tensor_parallel_size > 1:
            comm_id = self._next_id(f"comm_attn_L{layer_idx}")
            comm_node = self._build_comm_node(comm_id, layer_idx, OpType.COMM_ALLREDUCE)
            nodes.append(comm_node)

        # LayerNorm
        ln2_id = self._next_id(f"ln2_L{layer_idx}")
        ln2_node = ExecutionNode(
            node_id=ln2_id,
            op_type=OpType.LAYERNORM,
            layer_id=layer_idx,
            metadata=batch_info,
            estimated_time_us=self._estimate_layernorm_time(batch.batch_size),
        )
        nodes.append(ln2_node)

        # FFN
        ffn_id = self._next_id(f"ffn_L{layer_idx}")
        ffn_node = self._build_ffn_node(ffn_id, layer_idx, batch_info)
        nodes.append(ffn_node)

        # AllReduce for FFN output (if TP > 1)
        if self.parallel_config.tensor_parallel_size > 1:
            comm_id = self._next_id(f"comm_ffn_L{layer_idx}")
            comm_node = self._build_comm_node(comm_id, layer_idx, OpType.COMM_ALLREDUCE)
            nodes.append(comm_node)

        return nodes

    def _build_embedding_node(
        self,
        node_id: str,
        num_tokens: int,
    ) -> ExecutionNode:
        """Build embedding lookup node."""
        return ExecutionNode(
            node_id=node_id,
            op_type=OpType.EMBEDDING,
            inputs=[
                TensorRef(
                    "token_ids",
                    (num_tokens,),
                    "int64",
                ),
            ],
            outputs=[
                TensorRef(
                    "hidden",
                    (num_tokens, self.model_config.hidden_size),
                    self.model_config.dtype,
                ),
            ],
            metadata={"num_tokens": num_tokens},
            estimated_time_us=self._estimate_embedding_time(num_tokens),
            memory_bytes=num_tokens * self.model_config.hidden_size * 2,  # float16
        )

    def _build_attention_node(
        self,
        node_id: str,
        layer_id: int,
        is_prefill: bool,
        batch_info: dict[str, Any],
    ) -> ExecutionNode:
        """Build attention operation node."""
        op_type = OpType.PREFILL_ATTN if is_prefill else OpType.DECODE_ATTN

        if is_prefill:
            seq_len = batch_info.get("seq_len", 1)
            time_us = self._estimate_prefill_attention_time(
                batch_info.get("batch_size", 1),
                seq_len,
            )
        else:
            avg_ctx = batch_info.get("avg_context_len", 1)
            time_us = self._estimate_decode_attention_time(
                batch_info.get("batch_size", 1),
                avg_ctx,
            )

        return ExecutionNode(
            node_id=node_id,
            op_type=op_type,
            layer_id=layer_id,
            metadata={
                "is_prefill": is_prefill,
                **batch_info,
            },
            estimated_time_us=time_us,
        )

    def _build_ffn_node(
        self,
        node_id: str,
        layer_id: int,
        batch_info: dict[str, Any],
    ) -> ExecutionNode:
        """Build FFN operation node."""
        batch_size = batch_info.get("batch_size", 1)
        seq_len = batch_info.get("seq_len", batch_info.get("total_tokens", 1))

        return ExecutionNode(
            node_id=node_id,
            op_type=OpType.FFN,
            layer_id=layer_id,
            metadata=batch_info,
            estimated_time_us=self._estimate_ffn_time(batch_size, seq_len),
        )

    def _build_comm_node(
        self,
        node_id: str,
        layer_id: int,
        comm_type: OpType,
    ) -> ExecutionNode:
        """Build communication operation node."""
        return ExecutionNode(
            node_id=node_id,
            op_type=comm_type,
            layer_id=layer_id,
            metadata={
                "tp_size": self.parallel_config.tensor_parallel_size,
            },
            estimated_time_us=self._estimate_comm_time(),
        )

    def _build_lm_head_node(
        self,
        node_id: str,
        num_tokens: int,
    ) -> ExecutionNode:
        """Build language model head node."""
        return ExecutionNode(
            node_id=node_id,
            op_type=OpType.LM_HEAD,
            inputs=[
                TensorRef(
                    "hidden",
                    (num_tokens, self.model_config.hidden_size),
                    self.model_config.dtype,
                ),
            ],
            outputs=[
                TensorRef(
                    "logits",
                    (num_tokens, self.model_config.vocab_size),
                    self.model_config.dtype,
                ),
            ],
            metadata={"num_tokens": num_tokens},
            estimated_time_us=self._estimate_lm_head_time(num_tokens),
        )

    # ======================= Time Estimation =======================
    # Simple heuristic estimates. In production, these would be calibrated
    # based on actual hardware profiling.

    def _estimate_embedding_time(self, num_tokens: int) -> float:
        """Estimate embedding lookup time in microseconds."""
        # Embedding is memory-bound, ~0.1us per token
        return num_tokens * 0.1

    def _estimate_layernorm_time(self, num_tokens: int) -> float:
        """Estimate LayerNorm time in microseconds."""
        # LayerNorm is memory-bound, ~0.05us per token
        return num_tokens * 0.05

    def _estimate_prefill_attention_time(
        self,
        batch_size: int,
        seq_len: int,
    ) -> float:
        """Estimate prefill attention time in microseconds.

        Prefill attention is compute-bound with O(seq_len^2) complexity.
        """
        # Rough estimate: 0.001us * seq_len^2 * batch_size
        return 0.001 * seq_len * seq_len * batch_size

    def _estimate_decode_attention_time(
        self,
        batch_size: int,
        context_len: int,
    ) -> float:
        """Estimate decode attention time in microseconds.

        Decode attention is memory-bound with O(context_len) complexity.
        """
        # Rough estimate: 0.1us * context_len * batch_size
        return 0.1 * context_len * batch_size

    def _estimate_ffn_time(self, batch_size: int, seq_len: int) -> float:
        """Estimate FFN time in microseconds."""
        # FFN is compute-bound, O(hidden * intermediate * tokens)
        tokens = batch_size * seq_len if seq_len > 1 else batch_size
        return 0.01 * tokens

    def _estimate_kv_time(self, num_tokens: int, is_store: bool) -> float:
        """Estimate KV cache load/store time in microseconds."""
        # KV operations are memory-bound
        # Store is slightly faster than load due to write combining
        factor = 0.05 if is_store else 0.1
        return factor * num_tokens

    def _estimate_comm_time(self) -> float:
        """Estimate communication time in microseconds."""
        # AllReduce latency + bandwidth overhead
        # Simplified estimate: 100us base + size-dependent
        hidden = self.model_config.hidden_size
        tp = self.parallel_config.tensor_parallel_size
        return 100.0 + 0.01 * hidden * 2 / tp  # 2 bytes per float16

    def _estimate_lm_head_time(self, num_tokens: int) -> float:
        """Estimate LM head (vocab projection) time in microseconds."""
        # Large matmul: hidden_size x vocab_size
        return 0.01 * num_tokens * self.model_config.vocab_size / 1000


# Legacy builder classes for backward compatibility


class GraphBuilder:
    """Legacy fluent builder for constructing execution graphs."""

    def __init__(self) -> None:
        """Initialize an empty graph builder."""
        self._graph = ExecutionGraph(graph_id="legacy_graph")
        self._node_counter = 0

    def _next_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def add_prefill(
        self,
        request_id: str,
        seq_len: int,
        *,
        chunked: bool = False,
        chunk_size: int = 512,
        device_hint: str | None = None,
    ) -> str:
        """Add prefill operation(s) to the graph."""
        if not chunked or seq_len <= chunk_size:
            node_id = self._next_id(f"prefill_{request_id}")
            self._graph.add_node(
                node_id=node_id,
                node_type=NodeType.PREFILL,
                config={"request_id": request_id, "seq_len": seq_len},
            )
            self._graph.prefill_nodes.add(node_id)
            return node_id

        # Chunked prefill
        prev_node_id = None
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        node_id = ""
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            node_id = self._next_id(f"prefill_chunk_{request_id}")
            self._graph.add_node(
                node_id=node_id,
                node_type=NodeType.CHUNKED_PREFILL,
                config={
                    "request_id": request_id,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_idx": i,
                },
            )
            self._graph.prefill_nodes.add(node_id)
            if prev_node_id:
                self._graph.add_edge(prev_node_id, node_id, EdgeType.KV_CACHE)
            prev_node_id = node_id

        return node_id

    def add_decode(
        self,
        request_id: str,
        num_steps: int = 1,
        *,
        prefill_node_id: str | None = None,
        device_hint: str | None = None,
    ) -> list[str]:
        """Add decode step(s) to the graph."""
        node_ids = []
        prev_node_id = prefill_node_id

        for i in range(num_steps):
            node_id = self._next_id(f"decode_{request_id}")
            self._graph.add_node(
                node_id=node_id,
                node_type=NodeType.DECODE,
                config={"request_id": request_id, "step": i},
            )
            self._graph.decode_nodes.add(node_id)
            if prev_node_id:
                self._graph.add_edge(prev_node_id, node_id, EdgeType.KV_CACHE)
            node_ids.append(node_id)
            prev_node_id = node_id

        return node_ids

    def add_kv_migration(
        self,
        source_device: str,
        target_device: str,
        *,
        depends_on: str | None = None,
    ) -> str:
        """Add a KV cache migration operation."""
        node_id = self._next_id("kv_migrate")
        self._graph.add_node(
            node_id=node_id,
            node_type=NodeType.KV_CACHE_MIGRATE,
            config={"source_device": source_device, "target_device": target_device},
        )
        self._graph.kv_nodes.add(node_id)
        if depends_on:
            self._graph.add_edge(depends_on, node_id, EdgeType.MEMORY)
        return node_id

    def add_communication(
        self,
        comm_type: NodeType,
        participants: list[str],
        *,
        depends_on: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Add a communication operation."""
        node_id = self._next_id("comm")
        self._graph.add_node(
            node_id=node_id,
            node_type=comm_type,
            config={
                "participants": participants,
                **(config or {}),
            },
        )
        self._graph.comm_nodes.add(node_id)
        if depends_on:
            self._graph.add_edge(depends_on, node_id, EdgeType.CONTROL)
        return node_id

    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.CONTROL,
    ) -> None:
        """Add an explicit dependency between two nodes."""
        self._graph.add_edge(source_id, target_id, edge_type)

    def build(self) -> ExecutionGraph:
        """Build and return the execution graph."""
        return self._graph

    def reset(self) -> None:
        """Reset the builder for reuse."""
        self._graph = ExecutionGraph(graph_id="legacy_graph")
        self._node_counter = 0


class PrefillDecodeBuilder(GraphBuilder):
    """Specialized builder for standard prefill-decode workloads."""

    def build_request(
        self,
        request_id: str,
        prefill_len: int,
        max_decode_steps: int,
        *,
        chunked_prefill: bool = False,
        chunk_size: int = 512,
    ) -> ExecutionGraph:
        """Build a complete prefill-decode graph for a single request."""
        prefill_id = self.add_prefill(
            request_id,
            prefill_len,
            chunked=chunked_prefill,
            chunk_size=chunk_size,
        )
        self.add_decode(
            request_id,
            max_decode_steps,
            prefill_node_id=prefill_id,
        )
        return self.build()


__all__ = [
    # New API
    "ModelConfig",
    "ParallelConfig",
    "PrefillBatch",
    "DecodeBatch",
    "ExecutionGraphBuilder",
    # Legacy API
    "GraphBuilder",
    "PrefillDecodeBuilder",
]
