# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for runtime execution graph module.

Tests cover:
- ExecutionGraph IR: Node/edge creation, topological sort, critical path
- GraphBuilder: Prefill/Decode/Hybrid graph construction
- GraphOptimizer: Optimization passes
- PDScheduler: Three scheduling modes (strict, time_share, hybrid)
"""

from __future__ import annotations

import pytest

from sage.common.components.sage_llm.sageLLM.runtime.execution_graph.builder import (
    DecodeBatch,
    ExecutionGraphBuilder,
    ModelConfig,
    ParallelConfig,
    PrefillBatch,
)
from sage.common.components.sage_llm.sageLLM.runtime.execution_graph.ir import (
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    OpType,
    TensorRef,
)
from sage.common.components.sage_llm.sageLLM.runtime.execution_graph.optimizer import (
    CommunicationFusionPass,
    ComputeCommOverlapPass,
    GraphOptimizer,
    KVPrefetchPass,
    MemoryOptimizationPass,
)
from sage.common.components.sage_llm.sageLLM.runtime.scheduler.base import (
    Request,
    RequestStatus,
)
from sage.common.components.sage_llm.sageLLM.runtime.scheduler.pd_scheduler import (
    PDScheduler,
    PDSchedulerConfig,
)

# ======================= Execution Graph IR Tests =======================


class TestTensorRef:
    """Tests for TensorRef class."""

    def test_create_tensor_ref(self) -> None:
        """Test creating a tensor reference."""
        tensor = TensorRef(
            name="hidden",
            shape=(32, 128, 4096),
            dtype="float16",
            device_id=0,
        )
        assert tensor.name == "hidden"
        assert tensor.shape == (32, 128, 4096)
        assert tensor.dtype == "float16"
        assert tensor.device_id == 0

    def test_tensor_numel(self) -> None:
        """Test tensor element count."""
        tensor = TensorRef("x", (2, 3, 4), "float16")
        assert tensor.numel == 24

    def test_tensor_numel_symbolic(self) -> None:
        """Test tensor with symbolic dimensions."""
        tensor = TensorRef("x", ("batch", 128, 4096), "float16")
        assert tensor.numel is None  # Cannot compute with symbolic dims

    def test_tensor_size_bytes(self) -> None:
        """Test tensor size calculation."""
        tensor = TensorRef("x", (100,), "float16")
        assert tensor.size_bytes == 200  # 100 * 2 bytes


class TestExecutionNode:
    """Tests for ExecutionNode class."""

    def test_create_node(self) -> None:
        """Test creating an execution node."""
        node = ExecutionNode(
            node_id="attn_0",
            op_type=OpType.PREFILL_ATTN,
            inputs=[TensorRef("in", (1,), "float16")],
            outputs=[TensorRef("out", (1,), "float16")],
            layer_id=0,
        )
        assert node.node_id == "attn_0"
        assert node.op_type == OpType.PREFILL_ATTN
        assert node.layer_id == 0
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1

    def test_node_dependencies(self) -> None:
        """Test node dependency tracking."""
        node = ExecutionNode(
            node_id="ffn_0",
            op_type=OpType.FFN,
            dependencies={"attn_0", "layernorm_0"},
        )
        assert "attn_0" in node.dependencies
        assert "layernorm_0" in node.dependencies
        assert len(node.dependencies) == 2

    def test_node_legacy_properties(self) -> None:
        """Test legacy compatibility properties."""
        node = ExecutionNode(
            node_id="test",
            op_type=OpType.FFN,
            device_id=1,
            estimated_time_us=1000.0,
            memory_bytes=4096,
        )
        assert node.id == "test"  # Legacy alias
        assert node.node_type == OpType.FFN  # Legacy alias
        assert node.estimated_latency_ms == 1.0  # 1000us = 1ms


class TestExecutionGraph:
    """Tests for ExecutionGraph class."""

    def test_create_graph(self) -> None:
        """Test creating an empty graph."""
        graph = ExecutionGraph(graph_id="test")
        assert graph.graph_id == "test"
        assert len(graph.nodes) == 0

    def test_add_node(self) -> None:
        """Test adding nodes to graph."""
        graph = ExecutionGraph(graph_id="test")
        node = ExecutionNode(
            node_id="attn_0",
            op_type=OpType.PREFILL_ATTN,
        )
        graph.add_node(node)
        assert "attn_0" in graph.nodes
        assert "attn_0" in graph.prefill_nodes  # Auto-classified

    def test_add_node_legacy_api(self) -> None:
        """Test legacy add_node API."""
        from sage.common.components.sage_llm.sageLLM.runtime.execution_graph.ir import (
            NodeType,
        )

        graph = ExecutionGraph(graph_id="test")

        node = graph.add_node(
            node_id="prefill_1",
            node_type=NodeType.PREFILL,
            config={"seq_len": 128},
        )
        assert node.node_id == "prefill_1"
        assert "prefill_1" in graph.nodes

    def test_add_edge(self) -> None:
        """Test adding edges between nodes."""
        graph = ExecutionGraph(graph_id="test")
        node1 = ExecutionNode(node_id="n1", op_type=OpType.EMBEDDING)
        node2 = ExecutionNode(node_id="n2", op_type=OpType.PREFILL_ATTN)
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge("n1", "n2", EdgeType.DATA)

        assert "n1" in graph.nodes["n2"].dependencies

    def test_topological_sort_simple(self) -> None:
        """Test topological sort with simple DAG."""
        graph = ExecutionGraph(graph_id="test")

        # Create a chain: n1 -> n2 -> n3
        n1 = ExecutionNode(node_id="n1", op_type=OpType.EMBEDDING)
        n2 = ExecutionNode(node_id="n2", op_type=OpType.PREFILL_ATTN)
        n3 = ExecutionNode(node_id="n3", op_type=OpType.FFN)

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")

        sorted_nodes = graph.topological_sort()
        node_ids = [n.node_id for n in sorted_nodes]

        # n1 must come before n2, n2 before n3
        assert node_ids.index("n1") < node_ids.index("n2")
        assert node_ids.index("n2") < node_ids.index("n3")

    def test_topological_sort_diamond(self) -> None:
        """Test topological sort with diamond DAG."""
        graph = ExecutionGraph(graph_id="test")

        #     n1
        #    /  \
        #   n2  n3
        #    \  /
        #     n4
        for i, op in enumerate([OpType.EMBEDDING, OpType.FFN, OpType.FFN, OpType.LM_HEAD]):
            graph.add_node(ExecutionNode(node_id=f"n{i+1}", op_type=op))

        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")
        graph.add_edge("n2", "n4")
        graph.add_edge("n3", "n4")

        sorted_nodes = graph.topological_sort()
        node_ids = [n.node_id for n in sorted_nodes]

        assert node_ids.index("n1") < node_ids.index("n2")
        assert node_ids.index("n1") < node_ids.index("n3")
        assert node_ids.index("n2") < node_ids.index("n4")
        assert node_ids.index("n3") < node_ids.index("n4")

    def test_topological_sort_cycle_detection(self) -> None:
        """Test that cycles are detected."""
        graph = ExecutionGraph(graph_id="test")
        n1 = ExecutionNode(node_id="n1", op_type=OpType.FFN)
        n2 = ExecutionNode(node_id="n2", op_type=OpType.FFN)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n1")  # Creates cycle

        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()

    def test_critical_path(self) -> None:
        """Test critical path computation."""
        graph = ExecutionGraph(graph_id="test")

        # n1 (100us) -> n2 (200us) -> n4 (100us)
        #            -> n3 (50us)  ->
        n1 = ExecutionNode(node_id="n1", op_type=OpType.EMBEDDING, estimated_time_us=100)
        n2 = ExecutionNode(node_id="n2", op_type=OpType.PREFILL_ATTN, estimated_time_us=200)
        n3 = ExecutionNode(node_id="n3", op_type=OpType.FFN, estimated_time_us=50)
        n4 = ExecutionNode(node_id="n4", op_type=OpType.LM_HEAD, estimated_time_us=100)

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_node(n4)
        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")
        graph.add_edge("n2", "n4")
        graph.add_edge("n3", "n4")

        critical_path = graph.get_critical_path()
        path_ids = [n.node_id for n in critical_path]

        # Critical path should be n1 -> n2 -> n4 (longer than n1 -> n3 -> n4)
        assert "n2" in path_ids  # n2 is on critical path

    def test_estimate_total_time(self) -> None:
        """Test total time estimation."""
        graph = ExecutionGraph(graph_id="test")
        n1 = ExecutionNode(node_id="n1", op_type=OpType.FFN, estimated_time_us=100)
        n2 = ExecutionNode(node_id="n2", op_type=OpType.FFN, estimated_time_us=200)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge("n1", "n2")

        # Total time = sum of critical path
        total = graph.estimate_total_time()
        assert total >= 200  # At least the longest node

    def test_get_subgraph(self) -> None:
        """Test subgraph extraction."""
        graph = ExecutionGraph(graph_id="test")
        n1 = ExecutionNode(node_id="n1", op_type=OpType.PREFILL_ATTN)
        n2 = ExecutionNode(node_id="n2", op_type=OpType.FFN)
        n3 = ExecutionNode(node_id="n3", op_type=OpType.COMM_ALLREDUCE)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")

        # Extract subgraph with just n1 and n2
        subgraph = graph.get_subgraph({"n1", "n2"})
        assert len(subgraph.nodes) == 2
        assert "n3" not in subgraph.nodes

    def test_auto_classification(self) -> None:
        """Test automatic node classification."""
        graph = ExecutionGraph(graph_id="test")

        prefill_node = ExecutionNode(node_id="p1", op_type=OpType.PREFILL_ATTN)
        decode_node = ExecutionNode(node_id="d1", op_type=OpType.DECODE_ATTN)
        comm_node = ExecutionNode(node_id="c1", op_type=OpType.COMM_ALLREDUCE)
        kv_node = ExecutionNode(node_id="k1", op_type=OpType.KV_LOAD)

        graph.add_node(prefill_node)
        graph.add_node(decode_node)
        graph.add_node(comm_node)
        graph.add_node(kv_node)

        assert "p1" in graph.prefill_nodes
        assert "d1" in graph.decode_nodes
        assert "c1" in graph.comm_nodes
        assert "k1" in graph.kv_nodes


# ======================= Graph Builder Tests =======================


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_create_config(self) -> None:
        """Test creating model config."""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
        )
        assert config.num_layers == 32
        assert config.head_dim == 128  # 4096 / 32


class TestExecutionGraphBuilder:
    """Tests for ExecutionGraphBuilder class."""

    @pytest.fixture
    def builder(self) -> ExecutionGraphBuilder:
        """Create a builder fixture."""
        config = ModelConfig(
            num_layers=4,  # Small for testing
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
        )
        parallel = ParallelConfig(tensor_parallel_size=1)
        return ExecutionGraphBuilder(config, parallel)

    def test_build_prefill_graph(self, builder: ExecutionGraphBuilder) -> None:
        """Test building prefill graph."""
        batch = PrefillBatch(
            request_ids=["req_1"],
            input_lengths=[128],
            total_tokens=128,
        )
        graph = builder.build_prefill_graph(batch)

        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.prefill_nodes) > 0
        assert graph.num_layers == 4

    def test_build_decode_graph(self, builder: ExecutionGraphBuilder) -> None:
        """Test building decode graph."""
        batch = DecodeBatch(
            request_ids=["req_1"],
            context_lengths=[128],
            batch_size=1,
        )
        graph = builder.build_decode_graph(batch)

        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.decode_nodes) > 0

    def test_build_hybrid_graph(self, builder: ExecutionGraphBuilder) -> None:
        """Test building hybrid graph."""
        prefill_batch = PrefillBatch(
            request_ids=["p1"],
            input_lengths=[64],
            total_tokens=64,
        )
        decode_batch = DecodeBatch(
            request_ids=["d1", "d2"],
            context_lengths=[128, 256],
            batch_size=2,
        )
        graph = builder.build_hybrid_graph(prefill_batch, decode_batch)

        assert graph is not None
        assert len(graph.prefill_nodes) > 0
        assert len(graph.decode_nodes) > 0

    def test_parallel_config_adds_comm_nodes(self) -> None:
        """Test that TP > 1 adds communication nodes."""
        config = ModelConfig(
            num_layers=2,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
        )
        parallel = ParallelConfig(tensor_parallel_size=2)  # TP=2
        builder = ExecutionGraphBuilder(config, parallel)

        batch = PrefillBatch(
            request_ids=["req_1"],
            input_lengths=[64],
            total_tokens=64,
        )
        graph = builder.build_prefill_graph(batch)

        # Should have communication nodes
        assert len(graph.comm_nodes) > 0


# ======================= Graph Optimizer Tests =======================


class TestGraphOptimizer:
    """Tests for GraphOptimizer class."""

    def test_default_optimizer(self) -> None:
        """Test creating default optimizer."""
        optimizer = GraphOptimizer.default()
        assert len(optimizer.passes) == 4

    def test_optimize_empty_graph(self) -> None:
        """Test optimizing empty graph."""
        optimizer = GraphOptimizer()
        graph = ExecutionGraph(graph_id="test")
        optimized = optimizer.optimize(graph)
        assert optimized is not None

    def test_communication_fusion_pass(self) -> None:
        """Test communication fusion pass."""
        pass_ = CommunicationFusionPass()
        assert pass_.name == "comm_fusion"

        graph = ExecutionGraph(graph_id="test")
        optimized = pass_.apply(graph)
        assert optimized is not None

    def test_compute_comm_overlap_pass(self) -> None:
        """Test compute-communication overlap pass."""
        pass_ = ComputeCommOverlapPass()
        assert pass_.name == "compute_comm_overlap"

    def test_kv_prefetch_pass(self) -> None:
        """Test KV prefetch pass."""
        pass_ = KVPrefetchPass()
        assert pass_.name == "kv_prefetch"

    def test_memory_optimization_pass(self) -> None:
        """Test memory optimization pass."""
        pass_ = MemoryOptimizationPass()
        assert pass_.name == "memory_opt"


# ======================= Scheduler Tests =======================


class TestRequest:
    """Tests for Request class."""

    def test_create_request(self) -> None:
        """Test creating a request."""
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3, 4],
            max_new_tokens=128,
        )
        assert req.request_id == "r1"
        assert req.num_prompt_tokens == 4
        assert req.status == RequestStatus.WAITING

    def test_request_properties(self) -> None:
        """Test request property calculations."""
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3, 4],
        )
        req.output_token_ids = [5, 6]

        assert req.num_output_tokens == 2
        assert req.total_tokens == 6
        assert req.context_len == 6


class TestPDScheduler:
    """Tests for PDScheduler class."""

    def test_strict_mode(self) -> None:
        """Test strict separation mode."""
        config = PDSchedulerConfig(
            mode="strict",
            prefill_device_ids=[0],
            decode_device_ids=[1],
        )
        scheduler = PDScheduler(config)

        # Add request
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)

        # Schedule should return prefill batch
        output = scheduler.schedule()
        assert output.prefill_batch is not None
        assert output.prefill_batch.batch_size == 1

    def test_time_share_mode(self) -> None:
        """Test time-share mode."""
        config = PDSchedulerConfig(mode="time_share")
        scheduler = PDScheduler(config)

        # Add request for prefill
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)

        # First schedule returns prefill (no decode work)
        output = scheduler.schedule()
        assert output.prefill_batch is not None

    def test_hybrid_mode(self) -> None:
        """Test hybrid (chunked prefill) mode."""
        config = PDSchedulerConfig(
            mode="hybrid",
            prefill_chunk_size=2,
            max_decode_tokens=1024,
        )
        scheduler = PDScheduler(config)

        # Add request with 4 tokens (will be chunked)
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)

        # First chunk
        output1 = scheduler.schedule()
        assert output1.prefill_batch is not None
        # After first chunk, only 2 tokens computed
        assert scheduler.prefill_queue[0].num_computed_tokens == 2

        # Second chunk should complete prefill
        output2 = scheduler.schedule()
        assert output2.prefill_batch is not None
        # Request should now be in decode queue
        assert len(scheduler.decode_queue) == 1

    def test_prefill_to_decode_transition(self) -> None:
        """Test request transition from prefill to decode."""
        config = PDSchedulerConfig(mode="time_share")
        scheduler = PDScheduler(config)

        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)

        # Schedule prefill
        output = scheduler.schedule()
        assert output.prefill_batch is not None

        # Manually transition to decode
        scheduler.prefill_to_decode(["r1"])

        assert req.status == RequestStatus.DECODING
        assert len(scheduler.decode_queue) == 1

    def test_abort_request(self) -> None:
        """Test aborting a request."""
        config = PDSchedulerConfig(mode="time_share")
        scheduler = PDScheduler(config)

        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)

        # Abort
        result = scheduler.abort_request("r1")
        assert result is True
        assert len(scheduler.prefill_queue) == 0
        assert req.status == RequestStatus.FINISHED

    def test_scheduler_stats(self) -> None:
        """Test scheduler statistics."""
        config = PDSchedulerConfig(mode="strict")
        scheduler = PDScheduler(config)

        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3, 4])
        scheduler.add_request(req)
        scheduler.schedule()

        stats = scheduler.get_stats()
        assert "total_prefill_batches" in stats
        assert stats["total_prefill_batches"] == 1

    def test_continuous_batching(self) -> None:
        """Test continuous batching in decode phase."""
        config = PDSchedulerConfig(
            mode="hybrid",  # Use hybrid mode which supports chunked prefill
            max_decode_batch_size=256,
            max_prefill_batch_size=8,
        )
        scheduler = PDScheduler(config)

        # Add all 3 requests first
        for i in range(3):
            req = Request(request_id=f"r{i}", prompt_token_ids=[1, 2, 3, 4])
            scheduler.add_request(req)

        # One schedule call will prefill all 3 (batch_size=8 > 3)
        output = scheduler.schedule()
        assert output.prefill_batch is not None
        assert len(output.prefill_batch.requests) == 3

        # Transition all to decode
        req_ids = [r.request_id for r in output.prefill_batch.requests]
        scheduler.prefill_to_decode(req_ids)

        # All 3 should now be in decode queue
        assert len(scheduler.decode_queue) == 3

        # Schedule should batch them together
        output = scheduler.schedule()
        assert output.decode_batch is not None
        assert output.decode_batch.batch_size == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
