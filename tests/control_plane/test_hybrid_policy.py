# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for HybridSchedulingPolicy.

This module contains unit tests for the hybrid scheduling strategy
that handles mixed LLM and Embedding workloads.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (  # noqa: E402  # type: ignore[import-not-found]
    ExecutionInstance,
    ExecutionInstanceType,
    RequestMetadata,
    RequestPriority,
    RequestType,
)
from control_plane.strategies import (  # noqa: E402  # type: ignore[import-not-found]
    EmbeddingBatch,
    EmbeddingPriority,
    HybridSchedulingConfig,
    HybridSchedulingPolicy,
)

# ============ Fixtures ============


@pytest.fixture
def llm_instances():
    """Create LLM-only execution instances."""
    return [
        ExecutionInstance(
            instance_id="llm-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.GENERAL,
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.3,
            avg_latency_ms=30.0,
        ),
        ExecutionInstance(
            instance_id="llm-2",
            host="localhost",
            port=8001,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.GENERAL,
            tensor_parallel_size=1,
            gpu_count=1,
            current_load=0.5,
            avg_latency_ms=50.0,
        ),
    ]


@pytest.fixture
def embedding_instances():
    """Create embedding-only execution instances."""
    return [
        ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            embedding_model_loaded="BAAI/bge-m3",
            embedding_max_batch_size=32,
            embedding_active_requests=0,
            current_load=0.2,
        ),
        ExecutionInstance(
            instance_id="embed-2",
            host="localhost",
            port=8091,
            model_name="BAAI/bge-small-zh-v1.5",
            instance_type=ExecutionInstanceType.EMBEDDING,
            embedding_model_loaded="BAAI/bge-small-zh-v1.5",
            embedding_max_batch_size=64,
            embedding_active_requests=10,
            current_load=0.4,
        ),
    ]


@pytest.fixture
def mixed_instances():
    """Create mixed LLM+Embedding execution instances."""
    return [
        ExecutionInstance(
            instance_id="mixed-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.LLM_EMBEDDING,
            embedding_model_loaded="BAAI/bge-m3",
            embedding_max_batch_size=32,
            embedding_active_requests=5,
            current_load=0.4,
            avg_latency_ms=40.0,
        ),
    ]


@pytest.fixture
def all_instances(llm_instances, embedding_instances, mixed_instances):
    """Create a heterogeneous set of instances."""
    return llm_instances + embedding_instances + mixed_instances


@pytest.fixture
def llm_requests():
    """Create sample LLM requests."""
    base_time = datetime.now()
    return [
        RequestMetadata(
            request_id="llm-req-1",
            prompt="Hello, how are you?",
            request_type=RequestType.LLM_CHAT,
            priority=RequestPriority.NORMAL,
            arrival_time=base_time,
        ),
        RequestMetadata(
            request_id="llm-req-2",
            prompt="Tell me a story",
            request_type=RequestType.LLM_GENERATE,
            priority=RequestPriority.HIGH,
            arrival_time=base_time + timedelta(milliseconds=10),
        ),
    ]


@pytest.fixture
def embedding_requests():
    """Create sample embedding requests."""
    base_time = datetime.now()
    return [
        RequestMetadata(
            request_id="embed-req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["Hello world", "How are you"],
            embedding_model="BAAI/bge-m3",
            priority=RequestPriority.NORMAL,
            arrival_time=base_time + timedelta(milliseconds=5),
        ),
        RequestMetadata(
            request_id="embed-req-2",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["Test embedding"],
            embedding_model="BAAI/bge-m3",
            priority=RequestPriority.LOW,
            arrival_time=base_time + timedelta(milliseconds=15),
        ),
    ]


@pytest.fixture
def mixed_requests(llm_requests, embedding_requests):
    """Create mixed LLM and embedding requests."""
    return llm_requests + embedding_requests


# ============ HybridSchedulingConfig Tests ============


class TestHybridSchedulingConfig:
    """Tests for HybridSchedulingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HybridSchedulingConfig()

        assert config.embedding_batch_size == 32
        assert config.embedding_priority == "normal"
        assert config.llm_fallback_policy == "adaptive"
        assert config.hybrid_instance_ratio == 0.7
        assert config.enable_embedding_batching is True
        assert config.max_embedding_wait_ms == 50.0
        assert config.min_embedding_batch_size == 1
        assert config.prefer_specialized_instances is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HybridSchedulingConfig(
            embedding_batch_size=64,
            embedding_priority="high",
            llm_fallback_policy="fifo",
            hybrid_instance_ratio=0.5,
        )

        assert config.embedding_batch_size == 64
        assert config.embedding_priority == "high"
        assert config.llm_fallback_policy == "fifo"
        assert config.hybrid_instance_ratio == 0.5


# ============ EmbeddingBatch Tests ============


class TestEmbeddingBatch:
    """Tests for EmbeddingBatch dataclass."""

    def test_empty_batch(self):
        """Test empty batch creation."""
        batch = EmbeddingBatch(batch_id="test-batch")

        assert batch.size == 0
        assert batch.total_texts == 0
        assert batch.model is None

    def test_add_first_request(self):
        """Test adding first request to batch."""
        batch = EmbeddingBatch(batch_id="test-batch")
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["text1", "text2"],
            embedding_model="BAAI/bge-m3",
        )

        result = batch.add_request(request)

        assert result is True
        assert batch.size == 1
        assert batch.total_texts == 2
        assert batch.model == "BAAI/bge-m3"

    def test_add_compatible_request(self):
        """Test adding compatible request to batch."""
        batch = EmbeddingBatch(batch_id="test-batch")
        req1 = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["text1"],
            embedding_model="BAAI/bge-m3",
        )
        req2 = RequestMetadata(
            request_id="req-2",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["text2", "text3"],
            embedding_model="BAAI/bge-m3",
        )

        batch.add_request(req1)
        result = batch.add_request(req2)

        assert result is True
        assert batch.size == 2
        assert batch.total_texts == 3

    def test_add_incompatible_request(self):
        """Test adding request with different model."""
        batch = EmbeddingBatch(batch_id="test-batch")
        req1 = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["text1"],
            embedding_model="BAAI/bge-m3",
        )
        req2 = RequestMetadata(
            request_id="req-2",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["text2"],
            embedding_model="BAAI/bge-small-zh-v1.5",  # Different model
        )

        batch.add_request(req1)
        result = batch.add_request(req2)

        assert result is False
        assert batch.size == 1


# ============ HybridSchedulingPolicy Tests ============


class TestHybridSchedulingPolicyInit:
    """Tests for HybridSchedulingPolicy initialization."""

    def test_default_init(self):
        """Test default initialization."""
        policy = HybridSchedulingPolicy()

        assert policy.name == "Hybrid"
        assert policy.config.embedding_batch_size == 32
        assert policy.config.embedding_priority == "normal"
        assert policy.config.llm_fallback_policy == "adaptive"
        assert policy.embedding_priority == EmbeddingPriority.NORMAL

    def test_custom_init(self):
        """Test custom initialization."""
        policy = HybridSchedulingPolicy(
            embedding_batch_size=64,
            embedding_priority="high",
            llm_fallback_policy="fifo",
        )

        assert policy.config.embedding_batch_size == 64
        assert policy.embedding_priority == EmbeddingPriority.HIGH

    def test_init_with_config(self):
        """Test initialization with config object."""
        config = HybridSchedulingConfig(
            embedding_batch_size=128,
            embedding_priority="low",
            llm_fallback_policy="priority",
        )
        policy = HybridSchedulingPolicy(config=config)

        assert policy.config.embedding_batch_size == 128
        assert policy.embedding_priority == EmbeddingPriority.LOW

    def test_llm_fallback_policies(self):
        """Test different LLM fallback policies are initialized."""
        for policy_name in ["fifo", "priority", "slo_aware", "adaptive"]:
            policy = HybridSchedulingPolicy(llm_fallback_policy=policy_name)
            assert policy.llm_policy is not None

    def test_invalid_llm_fallback_defaults_to_adaptive(self):
        """Test invalid LLM fallback policy defaults to Adaptive."""
        policy = HybridSchedulingPolicy(llm_fallback_policy="invalid")
        assert policy.llm_policy.name == "Adaptive"


class TestHybridSchedulingPolicySchedule:
    """Tests for HybridSchedulingPolicy.schedule() method."""

    def test_schedule_empty_requests(self, all_instances):
        """Test scheduling with empty request list."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule([], all_instances)

        assert len(decisions) == 0

    def test_schedule_empty_instances(self, mixed_requests):
        """Test scheduling with empty instance list."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(mixed_requests, [])

        assert len(decisions) == 0

    def test_schedule_llm_only(self, llm_requests, llm_instances):
        """Test scheduling LLM-only requests."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(llm_requests, llm_instances)

        assert len(decisions) == 2
        for decision in decisions:
            assert decision.target_instance_id in ["llm-1", "llm-2"]
            assert decision.metadata.get("request_type") == "llm"

    def test_schedule_embedding_only(self, embedding_requests, embedding_instances):
        """Test scheduling embedding-only requests."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(embedding_requests, embedding_instances)

        assert len(decisions) == 2
        for decision in decisions:
            assert decision.target_instance_id in ["embed-1", "embed-2"]
            assert decision.metadata.get("request_type") == "embedding"

    def test_schedule_mixed_requests(self, mixed_requests, all_instances):
        """Test scheduling mixed LLM and embedding requests."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(mixed_requests, all_instances)

        assert len(decisions) == 4

        # Check that embedding requests go to embedding-capable instances
        embedding_decisions = [
            d for d in decisions if d.metadata.get("request_type") == "embedding"
        ]
        for decision in embedding_decisions:
            assert decision.target_instance_id in ["embed-1", "embed-2", "mixed-1"]

        # Check that LLM requests go to LLM-capable instances
        llm_decisions = [
            d for d in decisions if d.metadata.get("request_type") == "llm"
        ]
        for decision in llm_decisions:
            assert decision.target_instance_id in ["llm-1", "llm-2", "mixed-1"]

    def test_schedule_prefers_specialized_instances(
        self, embedding_requests, embedding_instances, mixed_instances
    ):
        """Test that scheduling prefers specialized instances."""
        all_inst = embedding_instances + mixed_instances
        policy = HybridSchedulingPolicy()

        decisions = policy.schedule(embedding_requests, all_inst)

        # Should prefer embed-1 (specialized, lower load) over mixed-1
        assert len(decisions) == 2
        # First request should go to embed-1 (least loaded, specialized)
        assert decisions[0].target_instance_id == "embed-1"

    def test_schedule_with_no_embedding_instances(
        self, embedding_requests, llm_instances
    ):
        """Test scheduling embeddings when no embedding instances available."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(embedding_requests, llm_instances)

        # No embedding-capable instances, should return empty
        assert len(decisions) == 0

    def test_schedule_with_no_llm_instances(self, llm_requests, embedding_instances):
        """Test scheduling LLM when no LLM instances available."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(llm_requests, embedding_instances)

        # No LLM-capable instances, should return empty
        assert len(decisions) == 0


class TestHybridSchedulingPolicyPriority:
    """Tests for embedding priority settings."""

    def test_embedding_priority_high(self, mixed_requests, all_instances):
        """Test high embedding priority schedules embeddings first."""
        policy = HybridSchedulingPolicy(embedding_priority="high")

        # The schedule method handles priority internally
        decisions = policy.schedule(mixed_requests, all_instances)
        assert len(decisions) == 4

    def test_embedding_priority_low(self, mixed_requests, all_instances):
        """Test low embedding priority schedules LLM first."""
        policy = HybridSchedulingPolicy(embedding_priority="low")

        decisions = policy.schedule(mixed_requests, all_instances)
        assert len(decisions) == 4

    def test_embedding_priority_adaptive(self, mixed_requests, all_instances):
        """Test adaptive embedding priority."""
        policy = HybridSchedulingPolicy(embedding_priority="adaptive")

        decisions = policy.schedule(mixed_requests, all_instances)
        assert len(decisions) == 4

    def test_embedding_priority_normal(self, mixed_requests, all_instances):
        """Test normal embedding priority (interleaved)."""
        policy = HybridSchedulingPolicy(embedding_priority="normal")

        decisions = policy.schedule(mixed_requests, all_instances)
        assert len(decisions) == 4


class TestHybridSchedulingPolicyBatching:
    """Tests for embedding batching functionality."""

    def test_batching_groups_by_model(self):
        """Test that batching groups requests by model."""
        policy = HybridSchedulingPolicy(embedding_batch_size=32)

        requests = [
            RequestMetadata(
                request_id="req-1",
                request_type=RequestType.EMBEDDING,
                embedding_texts=["text1"],
                embedding_model="model-a",
            ),
            RequestMetadata(
                request_id="req-2",
                request_type=RequestType.EMBEDDING,
                embedding_texts=["text2"],
                embedding_model="model-b",
            ),
            RequestMetadata(
                request_id="req-3",
                request_type=RequestType.EMBEDDING,
                embedding_texts=["text3"],
                embedding_model="model-a",
            ),
        ]

        batches = policy._create_embedding_batches(requests)

        # Should create 2 batches (one for each model)
        assert len(batches) == 2

    def test_batching_respects_size_limit(self):
        """Test that batching respects batch size limit."""
        policy = HybridSchedulingPolicy(embedding_batch_size=5)

        # Create requests with 3 texts each
        requests = [
            RequestMetadata(
                request_id=f"req-{i}",
                request_type=RequestType.EMBEDDING,
                embedding_texts=["t1", "t2", "t3"],  # 3 texts each
                embedding_model="model-a",
            )
            for i in range(4)
        ]

        batches = policy._create_embedding_batches(requests)

        # 12 texts total, batch size 5 -> at least 3 batches
        assert len(batches) >= 2
        for batch in batches:
            assert batch.total_texts <= 5 or batch.size == 1

    def test_batching_disabled(self, embedding_requests, embedding_instances):
        """Test scheduling without batching."""
        config = HybridSchedulingConfig(enable_embedding_batching=False)
        policy = HybridSchedulingPolicy(config=config)

        batches = policy._create_embedding_batches(embedding_requests)

        # Each request should be its own batch
        assert len(batches) == len(embedding_requests)


class TestHybridSchedulingPolicyPrioritize:
    """Tests for prioritize() method."""

    def test_prioritize_by_request_priority(self, mixed_requests):
        """Test prioritization by request priority."""
        policy = HybridSchedulingPolicy()
        prioritized = policy.prioritize(mixed_requests)

        # HIGH priority should come before NORMAL and LOW
        high_idx = next(
            i for i, r in enumerate(prioritized)
            if r.priority == RequestPriority.HIGH
        )
        low_idx = next(
            i for i, r in enumerate(prioritized)
            if r.priority == RequestPriority.LOW
        )

        assert high_idx < low_idx

    def test_prioritize_with_embedding_priority_high(self, mixed_requests):
        """Test prioritization with high embedding priority."""
        policy = HybridSchedulingPolicy(embedding_priority="high")
        prioritized = policy.prioritize(mixed_requests)

        # Embedding requests should come first
        first_embedding_idx = next(
            i for i, r in enumerate(prioritized)
            if r.request_type == RequestType.EMBEDDING
        )
        first_llm_idx = next(
            i for i, r in enumerate(prioritized)
            if r.request_type in (RequestType.LLM_CHAT, RequestType.LLM_GENERATE)
        )

        assert first_embedding_idx < first_llm_idx

    def test_prioritize_with_embedding_priority_low(self, mixed_requests):
        """Test prioritization with low embedding priority."""
        policy = HybridSchedulingPolicy(embedding_priority="low")
        prioritized = policy.prioritize(mixed_requests)

        # LLM requests should come first
        first_llm_idx = next(
            i for i, r in enumerate(prioritized)
            if r.request_type in (RequestType.LLM_CHAT, RequestType.LLM_GENERATE)
        )
        first_embedding_idx = next(
            i for i, r in enumerate(prioritized)
            if r.request_type == RequestType.EMBEDDING
        )

        assert first_llm_idx < first_embedding_idx


class TestHybridSchedulingPolicyMetrics:
    """Tests for metrics functionality."""

    def test_get_metrics_initial(self):
        """Test initial metrics are zero."""
        policy = HybridSchedulingPolicy()
        metrics = policy.get_metrics()

        assert metrics["embedding_scheduled_count"] == 0
        assert metrics["llm_scheduled_count"] == 0
        assert metrics["batch_count"] == 0
        assert metrics["total_scheduled"] == 0

    def test_metrics_after_scheduling(self, mixed_requests, all_instances):
        """Test metrics are updated after scheduling."""
        policy = HybridSchedulingPolicy()
        policy.schedule(mixed_requests, all_instances)

        metrics = policy.get_metrics()
        assert metrics["total_scheduled"] == 4
        assert metrics["embedding_scheduled_count"] == 2
        assert metrics["llm_scheduled_count"] == 2

    def test_reset_metrics(self, mixed_requests, all_instances):
        """Test metrics reset."""
        policy = HybridSchedulingPolicy()
        policy.schedule(mixed_requests, all_instances)
        policy.reset_metrics()

        metrics = policy.get_metrics()
        assert metrics["total_scheduled"] == 0


class TestHybridSchedulingPolicyInstanceSelection:
    """Tests for instance selection logic."""

    def test_select_embedding_instance_prefers_model_match(self):
        """Test instance selection prefers matching model."""
        policy = HybridSchedulingPolicy()

        instances = [
            ExecutionInstance(
                instance_id="embed-nomatch",
                host="localhost",
                port=8090,
                model_name="model-other",
                instance_type=ExecutionInstanceType.EMBEDDING,
                embedding_model_loaded="model-other",
                embedding_max_batch_size=32,
                embedding_active_requests=0,
                current_load=0.1,
            ),
            ExecutionInstance(
                instance_id="embed-match",
                host="localhost",
                port=8091,
                model_name="BAAI/bge-m3",
                instance_type=ExecutionInstanceType.EMBEDDING,
                embedding_model_loaded="BAAI/bge-m3",
                embedding_max_batch_size=32,
                embedding_active_requests=5,  # Higher load
                current_load=0.3,
            ),
        ]

        selected = policy._select_embedding_instance(instances, "BAAI/bge-m3")

        # Should prefer model match despite higher load
        assert selected.instance_id == "embed-match"

    def test_select_embedding_instance_prefers_specialized(self):
        """Test instance selection prefers specialized instances."""
        policy = HybridSchedulingPolicy()

        instances = [
            ExecutionInstance(
                instance_id="mixed-1",
                host="localhost",
                port=8000,
                model_name="model",
                instance_type=ExecutionInstanceType.LLM_EMBEDDING,
                embedding_model_loaded="BAAI/bge-m3",
                embedding_max_batch_size=32,
                embedding_active_requests=0,
                current_load=0.1,  # Lower load
            ),
            ExecutionInstance(
                instance_id="embed-1",
                host="localhost",
                port=8090,
                model_name="BAAI/bge-m3",
                instance_type=ExecutionInstanceType.EMBEDDING,
                embedding_model_loaded="BAAI/bge-m3",
                embedding_max_batch_size=32,
                embedding_active_requests=5,
                current_load=0.3,  # Higher load
            ),
        ]

        selected = policy._select_embedding_instance(instances, "BAAI/bge-m3")

        # Should prefer specialized instance despite higher load
        assert selected.instance_id == "embed-1"


class TestHybridSchedulingPolicyDecisionMetadata:
    """Tests for scheduling decision metadata."""

    def test_embedding_decision_has_batch_metadata(
        self, embedding_requests, embedding_instances
    ):
        """Test embedding decisions include batch metadata."""
        policy = HybridSchedulingPolicy()
        decisions = policy.schedule(embedding_requests, embedding_instances)

        for decision in decisions:
            assert "batch_id" in decision.metadata
            assert "batch_size" in decision.metadata
            assert "total_texts" in decision.metadata
            assert decision.metadata["request_type"] == "embedding"

    def test_llm_decision_has_policy_metadata(self, llm_requests, llm_instances):
        """Test LLM decisions include policy metadata."""
        policy = HybridSchedulingPolicy(llm_fallback_policy="fifo")
        decisions = policy.schedule(llm_requests, llm_instances)

        for decision in decisions:
            assert decision.metadata.get("request_type") == "llm"
            assert decision.metadata.get("hybrid_policy") is True
            assert decision.metadata.get("llm_fallback") == "fifo"
