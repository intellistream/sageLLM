# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Control Plane types and data structures."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane.types import (  # noqa: E402
    DecodingConfig,
    ExecutionInstance,
    ExecutionInstanceType,
    ParallelismType,
    PerformanceMetrics,
    PrefillingConfig,
    RequestMetadata,
    RequestPriority,
    RequestStatus,
    RequestType,
    SchedulingDecision,
)


class TestEnums:
    """Tests for Enum types."""

    def test_request_priority_ordering(self):
        """Test priority levels are ordered correctly."""
        assert RequestPriority.CRITICAL.value < RequestPriority.HIGH.value
        assert RequestPriority.HIGH.value < RequestPriority.NORMAL.value
        assert RequestPriority.NORMAL.value < RequestPriority.LOW.value
        assert RequestPriority.LOW.value < RequestPriority.BACKGROUND.value

    def test_request_status_values(self):
        """Test request status enum values."""
        assert RequestStatus.PENDING.value == "pending"
        assert RequestStatus.RUNNING.value == "running"
        assert RequestStatus.COMPLETED.value == "completed"
        assert RequestStatus.FAILED.value == "failed"

    def test_parallelism_type_values(self):
        """Test parallelism type enum values."""
        assert ParallelismType.TENSOR_PARALLEL.value == "tp"
        assert ParallelismType.PIPELINE_PARALLEL.value == "pp"
        assert ParallelismType.DATA_PARALLEL.value == "dp"
        assert ParallelismType.EXPERT_PARALLEL.value == "ep"
        assert ParallelismType.HYBRID.value == "hybrid"

    def test_execution_instance_type_values(self):
        """Test execution instance type enum values."""
        assert ExecutionInstanceType.GENERAL.value == "general"
        assert ExecutionInstanceType.PREFILLING.value == "prefilling"
        assert ExecutionInstanceType.DECODING.value == "decoding"
        assert ExecutionInstanceType.HYBRID.value == "hybrid"
        # T1.2: New instance types for hybrid scheduling
        assert ExecutionInstanceType.EMBEDDING.value == "embedding"
        assert ExecutionInstanceType.LLM_EMBEDDING.value == "llm_embedding"

    def test_execution_instance_type_all_values(self):
        """Test that all expected execution instance types exist."""
        all_types = list(ExecutionInstanceType)
        assert len(all_types) == 6
        assert ExecutionInstanceType.GENERAL in all_types
        assert ExecutionInstanceType.PREFILLING in all_types
        assert ExecutionInstanceType.DECODING in all_types
        assert ExecutionInstanceType.HYBRID in all_types
        assert ExecutionInstanceType.EMBEDDING in all_types
        assert ExecutionInstanceType.LLM_EMBEDDING in all_types

    def test_request_type_values(self):
        """Test request type enum values for hybrid scheduling."""
        assert RequestType.LLM_CHAT.value == "llm_chat"
        assert RequestType.LLM_GENERATE.value == "llm_generate"
        assert RequestType.EMBEDDING.value == "embedding"

    def test_request_type_all_values(self):
        """Test that all expected request types exist."""
        all_types = list(RequestType)
        assert len(all_types) == 3
        assert RequestType.LLM_CHAT in all_types
        assert RequestType.LLM_GENERATE in all_types
        assert RequestType.EMBEDDING in all_types


class TestRequestMetadata:
    """Tests for RequestMetadata."""

    def test_basic_creation(self):
        """Test basic request metadata creation."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello, world!",
            priority=RequestPriority.NORMAL,
            max_tokens=100,
        )

        assert request.request_id == "req-1"
        assert request.prompt == "Hello, world!"
        assert request.priority == RequestPriority.NORMAL
        assert request.max_tokens == 100

    def test_default_values(self):
        """Test default values are set correctly."""
        request = RequestMetadata(request_id="req-1")

        assert request.priority == RequestPriority.NORMAL
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.billing_tier == "standard"
        assert isinstance(request.tags, dict)

    def test_latency_calculation(self):
        """Test latency calculation from arrival to end time."""
        request = RequestMetadata(request_id="req-1")
        request.arrival_time = datetime.now()
        request.end_time = request.arrival_time + timedelta(milliseconds=150)

        latency = request.latency_ms
        assert latency is not None
        assert 149 <= latency <= 151  # Allow small floating point variance

    def test_latency_none_when_incomplete(self):
        """Test latency is None when request is incomplete."""
        request = RequestMetadata(request_id="req-1")
        request.arrival_time = datetime.now()
        request.end_time = None

        assert request.latency_ms is None

    def test_queue_wait_calculation(self):
        """Test queue waiting time calculation."""
        request = RequestMetadata(request_id="req-1")
        base_time = datetime.now()
        request.queue_time = base_time
        request.schedule_time = base_time + timedelta(milliseconds=50)

        queue_wait = request.queue_wait_ms
        assert queue_wait is not None
        assert 49 <= queue_wait <= 51

    def test_queue_wait_none_when_not_scheduled(self):
        """Test queue wait is None when not scheduled."""
        request = RequestMetadata(request_id="req-1")
        request.queue_time = datetime.now()
        request.schedule_time = None

        assert request.queue_wait_ms is None

    def test_with_slo_deadline(self):
        """Test request with SLO deadline."""
        request = RequestMetadata(
            request_id="req-1", slo_deadline_ms=100.0, priority=RequestPriority.HIGH
        )

        assert request.slo_deadline_ms == 100.0
        assert request.priority == RequestPriority.HIGH

    def test_with_tags(self):
        """Test request with custom tags."""
        request = RequestMetadata(request_id="req-1", tags={"experiment": "test", "version": "1.0"})

        assert request.tags["experiment"] == "test"
        assert request.tags["version"] == "1.0"

    # ============ Tests for Hybrid Scheduling Extensions (T1.1) ============

    def test_request_type_default(self):
        """Test default request type is LLM_CHAT for backward compatibility."""
        request = RequestMetadata(request_id="req-1")
        assert request.request_type == RequestType.LLM_CHAT

    def test_request_type_llm_chat(self):
        """Test LLM chat request type."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello, how are you?",
            request_type=RequestType.LLM_CHAT,
        )
        assert request.request_type == RequestType.LLM_CHAT
        assert request.is_llm_request is True
        assert request.is_embedding_request is False

    def test_request_type_llm_generate(self):
        """Test LLM generate request type."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Once upon a time",
            request_type=RequestType.LLM_GENERATE,
        )
        assert request.request_type == RequestType.LLM_GENERATE
        assert request.is_llm_request is True
        assert request.is_embedding_request is False

    def test_request_type_embedding(self):
        """Test embedding request type."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["text1", "text2"],
            embedding_model="BAAI/bge-m3",
        )
        assert request.request_type == RequestType.EMBEDDING
        assert request.is_embedding_request is True
        assert request.is_llm_request is False

    def test_embedding_texts_field(self):
        """Test embedding_texts field for embedding requests."""
        texts = ["Hello world", "How are you", "Test embedding"]
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=texts,
        )
        assert request.embedding_texts == texts
        assert len(request.embedding_texts) == 3

    def test_embedding_texts_default_none(self):
        """Test embedding_texts defaults to None."""
        request = RequestMetadata(request_id="req-1")
        assert request.embedding_texts is None

    def test_embedding_model_field(self):
        """Test embedding_model field."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        assert request.embedding_model == "BAAI/bge-small-zh-v1.5"

    def test_embedding_model_default_none(self):
        """Test embedding_model defaults to None."""
        request = RequestMetadata(request_id="req-1")
        assert request.embedding_model is None

    def test_embedding_batch_size_default(self):
        """Test embedding_batch_size default value."""
        request = RequestMetadata(request_id="req-1")
        assert request.embedding_batch_size == 32

    def test_embedding_batch_size_custom(self):
        """Test custom embedding_batch_size."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_batch_size=64,
        )
        assert request.embedding_batch_size == 64

    def test_effective_model_name_llm(self):
        """Test effective_model_name for LLM requests."""
        request = RequestMetadata(
            request_id="req-1",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            request_type=RequestType.LLM_CHAT,
        )
        assert request.effective_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_effective_model_name_embedding_with_embedding_model(self):
        """Test effective_model_name for embedding with embedding_model set."""
        request = RequestMetadata(
            request_id="req-1",
            model_name="Qwen/Qwen2.5-7B-Instruct",  # This should be ignored
            embedding_model="BAAI/bge-m3",
            request_type=RequestType.EMBEDDING,
        )
        assert request.effective_model_name == "BAAI/bge-m3"

    def test_effective_model_name_embedding_without_embedding_model(self):
        """Test effective_model_name for embedding without embedding_model set."""
        request = RequestMetadata(
            request_id="req-1",
            model_name="BAAI/bge-m3",
            request_type=RequestType.EMBEDDING,
        )
        assert request.effective_model_name == "BAAI/bge-m3"

    def test_backward_compatibility_existing_fields(self):
        """Test that existing fields still work (backward compatibility)."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Test prompt",
            user_id="user-123",
            priority=RequestPriority.HIGH,
            slo_deadline_ms=100.0,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            model_name="llama-7b",
        )
        assert request.prompt == "Test prompt"
        assert request.user_id == "user-123"
        assert request.priority == RequestPriority.HIGH
        assert request.slo_deadline_ms == 100.0
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.model_name == "llama-7b"
        # New fields should have defaults
        assert request.request_type == RequestType.LLM_CHAT
        assert request.embedding_texts is None
        assert request.embedding_model is None
        assert request.embedding_batch_size == 32


class TestExecutionInstance:
    """Tests for ExecutionInstance."""

    def test_basic_creation(self):
        """Test basic instance creation."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            gpu_count=1,
        )

        assert instance.instance_id == "inst-1"
        assert instance.host == "localhost"
        assert instance.port == 8000
        assert instance.model_name == "llama-7b"
        assert instance.gpu_count == 1

    def test_default_values(self):
        """Test default values for execution instance."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
        )

        assert instance.tensor_parallel_size == 1
        assert instance.pipeline_parallel_size == 1
        assert instance.data_parallel_size == 1
        assert instance.instance_type == ExecutionInstanceType.GENERAL
        assert instance.is_available is True
        assert instance.is_healthy is True
        assert instance.current_load == 0.0

    def test_available_capacity(self):
        """Test available capacity calculation."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            current_load=0.3,
        )

        assert instance.available_capacity == 0.7

    def test_available_capacity_floor(self):
        """Test available capacity doesn't go negative."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            current_load=1.5,  # Over capacity
        )

        assert instance.available_capacity == 0.0

    def test_can_accept_request_healthy(self):
        """Test can_accept_request when instance is healthy."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            is_available=True,
            is_healthy=True,
            current_load=0.5,
            active_requests=10,
            max_concurrent_requests=100,
        )

        assert instance.can_accept_request is True

    def test_can_accept_request_overloaded(self):
        """Test can_accept_request when instance is overloaded."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            current_load=0.96,  # Over 0.95 threshold
        )

        assert instance.can_accept_request is False

    def test_can_accept_request_at_max_concurrent(self):
        """Test can_accept_request when at max concurrent requests."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            active_requests=100,
            max_concurrent_requests=100,
        )

        assert instance.can_accept_request is False

    def test_can_accept_request_unhealthy(self):
        """Test can_accept_request when instance is unhealthy."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            is_healthy=False,
        )

        assert instance.can_accept_request is False

    def test_can_accept_prefilling_request(self):
        """Test prefilling request acceptance for different instance types."""
        # General instance should accept
        general = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
        )
        assert general.can_accept_prefilling_request() is True

        # Prefilling instance should accept
        prefilling = ExecutionInstance(
            instance_id="inst-2",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.PREFILLING,
        )
        assert prefilling.can_accept_prefilling_request() is True

        # Decoding instance should NOT accept
        decoding = ExecutionInstance(
            instance_id="inst-3",
            host="localhost",
            port=8002,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.DECODING,
        )
        assert decoding.can_accept_prefilling_request() is False

    def test_can_accept_decoding_request(self):
        """Test decoding request acceptance for different instance types."""
        # General instance should accept
        general = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
        )
        assert general.can_accept_decoding_request() is True

        # Decoding instance should accept
        decoding = ExecutionInstance(
            instance_id="inst-2",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.DECODING,
        )
        assert decoding.can_accept_decoding_request() is True

        # Prefilling instance should NOT accept
        prefilling = ExecutionInstance(
            instance_id="inst-3",
            host="localhost",
            port=8002,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.PREFILLING,
        )
        assert prefilling.can_accept_decoding_request() is False

    def test_affinity_score_nvlink(self):
        """Test affinity score for NVLINK-connected instances."""
        inst1 = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            machine_id="machine-1",
            nvlink_peers=["inst-2"],
        )
        inst2 = ExecutionInstance(
            instance_id="inst-2",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            machine_id="machine-1",
        )

        score = inst1.get_affinity_score(inst2)
        assert score == 1.0  # NVLINK connected

    def test_affinity_score_same_machine(self):
        """Test affinity score for same machine without NVLINK."""
        inst1 = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            machine_id="machine-1",
        )
        inst2 = ExecutionInstance(
            instance_id="inst-2",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            machine_id="machine-1",
        )

        score = inst1.get_affinity_score(inst2)
        assert score == 0.5  # Same machine, no NVLINK

    def test_affinity_score_same_rack(self):
        """Test affinity score for same rack."""
        inst1 = ExecutionInstance(
            instance_id="inst-1",
            host="host1",
            port=8000,
            model_name="llama-7b",
            machine_id="machine-1",
            rack_id="rack-1",
        )
        inst2 = ExecutionInstance(
            instance_id="inst-2",
            host="host2",
            port=8001,
            model_name="llama-7b",
            machine_id="machine-2",
            rack_id="rack-1",
        )

        score = inst1.get_affinity_score(inst2)
        assert score == 0.1  # Same rack

    def test_affinity_score_different_rack(self):
        """Test affinity score for different racks."""
        inst1 = ExecutionInstance(
            instance_id="inst-1",
            host="host1",
            port=8000,
            model_name="llama-7b",
            rack_id="rack-1",
        )
        inst2 = ExecutionInstance(
            instance_id="inst-2",
            host="host2",
            port=8001,
            model_name="llama-7b",
            rack_id="rack-2",
        )

        score = inst1.get_affinity_score(inst2)
        assert score == 0.01  # Different rack

    def test_is_local_to(self):
        """Test locality check between instances."""
        inst1 = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            machine_id="machine-1",
        )
        inst2 = ExecutionInstance(
            instance_id="inst-2",
            host="localhost",
            port=8001,
            model_name="llama-7b",
            machine_id="machine-1",
        )
        inst3 = ExecutionInstance(
            instance_id="inst-3",
            host="host2",
            port=8002,
            model_name="llama-7b",
            machine_id="machine-2",
        )

        assert inst1.is_local_to(inst2) is True
        assert inst1.is_local_to(inst3) is False

    # ============ Tests for Hybrid Scheduling Extensions (T1.2) ============

    def test_embedding_instance_type(self):
        """Test EMBEDDING instance type creation."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            embedding_model_loaded="BAAI/bge-m3",
        )
        assert instance.instance_type == ExecutionInstanceType.EMBEDDING
        assert instance.embedding_model_loaded == "BAAI/bge-m3"

    def test_llm_embedding_instance_type(self):
        """Test LLM_EMBEDDING mixed instance type creation."""
        instance = ExecutionInstance(
            instance_id="mixed-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.LLM_EMBEDDING,
            embedding_model_loaded="BAAI/bge-m3",
        )
        assert instance.instance_type == ExecutionInstanceType.LLM_EMBEDDING
        assert instance.embedding_model_loaded == "BAAI/bge-m3"

    def test_embedding_fields_defaults(self):
        """Test default values for embedding-related fields."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
        )
        assert instance.supported_request_types is None
        assert instance.embedding_model_loaded is None
        assert instance.embedding_max_batch_size == 32
        assert instance.embedding_active_requests == 0

    def test_embedding_fields_custom(self):
        """Test custom values for embedding-related fields."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            supported_request_types=[RequestType.EMBEDDING],
            embedding_model_loaded="BAAI/bge-m3",
            embedding_max_batch_size=64,
            embedding_active_requests=10,
        )
        assert instance.supported_request_types == [RequestType.EMBEDDING]
        assert instance.embedding_model_loaded == "BAAI/bge-m3"
        assert instance.embedding_max_batch_size == 64
        assert instance.embedding_active_requests == 10

    def test_get_effective_supported_request_types_general(self):
        """Test effective supported types for GENERAL instance."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
        )
        types = instance.get_effective_supported_request_types()
        assert RequestType.LLM_CHAT in types
        assert RequestType.LLM_GENERATE in types
        assert RequestType.EMBEDDING not in types

    def test_get_effective_supported_request_types_embedding(self):
        """Test effective supported types for EMBEDDING instance."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
        )
        types = instance.get_effective_supported_request_types()
        assert RequestType.EMBEDDING in types
        assert RequestType.LLM_CHAT not in types
        assert RequestType.LLM_GENERATE not in types

    def test_get_effective_supported_request_types_llm_embedding(self):
        """Test effective supported types for LLM_EMBEDDING mixed instance."""
        instance = ExecutionInstance(
            instance_id="mixed-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.LLM_EMBEDDING,
        )
        types = instance.get_effective_supported_request_types()
        assert RequestType.LLM_CHAT in types
        assert RequestType.LLM_GENERATE in types
        assert RequestType.EMBEDDING in types

    def test_get_effective_supported_request_types_explicit(self):
        """Test that explicit supported_request_types overrides defaults."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
            supported_request_types=[RequestType.LLM_CHAT],  # Only chat
        )
        types = instance.get_effective_supported_request_types()
        assert types == [RequestType.LLM_CHAT]
        assert RequestType.LLM_GENERATE not in types

    def test_can_handle_request_type_llm(self):
        """Test can_handle_request_type for LLM requests."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
        )
        assert instance.can_handle_request_type(RequestType.LLM_CHAT) is True
        assert instance.can_handle_request_type(RequestType.LLM_GENERATE) is True
        assert instance.can_handle_request_type(RequestType.EMBEDDING) is False

    def test_can_handle_request_type_embedding(self):
        """Test can_handle_request_type for EMBEDDING instance."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
        )
        assert instance.can_handle_request_type(RequestType.EMBEDDING) is True
        assert instance.can_handle_request_type(RequestType.LLM_CHAT) is False
        assert instance.can_handle_request_type(RequestType.LLM_GENERATE) is False

    def test_can_handle_request_type_mixed(self):
        """Test can_handle_request_type for LLM_EMBEDDING mixed instance."""
        instance = ExecutionInstance(
            instance_id="mixed-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.LLM_EMBEDDING,
        )
        assert instance.can_handle_request_type(RequestType.LLM_CHAT) is True
        assert instance.can_handle_request_type(RequestType.LLM_GENERATE) is True
        assert instance.can_handle_request_type(RequestType.EMBEDDING) is True

    def test_can_accept_embedding_request_embedding_instance(self):
        """Test can_accept_embedding_request for EMBEDDING instance."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            embedding_max_batch_size=32,
            embedding_active_requests=10,
        )
        assert instance.can_accept_embedding_request() is True

    def test_can_accept_embedding_request_at_max_batch(self):
        """Test can_accept_embedding_request when at max batch size."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            embedding_max_batch_size=32,
            embedding_active_requests=32,  # At max
        )
        assert instance.can_accept_embedding_request() is False

    def test_can_accept_embedding_request_general_instance(self):
        """Test can_accept_embedding_request for GENERAL instance (no embedding support)."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
        )
        assert instance.can_accept_embedding_request() is False

    def test_can_accept_embedding_request_mixed_instance(self):
        """Test can_accept_embedding_request for LLM_EMBEDDING mixed instance."""
        instance = ExecutionInstance(
            instance_id="mixed-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.LLM_EMBEDDING,
            embedding_max_batch_size=32,
            embedding_active_requests=10,
        )
        assert instance.can_accept_embedding_request() is True

    def test_can_accept_embedding_request_unhealthy(self):
        """Test can_accept_embedding_request when instance is unhealthy."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            is_healthy=False,
        )
        assert instance.can_accept_embedding_request() is False

    def test_can_accept_embedding_request_unavailable(self):
        """Test can_accept_embedding_request when instance is unavailable."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            is_available=False,
        )
        assert instance.can_accept_embedding_request() is False

    def test_can_accept_embedding_request_overloaded(self):
        """Test can_accept_embedding_request when instance is overloaded."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
            current_load=0.96,  # Over 0.95 threshold
        )
        assert instance.can_accept_embedding_request() is False

    def test_can_accept_llm_request_general(self):
        """Test can_accept_llm_request for GENERAL instance."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
        )
        assert instance.can_accept_llm_request() is True

    def test_can_accept_llm_request_embedding_only(self):
        """Test can_accept_llm_request for EMBEDDING-only instance."""
        instance = ExecutionInstance(
            instance_id="embed-1",
            host="localhost",
            port=8090,
            model_name="BAAI/bge-m3",
            instance_type=ExecutionInstanceType.EMBEDDING,
        )
        assert instance.can_accept_llm_request() is False

    def test_can_accept_llm_request_mixed(self):
        """Test can_accept_llm_request for LLM_EMBEDDING mixed instance."""
        instance = ExecutionInstance(
            instance_id="mixed-1",
            host="localhost",
            port=8000,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            instance_type=ExecutionInstanceType.LLM_EMBEDDING,
        )
        assert instance.can_accept_llm_request() is True

    def test_can_accept_llm_request_unhealthy(self):
        """Test can_accept_llm_request when instance is unhealthy."""
        instance = ExecutionInstance(
            instance_id="inst-1",
            host="localhost",
            port=8000,
            model_name="llama-7b",
            instance_type=ExecutionInstanceType.GENERAL,
            is_healthy=False,
        )
        assert instance.can_accept_llm_request() is False


class TestSchedulingDecision:
    """Tests for SchedulingDecision."""

    def test_basic_creation(self):
        """Test basic scheduling decision creation."""
        decision = SchedulingDecision(
            request_id="req-1",
            target_instance_id="inst-1",
            parallelism_strategy=ParallelismType.TENSOR_PARALLEL,
            estimated_latency_ms=50.0,
            estimated_cost=0.01,
        )

        assert decision.request_id == "req-1"
        assert decision.target_instance_id == "inst-1"
        assert decision.parallelism_strategy == ParallelismType.TENSOR_PARALLEL
        assert decision.estimated_latency_ms == 50.0
        assert decision.estimated_cost == 0.01

    def test_with_reason_and_confidence(self):
        """Test scheduling decision with reason and confidence."""
        decision = SchedulingDecision(
            request_id="req-1",
            target_instance_id="inst-1",
            parallelism_strategy=ParallelismType.HYBRID,
            estimated_latency_ms=100.0,
            estimated_cost=0.05,
            reason="Load balanced selection",
            confidence=0.85,
        )

        assert decision.reason == "Load balanced selection"
        assert decision.confidence == 0.85


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = PerformanceMetrics(
            total_requests=100,
            completed_requests=90,
            failed_requests=5,
            active_requests=5,
        )

        assert metrics.total_requests == 100
        assert metrics.completed_requests == 90
        assert metrics.failed_requests == 5
        assert metrics.active_requests == 5

    def test_latency_metrics(self):
        """Test latency metrics."""
        metrics = PerformanceMetrics(
            avg_latency_ms=50.0,
            p50_latency_ms=45.0,
            p95_latency_ms=80.0,
            p99_latency_ms=120.0,
        )

        assert metrics.avg_latency_ms == 50.0
        assert metrics.p50_latency_ms == 45.0
        assert metrics.p95_latency_ms == 80.0
        assert metrics.p99_latency_ms == 120.0


class TestPDConfigs:
    """Tests for Prefilling and Decoding configurations."""

    def test_prefilling_config_defaults(self):
        """Test prefilling config default values."""
        config = PrefillingConfig()

        assert config.target_batch_size == 64
        assert config.tensor_parallel_size == 4
        assert config.enable_kv_cache is True
        assert config.enable_chunked_prefill is True

    def test_prefilling_config_custom(self):
        """Test prefilling config with custom values."""
        config = PrefillingConfig(
            target_batch_size=128,
            tensor_parallel_size=8,
            enable_chunked_prefill=False,
        )

        assert config.target_batch_size == 128
        assert config.tensor_parallel_size == 8
        assert config.enable_chunked_prefill is False

    def test_decoding_config_defaults(self):
        """Test decoding config default values."""
        config = DecodingConfig()

        assert config.target_latency_ms == 50.0
        assert config.tensor_parallel_size == 1
        assert config.enable_prefix_caching is True
        assert config.max_parallel_requests == 200

    def test_decoding_config_custom(self):
        """Test decoding config with custom values."""
        config = DecodingConfig(
            target_latency_ms=30.0,
            max_parallel_requests=500,
            kv_cache_memory_fraction=0.9,
        )

        assert config.target_latency_ms == 30.0
        assert config.max_parallel_requests == 500
        assert config.kv_cache_memory_fraction == 0.9
