# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Unit tests for sageLLM core module - types."""

from sage.common.components.sage_llm.sageLLM.core.types import (
    EngineInfo,
    EngineKind,
    EngineState,
    RequestMetadata,
    RequestPriority,
    RequestType,
)


class TestEngineState:
    """Tests for EngineState enum."""

    def test_state_values(self) -> None:
        """Test that all expected states exist."""
        assert EngineState.STARTING.value == "STARTING"
        assert EngineState.READY.value == "READY"
        assert EngineState.DRAINING.value == "DRAINING"
        assert EngineState.STOPPED.value == "STOPPED"
        assert EngineState.ERROR.value == "ERROR"

    def test_state_is_string_enum(self) -> None:
        """Test that state can be used as string."""
        # EngineState inherits from str, so .value gives the string
        assert EngineState.READY.value == "READY"
        # Can compare directly with string
        assert EngineState.READY == "READY"


class TestEngineInfo:
    """Tests for EngineInfo dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating EngineInfo with minimal parameters."""
        info = EngineInfo(
            engine_id="test-1",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            host="localhost",
            port=8001,
        )
        assert info.engine_id == "test-1"
        assert info.model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert info.state == EngineState.STARTING
        assert info.engine_kind == EngineKind.LLM
        assert info.backend_type == "lmdeploy"

    def test_create_full(self) -> None:
        """Test creating EngineInfo with all parameters."""
        info = EngineInfo(
            engine_id="test-2",
            model_id="BAAI/bge-m3",
            host="192.168.1.100",
            port=8090,
            state=EngineState.READY,
            engine_kind=EngineKind.EMBEDDING,
            backend_type="tei",
            metadata={"gpu": "A100"},
        )
        assert info.engine_kind == EngineKind.EMBEDDING
        assert info.metadata["gpu"] == "A100"

    def test_is_healthy(self) -> None:
        """Test is_healthy property."""
        info = EngineInfo(
            engine_id="test",
            model_id="model",
            host="localhost",
            port=8001,
            state=EngineState.READY,
        )
        assert info.is_healthy is True

        info.state = EngineState.ERROR
        assert info.is_healthy is False

    def test_is_accepting_requests(self) -> None:
        """Test is_accepting_requests property."""
        info = EngineInfo(
            engine_id="test",
            model_id="model",
            host="localhost",
            port=8001,
            state=EngineState.READY,
        )
        assert info.is_accepting_requests is True

        info.state = EngineState.DRAINING
        assert info.is_accepting_requests is False

    def test_is_terminal(self) -> None:
        """Test is_terminal property."""
        info = EngineInfo(
            engine_id="test",
            model_id="model",
            host="localhost",
            port=8001,
        )

        info.state = EngineState.STOPPED
        assert info.is_terminal is True

        info.state = EngineState.ERROR
        assert info.is_terminal is True

        info.state = EngineState.READY
        assert info.is_terminal is False

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        info = EngineInfo(
            engine_id="test",
            model_id="model",
            host="localhost",
            port=8001,
        )
        data = info.to_dict()
        assert data["engine_id"] == "test"
        assert data["state"] == "STARTING"
        assert "created_at" in data


class TestRequestMetadata:
    """Tests for RequestMetadata dataclass."""

    def test_create_llm_request(self) -> None:
        """Test creating LLM request metadata."""
        meta = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.LLM_CHAT,
            model_name="Qwen/Qwen2.5-7B-Instruct",
        )
        assert meta.request_id == "req-1"
        assert meta.request_type == RequestType.LLM_CHAT
        assert meta.is_llm_request is True
        assert meta.is_embedding_request is False

    def test_create_embedding_request(self) -> None:
        """Test creating embedding request metadata."""
        meta = RequestMetadata(
            request_id="req-2",
            request_type=RequestType.EMBEDDING,
            model_name="BAAI/bge-m3",
        )
        assert meta.is_embedding_request is True
        assert meta.is_llm_request is False
        assert meta.model_name == "BAAI/bge-m3"

    def test_model_name(self) -> None:
        """Test model_name property."""
        # LLM request uses model_name
        llm_meta = RequestMetadata(
            request_id="req-1",
            model_name="qwen",
        )
        assert llm_meta.model_name == "qwen"

        # Embedding request also uses model_name
        embed_meta = RequestMetadata(
            request_id="req-2",
            request_type=RequestType.EMBEDDING,
            model_name="bge-m3",
        )
        assert embed_meta.model_name == "bge-m3"

    def test_priority_default(self) -> None:
        """Test default priority."""
        meta = RequestMetadata(request_id="req")
        assert meta.priority == RequestPriority.NORMAL

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        meta = RequestMetadata(
            request_id="req-1",
            model_name="test-model",
            priority=RequestPriority.HIGH,
        )
        data = meta.to_dict()
        assert data["request_id"] == "req-1"
        # RequestPriority is int enum, so value is the int
        assert data["priority"] == RequestPriority.HIGH.value
        assert data["priority"] == 1
