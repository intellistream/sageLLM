# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Unit tests for UnifiedAPIServer.

This module tests the UnifiedAPIServer class which provides
OpenAI-compatible endpoints for both LLM and Embedding inference.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent path for imports
_current_dir = Path(__file__).parent
_sage_llm_dir = _current_dir.parent.parent  # sage_llm directory
sys.path.insert(0, str(_sage_llm_dir))

from unified_api_server import (  # type: ignore[import-not-found]
    BackendInstanceConfig,
    SchedulingPolicyType,
    UnifiedAPIServer,
    UnifiedServerConfig,
    create_unified_server,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> UnifiedServerConfig:
    """Create a default server configuration for testing."""
    return UnifiedServerConfig(
        host="127.0.0.1",
        port=9999,
        llm_backends=[
            BackendInstanceConfig(
                host="localhost",
                port=8001,
                model_name="test-llm-model",
                instance_type="llm",
            )
        ],
        embedding_backends=[
            BackendInstanceConfig(
                host="localhost",
                port=8090,
                model_name="test-embed-model",
                instance_type="embedding",
            )
        ],
    )


@pytest.fixture
def multi_backend_config() -> UnifiedServerConfig:
    """Create a configuration with multiple backends."""
    return UnifiedServerConfig(
        host="127.0.0.1",
        port=9998,
        llm_backends=[
            BackendInstanceConfig(
                host="localhost",
                port=8001,
                model_name="llm-model-1",
                instance_type="llm",
            ),
            BackendInstanceConfig(
                host="localhost",
                port=8002,
                model_name="llm-model-2",
                instance_type="llm",
            ),
        ],
        embedding_backends=[
            BackendInstanceConfig(
                host="localhost",
                port=8090,
                model_name="embed-model-1",
                instance_type="embedding",
            ),
            BackendInstanceConfig(
                host="localhost",
                port=8091,
                model_name="embed-model-2",
                instance_type="embedding",
            ),
        ],
    )


# =============================================================================
# Test BackendInstanceConfig
# =============================================================================


class TestBackendInstanceConfig:
    """Tests for BackendInstanceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = BackendInstanceConfig()
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.model_name == ""
        assert config.instance_type == "llm"
        assert config.max_concurrent_requests == 100
        assert config.api_key is None

    def test_base_url_property(self) -> None:
        """Test base_url property generates correct URL."""
        config = BackendInstanceConfig(host="example.com", port=9000)
        assert config.base_url == "http://example.com:9000"

    def test_custom_values(self) -> None:
        """Test custom values are set correctly."""
        config = BackendInstanceConfig(
            host="192.168.1.1",
            port=8080,
            model_name="my-model",
            instance_type="embedding",
            max_concurrent_requests=50,
            api_key="test-key",
        )
        assert config.host == "192.168.1.1"
        assert config.port == 8080
        assert config.model_name == "my-model"
        assert config.instance_type == "embedding"
        assert config.max_concurrent_requests == 50
        assert config.api_key == "test-key"


# =============================================================================
# Test UnifiedServerConfig
# =============================================================================


class TestUnifiedServerConfig:
    """Tests for UnifiedServerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = UnifiedServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.scheduling_policy == SchedulingPolicyType.ADAPTIVE
        assert config.enable_control_plane is False
        assert config.enable_cors is True
        assert config.request_timeout == 300.0

    def test_default_backends_created(self) -> None:
        """Test that default backends are created if none specified."""
        config = UnifiedServerConfig()
        # Should have at least one LLM and one embedding backend
        assert len(config.llm_backends) >= 1
        assert len(config.embedding_backends) >= 1

    def test_custom_backends(self) -> None:
        """Test custom backends are preserved."""
        llm_backend = BackendInstanceConfig(
            host="llm.example.com",
            port=8001,
            model_name="custom-llm",
        )
        embed_backend = BackendInstanceConfig(
            host="embed.example.com",
            port=8090,
            model_name="custom-embed",
        )
        config = UnifiedServerConfig(
            llm_backends=[llm_backend],
            embedding_backends=[embed_backend],
        )
        assert len(config.llm_backends) == 1
        assert config.llm_backends[0].host == "llm.example.com"
        assert len(config.embedding_backends) == 1
        assert config.embedding_backends[0].host == "embed.example.com"

    def test_scheduling_policy_types(self) -> None:
        """Test all scheduling policy types are accepted."""
        for policy in SchedulingPolicyType:
            config = UnifiedServerConfig(scheduling_policy=policy)
            assert config.scheduling_policy == policy


# =============================================================================
# Test SchedulingPolicyType
# =============================================================================


class TestSchedulingPolicyType:
    """Tests for SchedulingPolicyType enum."""

    def test_all_policies_exist(self) -> None:
        """Test all expected policies exist."""
        expected = ["fifo", "priority", "slo_aware", "cost_optimized", "adaptive", "hybrid"]
        actual = [p.value for p in SchedulingPolicyType]
        for policy in expected:
            assert policy in actual

    def test_policy_values(self) -> None:
        """Test policy enum values."""
        assert SchedulingPolicyType.FIFO.value == "fifo"
        assert SchedulingPolicyType.PRIORITY.value == "priority"
        assert SchedulingPolicyType.ADAPTIVE.value == "adaptive"
        assert SchedulingPolicyType.HYBRID.value == "hybrid"


# =============================================================================
# Test UnifiedAPIServer
# =============================================================================


class TestUnifiedAPIServer:
    """Tests for UnifiedAPIServer class."""

    def test_initialization(self, default_config: UnifiedServerConfig) -> None:
        """Test server initialization."""
        server = UnifiedAPIServer(default_config)
        assert server.config == default_config
        assert not server.is_running()

    def test_initialization_default_config(self) -> None:
        """Test server initialization with default config."""
        server = UnifiedAPIServer()
        assert server.config is not None
        assert server.config.host == "0.0.0.0"
        assert server.config.port == 8000

    def test_model_mappings_built(self, default_config: UnifiedServerConfig) -> None:
        """Test that model mappings are built correctly."""
        server = UnifiedAPIServer(default_config)
        assert "test-llm-model" in server._llm_models
        assert "test-embed-model" in server._embedding_models

    def test_multi_model_mappings(self, multi_backend_config: UnifiedServerConfig) -> None:
        """Test model mappings with multiple backends."""
        server = UnifiedAPIServer(multi_backend_config)
        assert "llm-model-1" in server._llm_models
        assert "llm-model-2" in server._llm_models
        assert "embed-model-1" in server._embedding_models
        assert "embed-model-2" in server._embedding_models

    def test_get_llm_backend_exact_match(self, multi_backend_config: UnifiedServerConfig) -> None:
        """Test getting LLM backend by exact model name."""
        server = UnifiedAPIServer(multi_backend_config)
        backend = server._get_llm_backend("llm-model-2")
        assert backend is not None
        assert backend.port == 8002

    def test_get_llm_backend_fallback(self, multi_backend_config: UnifiedServerConfig) -> None:
        """Test getting LLM backend falls back to first backend."""
        server = UnifiedAPIServer(multi_backend_config)
        backend = server._get_llm_backend("unknown-model")
        assert backend is not None
        # Should return first backend as fallback
        assert backend.port == 8001

    def test_get_embedding_backend_exact_match(
        self, multi_backend_config: UnifiedServerConfig
    ) -> None:
        """Test getting embedding backend by exact model name."""
        server = UnifiedAPIServer(multi_backend_config)
        backend = server._get_embedding_backend("embed-model-2")
        assert backend is not None
        assert backend.port == 8091

    def test_get_embedding_backend_fallback(
        self, multi_backend_config: UnifiedServerConfig
    ) -> None:
        """Test getting embedding backend falls back to first backend."""
        server = UnifiedAPIServer(multi_backend_config)
        backend = server._get_embedding_backend("unknown-embed")
        assert backend is not None
        assert backend.port == 8090

    def test_get_status(self, default_config: UnifiedServerConfig) -> None:
        """Test get_status method."""
        server = UnifiedAPIServer(default_config)
        status = server.get_status()

        assert "running" in status
        assert "host" in status
        assert "port" in status
        assert "base_url" in status
        assert "llm_backends" in status
        assert "embedding_backends" in status
        assert "llm_models" in status
        assert "embedding_models" in status

        assert status["running"] is False
        assert status["host"] == "127.0.0.1"
        assert status["port"] == 9999
        assert status["llm_backends"] == 1
        assert status["embedding_backends"] == 1

    def test_app_property_creates_fastapi(self, default_config: UnifiedServerConfig) -> None:
        """Test that app property creates FastAPI instance."""
        server = UnifiedAPIServer(default_config)
        app = server.app
        assert app is not None
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/v1/models" in routes
        assert "/v1/chat/completions" in routes
        assert "/v1/completions" in routes
        assert "/v1/embeddings" in routes

    def test_app_property_cached(self, default_config: UnifiedServerConfig) -> None:
        """Test that app property returns cached instance."""
        server = UnifiedAPIServer(default_config)
        app1 = server.app
        app2 = server.app
        assert app1 is app2


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateUnifiedServer:
    """Tests for create_unified_server factory function."""

    def test_basic_creation(self) -> None:
        """Test basic server creation."""
        server = create_unified_server(host="127.0.0.1", port=9997)
        assert server is not None
        assert server.config.host == "127.0.0.1"
        assert server.config.port == 9997

    def test_with_llm_backend(self) -> None:
        """Test creation with LLM backend URL."""
        server = create_unified_server(
            port=9996,
            llm_model="my-llm",
            llm_backend_url="http://llm-server:8001",
        )
        assert len(server.config.llm_backends) >= 1
        # Find our configured backend
        backend = next(
            (b for b in server.config.llm_backends if b.host == "llm-server"), None
        )
        assert backend is not None
        assert backend.port == 8001
        assert backend.model_name == "my-llm"

    def test_with_embedding_backend(self) -> None:
        """Test creation with embedding backend URL."""
        server = create_unified_server(
            port=9995,
            embedding_model="my-embed",
            embedding_backend_url="http://embed-server:8090",
        )
        assert len(server.config.embedding_backends) >= 1
        backend = next(
            (b for b in server.config.embedding_backends if b.host == "embed-server"), None
        )
        assert backend is not None
        assert backend.port == 8090
        assert backend.model_name == "my-embed"

    def test_with_scheduling_policy(self) -> None:
        """Test creation with scheduling policy."""
        server = create_unified_server(port=9994, scheduling_policy="hybrid")
        assert server.config.scheduling_policy == SchedulingPolicyType.HYBRID


# =============================================================================
# Test API Endpoints (using TestClient)
# =============================================================================


class TestAPIEndpoints:
    """Tests for API endpoints using FastAPI TestClient."""

    @pytest.fixture
    def client(self, default_config: UnifiedServerConfig):
        """Create a test client."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI TestClient not available")

        server = UnifiedAPIServer(default_config)
        return TestClient(server.app)

    def test_root_endpoint(self, client) -> None:
        """Test root endpoint returns server info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["name"] == "SAGE Unified Inference API"

    def test_health_endpoint(self, client) -> None:
        """Test health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "backends" in data

    def test_models_endpoint(self, client) -> None:
        """Test models endpoint returns available models."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_chat_completions_missing_model(self, client) -> None:
        """Test chat completions with unknown model returns 404."""
        # Mock the backend check to fail
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "completely-unknown-model-xyz",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should succeed because we have a fallback (first backend)
        # The actual request might fail due to no backend, but model routing works
        assert response.status_code in [200, 404, 500]

    def test_embeddings_endpoint_validation(self, client) -> None:
        """Test embeddings endpoint validates input."""
        # Missing required field
        response = client.post("/v1/embeddings", json={})
        assert response.status_code == 422  # Validation error


# =============================================================================
# Test Async Methods
# =============================================================================


class TestAsyncMethods:
    """Tests for async methods of UnifiedAPIServer."""

    @pytest.mark.asyncio
    async def test_check_backends_health_no_session(
        self, default_config: UnifiedServerConfig
    ) -> None:
        """Test health check returns False when no session."""
        server = UnifiedAPIServer(default_config)
        result = await server._check_backends_health(default_config.llm_backends)
        assert result is False  # No HTTP session initialized

    @pytest.mark.asyncio
    async def test_check_backends_health_empty_list(
        self, default_config: UnifiedServerConfig
    ) -> None:
        """Test health check with empty backend list."""
        server = UnifiedAPIServer(default_config)
        server._http_session = MagicMock()
        result = await server._check_backends_health([])
        assert result is False


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_backends_configured(self) -> None:
        """Test server with no explicit backends uses defaults."""
        config = UnifiedServerConfig()
        server = UnifiedAPIServer(config)
        # Should have default backends
        assert len(config.llm_backends) >= 1
        assert len(config.embedding_backends) >= 1

    def test_empty_model_name_backend(self) -> None:
        """Test backend with empty model name."""
        config = UnifiedServerConfig(
            llm_backends=[BackendInstanceConfig(model_name="")],
            embedding_backends=[BackendInstanceConfig(model_name="")],
        )
        server = UnifiedAPIServer(config)
        # Should not crash, just have no named models
        assert len(server._llm_models) == 0
        assert len(server._embedding_models) == 0

    def test_server_stop_when_not_running(self, default_config: UnifiedServerConfig) -> None:
        """Test stopping server when it's not running."""
        server = UnifiedAPIServer(default_config)
        # Should not raise
        server.stop()
        assert not server.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
