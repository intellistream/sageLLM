# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for EmbeddingExecutor module.

This module contains unit tests for the EmbeddingExecutor class that handles
embedding requests via OpenAI-compatible HTTP API endpoints.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane.executors.embedding_executor import (  # noqa: E402  # type: ignore[import-not-found]
    EmbeddingExecutor,
    EmbeddingExecutorError,
    EmbeddingInstanceUnavailableError,
    EmbeddingMetrics,
    EmbeddingRequestError,
    EmbeddingResult,
    EmbeddingTimeoutError,
)


# ============ EmbeddingResult Tests ============


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_basic_creation(self):
        """Test basic EmbeddingResult creation."""
        result = EmbeddingResult(
            request_id="emb-123",
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="BAAI/bge-m3",
            usage={"prompt_tokens": 10, "total_tokens": 10},
            latency_ms=50.5,
        )

        assert result.request_id == "emb-123"
        assert len(result.embeddings) == 2
        assert result.model == "BAAI/bge-m3"
        assert result.usage["prompt_tokens"] == 10
        assert result.latency_ms == 50.5
        assert result.success is True
        assert result.error_message is None

    def test_embedding_count_property(self):
        """Test embedding_count property."""
        result = EmbeddingResult(
            request_id="emb-1",
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            model="model",
            usage={},
            latency_ms=10.0,
        )

        assert result.embedding_count == 3

    def test_embedding_count_empty(self):
        """Test embedding_count with empty embeddings."""
        result = EmbeddingResult(
            request_id="emb-1",
            embeddings=[],
            model="model",
            usage={},
            latency_ms=10.0,
        )

        assert result.embedding_count == 0

    def test_embedding_dimension_property(self):
        """Test embedding_dimension property."""
        result = EmbeddingResult(
            request_id="emb-1",
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            model="model",
            usage={},
            latency_ms=10.0,
        )

        assert result.embedding_dimension == 5

    def test_embedding_dimension_empty(self):
        """Test embedding_dimension with empty embeddings."""
        result = EmbeddingResult(
            request_id="emb-1",
            embeddings=[],
            model="model",
            usage={},
            latency_ms=10.0,
        )

        assert result.embedding_dimension is None

    def test_failed_result(self):
        """Test creating a failed EmbeddingResult."""
        result = EmbeddingResult(
            request_id="emb-fail",
            embeddings=[],
            model="model",
            usage={},
            latency_ms=100.0,
            success=False,
            error_message="Connection refused",
        )

        assert result.success is False
        assert "Connection refused" in result.error_message

    def test_result_with_metadata(self):
        """Test EmbeddingResult with custom metadata."""
        result = EmbeddingResult(
            request_id="emb-1",
            embeddings=[[0.1]],
            model="model",
            usage={},
            latency_ms=10.0,
            metadata={"host": "localhost", "port": 8090},
        )

        assert result.metadata["host"] == "localhost"
        assert result.metadata["port"] == 8090


# ============ EmbeddingMetrics Tests ============


class TestEmbeddingMetrics:
    """Tests for EmbeddingMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = EmbeddingMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_texts_embedded == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.retries_total == 0

    def test_avg_latency_ms(self):
        """Test avg_latency_ms calculation."""
        metrics = EmbeddingMetrics(
            successful_requests=10,
            total_latency_ms=500.0,
        )

        assert metrics.avg_latency_ms == 50.0

    def test_avg_latency_ms_no_requests(self):
        """Test avg_latency_ms with no requests."""
        metrics = EmbeddingMetrics()

        assert metrics.avg_latency_ms == 0.0

    def test_avg_texts_per_request(self):
        """Test avg_texts_per_request calculation."""
        metrics = EmbeddingMetrics(
            successful_requests=5,
            total_texts_embedded=50,
        )

        assert metrics.avg_texts_per_request == 10.0

    def test_avg_texts_per_request_no_requests(self):
        """Test avg_texts_per_request with no requests."""
        metrics = EmbeddingMetrics()

        assert metrics.avg_texts_per_request == 0.0

    def test_success_rate(self):
        """Test success_rate calculation."""
        metrics = EmbeddingMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
        )

        assert metrics.success_rate == 0.95

    def test_success_rate_no_requests(self):
        """Test success_rate with no requests."""
        metrics = EmbeddingMetrics()

        assert metrics.success_rate == 1.0

    def test_success_rate_all_failed(self):
        """Test success_rate when all requests failed."""
        metrics = EmbeddingMetrics(
            total_requests=10,
            successful_requests=0,
            failed_requests=10,
        )

        assert metrics.success_rate == 0.0


# ============ Exception Tests ============


class TestEmbeddingExceptions:
    """Tests for EmbeddingExecutor exception classes."""

    def test_embedding_executor_error(self):
        """Test base EmbeddingExecutorError."""
        error = EmbeddingExecutorError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_embedding_instance_unavailable_error(self):
        """Test EmbeddingInstanceUnavailableError."""
        error = EmbeddingInstanceUnavailableError("localhost", 8090)

        assert error.host == "localhost"
        assert error.port == 8090
        assert "localhost:8090" in str(error)
        assert "unavailable" in str(error).lower()

    def test_embedding_instance_unavailable_error_custom_message(self):
        """Test EmbeddingInstanceUnavailableError with custom message."""
        error = EmbeddingInstanceUnavailableError(
            "localhost", 8090, "Server is down for maintenance"
        )

        assert "Server is down for maintenance" in str(error)

    def test_embedding_timeout_error(self):
        """Test EmbeddingTimeoutError."""
        error = EmbeddingTimeoutError("localhost", 8090, 60.0)

        assert error.host == "localhost"
        assert error.port == 8090
        assert error.timeout_seconds == 60.0
        assert "timed out" in str(error).lower()
        assert "60" in str(error)

    def test_embedding_request_error(self):
        """Test EmbeddingRequestError."""
        error = EmbeddingRequestError(
            "localhost", 8090, 500, "Internal server error"
        )

        assert error.host == "localhost"
        assert error.port == 8090
        assert error.status_code == 500
        assert error.error_detail == "Internal server error"
        assert "500" in str(error)
        assert "Internal server error" in str(error)


# ============ EmbeddingExecutor Tests ============


class TestEmbeddingExecutorInit:
    """Tests for EmbeddingExecutor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        executor = EmbeddingExecutor()

        assert executor.max_retries == 3
        assert executor.retry_delay == 1.0
        assert executor._initialized is False
        assert executor.http_session is None

    def test_custom_init(self):
        """Test custom initialization."""
        executor = EmbeddingExecutor(
            timeout=120,
            max_retries=5,
            retry_delay=2.0,
        )

        assert executor.max_retries == 5
        assert executor.retry_delay == 2.0

    def test_metrics_initialized(self):
        """Test metrics are initialized."""
        executor = EmbeddingExecutor()
        metrics = executor.get_metrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0


class TestEmbeddingExecutorLifecycle:
    """Tests for EmbeddingExecutor lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialize creates HTTP session."""
        executor = EmbeddingExecutor()

        await executor.initialize()

        assert executor._initialized is True
        assert executor.http_session is not None

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test initialize is idempotent."""
        executor = EmbeddingExecutor()

        await executor.initialize()
        session1 = executor.http_session

        await executor.initialize()
        session2 = executor.http_session

        # Should be the same session
        assert session1 is session2

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup closes HTTP session."""
        executor = EmbeddingExecutor()

        await executor.initialize()
        await executor.cleanup()

        assert executor._initialized is False
        assert executor.http_session is None

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        """Test cleanup is idempotent."""
        executor = EmbeddingExecutor()

        await executor.initialize()
        await executor.cleanup()
        await executor.cleanup()  # Should not raise

        assert executor._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with EmbeddingExecutor() as executor:
            assert executor._initialized is True
            assert executor.http_session is not None

        assert executor._initialized is False
        assert executor.http_session is None


class TestEmbeddingExecutorExecute:
    """Tests for EmbeddingExecutor.execute_embedding method."""

    @pytest.fixture
    def mock_response_data(self):
        """Mock successful embedding response."""
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "BAAI/bge-m3",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

    @pytest.mark.asyncio
    async def test_execute_embedding_success(self, mock_response_data):
        """Test successful embedding execution."""
        executor = EmbeddingExecutor()

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor.initialize()
            executor.http_session = mock_session

            result = await executor.execute_embedding(
                texts=["Hello", "World"],
                model="BAAI/bge-m3",
                host="localhost",
                port=8090,
            )

            assert result.success is True
            assert len(result.embeddings) == 2
            assert result.model == "BAAI/bge-m3"
            assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_execute_embedding_auto_initialize(self):
        """Test auto-initialization when not initialized."""
        executor = EmbeddingExecutor()

        # Mock HTTP session and execute_http_request
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        async def mock_init():
            executor._initialized = True
            executor.http_session = mock_session

        with patch.object(executor, "initialize", side_effect=mock_init) as mock_init_method:
            with patch.object(
                executor,
                "_execute_http_request",
                new_callable=AsyncMock,
                return_value=EmbeddingResult(
                    request_id="test",
                    embeddings=[[0.1]],
                    model="model",
                    usage={},
                    latency_ms=10.0,
                ),
            ):
                await executor.execute_embedding(
                    texts=["test"],
                    model="model",
                    host="localhost",
                    port=8090,
                )
                mock_init_method.assert_called_once()

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_execute_embedding_generates_request_id(self):
        """Test request ID is generated if not provided."""
        executor = EmbeddingExecutor()
        await executor.initialize()

        with patch.object(
            executor,
            "_execute_http_request",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = EmbeddingResult(
                request_id="generated-id",
                embeddings=[[0.1]],
                model="model",
                usage={},
                latency_ms=10.0,
            )

            result = await executor.execute_embedding(
                texts=["test"],
                model="model",
                host="localhost",
                port=8090,
                # No request_id provided
            )

            # Should have a request_id
            call_args = mock_execute.call_args
            assert call_args is not None

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_execute_embedding_uses_provided_request_id(self):
        """Test provided request ID is used."""
        executor = EmbeddingExecutor()
        await executor.initialize()

        with patch.object(
            executor,
            "_execute_http_request",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = EmbeddingResult(
                request_id="my-custom-id",
                embeddings=[[0.1]],
                model="model",
                usage={},
                latency_ms=10.0,
            )

            await executor.execute_embedding(
                texts=["test"],
                model="model",
                host="localhost",
                port=8090,
                request_id="my-custom-id",
            )

            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs["request_id"] == "my-custom-id"

        await executor.cleanup()


class TestEmbeddingExecutorRetry:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on timeout."""
        executor = EmbeddingExecutor(max_retries=2, retry_delay=0.01)
        await executor.initialize()

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Timeout")
            return EmbeddingResult(
                request_id="test",
                embeddings=[[0.1]],
                model="model",
                usage={},
                latency_ms=10.0,
            )

        with patch.object(executor, "_execute_http_request", side_effect=mock_execute):
            result = await executor.execute_embedding(
                texts=["test"],
                model="model",
                host="localhost",
                port=8090,
            )

            assert result.success is True
            assert call_count == 3  # 1 initial + 2 retries

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_retry_exhausted_timeout(self):
        """Test all retries exhausted on timeout."""
        executor = EmbeddingExecutor(max_retries=2, retry_delay=0.01)
        await executor.initialize()

        with patch.object(
            executor,
            "_execute_http_request",
            side_effect=TimeoutError("Timeout"),
        ):
            with pytest.raises(EmbeddingTimeoutError):
                await executor.execute_embedding(
                    texts=["test"],
                    model="model",
                    host="localhost",
                    port=8090,
                )

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_no_retry_on_request_error(self):
        """Test no retry on server error response."""
        executor = EmbeddingExecutor(max_retries=2, retry_delay=0.01)
        await executor.initialize()

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise EmbeddingRequestError("localhost", 8090, 400, "Bad request")

        with patch.object(executor, "_execute_http_request", side_effect=mock_execute):
            with pytest.raises(EmbeddingRequestError):
                await executor.execute_embedding(
                    texts=["test"],
                    model="model",
                    host="localhost",
                    port=8090,
                )

            # Should not retry on EmbeddingRequestError
            assert call_count == 1

        await executor.cleanup()


class TestEmbeddingExecutorBatch:
    """Tests for batch embedding execution."""

    @pytest.mark.asyncio
    async def test_batch_small_list_no_batching(self):
        """Test small list doesn't trigger batching."""
        executor = EmbeddingExecutor()
        await executor.initialize()

        with patch.object(
            executor,
            "execute_embedding",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = EmbeddingResult(
                request_id="test",
                embeddings=[[0.1], [0.2]],
                model="model",
                usage={},
                latency_ms=10.0,
            )

            result = await executor.execute_embedding_batch(
                texts=["text1", "text2"],
                model="model",
                host="localhost",
                port=8090,
                batch_size=10,  # Batch size larger than text count
            )

            # Should call execute_embedding once (no batching)
            mock_execute.assert_called_once()
            assert len(result.embeddings) == 2

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_batch_large_list(self):
        """Test large list triggers batching."""
        executor = EmbeddingExecutor()
        await executor.initialize()

        call_count = 0

        async def mock_execute(texts, model, host, port, **kwargs):
            nonlocal call_count
            call_count += 1
            return EmbeddingResult(
                request_id=f"batch-{call_count}",
                embeddings=[[0.1] for _ in texts],
                model=model,
                usage={"prompt_tokens": len(texts), "total_tokens": len(texts)},
                latency_ms=10.0,
            )

        with patch.object(executor, "execute_embedding", side_effect=mock_execute):
            texts = [f"text-{i}" for i in range(10)]
            result = await executor.execute_embedding_batch(
                texts=texts,
                model="model",
                host="localhost",
                port=8090,
                batch_size=3,  # Should create 4 batches (3+3+3+1)
            )

            assert call_count == 4
            assert len(result.embeddings) == 10
            assert result.metadata["batch_count"] == 4

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_batch_combines_usage(self):
        """Test batch combines usage statistics."""
        executor = EmbeddingExecutor()
        await executor.initialize()

        async def mock_execute(texts, model, host, port, **kwargs):
            return EmbeddingResult(
                request_id="batch",
                embeddings=[[0.1] for _ in texts],
                model=model,
                usage={"prompt_tokens": 5, "total_tokens": 5},
                latency_ms=10.0,
            )

        with patch.object(executor, "execute_embedding", side_effect=mock_execute):
            texts = [f"text-{i}" for i in range(6)]
            result = await executor.execute_embedding_batch(
                texts=texts,
                model="model",
                host="localhost",
                port=8090,
                batch_size=2,  # 3 batches
            )

            # 3 batches * 5 tokens each
            assert result.usage["prompt_tokens"] == 15
            assert result.usage["total_tokens"] == 15

        await executor.cleanup()


class TestEmbeddingExecutorHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        executor = EmbeddingExecutor()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor.initialize()
            executor.http_session = mock_session

            is_healthy = await executor.health_check("localhost", 8090)

            assert is_healthy is True

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        executor = EmbeddingExecutor()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor.initialize()
            executor.http_session = mock_session

            is_healthy = await executor.health_check("localhost", 8090)

            assert is_healthy is False

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_health_check_auto_initialize(self):
        """Test health check auto-initializes if needed."""
        executor = EmbeddingExecutor()

        assert executor._initialized is False

        with patch.object(executor, "initialize", new_callable=AsyncMock):
            # Return False for simplicity since http_session won't be set
            result = await executor.health_check("localhost", 8090)
            # Without proper setup, should return False
            assert result is False


class TestEmbeddingExecutorMetrics:
    """Tests for metrics functionality."""

    def test_get_metrics(self):
        """Test get_metrics returns current metrics."""
        executor = EmbeddingExecutor()

        metrics = executor.get_metrics()

        assert isinstance(metrics, EmbeddingMetrics)
        assert metrics.total_requests == 0

    def test_reset_metrics(self):
        """Test reset_metrics clears all metrics."""
        executor = EmbeddingExecutor()

        # Simulate some activity
        executor.metrics.total_requests = 100
        executor.metrics.successful_requests = 95
        executor.metrics.failed_requests = 5
        executor.metrics.total_texts_embedded = 1000

        executor.reset_metrics()

        metrics = executor.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_texts_embedded == 0


class TestParseEmbeddingsResponse:
    """Tests for _parse_embeddings_response helper."""

    def test_parse_normal_response(self):
        """Test parsing normal OpenAI-format response."""
        executor = EmbeddingExecutor()

        response_data = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ]
        }

        embeddings = executor._parse_embeddings_response(response_data)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]

    def test_parse_out_of_order_response(self):
        """Test parsing response with out-of-order indices."""
        executor = EmbeddingExecutor()

        response_data = {
            "data": [
                {"embedding": [0.3, 0.4], "index": 1},
                {"embedding": [0.1, 0.2], "index": 0},
            ]
        }

        embeddings = executor._parse_embeddings_response(response_data)

        # Should be sorted by index
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        executor = EmbeddingExecutor()

        response_data = {"data": []}

        embeddings = executor._parse_embeddings_response(response_data)

        assert embeddings == []

    def test_parse_missing_data_key(self):
        """Test parsing response without data key."""
        executor = EmbeddingExecutor()

        response_data = {}

        embeddings = executor._parse_embeddings_response(response_data)

        assert embeddings == []

