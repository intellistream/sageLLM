# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine integration tests."""

import pytest

from sage.common.components.sage_llm.sageLLM import (
    BenchmarkConfig,
    GenerateRequest,
    KVCacheConfig,
    ModelConfig,
    SageLLMConfig,
    SageLLMEngine,
)


class TestEngineIntegration:
    """Engine integration tests."""

    @pytest.fixture
    def engine(self):
        """Create test engine."""
        config = SageLLMConfig(
            model=ModelConfig(
                model_id="test-model",
                num_layers=2,
                num_heads=2,
                hidden_size=64,
            ),
            kv_cache=KVCacheConfig(
                max_tokens=1024,
                enable_prefix_caching=True,
            ),
            benchmark=BenchmarkConfig(
                enable_metrics=True,
            ),
        )
        engine = SageLLMEngine(config)
        engine.initialize()
        yield engine
        engine.shutdown()

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        stats = engine.get_stats()
        assert stats["initialized"] is True
        assert stats["backend"] is not None

    def test_basic_generate(self, engine):
        """Test basic generation."""
        request = GenerateRequest(
            request_id="test_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )

        output = engine.generate(request)

        assert output.request_id == "test_1"
        assert len(output.output_tokens) == 10
        assert output.finish_reason == "length"

    def test_metrics_collection(self, engine):
        """Test metrics are collected."""
        request = GenerateRequest(
            request_id="test_metrics",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=20,
        )

        output = engine.generate(request)

        assert output.metrics is not None
        assert "throughput_tps" in output.metrics
        assert "ttft_ms" in output.metrics
        assert "tpot_ms" in output.metrics
        assert output.metrics["throughput_tps"] > 0

    def test_multiple_requests(self, engine):
        """Test handling multiple requests."""
        requests = [
            GenerateRequest(
                request_id=f"req_{i}",
                prompt_tokens=[i, i + 1, i + 2],
                max_new_tokens=5,
            )
            for i in range(3)
        ]

        outputs = [engine.generate(req) for req in requests]

        assert len(outputs) == 3
        for output in outputs:
            assert len(output.output_tokens) == 5

        stats = engine.get_stats()
        assert stats["total_requests"] == 3
        assert stats["total_tokens"] == 15  # 3 requests * 5 tokens

    def test_kv_reuse(self, engine):
        """Test KV cache reuse."""
        # First request
        request1 = GenerateRequest(
            request_id="req_1",
            prompt_tokens=[1, 2, 3, 4, 5],
            max_new_tokens=5,
        )
        output1 = engine.generate(request1)

        # Second request with overlapping prefix
        request2 = GenerateRequest(
            request_id="req_2",
            prompt_tokens=[1, 2, 3, 4, 5, 6, 7],
            max_new_tokens=5,
        )
        output2 = engine.generate(request2)

        assert output2.finish_reason == "length"
        # In a real implementation with KV cache, we would check reuse metrics
        # For now, just verify it completes successfully

    def test_different_generation_lengths(self, engine):
        """Test generation with different lengths."""
        lengths = [5, 10, 20, 50]

        for length in lengths:
            request = GenerateRequest(
                request_id=f"len_{length}",
                prompt_tokens=[1, 2, 3],
                max_new_tokens=length,
            )
            output = engine.generate(request)
            assert len(output.output_tokens) == length

    def test_engine_stats(self, engine):
        """Test engine statistics tracking."""
        # Generate some requests
        for i in range(5):
            request = GenerateRequest(
                request_id=f"stat_{i}",
                prompt_tokens=[1, 2, 3],
                max_new_tokens=10,
            )
            engine.generate(request)

        stats = engine.get_stats()
        assert stats["total_requests"] == 5
        assert stats["total_tokens"] == 50
        assert stats["avg_throughput_tps"] > 0
        assert stats["uptime_s"] > 0


@pytest.mark.asyncio
class TestAsyncEngine:
    """Async engine tests."""

    @pytest.fixture
    def engine(self):
        """Create test engine."""
        config = SageLLMConfig(
            model=ModelConfig(
                model_id="test-model",
                num_layers=2,
                num_heads=2,
                hidden_size=64,
            ),
        )
        engine = SageLLMEngine(config)
        engine.initialize()
        yield engine
        engine.shutdown()

    async def test_async_generate(self, engine):
        """Test asynchronous generation."""
        request = GenerateRequest(
            request_id="async_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )

        output = await engine.generate_async(request)

        assert output.request_id == "async_1"
        assert len(output.output_tokens) == 10

    async def test_streaming(self, engine):
        """Test streaming generation."""
        request = GenerateRequest(
            request_id="stream_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )

        tokens = []
        async for token in engine.generate_stream(request):
            tokens.append(token)

        assert len(tokens) == 5


class TestConfiguration:
    """Configuration tests."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = SageLLMConfig(
            model=ModelConfig(
                model_id="test-model",
                num_layers=4,
                num_heads=4,
                hidden_size=128,
            ),
            kv_cache=KVCacheConfig(
                max_tokens=2048,
                block_size=32,
            ),
        )

        assert config.model.model_id == "test-model"
        assert config.model.num_layers == 4
        assert config.kv_cache.max_tokens == 2048
        assert config.kv_cache.block_size == 32

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = SageLLMConfig(
            model=ModelConfig(
                model_id="test-model",
                num_layers=2,
            ),
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert config_dict["model"]["model_id"] == "test-model"
