# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Unit tests for sageLLM core module - config."""

import tempfile
from pathlib import Path

from sage.common.components.sage_llm.sageLLM.core.config import (
    ControlPlaneConfig,
    EngineConfig,
    KVCachePreset,
    SageLLMConfig,
)


class TestKVCachePreset:
    """Tests for KVCachePreset enum."""

    def test_preset_values(self) -> None:
        """Test that all expected presets exist."""
        assert KVCachePreset.FP16.value == "fp16"
        assert KVCachePreset.FP8.value == "fp8"
        assert KVCachePreset.INT8.value == "int8"
        assert KVCachePreset.CUSTOM.value == "custom"


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating EngineConfig with minimal parameters."""
        config = EngineConfig(
            engine_id="engine-1",
            model_id="Qwen/Qwen2.5-7B-Instruct",
        )
        assert config.engine_id == "engine-1"
        assert config.backend == "lmdeploy"
        assert config.tensor_parallel_size == 1

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = EngineConfig(
            engine_id="engine-1",
            model_id="model",
            kv_cache_preset=KVCachePreset.FP8,
        )
        data = config.to_dict()
        assert data["engine_id"] == "engine-1"
        assert data["kv_cache_preset"] == "fp8"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "engine_id": "engine-1",
            "model_id": "model",
            "kv_cache_preset": "fp8",
            "tensor_parallel_size": 2,
        }
        config = EngineConfig.from_dict(data)
        assert config.engine_id == "engine-1"
        assert config.kv_cache_preset == KVCachePreset.FP8
        assert config.tensor_parallel_size == 2


class TestControlPlaneConfig:
    """Tests for ControlPlaneConfig dataclass."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = ControlPlaneConfig.default()
        assert config.health_check_interval == 10.0
        assert config.auto_restart is True
        assert config.max_restart_attempts == 3

    def test_to_dict(self) -> None:
        """Test serialization."""
        config = ControlPlaneConfig(
            health_check_interval=5.0,
            auto_restart=False,
        )
        data = config.to_dict()
        assert data["health_check_interval"] == 5.0
        assert data["auto_restart"] is False

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "health_check_interval": 15.0,
            "consecutive_failures_threshold": 5,
        }
        config = ControlPlaneConfig.from_dict(data)
        assert config.health_check_interval == 15.0
        assert config.consecutive_failures_threshold == 5


class TestSageLLMConfig:
    """Tests for SageLLMConfig dataclass."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = SageLLMConfig()
        assert config.default_backend == "lmdeploy"
        assert config.kv_cache_preset == KVCachePreset.FP16
        assert len(config.engines) == 0

    def test_with_engines(self) -> None:
        """Test configuration with engines."""
        engine = EngineConfig(
            engine_id="engine-1",
            model_id="model",
        )
        config = SageLLMConfig(engines=[engine])
        assert len(config.engines) == 1
        assert config.engines[0].engine_id == "engine-1"

    def test_to_dict(self) -> None:
        """Test serialization."""
        config = SageLLMConfig(
            default_backend="vllm",
            kv_cache_preset=KVCachePreset.FP8,
        )
        data = config.to_dict()
        assert data["default_backend"] == "vllm"
        assert data["kv_cache_preset"] == "fp8"
        assert "control_plane" in data

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "default_backend": "vllm",
            "kv_cache_preset": "int8",
            "control_plane": {
                "health_check_interval": 20.0,
            },
            "engines": [
                {
                    "engine_id": "e1",
                    "model_id": "m1",
                }
            ],
        }
        config = SageLLMConfig.from_dict(data)
        assert config.default_backend == "vllm"
        assert config.kv_cache_preset == KVCachePreset.INT8
        assert config.control_plane.health_check_interval == 20.0
        assert len(config.engines) == 1

    def test_yaml_round_trip(self) -> None:
        """Test YAML save and load."""
        config = SageLLMConfig(
            default_backend="lmdeploy",
            engines=[
                EngineConfig(
                    engine_id="engine-1",
                    model_id="Qwen/Qwen2.5-7B-Instruct",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save_yaml(str(path))

            loaded = SageLLMConfig.load_yaml(str(path))
            assert loaded.default_backend == config.default_backend
            assert len(loaded.engines) == 1
            assert loaded.engines[0].model_id == "Qwen/Qwen2.5-7B-Instruct"
