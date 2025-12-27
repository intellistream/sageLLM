# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Core configuration types for sageLLM.

This module defines configuration dataclasses using pydantic-style
validation and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sage.common.config.ports import SagePorts


class KVCachePreset(str, Enum):
    """Preset KV cache configurations."""

    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    CUSTOM = "custom"


@dataclass
class EngineConfig:
    """Configuration for an inference engine.

    Attributes:
        engine_id: Unique identifier for the engine.
        model_id: Model identifier (HuggingFace ID or path).
        backend: Backend type (lmdeploy, vllm).
        host: Host address.
        port: Port number.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of stages for pipeline parallelism.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
        kv_cache_preset: KV cache configuration preset.
        metadata: Additional engine configuration.
    """

    engine_id: str
    model_id: str
    backend: str = "lmdeploy"
    host: str = "localhost"
    port: int = field(default_factory=lambda: SagePorts.LLM_DEFAULT)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_batch_size: int = 256
    max_seq_len: int = 8192
    kv_cache_preset: KVCachePreset = KVCachePreset.FP16
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "backend": self.backend,
            "host": self.host,
            "port": self.port,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "kv_cache_preset": self.kv_cache_preset.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        """Deserialize from dictionary."""
        data = data.copy()
        if "kv_cache_preset" in data and isinstance(data["kv_cache_preset"], str):
            data["kv_cache_preset"] = KVCachePreset(data["kv_cache_preset"])
        return cls(**data)


@dataclass
class ControlPlaneConfig:
    """Configuration for the Control Plane Manager.

    Attributes:
        health_check_interval: Interval between health checks (seconds).
        health_check_timeout: Timeout for health check requests (seconds).
        consecutive_failures_threshold: Failures before marking unhealthy.
        auto_restart: Whether to auto-restart failed engines.
        max_restart_attempts: Maximum restart attempts per engine.
        restart_backoff_base: Base time for exponential backoff (seconds).
        enable_metrics: Whether to collect metrics.
        metrics_port: Port for metrics endpoint.
        metadata: Additional configuration.
    """

    health_check_interval: float = 10.0
    health_check_timeout: float = 5.0
    consecutive_failures_threshold: int = 2
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_backoff_base: float = 1.0
    enable_metrics: bool = True
    metrics_port: int = field(default_factory=lambda: SagePorts.GATEWAY_DEFAULT + 1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "health_check_interval": self.health_check_interval,
            "health_check_timeout": self.health_check_timeout,
            "consecutive_failures_threshold": self.consecutive_failures_threshold,
            "auto_restart": self.auto_restart,
            "max_restart_attempts": self.max_restart_attempts,
            "restart_backoff_base": self.restart_backoff_base,
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlPlaneConfig:
        """Deserialize from dictionary."""
        return cls(**data)

    @classmethod
    def default(cls) -> ControlPlaneConfig:
        """Create default configuration."""
        return cls()


@dataclass
class SageLLMConfig:
    """Top-level sageLLM configuration.

    This is the main configuration class that aggregates all
    sageLLM settings.

    Attributes:
        control_plane: Control plane configuration.
        engines: List of engine configurations.
        default_backend: Default backend type.
        kv_cache_preset: Default KV cache preset.
        comm_backend: Communication backend type.
        accel_preset: Acceleration preset.
        metadata: Additional configuration.
    """

    control_plane: ControlPlaneConfig = field(default_factory=ControlPlaneConfig)
    engines: list[EngineConfig] = field(default_factory=list)
    default_backend: str = "lmdeploy"
    kv_cache_preset: KVCachePreset = KVCachePreset.FP16
    comm_backend: str = "nccl"
    accel_preset: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "control_plane": self.control_plane.to_dict(),
            "engines": [e.to_dict() for e in self.engines],
            "default_backend": self.default_backend,
            "kv_cache_preset": self.kv_cache_preset.value,
            "comm_backend": self.comm_backend,
            "accel_preset": self.accel_preset,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SageLLMConfig:
        """Deserialize from dictionary."""
        data = data.copy()
        if "control_plane" in data:
            data["control_plane"] = ControlPlaneConfig.from_dict(data["control_plane"])
        if "engines" in data:
            data["engines"] = [EngineConfig.from_dict(e) for e in data["engines"]]
        if "kv_cache_preset" in data and isinstance(data["kv_cache_preset"], str):
            data["kv_cache_preset"] = KVCachePreset(data["kv_cache_preset"])
        return cls(**data)

    @classmethod
    def load_yaml(cls, path: str) -> SageLLMConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded SageLLMConfig.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration.
        """
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
