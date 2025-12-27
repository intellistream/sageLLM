# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Engine types and data structures.

This module defines the core types for inference engines:
- EngineConfig: Engine configuration
- EngineCapability: Engine capability descriptor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..accel.types import AccelConfig
    from ..kv_runtime.types import KVCacheSchema


class EngineBackend(str, Enum):
    """Supported engine backends."""

    LMDEPLOY = "lmdeploy"
    VLLM = "vllm"
    MOCK = "mock"


class EngineRole(str, Enum):
    """Engine role in PD separation."""

    GENERAL = "general"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    HYBRID = "hybrid"


@dataclass
class EngineConfig:
    """Engine configuration.

    Attributes:
        model_id: Model identifier (HuggingFace ID or path).
        backend: Backend implementation to use.
        role: Engine role for PD separation.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of stages for pipeline parallelism.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
        gpu_memory_utilization: Target GPU memory utilization (0.0-1.0).
        kv_schema: KV cache configuration.
        accel_config: Acceleration configuration.
        host: Host address to bind.
        port: Port to bind.
        auto_download: Whether to auto-download model.
        trust_remote_code: Whether to trust remote code.
        metadata: Additional engine metadata.
    """

    model_id: str
    backend: EngineBackend = EngineBackend.LMDEPLOY
    role: EngineRole = EngineRole.GENERAL
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_batch_size: int = 256
    max_seq_len: int = 8192
    gpu_memory_utilization: float = 0.9
    kv_schema: KVCacheSchema | None = None
    accel_config: AccelConfig | None = None
    host: str = "0.0.0.0"
    port: int = 8001
    auto_download: bool = True
    trust_remote_code: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "backend": self.backend.value,
            "role": self.role.value,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_schema": self.kv_schema.to_dict() if self.kv_schema else None,
            "accel_config": self.accel_config.to_dict() if self.accel_config else None,
            "host": self.host,
            "port": self.port,
            "auto_download": self.auto_download,
            "trust_remote_code": self.trust_remote_code,
            "metadata": self.metadata,
        }


@dataclass
class EngineCapability:
    """Capability descriptor for an engine instance.

    Describes what an engine can do, used by Control Plane for routing.

    Attributes:
        supports_chat: Whether the engine supports chat API.
        supports_generate: Whether the engine supports generate API.
        supports_embedding: Whether the engine supports embedding API.
        supports_streaming: Whether the engine supports streaming.
        supports_prefix_caching: Whether prefix caching is enabled.
        supports_speculative: Whether speculative decoding is enabled.
        max_context_length: Maximum context length.
        max_new_tokens: Maximum new tokens per request.
        quantization: Quantization method (if any).
        tensor_parallel_size: Number of TP GPUs.
        pipeline_parallel_size: Number of PP stages.
        metadata: Additional capability metadata.
    """

    supports_chat: bool = True
    supports_generate: bool = True
    supports_embedding: bool = False
    supports_streaming: bool = True
    supports_prefix_caching: bool = True
    supports_speculative: bool = False
    max_context_length: int = 8192
    max_new_tokens: int = 4096
    quantization: str | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "supports_chat": self.supports_chat,
            "supports_generate": self.supports_generate,
            "supports_embedding": self.supports_embedding,
            "supports_streaming": self.supports_streaming,
            "supports_prefix_caching": self.supports_prefix_caching,
            "supports_speculative": self.supports_speculative,
            "max_context_length": self.max_context_length,
            "max_new_tokens": self.max_new_tokens,
            "quantization": self.quantization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "metadata": self.metadata,
        }

    def can_handle_request(
        self,
        request_type: str,
        context_length: int = 0,
        new_tokens: int = 0,
    ) -> bool:
        """Check if engine can handle a request.

        Args:
            request_type: Request type (chat, generate, embedding).
            context_length: Context length of the request.
            new_tokens: Requested new tokens.

        Returns:
            True if engine can handle the request.
        """
        # Check type support
        type_ok = {
            "chat": self.supports_chat,
            "generate": self.supports_generate,
            "embedding": self.supports_embedding,
        }.get(request_type, False)

        if not type_ok:
            return False

        # Check length constraints
        if context_length > self.max_context_length:
            return False
        if new_tokens > self.max_new_tokens:
            return False

        return True
