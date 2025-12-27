# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Engine Backend - Integration with vLLM inference engine.

This module provides a wrapper for vLLM, enabling:
- PagedAttention-based KV cache management
- Continuous batching
- Tensor/Pipeline parallelism

Requirements:
- vllm >= 0.4.0

Example:
    >>> from sageLLM.third_party_engines.vllm import VLLMEngine
    >>> engine = VLLMEngine(
    ...     model="meta-llama/Llama-2-7b-hf",
    ...     tensor_parallel_size=2,
    ... )
    >>> await engine.start()
    >>> response = await engine.generate("Hello, world")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from . import register_engine
from .base import BaseEngine
from .types import EngineCapability, EngineConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_engine("vllm")
class VLLMEngine(BaseEngine):
    """vLLM engine backend.

    This engine wraps vLLM and provides integration with sageLLM's
    control plane for request scheduling and routing.

    Attributes:
        config: Engine configuration.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of GPUs for pipeline parallelism.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        model: str | None = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the vLLM engine.

        Args:
            config: Engine configuration (alternative to individual params).
            model: Model name or path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            pipeline_parallel_size: Number of GPUs for pipeline parallelism.
            gpu_memory_utilization: GPU memory utilization ratio.
            max_model_len: Maximum model context length.
            **kwargs: Additional vLLM configuration.
        """
        if config is None:
            config = EngineConfig(
                model_id=model or "meta-llama/Llama-2-7b-hf",
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        super().__init__(config)

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._extra_kwargs = kwargs

        # vLLM internal state
        self._llm: Any = None
        self._sampling_params: Any = None
        self._is_vllm_available = False

        self._check_vllm()

    def _check_vllm(self) -> None:
        """Check if vLLM is available."""
        try:
            import vllm  # noqa: F401

            self._is_vllm_available = True
            logger.debug("vLLM is available")
        except ImportError:
            self._is_vllm_available = False
            logger.warning("vLLM is not installed. Install with: pip install vllm")

    async def start(self) -> None:
        """Start the vLLM engine."""
        if not self._is_vllm_available:
            msg = "vLLM is not installed"
            raise RuntimeError(msg)

        if self.is_running:
            logger.warning("Engine is already running")
            return

        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Starting vLLM engine with model: {self.config.model_id}")

            self._llm = LLM(
                model=self.config.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                **self._extra_kwargs,
            )

            self._sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )

            self.is_running = True
            logger.info("vLLM engine started successfully")

        except Exception as e:
            logger.error(f"Failed to start vLLM engine: {e}")
            raise

    async def stop(self) -> None:
        """Stop the vLLM engine."""
        if not self.is_running:
            return

        # vLLM doesn't have explicit cleanup, just release references
        self._llm = None
        self._sampling_params = None
        self.is_running = False
        logger.info("vLLM engine stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional sampling parameters.

        Returns:
            Generated text.
        """
        if not self.is_running:
            msg = "Engine is not running"
            raise RuntimeError(msg)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )

        outputs = self._llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate text with streaming.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional sampling parameters.

        Yields:
            Generated text chunks.
        """
        # vLLM's offline LLM doesn't support streaming directly
        # For streaming, use vllm.AsyncLLMEngine instead
        result = await self.generate(
            prompt, max_tokens, temperature, top_p, **kwargs
        )
        yield result

    def get_capability(self) -> EngineCapability:
        """Get engine capabilities."""
        return EngineCapability(
            supports_streaming=True,
            supports_batching=True,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=True,
            supports_prefix_caching=True,
            supports_speculative_decoding=False,  # Depends on vLLM version
            max_batch_size=256,
            max_sequence_length=self.max_model_len or 4096,
        )


__all__ = ["VLLMEngine"]
