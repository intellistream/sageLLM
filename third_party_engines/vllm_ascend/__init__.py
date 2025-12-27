# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-Ascend Engine Backend - vLLM optimized for Huawei Ascend NPUs.

This module provides a wrapper for vLLM-Ascend, a fork of vLLM
optimized for Huawei Ascend NPUs (Atlas 300/800/900 series).

Key differences from standard vLLM:
- Uses CANN instead of CUDA
- HCCL for collective communication
- Ascend-optimized attention kernels

Requirements:
- vllm-ascend (from https://github.com/vllm-project/vllm-ascend)
- CANN Toolkit 6.0+
- Ascend NPU driver

Example:
    >>> from sageLLM.third_party_engines.vllm_ascend import VLLMAscendEngine
    >>> engine = VLLMAscendEngine(
    ...     model="Qwen/Qwen2.5-7B-Instruct",
    ...     tensor_parallel_size=4,
    ... )
    >>> await engine.start()
    >>> response = await engine.generate("Hello, world")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from .. import register_engine
from ..base import BaseEngine
from ..types import EngineCapability, EngineConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_engine("vllm_ascend")
class VLLMAscendEngine(BaseEngine):
    """vLLM-Ascend engine backend for Huawei Ascend NPUs.

    This engine wraps vLLM-Ascend and provides integration with sageLLM's
    control plane for request scheduling and routing on Ascend hardware.

    Attributes:
        config: Engine configuration.
        tensor_parallel_size: Number of NPUs for tensor parallelism.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        model: str | None = None,
        tensor_parallel_size: int = 1,
        npu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the vLLM-Ascend engine.

        Args:
            config: Engine configuration (alternative to individual params).
            model: Model name or path.
            tensor_parallel_size: Number of NPUs for tensor parallelism.
            npu_memory_utilization: NPU memory utilization ratio.
            max_model_len: Maximum model context length.
            **kwargs: Additional vLLM-Ascend configuration.
        """
        if config is None:
            config = EngineConfig(
                model_id=model or "Qwen/Qwen2.5-7B-Instruct",
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=npu_memory_utilization,  # Reuse field
                max_model_len=max_model_len,
            )
        super().__init__(config)

        self.tensor_parallel_size = tensor_parallel_size
        self.npu_memory_utilization = npu_memory_utilization
        self.max_model_len = max_model_len
        self._extra_kwargs = kwargs

        # vLLM-Ascend internal state
        self._llm: Any = None
        self._sampling_params: Any = None
        self._is_available = False

        self._check_availability()

    def _check_availability(self) -> None:
        """Check if vLLM-Ascend and Ascend NPU are available."""
        try:
            # Check for Ascend NPU
            import torch_npu  # noqa: F401

            logger.debug("torch_npu is available")
        except ImportError:
            logger.warning(
                "torch_npu is not installed. "
                "Install CANN toolkit and torch_npu for Ascend support."
            )
            return

        try:
            # vLLM-Ascend uses the same API as vLLM
            import vllm  # noqa: F401

            self._is_available = True
            logger.debug("vLLM-Ascend is available")
        except ImportError:
            self._is_available = False
            logger.warning(
                "vLLM is not installed. "
                "Install vllm-ascend from https://github.com/vllm-project/vllm-ascend"
            )

    async def start(self) -> None:
        """Start the vLLM-Ascend engine."""
        if not self._is_available:
            msg = "vLLM-Ascend or Ascend NPU is not available"
            raise RuntimeError(msg)

        if self.is_running:
            logger.warning("Engine is already running")
            return

        try:
            from vllm import LLM, SamplingParams

            logger.info(
                f"Starting vLLM-Ascend engine with model: {self.config.model_id}"
            )

            # vLLM-Ascend automatically detects Ascend NPUs
            self._llm = LLM(
                model=self.config.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.npu_memory_utilization,
                max_model_len=self.max_model_len,
                device="npu",  # Ascend-specific
                **self._extra_kwargs,
            )

            self._sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )

            self.is_running = True
            logger.info("vLLM-Ascend engine started successfully")

        except Exception as e:
            logger.error(f"Failed to start vLLM-Ascend engine: {e}")
            raise

    async def stop(self) -> None:
        """Stop the vLLM-Ascend engine."""
        if not self.is_running:
            return

        self._llm = None
        self._sampling_params = None
        self.is_running = False
        logger.info("vLLM-Ascend engine stopped")

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
            supports_pipeline_parallel=False,  # Not yet in vLLM-Ascend
            supports_prefix_caching=True,
            supports_speculative_decoding=False,
            max_batch_size=128,  # Conservative for Ascend
            max_sequence_length=self.max_model_len or 4096,
        )


__all__ = ["VLLMAscendEngine"]
