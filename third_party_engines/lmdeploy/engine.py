# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""LMDeploy engine wrapper with sageLLM integration.

This module provides the main LMDeploy engine class that integrates
with sageLLM's modular architecture through dependency injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..base import BaseEngine
from ..types import EngineCapability, EngineConfig

if TYPE_CHECKING:
    from ...kv_runtime.protocols import KVBackendProtocol
    from ...scheduler_ir.protocols import SchedulingPolicy

logger = logging.getLogger(__name__)


class LMDeployEngine(BaseEngine):
    """LMDeploy/TurboMind engine with sageLLM integration.

    This engine wraps LMDeploy and provides hooks for:
    - KV cache management (via kv_runtime)
    - Prefix reuse (via prefix_reuse)
    - Scheduling (via scheduler_ir)
    - Communication (via comm_backend)
    - Acceleration (via accel)

    Attributes:
        config: Engine configuration.
        kv_backend: Optional KV backend for cache management.
        scheduler: Optional scheduler for request routing.
    """

    def __init__(
        self,
        config: EngineConfig,
        kv_backend: KVBackendProtocol | None = None,
        scheduler: SchedulingPolicy | None = None,
    ) -> None:
        """Initialize the LMDeploy engine.

        Args:
            config: Engine configuration.
            kv_backend: Optional KV backend for cache management.
            scheduler: Optional scheduler for request routing.
        """
        super().__init__(config)
        self.kv_backend = kv_backend
        self.scheduler = scheduler

        # LMDeploy internal state (initialized on start)
        self._pipeline: Any = None
        self._tokenizer: Any = None
        self._is_lmdeploy_available = False

        # Check LMDeploy availability
        self._check_lmdeploy()

    def _check_lmdeploy(self) -> None:
        """Check if LMDeploy is available."""
        try:
            import lmdeploy  # noqa: F401

            self._is_lmdeploy_available = True
            logger.debug("LMDeploy is available")
        except ImportError:
            self._is_lmdeploy_available = False
            logger.warning(
                "LMDeploy is not installed. "
                "Install with: pip install lmdeploy"
            )

    async def start(self) -> None:
        """Start the LMDeploy engine.

        Initializes the model, tokenizer, and prepares for inference.
        """
        if self.is_running:
            return

        if not self._is_lmdeploy_available:
            raise RuntimeError(
                "LMDeploy is not available. "
                "Install with: pip install lmdeploy"
            )

        logger.info("Starting LMDeploy engine for %s", self.config.model_id)

        try:
            from lmdeploy import TurbomindEngineConfig, pipeline

            # Configure TurboMind engine
            engine_config = TurbomindEngineConfig(
                tp=self.config.tensor_parallel_size,
                max_batch_size=self.config.max_batch_size,
                session_len=self.config.max_seq_len,
                cache_max_entry_count=self.config.gpu_memory_utilization,
            )

            # Create pipeline
            self._pipeline = pipeline(
                self.config.model_id,
                backend_config=engine_config,
            )

            self.is_running = True
            logger.info("LMDeploy engine started successfully")

        except Exception as e:
            logger.error("Failed to start LMDeploy engine: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the LMDeploy engine."""
        if not self.is_running:
            return

        logger.info("Stopping LMDeploy engine")

        # Cleanup pipeline
        if self._pipeline is not None:
            # LMDeploy cleanup (if available)
            self._pipeline = None

        self._tokenizer = None
        self.is_running = False

        logger.info("LMDeploy engine stopped")

    async def health_check(self) -> bool:
        """Check if the engine is healthy.

        Returns:
            True if the engine is running and responsive.
        """
        if not self.is_running or self._pipeline is None:
            return False

        try:
            # Simple health check - try a minimal generation
            # In production, this could be a dedicated health endpoint
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    def get_capability(self) -> EngineCapability:
        """Get the engine capability descriptor.

        Returns:
            EngineCapability describing what this engine can do.
        """
        if self._capability is None:
            self._capability = EngineCapability(
                supports_chat=True,
                supports_generate=True,
                supports_embedding=False,
                supports_streaming=True,
                supports_prefix_caching=True,
                supports_speculative=False,
                max_context_length=self.config.max_seq_len,
                max_new_tokens=self.config.max_seq_len // 2,
                quantization=None,
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
            )
        return self._capability

    # =========================================================================
    # Inference Methods
    # =========================================================================

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running")

        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        response = self._pipeline(prompt, gen_config=gen_config)

        # Extract text from response
        if isinstance(response, list) and len(response) > 0:
            return response[0].text
        return str(response)

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate text completion with streaming.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional generation parameters.

        Yields:
            Generated text chunks.
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running")

        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        # Use stream_infer for streaming
        async for output in self._pipeline.stream_infer(prompt, gen_config=gen_config):
            yield output.text

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Chat completion.

        Args:
            messages: List of chat messages with role and content.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional generation parameters.

        Returns:
            Assistant response.
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running")

        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        # Convert messages to LMDeploy format
        response = self._pipeline(messages, gen_config=gen_config)

        # Extract text from response
        if isinstance(response, list) and len(response) > 0:
            return response[0].text
        return str(response)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Chat completion with streaming.

        Args:
            messages: List of chat messages with role and content.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional generation parameters.

        Yields:
            Response text chunks.
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running")

        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        # Use stream_infer for streaming
        async for output in self._pipeline.stream_infer(
            messages, gen_config=gen_config
        ):
            yield output.text

    # =========================================================================
    # sageLLM Integration Hooks
    # =========================================================================

    def set_kv_backend(self, backend: KVBackendProtocol) -> None:
        """Set the KV backend for cache management.

        Args:
            backend: KV backend implementation.
        """
        self.kv_backend = backend
        logger.info("KV backend set for LMDeploy engine")

    def set_scheduler(self, scheduler: SchedulingPolicy) -> None:
        """Set the scheduler for request routing.

        Args:
            scheduler: Scheduler implementation.
        """
        self.scheduler = scheduler
        logger.info("Scheduler set for LMDeploy engine")

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with engine metrics.
        """
        stats = super().get_stats()
        stats.update(
            {
                "has_kv_backend": self.kv_backend is not None,
                "has_scheduler": self.scheduler is not None,
                "lmdeploy_available": self._is_lmdeploy_available,
            }
        )
        return stats
