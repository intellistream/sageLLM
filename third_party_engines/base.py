# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Base engine class and protocols.

This module defines the base engine abstraction for all
inference engine implementations.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from .types import EngineCapability, EngineConfig

logger = logging.getLogger(__name__)


class BaseEngine(abc.ABC):
    """Abstract base class for inference engines.

    All engine implementations (LMDeploy, vLLM, etc.) must inherit
    from this class and implement the required methods.

    Attributes:
        config: Engine configuration.
        is_running: Whether the engine is currently running.
    """

    def __init__(self, config: EngineConfig) -> None:
        """Initialize the engine.

        Args:
            config: Engine configuration.
        """
        self.config = config
        self.is_running = False
        self._capability: EngineCapability | None = None

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the engine.

        This method should initialize the model, allocate resources,
        and prepare the engine for inference.
        """
        ...

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the engine.

        This method should gracefully shutdown the engine,
        release resources, and cleanup.
        """
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the engine is healthy.

        Returns:
            True if the engine is healthy and ready for requests.
        """
        ...

    @abc.abstractmethod
    def get_capability(self) -> EngineCapability:
        """Get the engine capability descriptor.

        Returns:
            EngineCapability describing what this engine can do.
        """
        ...

    # =========================================================================
    # Inference Methods
    # =========================================================================

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
        ...

    # =========================================================================
    # Optional Methods (Override as needed)
    # =========================================================================

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            NotImplementedError: If embedding is not supported.
        """
        raise NotImplementedError("Embedding not supported by this engine")

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with engine metrics.
        """
        return {
            "is_running": self.is_running,
            "model_id": self.config.model_id,
            "backend": self.config.backend.value,
        }

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> BaseEngine:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.stop()


# Re-export EngineCapability for convenience
from .types import EngineCapability  # noqa: E402

__all__ = ["BaseEngine", "EngineCapability"]
