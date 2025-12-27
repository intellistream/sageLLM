# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Acceleration protocols and interfaces.

This module defines the protocol interfaces for acceleration:
- AccelControllerProtocol: Unified acceleration control
- QuantizerProtocol: Quantization interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .types import AccelConfig, QuantizationProfile


@runtime_checkable
class QuantizerProtocol(Protocol):
    """Protocol for quantization implementations.

    This protocol defines the interface for model quantization.

    Implementations:
        - AWQQuantizer: AWQ quantization
        - GPTQQuantizer: GPTQ quantization
        - FP8Quantizer: FP8 quantization
    """

    @property
    def profile(self) -> QuantizationProfile:
        """Get the quantization profile."""
        ...

    def quantize(
        self,
        model_path: str,
        output_path: str,
        calibration_data: list[str] | None = None,
    ) -> str:
        """Quantize a model.

        Args:
            model_path: Path to the original model.
            output_path: Path for the quantized model.
            calibration_data: Optional calibration data.

        Returns:
            Path to the quantized model.
        """
        ...

    def load_quantized(self, model_path: str) -> Any:
        """Load a quantized model.

        Args:
            model_path: Path to the quantized model.

        Returns:
            Loaded quantized model.
        """
        ...

    def get_compression_ratio(self) -> float:
        """Get the compression ratio achieved.

        Returns:
            Compression ratio (e.g., 4.0 for 4x compression).
        """
        ...


@runtime_checkable
class DraftModelProtocol(Protocol):
    """Protocol for speculative decoding draft models."""

    def generate_draft(
        self,
        input_ids: list[int],
        num_tokens: int,
    ) -> list[int]:
        """Generate draft tokens.

        Args:
            input_ids: Input token IDs.
            num_tokens: Number of draft tokens to generate.

        Returns:
            List of draft token IDs.
        """
        ...

    def get_draft_logits(
        self,
        input_ids: list[int],
    ) -> Any:
        """Get logits for draft tokens.

        Args:
            input_ids: Input token IDs.

        Returns:
            Logits tensor.
        """
        ...


@runtime_checkable
class VerifierProtocol(Protocol):
    """Protocol for speculative decoding verification."""

    def verify(
        self,
        input_ids: list[int],
        draft_ids: list[int],
        draft_logits: Any,
    ) -> tuple[list[int], int]:
        """Verify draft tokens against target model.

        Args:
            input_ids: Original input token IDs.
            draft_ids: Draft token IDs to verify.
            draft_logits: Logits from draft model.

        Returns:
            Tuple of (accepted_ids, num_accepted).
        """
        ...


@runtime_checkable
class AccelControllerProtocol(Protocol):
    """Protocol for unified acceleration control.

    This protocol defines the interface for managing all acceleration
    features (quantization, sparsity, speculative decoding, CoT).
    """

    @property
    def config(self) -> AccelConfig:
        """Get the acceleration configuration."""
        ...

    def setup(self, config: AccelConfig) -> None:
        """Setup acceleration with configuration.

        Args:
            config: Acceleration configuration.
        """
        ...

    def get_quantized_model(self, model_path: str) -> Any:
        """Get or create quantized model.

        Args:
            model_path: Path to the original model.

        Returns:
            Quantized model (or original if no quantization).
        """
        ...

    def run_speculative_decode(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[int]:
        """Run speculative decoding.

        Args:
            input_ids: Input token IDs.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            Generated token IDs.
        """
        ...

    def accelerate_cot(
        self,
        input_ids: list[int],
        cot_config: dict[str, Any],
    ) -> list[int]:
        """Run CoT-accelerated generation.

        Args:
            input_ids: Input token IDs.
            cot_config: CoT-specific configuration.

        Returns:
            Generated token IDs.
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get acceleration statistics.

        Returns:
            Dictionary with speedup, acceptance rate, etc.
        """
        ...
