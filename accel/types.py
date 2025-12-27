# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Acceleration types and data structures.

This module defines the core types for inference acceleration:
- AccelConfig: Acceleration configuration
- QuantizationProfile: Quantization settings
- SpeculativeConfig: Speculative decoding settings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuantizationMethod(str, Enum):
    """Quantization method type."""

    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    SMOOTH_QUANT = "smooth_quant"


class SparsityPattern(str, Enum):
    """Sparsity pattern type."""

    NONE = "none"
    UNSTRUCTURED = "unstructured"
    STRUCTURED_2_4 = "structured_2_4"  # 2:4 structured sparsity
    BLOCK = "block"


class SpeculativeMethod(str, Enum):
    """Speculative decoding method."""

    NONE = "none"
    DRAFT_MODEL = "draft_model"
    SELF_SPECULATIVE = "self_speculative"
    MEDUSA = "medusa"
    EAGLE = "eagle"


@dataclass
class QuantizationProfile:
    """Quantization configuration profile.

    Attributes:
        method: Quantization method.
        bits: Number of bits for quantization.
        group_size: Group size for grouped quantization.
        symmetric: Whether to use symmetric quantization.
        calibration_samples: Number of calibration samples.
        model_path: Path to quantized model weights.
        metadata: Additional quantization metadata.
    """

    method: QuantizationMethod = QuantizationMethod.NONE
    bits: int = 16
    group_size: int = 128
    symmetric: bool = True
    calibration_samples: int = 128
    model_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "method": self.method.value,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "calibration_samples": self.calibration_samples,
            "model_path": self.model_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationProfile:
        """Deserialize from dictionary."""
        data = data.copy()
        if "method" in data and isinstance(data["method"], str):
            data["method"] = QuantizationMethod(data["method"])
        return cls(**data)

    @classmethod
    def preset(cls, name: str) -> QuantizationProfile:
        """Create a preset quantization profile.

        Args:
            name: Preset name (fp16, fp8, int8, int4, awq, gptq).

        Returns:
            Configured QuantizationProfile.
        """
        presets: dict[str, dict[str, Any]] = {
            "fp16": {"method": QuantizationMethod.NONE, "bits": 16},
            "fp8": {"method": QuantizationMethod.FP8, "bits": 8},
            "int8": {"method": QuantizationMethod.INT8, "bits": 8},
            "int4": {"method": QuantizationMethod.INT4, "bits": 4},
            "awq": {"method": QuantizationMethod.AWQ, "bits": 4, "group_size": 128},
            "gptq": {"method": QuantizationMethod.GPTQ, "bits": 4, "group_size": 128},
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets)}")
        return cls(**presets[name])


@dataclass
class SparsityConfig:
    """Sparsity configuration.

    Attributes:
        pattern: Sparsity pattern type.
        ratio: Target sparsity ratio (0.0 to 1.0).
        block_size: Block size for block sparsity.
        metadata: Additional sparsity metadata.
    """

    pattern: SparsityPattern = SparsityPattern.NONE
    ratio: float = 0.0
    block_size: int = 64
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeculativeConfig:
    """Speculative decoding configuration.

    Attributes:
        method: Speculative decoding method.
        draft_model_id: Model ID for draft model.
        num_speculative_tokens: Number of tokens to speculate.
        acceptance_threshold: Minimum acceptance probability.
        max_draft_tokens: Maximum draft tokens per step.
        metadata: Additional speculative decoding metadata.
    """

    method: SpeculativeMethod = SpeculativeMethod.NONE
    draft_model_id: str | None = None
    num_speculative_tokens: int = 5
    acceptance_threshold: float = 0.9
    max_draft_tokens: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "method": self.method.value,
            "draft_model_id": self.draft_model_id,
            "num_speculative_tokens": self.num_speculative_tokens,
            "acceptance_threshold": self.acceptance_threshold,
            "max_draft_tokens": self.max_draft_tokens,
            "metadata": self.metadata,
        }


@dataclass
class CoTAccelConfig:
    """Chain-of-Thought acceleration configuration.

    Attributes:
        enable: Whether CoT acceleration is enabled.
        draft_ratio: Ratio of draft tokens in CoT.
        verification_interval: Steps between verifications.
        fallback_threshold: Accuracy threshold for fallback.
        metadata: Additional CoT acceleration metadata.
    """

    enable: bool = False
    draft_ratio: float = 0.5
    verification_interval: int = 5
    fallback_threshold: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccelConfig:
    """Unified acceleration configuration.

    Combines all acceleration features into a single configuration.

    Attributes:
        quantization: Quantization configuration.
        sparsity: Sparsity configuration.
        speculative: Speculative decoding configuration.
        cot_accel: CoT acceleration configuration.
        enable_flash_attention: Whether to enable Flash Attention.
        enable_cuda_graphs: Whether to enable CUDA graphs.
        metadata: Additional acceleration metadata.
    """

    quantization: QuantizationProfile = field(default_factory=QuantizationProfile)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    cot_accel: CoTAccelConfig = field(default_factory=CoTAccelConfig)
    enable_flash_attention: bool = True
    enable_cuda_graphs: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "quantization": self.quantization.to_dict(),
            "sparsity": {
                "pattern": self.sparsity.pattern.value,
                "ratio": self.sparsity.ratio,
                "block_size": self.sparsity.block_size,
                "metadata": self.sparsity.metadata,
            },
            "speculative": self.speculative.to_dict(),
            "cot_accel": {
                "enable": self.cot_accel.enable,
                "draft_ratio": self.cot_accel.draft_ratio,
                "verification_interval": self.cot_accel.verification_interval,
                "fallback_threshold": self.cot_accel.fallback_threshold,
                "metadata": self.cot_accel.metadata,
            },
            "enable_flash_attention": self.enable_flash_attention,
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "metadata": self.metadata,
        }

    @classmethod
    def default(cls) -> AccelConfig:
        """Create default acceleration config (no acceleration)."""
        return cls()

    @classmethod
    def for_inference(cls, quantization: str = "fp16") -> AccelConfig:
        """Create config optimized for inference.

        Args:
            quantization: Quantization preset name.

        Returns:
            AccelConfig optimized for inference.
        """
        return cls(
            quantization=QuantizationProfile.preset(quantization),
            enable_flash_attention=True,
            enable_cuda_graphs=True,
        )
