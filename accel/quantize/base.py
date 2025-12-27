# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Base quantization types and protocols.

Defines the core quantization interface and data structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class QuantizationType(Enum):
    """Quantization type."""

    NONE = auto()  # No quantization (FP16/BF16)
    FP32 = auto()  # FP32 (baseline)
    FP16 = auto()  # FP16
    BF16 = auto()  # BF16
    FP8_E4M3 = auto()  # FP8 (E4M3 format)
    FP8_E5M2 = auto()  # FP8 (E5M2 format)
    INT8 = auto()  # INT8 symmetric quantization
    INT4 = auto()  # INT4 grouped quantization
    NF4 = auto()  # NormalFloat 4-bit (QLoRA)


class QuantizationGranularity(Enum):
    """Quantization granularity."""

    PER_TENSOR = auto()  # Single scale for entire tensor
    PER_CHANNEL = auto()  # One scale per channel
    PER_GROUP = auto()  # Group quantization (e.g., every 128 elements)
    PER_TOKEN = auto()  # One scale per token (for activations)


@dataclass
class QuantizationConfig:
    """Quantization configuration.

    Attributes:
        quant_type: Quantization type.
        granularity: Quantization granularity.
        group_size: Group size for grouped quantization.
        calibration_samples: Number of calibration samples.
        sensitive_layers: Layers to keep in high precision.
        use_symmetric: Use symmetric quantization.
        clip_ratio: Clipping ratio.
    """

    quant_type: QuantizationType
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR

    # Group quantization parameters
    group_size: int = 128

    # Calibration parameters
    calibration_samples: int = 128

    # Mixed precision parameters
    sensitive_layers: list[str] = field(default_factory=list)

    # Algorithm parameters
    use_symmetric: bool = True  # Symmetric quantization
    clip_ratio: float = 1.0  # Clipping ratio

    def __post_init__(self) -> None:
        if not (0.0 < self.clip_ratio <= 1.0):
            raise ValueError(f"clip_ratio must be in (0, 1], got {self.clip_ratio}")

        if self.group_size is None or self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

        if self.calibration_samples < 1:
            raise ValueError(
                f"calibration_samples must be positive, got {self.calibration_samples}"
            )

        if self.granularity == QuantizationGranularity.PER_GROUP and self.group_size is None:
            raise ValueError("group_size is required for PER_GROUP granularity")


@dataclass
class QuantizationOutput:
    """Quantization output.

    Attributes:
        quantized_weight: Quantized weight tensor.
        scales: Scale factors.
        zeros: Zero points for asymmetric quantization.
        group_size: Group size used.
        quant_type: Quantization type used.
    """

    quantized_weight: Any  # torch.Tensor
    scales: Any  # torch.Tensor
    zeros: Any | None = None  # torch.Tensor for asymmetric quantization
    group_size: int = 128
    quant_type: QuantizationType = QuantizationType.INT8


class Quantizer(ABC):
    """Quantizer base class."""

    @property
    @abstractmethod
    def quant_type(self) -> QuantizationType:
        """Return quantization type."""
        ...

    @abstractmethod
    def quantize(
        self,
        weight: Any,  # torch.Tensor
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """Quantize weight.

        Args:
            weight: Original weight [out_features, in_features].
            config: Quantization configuration.

        Returns:
            Quantization output.
        """
        ...

    @abstractmethod
    def dequantize(
        self,
        output: QuantizationOutput,
    ) -> Any:  # torch.Tensor
        """Dequantize weight.

        Args:
            output: Quantization output.

        Returns:
            Dequantized weight.
        """
        ...


class QuantizerRegistry:
    """Quantizer registry."""

    _quantizers: dict[QuantizationType, type] = {}

    @classmethod
    def register(cls, quant_type: QuantizationType):
        """Decorator: register quantizer."""

        def decorator(quantizer_cls):
            cls._quantizers[quant_type] = quantizer_cls
            return quantizer_cls

        return decorator

    @classmethod
    def get(cls, quant_type: QuantizationType) -> Quantizer:
        """Get quantizer instance."""
        if quant_type not in cls._quantizers:
            raise ValueError(f"Unknown quantization type: {quant_type}")
        _require_hardware_support(quant_type)
        return cls._quantizers[quant_type]()

    @classmethod
    def list_available(cls) -> list[QuantizationType]:
        """List available quantization types."""
        return list(cls._quantizers.keys())


def _require_hardware_support(quant_type: QuantizationType) -> None:
    """Validate hardware support for a quantization type.

    Currently enforces GPU capability for FP8 paths to avoid silent misconfigurations
    on unsupported devices.
    """

    if quant_type not in {QuantizationType.FP8_E4M3, QuantizationType.FP8_E5M2}:
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Quantization {quant_type.name} requires torch with CUDA support; import failed: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Quantization {quant_type.name} requires a CUDA GPU (compute capability >= 8.0); CUDA not available"
        )

    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        raise RuntimeError(
            f"Quantization {quant_type.name} requires compute capability >= 8.0; detected {capability}"
        )
