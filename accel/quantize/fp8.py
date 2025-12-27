# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""FP8 quantization implementation.

Provides FP8 E4M3 and E5M2 quantization for weights and activations.

References:
- FP8 Formats for Deep Learning: https://arxiv.org/abs/2209.05433
- NVIDIA H100 FP8 Training: https://docs.nvidia.com/deeplearning/transformer-engine/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import (
    QuantizationConfig,
    QuantizationGranularity,
    QuantizationOutput,
    QuantizationType,
    Quantizer,
    QuantizerRegistry,
)

if TYPE_CHECKING:
    pass


@dataclass
class FP8Format:
    """FP8 format definition."""

    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    max_value: float
    min_value: float  # Minimum positive value


# E4M3: 4-bit exponent, 3-bit mantissa
FP8_E4M3 = FP8Format(
    name="E4M3",
    exponent_bits=4,
    mantissa_bits=3,
    exponent_bias=7,
    max_value=448.0,  # 2^8 * (1 + 7/8)
    min_value=2**-9,  # Minimum non-zero positive value
)

# E5M2: 5-bit exponent, 2-bit mantissa
FP8_E5M2 = FP8Format(
    name="E5M2",
    exponent_bits=5,
    mantissa_bits=2,
    exponent_bias=15,
    max_value=57344.0,  # 2^15 * (1 + 3/4)
    min_value=2**-16,  # Minimum non-zero positive value
)


@QuantizerRegistry.register(QuantizationType.FP8_E4M3)
class FP8E4M3Quantizer(Quantizer):
    """FP8 E4M3 quantizer.

    E4M3 is more suitable for weight quantization:
    - Larger dynamic range
    - Higher precision for values in [-1, 1] range
    """

    def __init__(self):
        self.format = FP8_E4M3

    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.FP8_E4M3

    def quantize(
        self,
        weight,  # torch.Tensor
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """FP8 E4M3 quantization with input validation."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for quantization")

        # Input validation
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")

        if weight.dim() < 2:
            raise ValueError(
                f"Weight must be at least 2D for quantization, got shape {weight.shape}"
            )

        if torch.isnan(weight).any():
            raise ValueError("Weight contains NaN values")

        if torch.isinf(weight).any():
            raise ValueError("Weight contains Inf values")

        # Compute scales
        if config.granularity == QuantizationGranularity.PER_TENSOR:
            scales = self._compute_scale_per_tensor(weight)
        elif config.granularity == QuantizationGranularity.PER_CHANNEL:
            scales = self._compute_scale_per_channel(weight)
        elif config.granularity == QuantizationGranularity.PER_GROUP:
            if config.group_size is None or config.group_size < 1:
                raise ValueError(
                    f"Valid group_size required for PER_GROUP, got {config.group_size}"
                )
            scales = self._compute_scale_per_group(weight, config.group_size)
        else:
            raise ValueError(f"Unsupported granularity: {config.granularity}")

        # Protect against zero division
        scales = scales.clamp(min=1e-8)

        # Ensure scale shape is compatible for broadcasting
        if scales.dim() == 1 and weight.dim() == 2:
            scales = scales.view(-1, 1)
        elif scales.dim() > 1:
            # For per-group quantization, reshape scales appropriately
            # scales shape: [out_features, num_groups]
            # Need to expand to [out_features, in_features]
            out_features = weight.shape[0]
            if scales.shape[0] != out_features:
                raise ValueError(
                    f"Scale shape {scales.shape} incompatible with weight shape {weight.shape}"
                )

        # Scale
        scaled_weight = weight / scales

        # Clip to FP8 range
        clipped = torch.clamp(
            scaled_weight,
            -self.format.max_value * config.clip_ratio,
            self.format.max_value * config.clip_ratio,
        )

        # Simulate FP8 rounding
        quantized = self._simulate_fp8_rounding(clipped)

        return QuantizationOutput(
            quantized_weight=quantized.to(torch.float16),
            scales=scales,
            quant_type=self.quant_type,
        )

    def _compute_scale_per_tensor(self, weight):  # -> torch.Tensor
        """Compute tensor-level scale."""
        abs_max = weight.abs().max()
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)

    def _compute_scale_per_channel(self, weight):  # -> torch.Tensor
        """Compute channel-level scale."""
        abs_max = weight.abs().amax(dim=1)
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)

    def _compute_scale_per_group(
        self,
        weight,  # torch.Tensor
        group_size: int,
    ):  # -> torch.Tensor
        """Compute group-level scale."""
        import torch

        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size

        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))

        # Reshape to [out_features, num_groups, group_size]
        grouped = weight.view(out_features, num_groups, group_size)

        # Compute scale per group
        abs_max = grouped.abs().amax(dim=-1)
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)

    def _simulate_fp8_rounding(self, x):  # -> torch.Tensor
        """Simulate FP8 rounding.

        On hardware without native FP8, use FP16 to simulate FP8 precision.
        """
        import torch

        # For E4M3, mantissa has 3 bits, precision is ~1/8
        precision = 2 ** (-self.format.mantissa_bits)

        # Round to nearest
        rounded = torch.round(x / precision) * precision
        return rounded

    def dequantize(self, output: QuantizationOutput):  # -> torch.Tensor
        """Dequantize."""
        return output.quantized_weight * output.scales.view(-1, 1)


@QuantizerRegistry.register(QuantizationType.FP8_E5M2)
class FP8E5M2Quantizer(Quantizer):
    """FP8 E5M2 quantizer.

    E5M2 is more suitable for activation quantization:
    - Larger dynamic range
    - More compatible with FP16 (same exponent bits)
    """

    def __init__(self):
        self.format = FP8_E5M2

    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.FP8_E5M2

    def quantize(
        self,
        weight,  # torch.Tensor
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """FP8 E5M2 quantization with input validation."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for quantization")

        # Input validation
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")

        if weight.dim() < 2:
            raise ValueError(
                f"Weight must be at least 2D for quantization, got shape {weight.shape}"
            )

        if torch.isnan(weight).any():
            raise ValueError("Weight contains NaN values")

        if torch.isinf(weight).any():
            raise ValueError("Weight contains Inf values")

        # Similar to E4M3 but using E5M2 format
        scales = self._compute_scale_per_tensor(weight)
        scales = scales.clamp(min=1e-8)  # Protect against zero division

        scaled_weight = weight / scales
        clipped = torch.clamp(scaled_weight, -self.format.max_value, self.format.max_value)
        quantized = self._simulate_fp8_rounding(clipped)

        return QuantizationOutput(
            quantized_weight=quantized.to(torch.float16),
            scales=scales,
            quant_type=self.quant_type,
        )

    def _compute_scale_per_tensor(self, weight):  # -> torch.Tensor
        abs_max = weight.abs().max()
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)

    def _simulate_fp8_rounding(self, x):  # -> torch.Tensor
        import torch

        precision = 2 ** (-self.format.mantissa_bits)
        rounded = torch.round(x / precision) * precision
        return rounded

    def dequantize(self, output: QuantizationOutput):  # -> torch.Tensor
        return output.quantized_weight * output.scales
