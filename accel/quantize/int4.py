# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""INT4/INT8 quantization implementation.

Provides INT4 grouped quantization and NF4 (NormalFloat 4-bit) quantization.

References:
- GPTQ: https://arxiv.org/abs/2210.17323
- QLoRA: https://arxiv.org/abs/2305.14314
- AWQ: https://arxiv.org/abs/2306.00978
"""

from __future__ import annotations

from .base import (
    QuantizationConfig,
    QuantizationOutput,
    QuantizationType,
    Quantizer,
    QuantizerRegistry,
)


@QuantizerRegistry.register(QuantizationType.INT4)
class INT4Quantizer(Quantizer):
    """INT4 grouped quantizer.

    Uses grouped quantization for better accuracy:
    - Divides weight into groups (e.g., 128 elements per group)
    - Computes scale/zero-point per group
    """

    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.INT4

    def quantize(
        self,
        weight,  # torch.Tensor
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """INT4 quantization with input validation."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for quantization")

        # Input validation
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")

        if weight.dim() != 2:
            raise ValueError(
                f"Weight must be 2D for INT4 quantization, got shape {weight.shape}"
            )

        if torch.isnan(weight).any():
            raise ValueError("Weight contains NaN values")

        if torch.isinf(weight).any():
            raise ValueError("Weight contains Inf values")

        if config.group_size is None or config.group_size < 1:
            raise ValueError(
                f"Valid group_size required for INT4 quantization, got {config.group_size}"
            )

        out_features, in_features = weight.shape
        group_size = config.group_size
        num_groups = (in_features + group_size - 1) // group_size

        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))

        # Reshape to [out_features, num_groups, group_size]
        grouped = weight.view(out_features, num_groups, group_size)

        # Compute scales and zeros per group
        if config.use_symmetric:
            # Symmetric: [-7, 7] for 4-bit signed
            abs_max = grouped.abs().amax(dim=-1, keepdim=True)
            scales = abs_max / 7.0
            scales = scales.clamp(min=1e-8)
            zeros = torch.zeros_like(scales)
        else:
            # Asymmetric: [0, 15] for 4-bit unsigned
            min_val = grouped.amin(dim=-1, keepdim=True)
            max_val = grouped.amax(dim=-1, keepdim=True)
            scales = (max_val - min_val) / 15.0
            scales = scales.clamp(min=1e-8)
            zeros = -torch.round(min_val / scales)

        # Quantize
        quantized = torch.round(grouped / scales + zeros).clamp(
            -7 if config.use_symmetric else 0, 7 if config.use_symmetric else 15
        )

        # Reshape back
        quantized = quantized.view(out_features, -1)[:, :in_features]

        return QuantizationOutput(
            quantized_weight=quantized.to(torch.int8),
            scales=scales.view(out_features, num_groups),
            zeros=zeros.view(out_features, num_groups) if not config.use_symmetric else None,
            group_size=group_size,
            quant_type=self.quant_type,
        )

    def dequantize(self, output: QuantizationOutput):  # -> torch.Tensor
        """Dequantize."""
        import torch

        quantized = output.quantized_weight.float()
        scales = output.scales
        zeros = output.zeros if output.zeros is not None else torch.zeros_like(scales)

        out_features, in_features = quantized.shape
        num_groups = scales.shape[1]
        group_size = output.group_size

        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            quantized = torch.nn.functional.pad(quantized, (0, pad_size))

        # Reshape and dequantize
        grouped = quantized.view(out_features, num_groups, group_size)
        dequantized = (grouped - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)

        # Reshape back
        return dequantized.view(out_features, -1)[:, :in_features]


@QuantizerRegistry.register(QuantizationType.INT8)
class INT8Quantizer(Quantizer):
    """INT8 symmetric quantizer."""

    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.INT8

    def quantize(
        self,
        weight,  # torch.Tensor
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """INT8 quantization with input validation."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for quantization")

        # Input validation
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")

        if weight.dim() != 2:
            raise ValueError(
                f"Weight must be 2D for INT8 quantization, got shape {weight.shape}"
            )

        if torch.isnan(weight).any():
            raise ValueError("Weight contains NaN values")

        if torch.isinf(weight).any():
            raise ValueError("Weight contains Inf values")

        # Per-channel symmetric quantization
        abs_max = weight.abs().amax(dim=1, keepdim=True)
        scales = abs_max / 127.0
        scales = scales.clamp(min=1e-8)

        quantized = torch.round(weight / scales).clamp(-127, 127)

        return QuantizationOutput(
            quantized_weight=quantized.to(torch.int8),
            scales=scales.squeeze(-1),
            quant_type=self.quant_type,
        )

    def dequantize(self, output: QuantizationOutput):  # -> torch.Tensor
        """Dequantize."""
        return output.quantized_weight.float() * output.scales.unsqueeze(-1)


@QuantizerRegistry.register(QuantizationType.NF4)
class NF4Quantizer(Quantizer):
    """NormalFloat 4-bit quantizer (QLoRA).

    Uses a non-linear quantization grid optimized for normally
    distributed weights (common in neural networks).
    """

    # NF4 quantization grid (16 levels)
    NF4_GRID = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]

    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.NF4

    def quantize(
        self,
        weight,  # torch.Tensor
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """NF4 quantization with input validation."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for quantization")

        # Input validation
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")

        if torch.isnan(weight).any():
            raise ValueError("Weight contains NaN values")

        if torch.isinf(weight).any():
            raise ValueError("Weight contains Inf values")

        # Normalize to [-1, 1]
        abs_max = weight.abs().max()
        if abs_max < 1e-8:
            # Handle near-zero weights
            scale = torch.tensor(1.0, dtype=weight.dtype, device=weight.device)
            normalized = weight
        else:
            scale = abs_max
            normalized = weight / scale

        # Find nearest NF4 grid point
        grid = torch.tensor(self.NF4_GRID, device=weight.device, dtype=weight.dtype)
        distances = torch.abs(normalized.unsqueeze(-1) - grid)
        quantized_idx = torch.argmin(distances, dim=-1)

        return QuantizationOutput(
            quantized_weight=quantized_idx.to(torch.int8),
            scales=scale.unsqueeze(0),
            quant_type=self.quant_type,
        )

    def dequantize(self, output: QuantizationOutput):  # -> torch.Tensor
        """Dequantize."""
        import torch

        grid = torch.tensor(self.NF4_GRID, device=output.quantized_weight.device)
        dequantized = grid[output.quantized_weight.long()]
        return dequantized * output.scales
