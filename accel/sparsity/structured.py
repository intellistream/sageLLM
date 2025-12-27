# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Structured sparsity implementation.

Supports N:M sparsity patterns (e.g., 2:4 for NVIDIA Ampere GPUs).

References:
- Accelerating Sparse Deep Neural Networks: https://arxiv.org/abs/2104.08378
- NVIDIA Sparse Tensor Cores: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class NMPattern(Enum):
    """Structured sparsity patterns."""

    TWO_FOUR = (2, 4)  # 2:4 sparsity (50% sparse, NVIDIA Ampere)
    FOUR_EIGHT = (4, 8)  # 4:8 sparsity (50% sparse)
    ONE_FOUR = (1, 4)  # 1:4 sparsity (75% sparse)


@dataclass
class SparseOutput:
    """Structured sparse weight output."""

    sparse_weight: Any  # torch.Tensor with zeros
    mask: Any  # torch.Tensor (bool)
    pattern: NMPattern
    sparsity_ratio: float


class StructuredPruner:
    """Structured pruner for N:M sparsity.

    Example:
        >>> pruner = StructuredPruner(NMPattern.TWO_FOUR)
        >>> sparse_output = pruner.prune(weight_tensor)
        >>> print(f"Sparsity: {sparse_output.sparsity_ratio:.2%}")
    """

    def __init__(self, pattern: NMPattern):
        """Initialize.

        Args:
            pattern: N:M sparsity pattern
        """
        self.pattern = pattern
        self.n, self.m = pattern.value

    def prune(self, weight) -> SparseOutput:  # torch.Tensor -> SparseOutput
        """Apply N:M structured pruning.

        Args:
            weight: Weight tensor to prune

        Returns:
            SparseOutput with pruned weight and mask
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for sparsity")

        # Shape compatibility: total elements must be divisible by M
        if weight.numel() % self.m != 0:
            raise ValueError(
                f"Weight size {weight.numel()} is not divisible by M={self.m}. "
                "Pad weights or adjust group size to apply N:M sparsity."
            )

        # Optional hardware capability hint (Ampere 8.0+ for structured sparsity)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability < (8, 0):
                import warnings

                warnings.warn(
                    f"GPU compute capability {capability} may not support structured sparsity "
                    "acceleration (requires >= 8.0, e.g., Ampere). Pruning will still run on CPU/GPU."
                )

        original_shape = weight.shape
        flattened = weight.view(-1, self.m)

        # Find top-N magnitudes in each M-group
        abs_weight = flattened.abs()
        _, indices = torch.topk(abs_weight, self.n, dim=-1)

        # Create mask: keep top-N, prune others
        mask = torch.zeros_like(flattened, dtype=torch.bool)
        mask.scatter_(1, indices, True)

        # Apply mask
        pruned = flattened * mask.float()

        # Reshape back
        sparse_weight = pruned.view(original_shape)
        mask_reshaped = mask.view(original_shape)

        # Compute actual sparsity
        sparsity_ratio = 1.0 - (sparse_weight != 0).float().mean().item()

        return SparseOutput(
            sparse_weight=sparse_weight,
            mask=mask_reshaped,
            pattern=self.pattern,
            sparsity_ratio=sparsity_ratio,
        )

    def prune_model(self, model: Any) -> dict[str, SparseOutput]:
        """Prune entire model.

        Args:
            model: PyTorch model or state dict

        Returns:
            Dict of layer_name -> SparseOutput
        """
        state_dict = model.state_dict() if hasattr(model, "state_dict") else model

        sparse_state = {}

        for name, weight in state_dict.items():
            # Only prune linear/conv layers
            if "weight" in name and weight.ndim >= 2:
                sparse_state[name] = self.prune(weight)
            else:
                # Keep bias/normalization as-is
                sparse_state[name] = weight

        return sparse_state
