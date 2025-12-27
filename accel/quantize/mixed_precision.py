# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Mixed precision quantization.

Allows different layers to use different quantization types based on sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import QuantizationConfig, QuantizationType, QuantizerRegistry


@dataclass
class LayerQuantizationPolicy:
    """Quantization policy for a single layer."""

    layer_name: str
    quant_type: QuantizationType
    config: QuantizationConfig


class MixedPrecisionQuantizer:
    """Mixed precision quantizer.

    Example:
        >>> policies = [
        ...     LayerQuantizationPolicy("embed", QuantizationType.FP8_E4M3, ...),
        ...     LayerQuantizationPolicy("attention", QuantizationType.INT8, ...),
        ...     LayerQuantizationPolicy("mlp", QuantizationType.INT4, ...),
        ... ]
        >>> quantizer = MixedPrecisionQuantizer(policies)
        >>> quantized_model = quantizer.quantize_model(model)
    """

    def __init__(self, policies: list[LayerQuantizationPolicy]):
        """Initialize.

        Args:
            policies: List of per-layer quantization policies
        """
        self.policies = {p.layer_name: p for p in policies}

    def quantize_model(self, model: Any) -> dict[str, Any]:
        """Quantize entire model with mixed precision.

        Args:
            model: PyTorch model or state dict

        Returns:
            Quantized state dict with metadata
        """
        state_dict = model.state_dict() if hasattr(model, "state_dict") else model

        quantized_state = {}

        for name, weight in state_dict.items():
            # Find matching policy
            policy = self._find_policy(name)

            if policy is None:
                # No policy: keep as FP16
                quantized_state[name] = weight
                continue

            # Get quantizer
            quantizer = QuantizerRegistry.get(policy.quant_type)

            # Quantize
            output = quantizer.quantize(weight, policy.config)
            quantized_state[name] = output

        return quantized_state

    def _find_policy(self, layer_name: str) -> LayerQuantizationPolicy | None:
        """Find policy for layer (supports wildcards)."""
        for pattern, policy in self.policies.items():
            if pattern in layer_name or pattern == "*":
                return policy
        return None
