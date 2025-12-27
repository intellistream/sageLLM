# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Acceleration module for sageLLM - quantization, sparsity, speculative decoding.

This module provides inference acceleration capabilities:
- AccelController: Unified acceleration control
- Quantization: AWQ/GPTQ/FP8 quantization support
- Sparsity: Sparse attention patterns
- Speculative decoding: Draft model + verification
- CoT acceleration: Chain-of-thought optimization

Example:
    >>> from sage.common.components.sage_llm.sageLLM.accel import (
    ...     AccelController,
    ...     QuantizationProfile,
    ... )
    >>> controller = AccelController(config=AccelConfig(
    ...     enable_quantization=True,
    ...     quantization_method="awq",
    ... ))
"""

from . import cost_model, quantize, sparsity
from .cost_model import CostEstimator, CostMetrics, ModelProfile
from .protocols import AccelControllerProtocol, QuantizerProtocol
from .quantize import (
    QuantizationConfig,
    QuantizationGranularity,
    QuantizationOutput,
    QuantizationType,
    Quantizer,
    QuantizerRegistry,
)
from .quantize.fp8 import FP8E4M3Quantizer, FP8E5M2Quantizer
from .quantize.int4 import INT4Quantizer, INT8Quantizer, NF4Quantizer
from .quantize.mixed_precision import LayerQuantizationPolicy, MixedPrecisionQuantizer
from .sparsity import NMPattern, StructuredPruner
from .types import (
    AccelConfig,
    QuantizationMethod,
    QuantizationProfile,
    SpeculativeConfig,
)

__all__ = [
    # Modules
    "quantize",
    "sparsity",
    "cost_model",
    # Protocols
    "AccelControllerProtocol",
    "QuantizerProtocol",
    # Types
    "AccelConfig",
    "QuantizationMethod",
    "QuantizationProfile",
    "SpeculativeConfig",
    # Quantization
    "Quantizer",
    "QuantizerRegistry",
    "QuantizationType",
    "QuantizationConfig",
    "QuantizationGranularity",
    "QuantizationOutput",
    "FP8E4M3Quantizer",
    "FP8E5M2Quantizer",
    "INT4Quantizer",
    "INT8Quantizer",
    "NF4Quantizer",
    "MixedPrecisionQuantizer",
    "LayerQuantizationPolicy",
    # Sparsity
    "StructuredPruner",
    "NMPattern",
    # Cost model
    "CostEstimator",
    "CostMetrics",
    "ModelProfile",
]
