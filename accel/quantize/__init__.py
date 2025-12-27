# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Quantization module.

Provides model quantization capabilities:
- FP8 (E4M3/E5M2) quantization
- INT4/INT8 quantization
- NF4 (NormalFloat 4-bit) quantization
- Mixed-precision quantization
"""

from .base import (
    QuantizationConfig,
    QuantizationGranularity,
    QuantizationOutput,
    QuantizationType,
    Quantizer,
    QuantizerRegistry,
)
from .fp8 import FP8E4M3Quantizer, FP8E5M2Quantizer
from .int4 import INT4Quantizer, INT8Quantizer, NF4Quantizer

__all__ = [
    "QuantizationType",
    "QuantizationGranularity",
    "QuantizationConfig",
    "QuantizationOutput",
    "Quantizer",
    "QuantizerRegistry",
    "FP8E4M3Quantizer",
    "FP8E5M2Quantizer",
    "INT4Quantizer",
    "INT8Quantizer",
    "NF4Quantizer",
]
