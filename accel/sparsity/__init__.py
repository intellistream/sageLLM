# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Structured sparsity for model acceleration."""

from .structured import NMPattern, StructuredPruner

__all__ = [
    "NMPattern",
    "StructuredPruner",
]
