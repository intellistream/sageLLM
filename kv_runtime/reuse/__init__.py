# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Cross-request KV reuse module.

This module provides KV cache sharing across requests:
- ReuseResult: Result of reuse query
- PrefixEntry: Prefix index entry for reuse lookup
- CrossRequestKVCache: Cross-request KV cache with prefix matching
"""

from .cross_request import (
    CrossRequestKVCache,
    PrefixEntry,
    ReuseResult,
)

__all__ = [
    "ReuseResult",
    "PrefixEntry",
    "CrossRequestKVCache",
]
