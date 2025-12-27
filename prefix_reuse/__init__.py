# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Prefix reuse module for sageLLM - KV cache prefix matching and reuse.

This module provides prefix-aware KV cache reuse capabilities:
- PrefixReuseIndex: Trie/Radix-based prefix indexing
- PrefixMatcher: Token sequence matching and verification
- PrefixReuseMetrics: Hit rate and performance tracking

Example:
    >>> from sage.common.components.sage_llm.sageLLM.prefix_reuse import (
    ...     PrefixReuseIndex,
    ...     PrefixMatcher,
    ... )
    >>> index = PrefixReuseIndex()
    >>> index.insert(token_ids=[1, 2, 3], kv_block_ids=[0, 1])
    >>> hit = index.lookup(token_ids=[1, 2, 3, 4, 5])
"""

# PR2: To be implemented
# from .index import PrefixReuseIndex
# from .matcher import PrefixMatcher
# from .metrics import PrefixReuseMetrics
# from .types import PrefixEntry, PrefixHit

__all__: list[str] = [
    # "PrefixReuseIndex",
    # "PrefixMatcher",
    # "PrefixReuseMetrics",
    # "PrefixEntry",
    # "PrefixHit",
]
