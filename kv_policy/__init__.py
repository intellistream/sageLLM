# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""KV policy module for sageLLM - eviction, migration, and lifecycle policies.

This module provides policy abstractions for KV cache management:
- EvictionPolicy: LRU/LFU/ARC/S3FIFO eviction strategies
- MigrationPolicy: Cross-tier migration decision making
- CostBenefitModel: Migration cost-benefit analysis
- LifetimePredictor: KV block TTL prediction

Example:
    >>> from sage.common.components.sage_llm.sageLLM.kv_policy import (
    ...     EvictionPolicy,
    ...     MigrationPolicy,
    ... )
    >>> policy = EvictionPolicy.create("lru")
    >>> victims = policy.select_victims(pool_stats, num_blocks=2)
"""

# PR4: To be implemented
# from .eviction import EvictionPolicy, LRUPolicy, LFUPolicy, ARCPolicy, S3FIFOPolicy
# from .migration import MigrationPolicy
# from .cost_benefit import CostBenefitModel
# from .lifetime import LifetimePredictor
# from .types import PolicyContext, EvictionDecision, MigrationDecision

__all__: list[str] = [
    # "EvictionPolicy",
    # "MigrationPolicy",
    # "CostBenefitModel",
    # "LifetimePredictor",
]
