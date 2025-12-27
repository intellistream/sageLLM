# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""KV runtime module for sageLLM - KV cache pool and hierarchy management.

This module provides unified KV cache runtime management:
- KVPool: Block allocation, deallocation, utilization tracking
- HierarchyManager: Tier management (GPU/CPU/NVMe)
- KVMigrator: Cross-tier block migration
- QuotaManager: Per-tenant resource quotas
- KVBackendProtocol: Backend abstraction for LMDeploy/vLLM

Multi-granularity KV management (Task 2 - 课题 4.2):
- MultiGranularKVPool: Block/Token/Head/Layer granularity support
- TieredKVStorage: HBM/DDR/NVMe three-tier storage hierarchy
- HotColdClassifier: Temperature-based block classification
- KVMigrator: Cross-tier migration with priority scheduling
- CrossRequestKVCache: Prefix-based cross-request KV reuse

Example:
    >>> from sage.common.components.sage_llm.sageLLM.kv_runtime import (
    ...     KVPool,
    ...     KVBackendProtocol,
    ... )
    >>> pool = KVPool(backend=lmdeploy_backend)
    >>> blocks = pool.allocate(num_blocks=4, tier="gpu")

Multi-granularity example:
    >>> from sage.common.components.sage_llm.sageLLM.kv_runtime import (
    ...     MultiGranularKVPool,
    ...     KVPoolConfig,
    ...     KVGranularity,
    ...     StorageTier,
    ... )
    >>> config = KVPoolConfig(block_size=16)
    >>> pool = MultiGranularKVPool(config)
    >>> blocks = pool.allocate(
    ...     sequence_id=1,
    ...     request_id="req_1",
    ...     num_tokens=64,
    ...     layer_ids=[0, 1, 2],
    ... )
"""

# Multi-granularity block management (Task 2)
from .blocks import (
    KVBlockDescriptor,
    KVGranularity,
    KVPoolConfig,
    MultiGranularKVPool,
    StorageTier,
)

# Three-tier storage hierarchy (Task 2)
from .hierarchy import (
    DDRBackend,
    HBMBackend,
    NVMeBackend,
    StorageBackend,
    TierConfig,
    TieredKVStorage,
    TierUsage,
)

# Hot/cold migration (Task 2)
from .migration import (
    HotColdClassifier,
    KVMigrator,
    MigrationResult,
)
from .migration import (
    MigrationPlan as MultiGranularMigrationPlan,
)
from .protocols import KVBackendProtocol

# Cross-request reuse (Task 2)
from .reuse import (
    CrossRequestKVCache,
    PrefixEntry,
    ReuseResult,
)
from .types import (
    AllocationResult,
    KVBlockInfo,
    KVCacheSchema,
    KVTier,
    MigrationPlan,
)

__all__ = [
    # Protocols
    "KVBackendProtocol",
    # Types (original)
    "KVBlockInfo",
    "KVCacheSchema",
    "KVTier",
    "AllocationResult",
    "MigrationPlan",
    # Multi-granularity blocks (Task 2)
    "KVGranularity",
    "StorageTier",
    "KVBlockDescriptor",
    "KVPoolConfig",
    "MultiGranularKVPool",
    # Three-tier storage (Task 2)
    "TierConfig",
    "TierUsage",
    "StorageBackend",
    "HBMBackend",
    "DDRBackend",
    "NVMeBackend",
    "TieredKVStorage",
    # Hot/cold migration (Task 2)
    "HotColdClassifier",
    "KVMigrator",
    "MultiGranularMigrationPlan",
    "MigrationResult",
    # Cross-request reuse (Task 2)
    "ReuseResult",
    "PrefixEntry",
    "CrossRequestKVCache",
]
