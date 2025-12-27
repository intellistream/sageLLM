# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""KV Manager extension for LMDeploy.

This module extends LMDeploy's SequenceManager with sageLLM hooks
for KV cache management, prefix reuse, and policy integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ...kv_runtime.protocols import KVBackendProtocol
    from ...kv_runtime.types import AllocationResult, KVBlockInfo, KVTier, MigrationPlan

logger = logging.getLogger(__name__)


class LMDeployKVManager:
    """Extended KV manager for LMDeploy with sageLLM integration.

    This class wraps LMDeploy's internal KV cache management and provides
    hooks for external policy control and prefix reuse.

    Attributes:
        backend: The underlying KV backend implementation.
    """

    def __init__(self, backend: KVBackendProtocol | None = None) -> None:
        """Initialize the KV manager.

        Args:
            backend: Optional KV backend for delegation.
        """
        self.backend = backend

        # Callbacks
        self._eviction_callbacks: list[Callable[[list[int]], None]] = []
        self._allocation_callbacks: list[Callable[[AllocationResult], None]] = []

        # Prefix hooks
        self._pre_fetch_hook: Callable[[list[int]], list[int] | None] | None = None
        self._post_store_hook: Callable[[list[int], list[int]], None] | None = None

        # Statistics
        self._stats = {
            "allocations": 0,
            "deallocations": 0,
            "evictions": 0,
            "migrations": 0,
            "prefix_hits": 0,
            "prefix_misses": 0,
        }

    # =========================================================================
    # Block Management
    # =========================================================================

    def allocate(
        self,
        num_blocks: int,
        tier: KVTier | None = None,
        tenant_id: str | None = None,
    ) -> AllocationResult:
        """Allocate KV cache blocks.

        Args:
            num_blocks: Number of blocks to allocate.
            tier: Target storage tier.
            tenant_id: Optional tenant identifier.

        Returns:
            AllocationResult with allocated block IDs.
        """
        if self.backend is None:
            raise RuntimeError("KV backend not set")

        result = self.backend.allocate(num_blocks, tier=tier, tenant_id=tenant_id)
        self._stats["allocations"] += 1

        # Invoke callbacks
        for callback in self._allocation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning("Allocation callback failed: %s", e)

        return result

    def free(self, block_ids: list[int]) -> bool:
        """Free allocated KV cache blocks.

        Args:
            block_ids: List of block IDs to free.

        Returns:
            True if all blocks were freed successfully.
        """
        if self.backend is None:
            raise RuntimeError("KV backend not set")

        result = self.backend.free(block_ids)
        self._stats["deallocations"] += 1
        return result

    def get_block_info(self, block_id: int) -> KVBlockInfo | None:
        """Get information about a specific block.

        Args:
            block_id: Block identifier.

        Returns:
            KVBlockInfo if block exists.
        """
        if self.backend is None:
            return None
        return self.backend.get_block_info(block_id)

    # =========================================================================
    # Migration
    # =========================================================================

    def migrate_blocks(self, plan: MigrationPlan) -> bool:
        """Migrate blocks according to a migration plan.

        Args:
            plan: Migration plan specification.

        Returns:
            True if migration completed successfully.
        """
        if self.backend is None:
            raise RuntimeError("KV backend not set")

        result = self.backend.migrate_blocks(plan)
        if result:
            self._stats["migrations"] += 1
        return result

    # =========================================================================
    # Callbacks and Hooks
    # =========================================================================

    def register_eviction_callback(
        self, callback: Callable[[list[int]], None]
    ) -> None:
        """Register callback for block eviction.

        Args:
            callback: Function called with list of evicted block IDs.
        """
        self._eviction_callbacks.append(callback)

    def register_allocation_callback(
        self, callback: Callable[[AllocationResult], None]
    ) -> None:
        """Register callback for allocation.

        Args:
            callback: Function called with allocation result.
        """
        self._allocation_callbacks.append(callback)

    def set_pre_fetch_hook(
        self, hook: Callable[[list[int]], list[int] | None]
    ) -> None:
        """Set hook called before KV fetch for prefix lookup.

        Args:
            hook: Function that takes token_ids and returns matched block_ids
                  if prefix hit, None otherwise.
        """
        self._pre_fetch_hook = hook

    def set_post_store_hook(
        self, hook: Callable[[list[int], list[int]], None]
    ) -> None:
        """Set hook called after KV store for prefix index update.

        Args:
            hook: Function that takes (token_ids, block_ids).
        """
        self._post_store_hook = hook

    # =========================================================================
    # Prefix Reuse Integration
    # =========================================================================

    def check_prefix_match(self, token_ids: list[int]) -> list[int] | None:
        """Check for prefix match before fetching KV cache.

        Args:
            token_ids: Input token IDs.

        Returns:
            List of matched block IDs if prefix hit, None otherwise.
        """
        if self._pre_fetch_hook is None:
            return None

        try:
            result = self._pre_fetch_hook(token_ids)
            if result is not None:
                self._stats["prefix_hits"] += 1
            else:
                self._stats["prefix_misses"] += 1
            return result
        except Exception as e:
            logger.warning("Pre-fetch hook failed: %s", e)
            self._stats["prefix_misses"] += 1
            return None

    def update_prefix_index(
        self, token_ids: list[int], block_ids: list[int]
    ) -> None:
        """Update prefix index after storing KV cache.

        Args:
            token_ids: Token IDs that were stored.
            block_ids: Block IDs where they were stored.
        """
        if self._post_store_hook is None:
            return

        try:
            self._post_store_hook(token_ids, block_ids)
        except Exception as e:
            logger.warning("Post-store hook failed: %s", e)

    def _trigger_eviction(self, block_ids: list[int]) -> None:
        """Trigger eviction callbacks.

        Args:
            block_ids: List of evicted block IDs.
        """
        self._stats["evictions"] += 1
        for callback in self._eviction_callbacks:
            try:
                callback(block_ids)
            except Exception as e:
                logger.warning("Eviction callback failed: %s", e)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get KV manager statistics.

        Returns:
            Dictionary with allocation, eviction, and prefix stats.
        """
        stats = dict(self._stats)

        if self.backend is not None:
            backend_stats = self.backend.get_stats()
            stats["backend"] = backend_stats

        # Calculate prefix hit rate
        total_prefix = stats["prefix_hits"] + stats["prefix_misses"]
        stats["prefix_hit_rate"] = (
            stats["prefix_hits"] / total_prefix if total_prefix > 0 else 0.0
        )

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0
