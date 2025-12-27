# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""KV runtime protocols and backend abstractions.

This module defines the protocol interfaces for KV cache backends,
enabling pluggable implementations (LMDeploy, vLLM, Mock).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import AllocationResult, KVBlockInfo, KVTier, MigrationPlan


@runtime_checkable
class KVBackendProtocol(Protocol):
    """Protocol for KV cache backend implementations.

    This protocol defines the interface that all KV cache backends must
    implement. It provides block-level operations for allocation, deallocation,
    migration, and statistics.

    Implementations:
        - LMDeployKVBackend: LMDeploy/TurboMind backend
        - VLLMKVBackend: vLLM backend (optional)
        - MockKVBackend: Mock backend for testing
    """

    # =========================================================================
    # Block Allocation and Deallocation
    # =========================================================================

    def allocate(
        self,
        num_blocks: int,
        tier: KVTier = ...,
        tenant_id: str | None = None,
    ) -> AllocationResult:
        """Allocate KV cache blocks.

        Args:
            num_blocks: Number of blocks to allocate.
            tier: Target storage tier.
            tenant_id: Optional tenant identifier for quota tracking.

        Returns:
            AllocationResult with allocated block IDs or error.
        """
        ...

    def free(self, block_ids: list[int]) -> bool:
        """Free allocated KV cache blocks.

        Args:
            block_ids: List of block IDs to free.

        Returns:
            True if all blocks were freed successfully.
        """
        ...

    # =========================================================================
    # Block Information and Statistics
    # =========================================================================

    def get_block_info(self, block_id: int) -> KVBlockInfo | None:
        """Get information about a specific block.

        Args:
            block_id: Block identifier.

        Returns:
            KVBlockInfo if block exists, None otherwise.
        """
        ...

    def get_utilization(self, tier: KVTier | None = None) -> float:
        """Get KV cache utilization ratio.

        Args:
            tier: Optional tier to query (None for overall).

        Returns:
            Utilization ratio (0.0 to 1.0).
        """
        ...

    def get_fragmentation(self, tier: KVTier | None = None) -> float:
        """Get fragmentation ratio.

        Args:
            tier: Optional tier to query (None for overall).

        Returns:
            Fragmentation ratio (0.0 to 1.0).
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive backend statistics.

        Returns:
            Dictionary with utilization, fragmentation, block counts, etc.
        """
        ...

    # =========================================================================
    # Migration Operations
    # =========================================================================

    def migrate_blocks(self, plan: MigrationPlan) -> bool:
        """Migrate blocks according to a migration plan.

        Args:
            plan: Migration plan specifying source, destination, and blocks.

        Returns:
            True if migration completed successfully.
        """
        ...

    # =========================================================================
    # Callback Registration (for Policy Integration)
    # =========================================================================

    def register_eviction_callback(
        self, callback: Callable[[list[int]], None]
    ) -> None:
        """Register callback to be invoked when blocks are evicted.

        Args:
            callback: Function called with list of evicted block IDs.
        """
        ...

    def register_allocation_callback(
        self, callback: Callable[[AllocationResult], None]
    ) -> None:
        """Register callback to be invoked after allocation.

        Args:
            callback: Function called with allocation result.
        """
        ...


@runtime_checkable
class KVPoolProtocol(Protocol):
    """Protocol for KV cache pool management.

    Higher-level interface built on top of KVBackendProtocol,
    providing pool-level operations with hierarchy awareness.
    """

    @property
    def backend(self) -> KVBackendProtocol:
        """Get the underlying backend."""
        ...

    def allocate_for_sequence(
        self,
        sequence_id: str,
        num_tokens: int,
        tier: KVTier = ...,
    ) -> AllocationResult:
        """Allocate blocks for a sequence.

        Args:
            sequence_id: Sequence identifier.
            num_tokens: Number of tokens to store.
            tier: Preferred storage tier.

        Returns:
            AllocationResult with allocated blocks.
        """
        ...

    def release_sequence(self, sequence_id: str) -> bool:
        """Release all blocks for a sequence.

        Args:
            sequence_id: Sequence identifier.

        Returns:
            True if sequence was found and released.
        """
        ...

    def get_sequence_blocks(self, sequence_id: str) -> list[KVBlockInfo]:
        """Get all blocks for a sequence.

        Args:
            sequence_id: Sequence identifier.

        Returns:
            List of KVBlockInfo for the sequence.
        """
        ...

    def defragment(self, tier: KVTier | None = None) -> int:
        """Defragment the pool to reduce fragmentation.

        Args:
            tier: Optional tier to defragment (None for all).

        Returns:
            Number of blocks moved during defragmentation.
        """
        ...
