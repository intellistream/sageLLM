# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Cross-request KV cache reuse.

This module provides KV cache sharing across requests:
- Prefix matching for common prefixes (system prompts, etc.)
- Reference counting for shared blocks
- Optional multi-tenant isolation

Integration with prefix_reuse module for advanced prefix matching.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..blocks.multi_granular import KVBlockDescriptor, MultiGranularKVPool


@dataclass
class ReuseResult:
    """Result of a KV reuse query.

    Attributes:
        reused: Whether any reuse was found.
        matched_blocks: List of matched KV blocks.
        matched_tokens: Number of tokens matched.
        total_tokens: Total tokens in query.
    """

    reused: bool
    matched_blocks: list[KVBlockDescriptor]
    matched_tokens: int
    total_tokens: int

    @property
    def reuse_ratio(self) -> float:
        """Calculate the reuse ratio.

        Returns:
            Fraction of tokens that were reused.
        """
        if self.total_tokens == 0:
            return 0.0
        return self.matched_tokens / self.total_tokens

    @property
    def new_tokens_needed(self) -> int:
        """Calculate tokens that need fresh computation.

        Returns:
            Number of tokens not covered by reuse.
        """
        return self.total_tokens - self.matched_tokens


@dataclass
class PrefixEntry:
    """Prefix index entry for reuse lookup.

    Attributes:
        token_hash: Hash of the token sequence.
        token_ids: The actual token sequence.
        block_ids: KV block IDs storing this prefix.
        ref_count: Number of active references.
        tenant_id: Optional tenant identifier for isolation.
        created_at: Timestamp when entry was created.
    """

    token_hash: str
    token_ids: list[int]
    block_ids: list[int]
    ref_count: int = 1
    tenant_id: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


class CrossRequestKVCache:
    """Cross-request KV cache with prefix matching.

    Enables KV cache sharing across requests:
    1. Same prefix (e.g., system prompts) can be reused
    2. Reference counting prevents premature deallocation
    3. Optional tenant isolation for multi-tenant deployments

    Example:
        >>> pool = MultiGranularKVPool(config)
        >>> cache = CrossRequestKVCache(pool)
        >>>
        >>> # First request computes KV and commits
        >>> blocks = pool.allocate(1, "req_1", 100, [0, 1, 2])
        >>> cache.commit("req_1", token_ids, blocks)
        >>>
        >>> # Second request can reuse
        >>> result = cache.try_reuse("req_2", token_ids)
        >>> if result.reused:
        ...     print(f"Reused {result.matched_tokens} tokens")
    """

    def __init__(
        self,
        pool: MultiGranularKVPool,
        enable_tenant_isolation: bool = False,
    ):
        """Initialize the cross-request cache.

        Args:
            pool: The underlying KV pool.
            enable_tenant_isolation: Whether to isolate by tenant.
        """
        self.pool = pool
        self.enable_tenant_isolation = enable_tenant_isolation

        # Prefix index: hash -> PrefixEntry
        self._prefix_index: dict[str, PrefixEntry] = {}

        # Token sequence to hash mapping (for fast lookups)
        self._token_to_hash: dict[tuple[int, ...], str] = {}

        # Statistics
        self._stats: dict[str, int | float] = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_reused_tokens": 0,
            "total_committed": 0,
            "total_released": 0,
        }

    def _compute_hash(self, token_ids: list[int]) -> str:
        """Compute a hash for a token sequence.

        Args:
            token_ids: Token sequence.

        Returns:
            16-character hex hash.
        """
        key = ",".join(map(str, token_ids))
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def try_reuse(
        self,
        request_id: str,
        token_ids: list[int],
        tenant_id: str | None = None,
    ) -> ReuseResult:
        """Try to reuse existing KV cache for given tokens.

        Finds the longest matching prefix in the cache and returns
        the associated KV blocks for reuse.

        Args:
            request_id: Request identifier.
            token_ids: Token sequence to match.
            tenant_id: Optional tenant identifier.

        Returns:
            ReuseResult with matched blocks and statistics.
        """
        self._stats["total_queries"] += 1

        if not token_ids:
            return ReuseResult(
                reused=False,
                matched_blocks=[],
                matched_tokens=0,
                total_tokens=0,
            )

        # Try to find longest matching prefix
        best_match: PrefixEntry | None = None
        best_length = 0

        # Search from longest to shortest prefix
        for length in range(len(token_ids), 0, -1):
            prefix = token_ids[:length]
            prefix_tuple = tuple(prefix)

            if prefix_tuple in self._token_to_hash:
                hash_key = self._token_to_hash[prefix_tuple]
                entry = self._prefix_index.get(hash_key)

                if entry:
                    # Check tenant isolation
                    if self.enable_tenant_isolation and entry.tenant_id != tenant_id:
                        continue

                    best_match = entry
                    best_length = length
                    break

        if best_match is None:
            self._stats["cache_misses"] += 1
            return ReuseResult(
                reused=False,
                matched_blocks=[],
                matched_tokens=0,
                total_tokens=len(token_ids),
            )

        # Found a match - increase reference count
        self._stats["cache_hits"] += 1
        self._stats["total_reused_tokens"] += best_length
        best_match.ref_count += 1

        # Get corresponding KV blocks
        matched_blocks = [
            self.pool._blocks[bid]
            for bid in best_match.block_ids
            if bid in self.pool._blocks
        ]

        # Update access statistics on reused blocks
        for block in matched_blocks:
            block.update_access()

        return ReuseResult(
            reused=True,
            matched_blocks=matched_blocks,
            matched_tokens=best_length,
            total_tokens=len(token_ids),
        )

    def commit(
        self,
        request_id: str,
        token_ids: list[int],
        blocks: list[KVBlockDescriptor],
        shareable: bool = True,
        tenant_id: str | None = None,
    ) -> None:
        """Commit KV cache for future reuse.

        Registers a token sequence and its KV blocks for potential
        reuse by future requests.

        Args:
            request_id: Request identifier.
            token_ids: Token sequence.
            blocks: KV blocks storing the cache.
            shareable: Whether this cache can be shared.
            tenant_id: Optional tenant identifier.
        """
        if not shareable or not token_ids or not blocks:
            return

        self._stats["total_committed"] += 1

        # Check if this prefix already exists
        prefix_tuple = tuple(token_ids)
        if prefix_tuple in self._token_to_hash:
            # Increment ref count of existing entry
            hash_key = self._token_to_hash[prefix_tuple]
            entry = self._prefix_index.get(hash_key)
            if entry:
                entry.ref_count += 1
                return

        # Create new entry
        hash_key = self._compute_hash(token_ids)
        entry = PrefixEntry(
            token_hash=hash_key,
            token_ids=token_ids.copy(),
            block_ids=[b.block_id for b in blocks],
            ref_count=1,
            tenant_id=tenant_id,
        )

        # Add to indices
        self._prefix_index[hash_key] = entry
        self._token_to_hash[prefix_tuple] = hash_key

        # Mark blocks as shared
        for block in blocks:
            block.is_shared = True

    def release(
        self,
        request_id: str,
        token_ids: list[int],
    ) -> None:
        """Release a reused KV cache reference.

        Called when a request completes to decrement the reference
        count. Cache is removed when ref_count reaches 0.

        Args:
            request_id: Request identifier.
            token_ids: Token sequence that was reused.
        """
        if not token_ids:
            return

        prefix_tuple = tuple(token_ids)
        if prefix_tuple not in self._token_to_hash:
            return

        hash_key = self._token_to_hash[prefix_tuple]
        entry = self._prefix_index.get(hash_key)

        if entry:
            entry.ref_count -= 1
            self._stats["total_released"] += 1

            # Remove when no more references
            if entry.ref_count <= 0:
                del self._prefix_index[hash_key]
                del self._token_to_hash[prefix_tuple]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total = self._stats["total_queries"]
        hit_rate = self._stats["cache_hits"] / total if total > 0 else 0.0

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "index_size": len(self._prefix_index),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._prefix_index.clear()
        self._token_to_hash.clear()

    def get_entry(self, token_ids: list[int]) -> PrefixEntry | None:
        """Get a cache entry by token sequence.

        Args:
            token_ids: Token sequence.

        Returns:
            PrefixEntry if found, None otherwise.
        """
        prefix_tuple = tuple(token_ids)
        if prefix_tuple not in self._token_to_hash:
            return None

        hash_key = self._token_to_hash[prefix_tuple]
        return self._prefix_index.get(hash_key)

    def get_all_entries(self) -> list[PrefixEntry]:
        """Get all cache entries.

        Returns:
            List of all PrefixEntry objects.
        """
        return list(self._prefix_index.values())

    def evict_by_tenant(self, tenant_id: str) -> int:
        """Evict all entries for a specific tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Number of entries evicted.
        """
        to_remove = []
        for hash_key, entry in self._prefix_index.items():
            if entry.tenant_id == tenant_id:
                to_remove.append((hash_key, tuple(entry.token_ids)))

        for hash_key, token_tuple in to_remove:
            del self._prefix_index[hash_key]
            if token_tuple in self._token_to_hash:
                del self._token_to_hash[token_tuple]

        return len(to_remove)

    def evict_by_age(self, max_age_seconds: float) -> int:
        """Evict entries older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds.

        Returns:
            Number of entries evicted.
        """
        import time

        now = time.time()
        to_remove = []

        for hash_key, entry in self._prefix_index.items():
            if now - entry.created_at > max_age_seconds:
                to_remove.append((hash_key, tuple(entry.token_ids)))

        for hash_key, token_tuple in to_remove:
            del self._prefix_index[hash_key]
            if token_tuple in self._token_to_hash:
                del self._token_to_hash[token_tuple]

        return len(to_remove)
