# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Multi-granularity KV block management.

This module provides multi-granularity KV Cache block management supporting:
- BLOCK: Traditional block-level granularity (e.g., 16 tokens per block)
- TOKEN: Token-level granularity (finest)
- HEAD: Attention head-level granularity (for MQA/GQA optimization)
- LAYER: Layer-level granularity (coarsest, for early exit)

References:
- vLLM BlockManager: https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager_v2.py
- PagedAttention: https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class KVGranularity(Enum):
    """KV block granularity levels.

    Traditional approaches only support BLOCK granularity (e.g., 16 tokens).
    We support finer granularities to improve reuse and memory efficiency.
    """

    BLOCK = auto()  # Block-level (traditional, e.g., 16 tokens)
    TOKEN = auto()  # Token-level (finest granularity)
    HEAD = auto()  # Attention head-level (for MQA/GQA)
    LAYER = auto()  # Layer-level (coarsest, for early exit)


class StorageTier(Enum):
    """Storage hierarchy tier.

    Defines the three-tier storage hierarchy:
    - HBM: GPU High Bandwidth Memory (fastest, most expensive)
    - DDR: CPU main memory (moderate speed and cost)
    - NVME: NVMe SSD (slowest, cheapest)
    """

    HBM = auto()  # GPU high-bandwidth memory
    DDR = auto()  # CPU main memory
    NVME = auto()  # NVMe SSD storage


@dataclass
class KVBlockDescriptor:
    """KV block descriptor (metadata only, no actual data).

    Describes metadata for a KV Cache block, enabling fine-grained
    tracking and management of KV cache resources.

    Attributes:
        block_id: Unique block identifier.
        granularity: Block granularity level.
        layer_ids: List of layer IDs covered by this block.
        head_ids: List of attention head IDs covered.
        token_range: Token range [start, end) covered.
        sequence_id: Owner sequence ID.
        request_id: Owner request ID.
        tier: Current storage tier.
        device_id: Device index within the tier.
        offset: Byte offset within storage.
        size_bytes: Block size in bytes.
        ref_count: Reference count for sharing.
        is_shared: Whether block is shared across requests.
        last_access_time: Last access timestamp.
        access_count: Total access count.
        access_frequency: Access frequency (accesses/second).
        token_hash: Hash for prefix matching.
        metadata: Additional metadata.
    """

    block_id: int
    granularity: KVGranularity

    # Location information
    layer_ids: list[int]
    head_ids: list[int]
    token_range: tuple[int, int]  # [start, end)

    # Owner information
    sequence_id: int
    request_id: str

    # Storage location
    tier: StorageTier = StorageTier.HBM
    device_id: int = 0
    offset: int = 0
    size_bytes: int = 0

    # Sharing state
    ref_count: int = 1
    is_shared: bool = False

    # Access statistics (for hot/cold classification)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    access_frequency: float = 0.0  # accesses per second

    # Metadata
    token_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_access(self) -> None:
        """Update access statistics.

        Called on every access to track usage patterns for hot/cold
        classification and migration decisions.
        """
        now = time.time()
        elapsed = now - self.last_access_time
        if elapsed > 0:
            # Exponential moving average for frequency
            alpha = 0.3
            instant_freq = 1.0 / elapsed if elapsed > 0 else 0.0
            self.access_frequency = (
                alpha * instant_freq + (1 - alpha) * self.access_frequency
            )
        self.last_access_time = now
        self.access_count += 1

    @property
    def num_tokens(self) -> int:
        """Number of tokens covered by this block."""
        return self.token_range[1] - self.token_range[0]

    @property
    def num_layers(self) -> int:
        """Number of layers covered by this block."""
        return len(self.layer_ids)

    @property
    def num_heads(self) -> int:
        """Number of attention heads covered by this block."""
        return len(self.head_ids)


@dataclass
class KVPoolConfig:
    """KV pool configuration.

    Attributes:
        hbm_capacity_bytes: HBM capacity in bytes.
        ddr_capacity_bytes: DDR capacity in bytes.
        nvme_capacity_bytes: NVMe capacity in bytes.
        block_size: Tokens per block (for BLOCK granularity).
        default_granularity: Default allocation granularity.
        enable_sharing: Allow cross-request sharing.
        enable_tiering: Enable multi-tier storage.
    """

    # Capacity configuration
    hbm_capacity_bytes: int = 16 * 1024**3  # 16 GB
    ddr_capacity_bytes: int = 64 * 1024**3  # 64 GB
    nvme_capacity_bytes: int = 256 * 1024**3  # 256 GB

    # Block configuration
    block_size: int = 16  # tokens per block
    default_granularity: KVGranularity = KVGranularity.BLOCK

    # Behavior configuration
    enable_sharing: bool = True
    enable_tiering: bool = True


class MultiGranularKVPool:
    """Multi-granularity KV pool.

    Supports different KV Cache granularities:
    - BLOCK: Traditional block-level, suitable for batch operations
    - TOKEN: Token-level, suitable for fine-grained reuse
    - HEAD: Head-level, suitable for MQA/GQA optimization
    - LAYER: Layer-level, suitable for early exit

    Example:
        >>> config = KVPoolConfig(block_size=16)
        >>> pool = MultiGranularKVPool(config)
        >>> blocks = pool.allocate(
        ...     sequence_id=1,
        ...     request_id="req_1",
        ...     num_tokens=64,
        ...     layer_ids=[0, 1, 2],
        ... )
        >>> print(f"Allocated {len(blocks)} blocks")
        Allocated 4 blocks
    """

    def __init__(self, config: KVPoolConfig):
        """Initialize the multi-granular KV pool.

        Args:
            config: Pool configuration.
        """
        self.config = config

        # Block index
        self._blocks: dict[int, KVBlockDescriptor] = {}
        self._next_block_id = 0

        # Index by sequence
        self._sequence_blocks: dict[int, list[int]] = {}

        # Index by tier (for fast lookups)
        self._tier_blocks: dict[StorageTier, list[int]] = {
            tier: [] for tier in StorageTier
        }

        # Free lists per tier
        self._free_blocks: dict[StorageTier, list[int]] = {
            tier: [] for tier in StorageTier
        }

        # Statistics
        self._stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "total_migrations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def allocate(
        self,
        sequence_id: int,
        request_id: str,
        num_tokens: int,
        layer_ids: list[int],
        head_ids: list[int] | None = None,
        granularity: KVGranularity | None = None,
        preferred_tier: StorageTier = StorageTier.HBM,
        token_ids: list[int] | None = None,
    ) -> list[KVBlockDescriptor]:
        """Allocate KV blocks.

        Args:
            sequence_id: Sequence identifier.
            request_id: Request identifier.
            num_tokens: Number of tokens to allocate.
            layer_ids: Layer IDs to cover.
            head_ids: Optional head IDs (for fine-grained allocation).
            granularity: Allocation granularity (defaults to config).
            preferred_tier: Preferred storage tier.
            token_ids: Optional token IDs for prefix matching (enables cache reuse).

        Returns:
            List of allocated KV block descriptors.
        """
        granularity = granularity or self.config.default_granularity

        # Compute token hash if token_ids provided
        token_hash = None
        if token_ids:
            import hashlib
            token_str = ",".join(map(str, token_ids[:num_tokens]))
            token_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]

        # Calculate number of blocks needed
        if granularity == KVGranularity.BLOCK:
            num_blocks = (num_tokens + self.config.block_size - 1) // self.config.block_size
        elif granularity == KVGranularity.TOKEN:
            num_blocks = num_tokens
        else:
            num_blocks = 1  # HEAD/LAYER granularity

        # Allocate blocks
        allocated = []
        for i in range(num_blocks):
            if granularity == KVGranularity.BLOCK:
                token_start = i * self.config.block_size
                token_end = min((i + 1) * self.config.block_size, num_tokens)
            elif granularity == KVGranularity.TOKEN:
                token_start = i
                token_end = i + 1
            else:
                token_start = 0
                token_end = num_tokens

            block = self._allocate_single_block(
                sequence_id=sequence_id,
                request_id=request_id,
                granularity=granularity,
                layer_ids=layer_ids,
                head_ids=head_ids or [],
                token_start=token_start,
                token_end=token_end,
                tier=preferred_tier,
                token_hash=token_hash,
            )
            allocated.append(block)

        self._stats["total_allocations"] += len(allocated)
        return allocated

    def _allocate_single_block(
        self,
        sequence_id: int,
        request_id: str,
        granularity: KVGranularity,
        layer_ids: list[int],
        head_ids: list[int],
        token_start: int,
        token_end: int,
        tier: StorageTier,
        token_hash: str | None = None,
    ) -> KVBlockDescriptor:
        """Allocate a single block.

        Args:
            sequence_id: Sequence identifier.
            request_id: Request identifier.
            granularity: Block granularity.
            layer_ids: Layer IDs covered.
            head_ids: Head IDs covered.
            token_start: Start token index.
            token_end: End token index (exclusive).
            tier: Storage tier.
            token_hash: Optional token sequence hash for prefix matching.

        Returns:
            Allocated block descriptor.
        """
        block_id = self._next_block_id
        self._next_block_id += 1

        block = KVBlockDescriptor(
            block_id=block_id,
            granularity=granularity,
            layer_ids=layer_ids.copy(),
            head_ids=head_ids.copy(),
            token_range=(token_start, token_end),
            sequence_id=sequence_id,
            request_id=request_id,
            tier=tier,
            token_hash=token_hash,
        )

        # Register block
        self._blocks[block_id] = block
        self._tier_blocks[tier].append(block_id)

        if sequence_id not in self._sequence_blocks:
            self._sequence_blocks[sequence_id] = []
        self._sequence_blocks[sequence_id].append(block_id)

        return block

    def deallocate(self, blocks: list[KVBlockDescriptor]) -> None:
        """Deallocate KV blocks.

        Decreases reference count. Block is freed when ref_count reaches 0.

        Args:
            blocks: Blocks to deallocate.
        """
        for block in blocks:
            if block.ref_count > 1:
                block.ref_count -= 1
            else:
                self._free_block(block)

        self._stats["total_deallocations"] += len(blocks)

    def _free_block(self, block: KVBlockDescriptor) -> None:
        """Free a single block.

        Args:
            block: Block to free.
        """
        block_id = block.block_id

        # Remove from index
        if block_id in self._blocks:
            del self._blocks[block_id]

        if block.sequence_id in self._sequence_blocks:
            seq_blocks = self._sequence_blocks[block.sequence_id]
            if block_id in seq_blocks:
                seq_blocks.remove(block_id)

        if block_id in self._tier_blocks[block.tier]:
            self._tier_blocks[block.tier].remove(block_id)

        # Add to free list
        self._free_blocks[block.tier].append(block_id)

    def get_blocks_by_sequence(self, sequence_id: int) -> list[KVBlockDescriptor]:
        """Get all blocks for a sequence.

        Args:
            sequence_id: Sequence identifier.

        Returns:
            List of block descriptors for the sequence.
        """
        block_ids = self._sequence_blocks.get(sequence_id, [])
        return [self._blocks[bid] for bid in block_ids if bid in self._blocks]

    def get_block(self, block_id: int) -> KVBlockDescriptor | None:
        """Get a block by ID.

        Args:
            block_id: Block identifier.

        Returns:
            Block descriptor if found, None otherwise.
        """
        return self._blocks.get(block_id)

    def query_by_prefix(
        self,
        token_ids: list[int],
        min_match_length: int = 1,
    ) -> list[KVBlockDescriptor] | None:
        """Query for reusable KV blocks by prefix.

        Finds blocks that match the given token prefix for potential reuse.
        Integration point with prefix_reuse module.

        Args:
            token_ids: Token sequence to match.
            min_match_length: Minimum prefix length to consider a match.

        Returns:
            List of matching blocks, or None if no match found.
        """
        if len(token_ids) < min_match_length:
            self._stats["cache_misses"] += 1
            return None

        # Compute hash for the query
        import hashlib
        token_str = ",".join(map(str, token_ids))
        query_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]

        # Search for blocks with matching token_hash
        # Try longest prefix first
        best_match_blocks: list[KVBlockDescriptor] = []
        best_match_length = 0

        for block in self._blocks.values():
            if not block.token_hash:
                continue

            # Check if this block's hash matches a prefix of our query
            # or vice versa
            if query_hash.startswith(block.token_hash) or block.token_hash.startswith(query_hash):
                # Estimate match length from token_range
                match_length = block.num_tokens
                if match_length >= min_match_length and match_length > best_match_length:
                    best_match_blocks = [block]
                    best_match_length = match_length
                elif match_length == best_match_length:
                    # Same length - add to candidates
                    best_match_blocks.append(block)

        if best_match_blocks:
            self._stats["cache_hits"] += 1
            # Increment ref_count for matched blocks
            for block in best_match_blocks:
                block.ref_count += 1
            return best_match_blocks

        self._stats["cache_misses"] += 1
        return None

    def get_tier_usage(self, tier: StorageTier) -> dict[str, Any]:
        """Get usage statistics for a storage tier.

        Args:
            tier: Storage tier to query.

        Returns:
            Dictionary with usage statistics.
        """
        blocks = self._tier_blocks[tier]
        total_bytes = sum(
            self._blocks[bid].size_bytes for bid in blocks if bid in self._blocks
        )

        capacity = {
            StorageTier.HBM: self.config.hbm_capacity_bytes,
            StorageTier.DDR: self.config.ddr_capacity_bytes,
            StorageTier.NVME: self.config.nvme_capacity_bytes,
        }[tier]

        return {
            "tier": tier.name,
            "num_blocks": len(blocks),
            "used_bytes": total_bytes,
            "capacity_bytes": capacity,
            "utilization": total_bytes / capacity if capacity > 0 else 0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with comprehensive pool statistics.
        """
        return {
            **self._stats,
            "total_blocks": len(self._blocks),
            "hbm_usage": self.get_tier_usage(StorageTier.HBM),
            "ddr_usage": self.get_tier_usage(StorageTier.DDR),
            "nvme_usage": self.get_tier_usage(StorageTier.NVME),
        }

    def update_block_tier(self, block_id: int, new_tier: StorageTier) -> bool:
        """Update the storage tier for a block.

        Args:
            block_id: Block identifier.
            new_tier: New storage tier.

        Returns:
            True if update succeeded, False if block not found.
        """
        block = self._blocks.get(block_id)
        if block is None:
            return False

        old_tier = block.tier

        # Update tier index
        if block_id in self._tier_blocks[old_tier]:
            self._tier_blocks[old_tier].remove(block_id)
        self._tier_blocks[new_tier].append(block_id)

        # Update block
        block.tier = new_tier
        self._stats["total_migrations"] += 1

        return True

    def get_all_blocks(self) -> list[KVBlockDescriptor]:
        """Get all blocks in the pool.

        Returns:
            List of all block descriptors.
        """
        return list(self._blocks.values())

    def clear(self) -> None:
        """Clear all blocks from the pool."""
        self._blocks.clear()
        self._sequence_blocks.clear()
        for tier in StorageTier:
            self._tier_blocks[tier].clear()
            self._free_blocks[tier].clear()
        self._next_block_id = 0
