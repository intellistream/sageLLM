# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tiered KV storage management.

This module provides HBM/DDR/NVMe three-tier storage management for KV cache:
- HBM: GPU high-bandwidth memory for hot data
- DDR: CPU pinned memory for warm data
- NVMe: SSD storage for cold data persistence

References:
- Infinite-LLM: https://arxiv.org/abs/2401.02669 (DistKV tiered storage)
- vLLM Prefix Caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..blocks.multi_granular import KVBlockDescriptor, StorageTier


# Import StorageTier at runtime to avoid circular imports
def _get_storage_tier() -> type:
    from ..blocks.multi_granular import StorageTier

    return StorageTier


@dataclass
class TierConfig:
    """Storage tier configuration.

    Attributes:
        tier: Storage tier type.
        capacity_bytes: Total capacity in bytes.
        bandwidth_gbps: Read/write bandwidth in GB/s.
        latency_us: Access latency in microseconds.
        device_id: Optional device index (for GPU).
        path: Optional file path (for NVMe).
    """

    tier: Any  # StorageTier (deferred to avoid circular import)
    capacity_bytes: int
    bandwidth_gbps: float
    latency_us: float

    # Optional device-specific configuration
    device_id: int | None = None
    path: str | None = None  # NVMe file path


@dataclass
class TierUsage:
    """Storage tier usage statistics.

    Attributes:
        tier: Storage tier type.
        used_bytes: Currently used bytes.
        free_bytes: Available bytes.
        capacity_bytes: Total capacity.
        num_blocks: Number of blocks stored.
    """

    tier: Any  # StorageTier
    used_bytes: int
    free_bytes: int
    capacity_bytes: int
    num_blocks: int

    @property
    def utilization(self) -> float:
        """Calculate utilization ratio."""
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes


class StorageBackend(ABC):
    """Abstract storage backend interface.

    Defines the common interface for all storage tier backends.
    Implementations handle the actual data storage and retrieval.
    """

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """Read data from storage.

        Args:
            offset: Byte offset to read from.
            size: Number of bytes to read.

        Returns:
            Read data as bytes.
        """
        ...

    @abstractmethod
    def write(self, offset: int, data: bytes) -> None:
        """Write data to storage.

        Args:
            offset: Byte offset to write to.
            data: Data to write.
        """
        ...

    @abstractmethod
    def get_free_space(self) -> int:
        """Get available free space.

        Returns:
            Number of free bytes.
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend."""
        ...

    def close(self) -> None:
        """Close and cleanup the backend."""
        pass


class HBMBackend(StorageBackend):
    """HBM (GPU high-bandwidth memory) storage backend.

    Uses PyTorch tensors on GPU for high-speed KV cache storage.
    Falls back to CPU if CUDA is not available.
    
    Uses free list allocation with first-fit strategy to avoid fragmentation.
    """

    def __init__(self, device_id: int, capacity_bytes: int):
        """Initialize HBM backend.

        Args:
            device_id: CUDA device index.
            capacity_bytes: Pool capacity in bytes.
        """
        self.device_id = device_id
        self.capacity_bytes = capacity_bytes
        self._pool: bytes | None = None
        self._cuda_available = False

        # Free list: [(offset, size), ...] sorted by offset
        self._free_chunks: list[tuple[int, int]] = [(0, capacity_bytes)]
        # Allocated chunks: {offset: size}
        self._allocated_chunks: dict[int, int] = {}

    def initialize(self) -> None:
        """Initialize the GPU memory pool.

        Falls back to CPU memory if CUDA is not available.
        """
        try:
            import torch

            if torch.cuda.is_available() and self.device_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{self.device_id}")
                self._pool = torch.empty(
                    self.capacity_bytes,
                    dtype=torch.uint8,
                    device=self.device,
                )
                self._cuda_available = True
            else:
                # Fallback to CPU for testing/non-GPU environments
                self._pool = bytearray(self.capacity_bytes)
                self._cuda_available = False
        except ImportError:
            # PyTorch not available, use bytes
            self._pool = bytearray(self.capacity_bytes)
            self._cuda_available = False

    def read(self, offset: int, size: int) -> bytes:
        """Read data from HBM.

        Args:
            offset: Byte offset.
            size: Number of bytes.

        Returns:
            Read data as bytes.
        """
        if self._pool is None:
            raise RuntimeError("HBM backend not initialized")

        if self._cuda_available:

            tensor_slice = self._pool[offset : offset + size]
            return tensor_slice.cpu().numpy().tobytes()
        else:
            return bytes(self._pool[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        """Write data to HBM.

        Args:
            offset: Byte offset.
            data: Data to write.
        """
        if self._pool is None:
            raise RuntimeError("HBM backend not initialized")

        if self._cuda_available:
            import torch

            tensor_data = torch.frombuffer(
                bytearray(data), dtype=torch.uint8
            ).to(self.device)
            self._pool[offset : offset + len(data)] = tensor_data
        else:
            self._pool[offset : offset + len(data)] = data

    def get_free_space(self) -> int:
        """Get available HBM space."""
        return sum(size for _, size in self._free_chunks)

    def allocate(self, size: int) -> int:
        """Allocate space using first-fit strategy.

        Args:
            size: Bytes to allocate.

        Returns:
            Offset of allocated space.

        Raises:
            MemoryError: If insufficient space.
        """
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")

        # Find first chunk that fits
        for i, (offset, chunk_size) in enumerate(self._free_chunks):
            if chunk_size >= size:
                # Allocate from this chunk
                self._allocated_chunks[offset] = size

                # Update free list
                if chunk_size == size:
                    # Exact fit - remove chunk
                    del self._free_chunks[i]
                else:
                    # Partial fit - shrink chunk
                    self._free_chunks[i] = (offset + size, chunk_size - size)

                return offset

        # No suitable chunk found
        free_space = self.get_free_space()
        raise MemoryError(
            f"Insufficient HBM space: need {size} bytes, have {free_space} bytes available. "
            f"Allocated: {sum(self._allocated_chunks.values())}/{self.capacity_bytes} bytes. "
            f"Fragmentation: {len(self._free_chunks)} free chunks."
        )

    def deallocate(self, offset: int) -> None:
        """Deallocate space and merge adjacent free chunks.

        Args:
            offset: Offset to deallocate (returned by allocate()).
        
        Raises:
            ValueError: If offset is not allocated.
        """
        if offset not in self._allocated_chunks:
            raise ValueError(f"Offset {offset} is not allocated or already freed")

        size = self._allocated_chunks.pop(offset)

        # Add to free list
        self._free_chunks.append((offset, size))
        self._free_chunks.sort()  # Keep sorted by offset

        # Merge adjacent chunks
        self._merge_free_chunks()

    def _merge_free_chunks(self) -> None:
        """Merge contiguous free chunks to reduce fragmentation."""
        if len(self._free_chunks) <= 1:
            return

        merged = []
        current_offset, current_size = self._free_chunks[0]

        for offset, size in self._free_chunks[1:]:
            if current_offset + current_size == offset:
                # Adjacent - merge
                current_size += size
            else:
                # Not adjacent - save current and start new
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size

        # Add last chunk
        merged.append((current_offset, current_size))
        self._free_chunks = merged


class DDRBackend(StorageBackend):
    """DDR (CPU main memory) storage backend.

    Uses pinned memory for efficient GPU transfers.
    Uses free list allocation with first-fit strategy.
    """

    def __init__(self, capacity_bytes: int):
        """Initialize DDR backend.

        Args:
            capacity_bytes: Pool capacity in bytes.
        """
        self.capacity_bytes = capacity_bytes
        self._pool: bytes | None = None
        self._pinned = False

        # Free list management
        self._free_chunks: list[tuple[int, int]] = [(0, capacity_bytes)]
        self._allocated_chunks: dict[int, int] = {}

    def initialize(self) -> None:
        """Initialize the CPU memory pool with pinned memory if available."""
        try:
            import torch

            if torch.cuda.is_available():
                self._pool = torch.empty(
                    self.capacity_bytes,
                    dtype=torch.uint8,
                    pin_memory=True,  # Pinned memory for faster GPU transfers
                )
                self._pinned = True
            else:
                self._pool = bytearray(self.capacity_bytes)
                self._pinned = False
        except ImportError:
            self._pool = bytearray(self.capacity_bytes)
            self._pinned = False

    def read(self, offset: int, size: int) -> bytes:
        """Read data from DDR.

        Args:
            offset: Byte offset.
            size: Number of bytes.

        Returns:
            Read data as bytes.
        """
        if self._pool is None:
            raise RuntimeError("DDR backend not initialized")

        if self._pinned:
            tensor_slice = self._pool[offset : offset + size]
            return tensor_slice.numpy().tobytes()
        else:
            return bytes(self._pool[offset : offset + size])

    def write(self, offset: int, data: bytes) -> None:
        """Write data to DDR.

        Args:
            offset: Byte offset.
            data: Data to write.
        """
        if self._pool is None:
            raise RuntimeError("DDR backend not initialized")

        if self._pinned:
            import torch

            tensor_data = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            self._pool[offset : offset + len(data)] = tensor_data
        else:
            self._pool[offset : offset + len(data)] = data

    def get_free_space(self) -> int:
        """Get available DDR space."""
        return sum(size for _, size in self._free_chunks)

    def allocate(self, size: int) -> int:
        """Allocate space using first-fit strategy.

        Args:
            size: Bytes to allocate.

        Returns:
            Offset of allocated space.
        """
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")

        for i, (offset, chunk_size) in enumerate(self._free_chunks):
            if chunk_size >= size:
                self._allocated_chunks[offset] = size

                if chunk_size == size:
                    del self._free_chunks[i]
                else:
                    self._free_chunks[i] = (offset + size, chunk_size - size)

                return offset

        free_space = self.get_free_space()
        raise MemoryError(
            f"Insufficient DDR space: need {size} bytes, have {free_space} bytes available. "
            f"Allocated: {sum(self._allocated_chunks.values())}/{self.capacity_bytes} bytes."
        )

    def deallocate(self, offset: int) -> None:
        """Deallocate space and merge adjacent free chunks.

        Args:
            offset: Offset to deallocate.
        """
        if offset not in self._allocated_chunks:
            raise ValueError(f"Offset {offset} is not allocated or already freed")

        size = self._allocated_chunks.pop(offset)
        self._free_chunks.append((offset, size))
        self._free_chunks.sort()
        self._merge_free_chunks()

    def _merge_free_chunks(self) -> None:
        """Merge contiguous free chunks."""
        if len(self._free_chunks) <= 1:
            return

        merged = []
        current_offset, current_size = self._free_chunks[0]

        for offset, size in self._free_chunks[1:]:
            if current_offset + current_size == offset:
                current_size += size
            else:
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size

        merged.append((current_offset, current_size))
        self._free_chunks = merged


class NVMeBackend(StorageBackend):
    """NVMe SSD storage backend.

    Uses memory-mapped file for large capacity overflow storage.
    Uses free list allocation with first-fit strategy.
    """

    def __init__(self, path: str, capacity_bytes: int):
        """Initialize NVMe backend.

        Args:
            path: File path for storage.
            capacity_bytes: Pool capacity in bytes.
        """
        self.path = path
        self.capacity_bytes = capacity_bytes
        self._file = None

        # Free list management
        self._free_chunks: list[tuple[int, int]] = [(0, capacity_bytes)]
        self._allocated_chunks: dict[int, int] = {}

    def initialize(self) -> None:
        """Initialize the NVMe file storage."""
        # Create directory if needed
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Open file in read-write binary mode
        self._file = open(self.path, "wb+")  # noqa: SIM115

        # Pre-allocate file to avoid fragmentation
        self._file.seek(self.capacity_bytes - 1)
        self._file.write(b"\0")
        self._file.flush()

    def read(self, offset: int, size: int) -> bytes:
        """Read data from NVMe.

        Args:
            offset: Byte offset.
            size: Number of bytes.

        Returns:
            Read data as bytes.
        """
        if self._file is None:
            raise RuntimeError("NVMe backend not initialized")

        self._file.seek(offset)
        return self._file.read(size)

    def write(self, offset: int, data: bytes) -> None:
        """Write data to NVMe.

        Args:
            offset: Byte offset.
            data: Data to write.
        """
        if self._file is None:
            raise RuntimeError("NVMe backend not initialized")

        self._file.seek(offset)
        self._file.write(data)
        self._file.flush()

    def get_free_space(self) -> int:
        """Get available NVMe space."""
        return sum(size for _, size in self._free_chunks)

    def allocate(self, size: int) -> int:
        """Allocate space using first-fit strategy.

        Args:
            size: Bytes to allocate.

        Returns:
            Offset of allocated space.
        """
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")

        for i, (offset, chunk_size) in enumerate(self._free_chunks):
            if chunk_size >= size:
                self._allocated_chunks[offset] = size

                if chunk_size == size:
                    del self._free_chunks[i]
                else:
                    self._free_chunks[i] = (offset + size, chunk_size - size)

                return offset

        free_space = self.get_free_space()
        raise MemoryError(
            f"Insufficient NVMe space: need {size} bytes, have {free_space} bytes available. "
            f"Allocated: {sum(self._allocated_chunks.values())}/{self.capacity_bytes} bytes."
        )

    def deallocate(self, offset: int) -> None:
        """Deallocate space and merge adjacent free chunks.

        Args:
            offset: Offset to deallocate.
        """
        if offset not in self._allocated_chunks:
            raise ValueError(f"Offset {offset} is not allocated or already freed")

        size = self._allocated_chunks.pop(offset)
        self._free_chunks.append((offset, size))
        self._free_chunks.sort()
        self._merge_free_chunks()

    def _merge_free_chunks(self) -> None:
        """Merge contiguous free chunks."""
        if len(self._free_chunks) <= 1:
            return

        merged = []
        current_offset, current_size = self._free_chunks[0]

        for offset, size in self._free_chunks[1:]:
            if current_offset + current_size == offset:
                current_size += size
            else:
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size

        merged.append((current_offset, current_size))
        self._free_chunks = merged

    def close(self) -> None:
        """Close the file handle."""
        if self._file:
            self._file.close()
            self._file = None


class TieredKVStorage:
    """Three-tier KV storage manager.

    Manages HBM -> DDR -> NVMe three-tier storage:
    - HBM: Hot data, high-speed access
    - DDR: Warm data, CPU pinned memory
    - NVMe: Cold data, persistent storage

    Example:
        >>> storage = TieredKVStorage(
        ...     hbm_config=TierConfig(
        ...         tier=StorageTier.HBM,
        ...         capacity_bytes=1024 * 1024,
        ...         bandwidth_gbps=900.0,
        ...         latency_us=1.0,
        ...         device_id=0,
        ...     ),
        ...     ddr_config=TierConfig(
        ...         tier=StorageTier.DDR,
        ...         capacity_bytes=4 * 1024 * 1024,
        ...         bandwidth_gbps=50.0,
        ...         latency_us=100.0,
        ...     ),
        ... )
        >>> usage = storage.get_tier_usage(StorageTier.HBM)
    """

    def __init__(
        self,
        hbm_config: TierConfig,
        ddr_config: TierConfig,
        nvme_config: TierConfig | None = None,
    ):
        """Initialize tiered storage.

        Args:
            hbm_config: HBM tier configuration.
            ddr_config: DDR tier configuration.
            nvme_config: Optional NVMe tier configuration.
        """
        StorageTier = _get_storage_tier()

        self.configs = {
            StorageTier.HBM: hbm_config,
            StorageTier.DDR: ddr_config,
        }
        if nvme_config:
            self.configs[StorageTier.NVME] = nvme_config

        # Initialize backends
        self.backends: dict[Any, StorageBackend] = {}  # StorageTier -> Backend
        self._init_backends()

        # Block location mapping: block_id -> (tier, offset)
        self._block_locations: dict[int, tuple[Any, int]] = {}

    def _init_backends(self) -> None:
        """Initialize storage backends for each tier."""
        StorageTier = _get_storage_tier()

        hbm_cfg = self.configs[StorageTier.HBM]
        self.backends[StorageTier.HBM] = HBMBackend(
            device_id=hbm_cfg.device_id or 0,
            capacity_bytes=hbm_cfg.capacity_bytes,
        )

        ddr_cfg = self.configs[StorageTier.DDR]
        self.backends[StorageTier.DDR] = DDRBackend(
            capacity_bytes=ddr_cfg.capacity_bytes,
        )

        if StorageTier.NVME in self.configs:
            nvme_cfg = self.configs[StorageTier.NVME]
            self.backends[StorageTier.NVME] = NVMeBackend(
                path=nvme_cfg.path or "/tmp/sagellm_kv_cache.bin",
                capacity_bytes=nvme_cfg.capacity_bytes,
            )

        # Initialize all backends
        for backend in self.backends.values():
            backend.initialize()

    def get_tier_usage(self, tier: StorageTier) -> TierUsage:
        """Get usage statistics for a storage tier.

        Args:
            tier: Storage tier to query.

        Returns:
            TierUsage with current statistics.

        Raises:
            ValueError: If tier is not configured.
        """
        if tier not in self.backends:
            available = ", ".join(t.name for t in self.backends)
            raise ValueError(
                f"Tier {tier} not configured. Available tiers: {available}. "
                "Ensure TieredKVStorage was initialized with this tier."
            )

        backend = self.backends[tier]
        config = self.configs[tier]
        free = backend.get_free_space()

        return TierUsage(
            tier=tier,
            used_bytes=config.capacity_bytes - free,
            free_bytes=free,
            capacity_bytes=config.capacity_bytes,
            num_blocks=sum(
                1 for loc in self._block_locations.values() if loc[0] == tier
            ),
        )

    def read_block(
        self,
        block: KVBlockDescriptor,
    ) -> bytes:
        """Read a KV block from storage.

        Args:
            block: Block descriptor to read.

        Returns:
            Block data as bytes.

        Raises:
            ValueError: If block not found in storage.
        """
        if block.block_id not in self._block_locations:
            raise ValueError(f"Block {block.block_id} not found in storage")

        tier, offset = self._block_locations[block.block_id]
        data = self.backends[tier].read(offset, block.size_bytes)
        block.update_access()
        return data

    def write_block(
        self,
        block: KVBlockDescriptor,
        data: bytes,
    ) -> None:
        """Write a KV block to storage.

        Args:
            block: Block descriptor.
            data: Data to write.
        """
        tier = block.tier
        backend = self.backends[tier]

        # Allocate space
        offset = backend.allocate(len(data))

        # Write data
        backend.write(offset, data)

        # Record location
        self._block_locations[block.block_id] = (tier, offset)
        block.offset = offset
        block.size_bytes = len(data)

    def migrate_block(
        self,
        block: KVBlockDescriptor,
        to_tier: StorageTier,
    ) -> bool:
        """Migrate a block to a different storage tier.

        Args:
            block: Block to migrate.
            to_tier: Destination tier.

        Returns:
            True if migration succeeded.
        """
        if block.block_id not in self._block_locations:
            return False

        from_tier, offset = self._block_locations[block.block_id]
        if from_tier == to_tier:
            return True  # Already in target tier

        # Read data from source
        data = self.backends[from_tier].read(offset, block.size_bytes)

        # Allocate and write to destination
        new_offset = self.backends[to_tier].allocate(block.size_bytes)
        self.backends[to_tier].write(new_offset, data)

        # Update location
        self._block_locations[block.block_id] = (to_tier, new_offset)
        block.tier = to_tier
        block.offset = new_offset

        # Deallocate from source
        # NVMe uses size-based deallocation, HBM/DDR use offset-based
        from ..blocks.multi_granular import StorageTier as _StorageTier

        if from_tier == _StorageTier.NVME:
            self.backends[from_tier].deallocate(block.size_bytes)
        else:
            self.backends[from_tier].deallocate(offset)

        return True

    def delete_block(self, block: KVBlockDescriptor) -> bool:
        """Delete a block from storage.

        Args:
            block: Block to delete.

        Returns:
            True if deletion succeeded.
        """
        if block.block_id not in self._block_locations:
            return False

        tier, offset = self._block_locations[block.block_id]
        self.backends[tier].deallocate(offset)
        del self._block_locations[block.block_id]
        return True

    def get_estimated_latency(
        self,
        tier: StorageTier,
        size_bytes: int,
    ) -> float:
        """Estimate access latency for a tier.

        Args:
            tier: Storage tier.
            size_bytes: Data size.

        Returns:
            Estimated latency in microseconds.
        """
        config = self.configs[tier]
        # Latency + transfer time
        transfer_time_us = (size_bytes / (config.bandwidth_gbps * 1e9)) * 1e6
        return config.latency_us + transfer_time_us

    def close(self) -> None:
        """Close all storage backends."""
        for backend in self.backends.values():
            backend.close()

    def get_block_tier(self, block_id: int) -> StorageTier | None:
        """Get the current tier of a block.

        Args:
            block_id: Block identifier.

        Returns:
            Storage tier or None if not found.
        """
        location = self._block_locations.get(block_id)
        return location[0] if location else None
