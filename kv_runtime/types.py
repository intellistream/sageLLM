# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""KV runtime types and data structures.

This module defines the core types for KV cache management:
- KVCacheSchema: KV cache configuration and format
- KVBlockInfo: Block-level metadata
- KVTier: Storage tier hierarchy
- AllocationResult: Block allocation response
- MigrationPlan: Cross-tier migration specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class KVTier(str, Enum):
    """Storage tier for KV cache blocks.

    Defines the memory hierarchy for KV cache storage:
    - GPU: High-bandwidth GPU memory (HBM)
    - CPU: System memory (DRAM)
    - NVME: NVMe storage for overflow
    """

    GPU = "gpu"
    CPU = "cpu"
    NVME = "nvme"


class KVDataType(str, Enum):
    """Data type for KV cache values."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class KVCacheSchema:
    """Schema defining KV cache format and configuration.

    This schema describes the KV cache layout and is used by all modules
    that interact with KV cache data (kv_runtime, prefix_reuse, engines).

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (per layer).
        head_dim: Dimension per attention head.
        block_size: Number of tokens per KV block.
        dtype: Data type for KV values.
        page_size_bytes: Size of each KV page in bytes.
        max_seq_len: Maximum sequence length supported.
        enable_prefix_caching: Whether prefix caching is enabled.
        metadata: Additional schema metadata.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    block_size: int = 16
    dtype: KVDataType = KVDataType.FP16
    page_size_bytes: int | None = None
    max_seq_len: int = 8192
    enable_prefix_caching: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        if self.page_size_bytes is None:
            # Calculate page size: 2 (K+V) * layers * heads * head_dim * dtype_size
            dtype_bytes = {
                KVDataType.FP16: 2,
                KVDataType.BF16: 2,
                KVDataType.FP8_E4M3: 1,
                KVDataType.FP8_E5M2: 1,
                KVDataType.INT8: 1,
                KVDataType.INT4: 0.5,
            }
            bytes_per_elem = dtype_bytes.get(self.dtype, 2)
            self.page_size_bytes = int(
                2
                * self.num_layers
                * self.num_heads
                * self.head_dim
                * self.block_size
                * bytes_per_elem
            )

    @property
    def tokens_per_block(self) -> int:
        """Number of tokens that fit in one KV block."""
        return self.block_size

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "dtype": self.dtype.value,
            "page_size_bytes": self.page_size_bytes,
            "max_seq_len": self.max_seq_len,
            "enable_prefix_caching": self.enable_prefix_caching,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KVCacheSchema:
        """Deserialize from dictionary."""
        data = data.copy()
        if "dtype" in data and isinstance(data["dtype"], str):
            data["dtype"] = KVDataType(data["dtype"])
        return cls(**data)

    @classmethod
    def for_model(cls, model_id: str, **kwargs: Any) -> KVCacheSchema:
        """Create schema for a known model.

        Args:
            model_id: Model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
            **kwargs: Override default parameters

        Returns:
            KVCacheSchema configured for the model
        """
        # Model-specific defaults (can be extended)
        model_configs: dict[str, dict[str, Any]] = {
            "qwen2.5-7b": {
                "num_layers": 28,
                "num_heads": 28,
                "head_dim": 128,
            },
            "qwen2.5-14b": {
                "num_layers": 40,
                "num_heads": 40,
                "head_dim": 128,
            },
            "qwen2.5-72b": {
                "num_layers": 80,
                "num_heads": 64,
                "head_dim": 128,
            },
            "llama3-8b": {
                "num_layers": 32,
                "num_heads": 32,
                "head_dim": 128,
            },
            "llama3-70b": {
                "num_layers": 80,
                "num_heads": 64,
                "head_dim": 128,
            },
        }

        # Find matching config
        model_lower = model_id.lower()
        config: dict[str, Any] = {}
        for key, value in model_configs.items():
            if key in model_lower:
                config = value.copy()
                break

        # Use defaults if no match
        if not config:
            config = {
                "num_layers": 32,
                "num_heads": 32,
                "head_dim": 128,
            }

        # Apply overrides
        config.update(kwargs)
        return cls(**config)


@dataclass
class KVBlockInfo:
    """Information about a KV cache block.

    Attributes:
        block_id: Unique block identifier.
        tier: Current storage tier.
        size_bytes: Block size in bytes.
        tenant_id: Owner tenant identifier (for quota management).
        sequence_id: Associated sequence identifier.
        token_range: Range of tokens stored (start, end).
        last_access: Last access timestamp.
        access_count: Total number of accesses.
        is_prefix: Whether this block is a shared prefix.
        ref_count: Reference count for shared blocks.
        metadata: Additional block metadata.
    """

    block_id: int
    tier: KVTier = KVTier.GPU
    size_bytes: int = 0
    tenant_id: str | None = None
    sequence_id: str | None = None
    token_range: tuple[int, int] | None = None
    last_access: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    is_prefix: bool = False
    ref_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "block_id": self.block_id,
            "tier": self.tier.value,
            "size_bytes": self.size_bytes,
            "tenant_id": self.tenant_id,
            "sequence_id": self.sequence_id,
            "token_range": self.token_range,
            "last_access": self.last_access.isoformat(),
            "access_count": self.access_count,
            "is_prefix": self.is_prefix,
            "ref_count": self.ref_count,
            "metadata": self.metadata,
        }


@dataclass
class AllocationResult:
    """Result of a KV block allocation request.

    Attributes:
        success: Whether allocation succeeded.
        block_ids: List of allocated block IDs.
        tier: Tier where blocks were allocated.
        total_bytes: Total bytes allocated.
        error_message: Error message if allocation failed.
    """

    success: bool
    block_ids: list[int] = field(default_factory=list)
    tier: KVTier = KVTier.GPU
    total_bytes: int = 0
    error_message: str | None = None


@dataclass
class MigrationPlan:
    """Plan for migrating KV blocks between tiers.

    Attributes:
        plan_id: Unique plan identifier.
        src_tier: Source storage tier.
        dst_tier: Destination storage tier.
        block_ids: List of block IDs to migrate.
        total_bytes: Total bytes to migrate.
        priority: Migration priority (lower = higher priority).
        deadline: Optional deadline for completion.
        metadata: Additional plan metadata.
    """

    plan_id: str
    src_tier: KVTier
    dst_tier: KVTier
    block_ids: list[int]
    total_bytes: int = 0
    priority: int = 10
    deadline: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "src_tier": self.src_tier.value,
            "dst_tier": self.dst_tier.value,
            "block_ids": self.block_ids,
            "total_bytes": self.total_bytes,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "metadata": self.metadata,
        }
