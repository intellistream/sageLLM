# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collective Communication Primitives - High-level collective operations.

Provides collective operations for distributed inference:
- AllReduce: Sum/average tensors across all devices
- AllGather: Gather tensors from all devices
- Broadcast: Send tensor from one device to all others
- ReduceScatter: Reduce and scatter result across devices

These primitives are used for:
- Tensor parallelism (AllReduce after attention/FFN)
- Pipeline parallelism (point-to-point for activations)
- KV cache synchronization in PD separation

Example:
    >>> from sageLLM.runtime.comm.collective import CollectiveGroup
    >>> group = CollectiveGroup(ranks=[0, 1, 2, 3])
    >>> await group.all_reduce(tensor, op=ReduceOp.SUM)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ReduceOp(Enum):
    """Reduction operations for collective communication."""

    SUM = auto()
    PRODUCT = auto()
    MIN = auto()
    MAX = auto()
    AVG = auto()


@dataclass
class CollectiveConfig:
    """Configuration for collective operations."""

    # Communication settings
    async_op: bool = False  # Whether to run asynchronously
    timeout_ms: int = 30000  # Timeout in milliseconds

    # Performance tuning
    chunk_size: int = 0  # 0 means auto-tune
    use_ring: bool = True  # Use ring algorithm when possible
    use_tree: bool = False  # Use tree algorithm

    # Hardware-specific
    use_nccl: bool = True  # Use NCCL on NVIDIA GPUs
    use_hccl: bool = True  # Use HCCL on Ascend NPUs


class CollectiveBackend(ABC):
    """Abstract backend for collective operations."""

    @abstractmethod
    async def all_reduce(
        self,
        tensor: Any,
        op: ReduceOp,
        group_ranks: list[int],
    ) -> Any:
        """Perform all-reduce across the group."""

    @abstractmethod
    async def all_gather(
        self,
        tensor: Any,
        group_ranks: list[int],
    ) -> list[Any]:
        """Gather tensors from all ranks in the group."""

    @abstractmethod
    async def broadcast(
        self,
        tensor: Any,
        src_rank: int,
        group_ranks: list[int],
    ) -> Any:
        """Broadcast tensor from src_rank to all ranks."""

    @abstractmethod
    async def reduce_scatter(
        self,
        tensors: list[Any],
        op: ReduceOp,
        group_ranks: list[int],
    ) -> Any:
        """Reduce tensors and scatter result."""

    @abstractmethod
    async def send(
        self,
        tensor: Any,
        dst_rank: int,
    ) -> None:
        """Send tensor to destination rank."""

    @abstractmethod
    async def recv(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        src_rank: int,
    ) -> Any:
        """Receive tensor from source rank."""


class SimulatedBackend(CollectiveBackend):
    """Simulated backend for testing (no actual communication)."""

    async def all_reduce(
        self,
        tensor: Any,
        op: ReduceOp,
        group_ranks: list[int],
    ) -> Any:
        """Simulated all-reduce (returns input unchanged)."""
        return tensor

    async def all_gather(
        self,
        tensor: Any,
        group_ranks: list[int],
    ) -> list[Any]:
        """Simulated all-gather (returns copies)."""
        return [tensor] * len(group_ranks)

    async def broadcast(
        self,
        tensor: Any,
        src_rank: int,
        group_ranks: list[int],
    ) -> Any:
        """Simulated broadcast (returns input)."""
        return tensor

    async def reduce_scatter(
        self,
        tensors: list[Any],
        op: ReduceOp,
        group_ranks: list[int],
    ) -> Any:
        """Simulated reduce-scatter (returns first tensor)."""
        return tensors[0] if tensors else None

    async def send(
        self,
        tensor: Any,
        dst_rank: int,
    ) -> None:
        """Simulated send (no-op)."""
        pass

    async def recv(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        src_rank: int,
    ) -> Any:
        """Simulated recv (returns None)."""
        return None


@dataclass
class CollectiveGroup:
    """
    A group of devices that participate in collective operations.

    Example:
        >>> group = CollectiveGroup(ranks=[0, 1, 2, 3])
        >>> result = await group.all_reduce(tensor, ReduceOp.SUM)
    """

    ranks: list[int]
    config: CollectiveConfig = field(default_factory=CollectiveConfig)
    backend: CollectiveBackend = field(default_factory=SimulatedBackend)

    @property
    def size(self) -> int:
        """Number of ranks in the group."""
        return len(self.ranks)

    async def all_reduce(
        self,
        tensor: Any,
        op: ReduceOp = ReduceOp.SUM,
    ) -> Any:
        """
        All-reduce tensor across all ranks in the group.

        Args:
            tensor: Input tensor (modified in place if possible)
            op: Reduction operation

        Returns:
            Reduced tensor (same shape as input)
        """
        return await self.backend.all_reduce(tensor, op, self.ranks)

    async def all_gather(
        self,
        tensor: Any,
    ) -> list[Any]:
        """
        Gather tensor from all ranks.

        Args:
            tensor: Local tensor to contribute

        Returns:
            List of tensors from all ranks
        """
        return await self.backend.all_gather(tensor, self.ranks)

    async def broadcast(
        self,
        tensor: Any,
        src_rank: int,
    ) -> Any:
        """
        Broadcast tensor from one rank to all others.

        Args:
            tensor: Tensor to broadcast (only valid on src_rank)
            src_rank: Source rank

        Returns:
            Broadcast tensor (same on all ranks)
        """
        return await self.backend.broadcast(tensor, src_rank, self.ranks)

    async def reduce_scatter(
        self,
        tensors: list[Any],
        op: ReduceOp = ReduceOp.SUM,
    ) -> Any:
        """
        Reduce and scatter result across ranks.

        Args:
            tensors: List of tensors to reduce (one per rank)
            op: Reduction operation

        Returns:
            Scattered chunk of reduced result
        """
        return await self.backend.reduce_scatter(tensors, op, self.ranks)

    async def send(
        self,
        tensor: Any,
        dst_rank: int,
    ) -> None:
        """
        Send tensor to destination rank.

        Args:
            tensor: Tensor to send
            dst_rank: Destination rank
        """
        await self.backend.send(tensor, dst_rank)

    async def recv(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        src_rank: int,
    ) -> Any:
        """
        Receive tensor from source rank.

        Args:
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            src_rank: Source rank

        Returns:
            Received tensor
        """
        return await self.backend.recv(shape, dtype, src_rank)


def create_tensor_parallel_group(
    world_size: int,
    tp_size: int,
) -> list[CollectiveGroup]:
    """
    Create tensor parallel groups.

    Args:
        world_size: Total number of ranks
        tp_size: Tensor parallel size (ranks per group)

    Returns:
        List of CollectiveGroup for tensor parallelism
    """
    if world_size % tp_size != 0:
        msg = f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        raise ValueError(msg)

    groups = []
    for i in range(0, world_size, tp_size):
        ranks = list(range(i, i + tp_size))
        groups.append(CollectiveGroup(ranks=ranks))
    return groups


def create_pipeline_parallel_group(
    world_size: int,
    pp_size: int,
) -> list[CollectiveGroup]:
    """
    Create pipeline parallel groups.

    Args:
        world_size: Total number of ranks
        pp_size: Pipeline parallel size (ranks per pipeline)

    Returns:
        List of CollectiveGroup for pipeline parallelism
    """
    if world_size % pp_size != 0:
        msg = f"world_size ({world_size}) must be divisible by pp_size ({pp_size})"
        raise ValueError(msg)

    groups = []
    num_pipelines = world_size // pp_size
    for i in range(num_pipelines):
        ranks = [i + j * num_pipelines for j in range(pp_size)]
        groups.append(CollectiveGroup(ranks=ranks))
    return groups


__all__ = [
    "ReduceOp",
    "CollectiveConfig",
    "CollectiveBackend",
    "SimulatedBackend",
    "CollectiveGroup",
    "create_tensor_parallel_group",
    "create_pipeline_parallel_group",
]
