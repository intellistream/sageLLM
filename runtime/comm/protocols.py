# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Communication Protocols - Abstract interfaces for different transport layers.

Supported protocols:
- RDMA: High-performance remote memory access (InfiniBand, RoCE)
- TCP: Standard TCP/IP networking
- SharedMemory: Intra-node shared memory communication
- NCCL: NVIDIA collective communication (GPU-optimized)
- HCCL: Huawei collective communication (Ascend-optimized)

Example:
    >>> from sageLLM.runtime.comm.protocols import create_protocol
    >>> protocol = create_protocol("rdma", device_id=0)
    >>> await protocol.send(tensor, dest_rank=1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ProtocolType(Enum):
    """Supported communication protocol types."""

    RDMA = auto()  # Remote Direct Memory Access
    TCP = auto()  # Standard TCP/IP
    SHARED_MEMORY = auto()  # Intra-node shared memory
    NCCL = auto()  # NVIDIA Collective Communication Library
    HCCL = auto()  # Huawei Collective Communication Library
    CNCL = auto()  # Cambricon Collective Communication Library


@dataclass
class EndpointAddress:
    """Address for a communication endpoint."""

    host: str
    port: int
    device_id: int = 0
    protocol: ProtocolType = ProtocolType.TCP

    def __str__(self) -> str:
        return f"{self.host}:{self.port}/dev{self.device_id}"


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""

    @property
    @abstractmethod
    def protocol_type(self) -> ProtocolType:
        """Return the protocol type."""

    @abstractmethod
    async def connect(self, remote: EndpointAddress) -> None:
        """Establish connection to a remote endpoint."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def send(self, data: bytes, dest_rank: int) -> None:
        """Send data to a remote rank."""

    @abstractmethod
    async def recv(self, src_rank: int) -> bytes:
        """Receive data from a remote rank."""

    @abstractmethod
    async def send_tensor(self, tensor: Any, dest_rank: int) -> None:
        """Send a tensor to a remote rank (zero-copy if possible)."""

    @abstractmethod
    async def recv_tensor(self, shape: tuple[int, ...], dtype: Any, src_rank: int) -> Any:
        """Receive a tensor from a remote rank."""


class TCPProtocol(CommunicationProtocol):
    """TCP/IP-based communication protocol."""

    @property
    def protocol_type(self) -> ProtocolType:
        return ProtocolType.TCP

    async def connect(self, remote: EndpointAddress) -> None:
        """Establish TCP connection."""
        # TODO: Implement TCP connection
        pass

    async def disconnect(self) -> None:
        """Close TCP connection."""
        # TODO: Implement disconnect
        pass

    async def send(self, data: bytes, dest_rank: int) -> None:
        """Send bytes over TCP."""
        # TODO: Implement send
        pass

    async def recv(self, src_rank: int) -> bytes:
        """Receive bytes over TCP."""
        # TODO: Implement recv
        return b""

    async def send_tensor(self, tensor: Any, dest_rank: int) -> None:
        """Send tensor over TCP (serialized)."""
        # TODO: Implement tensor send
        pass

    async def recv_tensor(self, shape: tuple[int, ...], dtype: Any, src_rank: int) -> Any:
        """Receive tensor over TCP."""
        # TODO: Implement tensor recv
        return None


class RDMAProtocol(CommunicationProtocol):
    """RDMA-based communication protocol for high-performance transfers."""

    @property
    def protocol_type(self) -> ProtocolType:
        return ProtocolType.RDMA

    async def connect(self, remote: EndpointAddress) -> None:
        """Establish RDMA connection."""
        # TODO: Implement RDMA QP setup
        pass

    async def disconnect(self) -> None:
        """Close RDMA connection."""
        # TODO: Implement RDMA cleanup
        pass

    async def send(self, data: bytes, dest_rank: int) -> None:
        """Send bytes via RDMA write."""
        # TODO: Implement RDMA send
        pass

    async def recv(self, src_rank: int) -> bytes:
        """Receive bytes via RDMA read."""
        # TODO: Implement RDMA recv
        return b""

    async def send_tensor(self, tensor: Any, dest_rank: int) -> None:
        """Send tensor via RDMA (zero-copy)."""
        # TODO: Implement zero-copy tensor send
        pass

    async def recv_tensor(self, shape: tuple[int, ...], dtype: Any, src_rank: int) -> Any:
        """Receive tensor via RDMA (zero-copy)."""
        # TODO: Implement zero-copy tensor recv
        return None


class SharedMemoryProtocol(CommunicationProtocol):
    """Shared memory protocol for intra-node communication."""

    @property
    def protocol_type(self) -> ProtocolType:
        return ProtocolType.SHARED_MEMORY

    async def connect(self, remote: EndpointAddress) -> None:
        """Map shared memory region."""
        # TODO: Implement shared memory setup
        pass

    async def disconnect(self) -> None:
        """Unmap shared memory region."""
        # TODO: Implement cleanup
        pass

    async def send(self, data: bytes, dest_rank: int) -> None:
        """Write to shared memory."""
        # TODO: Implement
        pass

    async def recv(self, src_rank: int) -> bytes:
        """Read from shared memory."""
        # TODO: Implement
        return b""

    async def send_tensor(self, tensor: Any, dest_rank: int) -> None:
        """Share tensor via shared memory (zero-copy)."""
        # TODO: Implement
        pass

    async def recv_tensor(self, shape: tuple[int, ...], dtype: Any, src_rank: int) -> Any:
        """Access tensor from shared memory."""
        # TODO: Implement
        return None


def create_protocol(
    protocol_type: str | ProtocolType,
    **kwargs: Any,
) -> CommunicationProtocol:
    """
    Factory function to create a communication protocol instance.

    Args:
        protocol_type: Type of protocol ("tcp", "rdma", "shared_memory", or ProtocolType)
        **kwargs: Protocol-specific configuration

    Returns:
        A CommunicationProtocol instance
    """
    if isinstance(protocol_type, str):
        protocol_type = ProtocolType[protocol_type.upper()]

    protocol_classes: dict[ProtocolType, type[CommunicationProtocol]] = {
        ProtocolType.TCP: TCPProtocol,
        ProtocolType.RDMA: RDMAProtocol,
        ProtocolType.SHARED_MEMORY: SharedMemoryProtocol,
        # NCCL and HCCL require specific backend implementations
    }

    if protocol_type not in protocol_classes:
        msg = f"Protocol {protocol_type} not yet implemented"
        raise NotImplementedError(msg)

    return protocol_classes[protocol_type]()


__all__ = [
    "ProtocolType",
    "EndpointAddress",
    "CommunicationProtocol",
    "TCPProtocol",
    "RDMAProtocol",
    "SharedMemoryProtocol",
    "create_protocol",
]
