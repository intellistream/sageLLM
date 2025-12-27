# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Topology Discovery and Management - Cluster topology for distributed inference.

This module provides:
- Automatic topology discovery (detect GPUs, network interfaces, NUMA nodes)
- Topology representation (nodes, devices, connections)
- Optimal communication path selection

Example:
    >>> from sageLLM.runtime.comm.topology import TopologyManager
    >>> topo = TopologyManager()
    >>> await topo.discover()
    >>> path = topo.get_optimal_path(src_device=0, dst_device=4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class DeviceType(Enum):
    """Types of compute devices."""

    CPU = auto()
    NVIDIA_GPU = auto()  # CUDA devices
    ASCEND_NPU = auto()  # Huawei Ascend NPUs
    CAMBRICON_MLU = auto()  # Cambricon MLUs
    HYGON_DCU = auto()  # Hygon DCUs


class ConnectionType(Enum):
    """Types of device connections."""

    PCIE = auto()  # PCIe connection
    NVLINK = auto()  # NVIDIA NVLink
    HCCS = auto()  # Huawei HCCS
    INFINIBAND = auto()  # InfiniBand network
    ETHERNET = auto()  # Ethernet network
    SHARED_MEMORY = auto()  # Same-node shared memory


@dataclass
class Device:
    """Representation of a compute device."""

    device_id: int
    device_type: DeviceType
    node_id: int  # Which physical node this device is on
    numa_node: int = 0  # NUMA node affinity
    memory_bytes: int = 0  # Device memory capacity
    compute_capability: str = ""  # e.g., "8.0" for A100

    # Hardware identifiers
    pci_bus_id: str = ""
    uuid: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Connection:
    """Representation of a connection between devices."""

    src_device_id: int
    dst_device_id: int
    connection_type: ConnectionType
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    is_bidirectional: bool = True


@dataclass
class Node:
    """Representation of a physical compute node."""

    node_id: int
    hostname: str
    ip_address: str
    devices: list[Device] = field(default_factory=list)
    cpu_count: int = 0
    memory_bytes: int = 0
    numa_nodes: int = 1


@dataclass
class Topology:
    """Complete cluster topology."""

    nodes: dict[int, Node] = field(default_factory=dict)
    devices: dict[int, Device] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)

    def get_device(self, device_id: int) -> Device | None:
        """Get a device by ID."""
        return self.devices.get(device_id)

    def get_node(self, node_id: int) -> Node | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_devices_on_node(self, node_id: int) -> list[Device]:
        """Get all devices on a specific node."""
        return [d for d in self.devices.values() if d.node_id == node_id]

    def get_connections_from(self, device_id: int) -> list[Connection]:
        """Get all connections originating from a device."""
        return [c for c in self.connections if c.src_device_id == device_id]

    def get_connection(self, src_id: int, dst_id: int) -> Connection | None:
        """Get the connection between two devices."""
        for conn in self.connections:
            if conn.src_device_id == src_id and conn.dst_device_id == dst_id:
                return conn
            if conn.is_bidirectional and conn.src_device_id == dst_id and conn.dst_device_id == src_id:
                return conn
        return None


class TopologyManager:
    """Manager for discovering and querying cluster topology."""

    def __init__(self) -> None:
        """Initialize the topology manager."""
        self._topology: Topology | None = None
        self._discovered = False

    @property
    def topology(self) -> Topology:
        """Get the discovered topology."""
        if not self._discovered or self._topology is None:
            msg = "Topology not discovered. Call discover() first."
            raise RuntimeError(msg)
        return self._topology

    async def discover(self) -> Topology:
        """
        Discover the cluster topology.

        This method:
        1. Detects local compute devices (GPUs, NPUs, etc.)
        2. Queries device properties and connections
        3. Discovers network interfaces and remote nodes
        4. Builds the complete topology graph

        Returns:
            The discovered Topology
        """
        # TODO: Implement actual discovery
        # For now, create a placeholder topology
        self._topology = Topology()
        self._discovered = True
        return self._topology

    def get_optimal_path(
        self,
        src_device: int,
        dst_device: int,
    ) -> list[Connection]:
        """
        Find the optimal communication path between two devices.

        Args:
            src_device: Source device ID
            dst_device: Destination device ID

        Returns:
            List of connections forming the optimal path
        """
        # TODO: Implement shortest path / highest bandwidth path
        topology = self.topology
        direct_conn = topology.get_connection(src_device, dst_device)
        if direct_conn:
            return [direct_conn]
        return []

    def estimate_transfer_time(
        self,
        src_device: int,
        dst_device: int,
        size_bytes: int,
    ) -> float:
        """
        Estimate transfer time between devices.

        Args:
            src_device: Source device ID
            dst_device: Destination device ID
            size_bytes: Data size in bytes

        Returns:
            Estimated transfer time in milliseconds
        """
        path = self.get_optimal_path(src_device, dst_device)
        if not path:
            return float("inf")

        # Use the bottleneck bandwidth
        min_bandwidth = min(c.bandwidth_gbps for c in path)
        if min_bandwidth <= 0:
            return float("inf")

        # bandwidth in GB/s, size in bytes
        bandwidth_bytes_per_ms = min_bandwidth * 1e9 / 8 / 1000
        return size_bytes / bandwidth_bytes_per_ms


__all__ = [
    "DeviceType",
    "ConnectionType",
    "Device",
    "Connection",
    "Node",
    "Topology",
    "TopologyManager",
]
