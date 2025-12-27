# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Backend Base - Abstract interface for hardware accelerator backends.

This module defines the HardwareBackend ABC that all backends must implement.
The interface provides:
- Device discovery and initialization
- Memory management (allocation, transfer, pinning)
- Kernel execution
- Synchronization primitives
- Performance profiling hooks

Design principles:
- Hardware-agnostic API: Same code works on CUDA, Ascend, Cambricon, etc.
- Zero-copy when possible: Minimize data movement overhead
- Async-first: Support for overlapping compute and communication
- Explicit resource management: Clear ownership and lifecycle
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class DeviceType(Enum):
    """Supported device types."""

    CPU = auto()
    NVIDIA_GPU = auto()  # CUDA
    ASCEND_NPU = auto()  # Huawei Ascend
    CAMBRICON_MLU = auto()  # Cambricon
    HYGON_DCU = auto()  # Hygon


class MemoryKind(Enum):
    """Types of memory regions."""

    DEVICE = auto()  # On-device memory (GPU/NPU)
    HOST = auto()  # CPU memory (pageable)
    HOST_PINNED = auto()  # CPU memory (pinned for fast DMA)
    UNIFIED = auto()  # Unified virtual memory


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    device_id: int
    device_type: DeviceType
    name: str
    compute_capability: str = ""

    # Memory
    total_memory_bytes: int = 0
    available_memory_bytes: int = 0

    # Compute
    num_compute_units: int = 0  # SMs, Cores, etc.
    max_threads_per_unit: int = 0
    clock_rate_mhz: int = 0

    # Identifiers
    pci_bus_id: str = ""
    uuid: str = ""

    # Additional properties
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryHandle:
    """Handle to an allocated memory region."""

    ptr: int  # Memory address (device pointer or host pointer)
    size_bytes: int
    memory_kind: MemoryKind
    device_id: int
    dtype: str = "float16"
    shape: tuple[int, ...] = ()

    # For tracking
    is_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelConfig:
    """Configuration for kernel execution."""

    grid_dim: tuple[int, int, int] = (1, 1, 1)
    block_dim: tuple[int, int, int] = (1, 1, 1)
    shared_memory_bytes: int = 0
    stream: Any = None  # Backend-specific stream handle


class HardwareBackend(ABC):
    """
    Abstract base class for hardware accelerator backends.

    Each backend implementation must provide methods for:
    - Device management
    - Memory allocation and transfer
    - Kernel execution
    - Synchronization
    """

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """Return the device type this backend supports."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'cuda', 'ascend')."""

    # Device Management

    @abstractmethod
    def get_device_count(self) -> int:
        """Return the number of available devices."""

    @abstractmethod
    def get_device_info(self, device_id: int) -> DeviceInfo:
        """Get information about a specific device."""

    @abstractmethod
    def set_device(self, device_id: int) -> None:
        """Set the current device for subsequent operations."""

    @abstractmethod
    def get_current_device(self) -> int:
        """Get the current device ID."""

    @abstractmethod
    def synchronize(self, device_id: int | None = None) -> None:
        """
        Synchronize the device (wait for all operations to complete).

        Args:
            device_id: Device to sync (None for current device)
        """

    # Memory Management

    @abstractmethod
    def allocate(
        self,
        size_bytes: int,
        memory_kind: MemoryKind = MemoryKind.DEVICE,
        device_id: int | None = None,
    ) -> MemoryHandle:
        """
        Allocate memory.

        Args:
            size_bytes: Size in bytes to allocate
            memory_kind: Type of memory to allocate
            device_id: Device to allocate on (None for current)

        Returns:
            MemoryHandle to the allocated region
        """

    @abstractmethod
    def free(self, handle: MemoryHandle) -> None:
        """
        Free allocated memory.

        Args:
            handle: Memory handle to free
        """

    @abstractmethod
    def copy(
        self,
        src: MemoryHandle,
        dst: MemoryHandle,
        size_bytes: int | None = None,
        *,
        async_op: bool = False,
        stream: Any = None,
    ) -> None:
        """
        Copy data between memory regions.

        Args:
            src: Source memory handle
            dst: Destination memory handle
            size_bytes: Bytes to copy (None for full src size)
            async_op: Whether to perform async copy
            stream: Stream for async copy
        """

    @abstractmethod
    def memset(
        self,
        handle: MemoryHandle,
        value: int,
        size_bytes: int | None = None,
    ) -> None:
        """
        Set memory to a value.

        Args:
            handle: Memory handle to fill
            value: Byte value to set
            size_bytes: Bytes to set (None for full handle size)
        """

    # Tensor Operations (high-level)

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: str = "float16",
        device_id: int | None = None,
    ) -> MemoryHandle:
        """
        Allocate a tensor with given shape and dtype.

        Args:
            shape: Tensor shape
            dtype: Data type string
            device_id: Device to allocate on

        Returns:
            MemoryHandle for the tensor
        """
        dtype_sizes = {
            "float16": 2,
            "float32": 4,
            "float64": 8,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "bfloat16": 2,
        }
        element_size = dtype_sizes.get(dtype, 4)
        import math

        total_elements = math.prod(shape)
        size_bytes = total_elements * element_size

        handle = self.allocate(size_bytes, MemoryKind.DEVICE, device_id)
        handle.dtype = dtype
        handle.shape = shape
        return handle

    # Kernel Execution

    @abstractmethod
    def launch_kernel(
        self,
        kernel: Any,  # Backend-specific kernel object
        config: KernelConfig,
        *args: Any,
    ) -> None:
        """
        Launch a kernel on the device.

        Args:
            kernel: The kernel to launch
            config: Execution configuration
            *args: Kernel arguments
        """

    # Stream Management

    @abstractmethod
    def create_stream(self, device_id: int | None = None) -> Any:
        """Create an execution stream."""

    @abstractmethod
    def destroy_stream(self, stream: Any) -> None:
        """Destroy an execution stream."""

    @abstractmethod
    def stream_synchronize(self, stream: Any) -> None:
        """Wait for all operations on a stream to complete."""

    # Events

    @abstractmethod
    def create_event(self) -> Any:
        """Create a synchronization event."""

    @abstractmethod
    def destroy_event(self, event: Any) -> None:
        """Destroy an event."""

    @abstractmethod
    def record_event(self, event: Any, stream: Any = None) -> None:
        """Record an event on a stream."""

    @abstractmethod
    def wait_event(self, event: Any, stream: Any = None) -> None:
        """Make a stream wait for an event."""

    # Profiling

    def profile_region(self, name: str) -> Callable:
        """
        Context manager or decorator for profiling a code region.

        Default implementation is a no-op.
        """
        from contextlib import contextmanager

        @contextmanager
        def noop_context():
            yield

        return noop_context()


class DummyBackend(HardwareBackend):
    """Dummy backend for testing (no actual hardware operations)."""

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def name(self) -> str:
        return "dummy"

    def get_device_count(self) -> int:
        return 1

    def get_device_info(self, device_id: int) -> DeviceInfo:
        return DeviceInfo(
            device_id=device_id,
            device_type=DeviceType.CPU,
            name="Dummy Device",
        )

    def set_device(self, device_id: int) -> None:
        pass

    def get_current_device(self) -> int:
        return 0

    def synchronize(self, device_id: int | None = None) -> None:
        pass

    def allocate(
        self,
        size_bytes: int,
        memory_kind: MemoryKind = MemoryKind.DEVICE,
        device_id: int | None = None,
    ) -> MemoryHandle:
        return MemoryHandle(
            ptr=0,
            size_bytes=size_bytes,
            memory_kind=memory_kind,
            device_id=device_id or 0,
        )

    def free(self, handle: MemoryHandle) -> None:
        handle.is_valid = False

    def copy(
        self,
        src: MemoryHandle,
        dst: MemoryHandle,
        size_bytes: int | None = None,
        *,
        async_op: bool = False,
        stream: Any = None,
    ) -> None:
        pass

    def memset(
        self,
        handle: MemoryHandle,
        value: int,
        size_bytes: int | None = None,
    ) -> None:
        pass

    def launch_kernel(
        self,
        kernel: Any,
        config: KernelConfig,
        *args: Any,
    ) -> None:
        pass

    def create_stream(self, device_id: int | None = None) -> Any:
        return None

    def destroy_stream(self, stream: Any) -> None:
        pass

    def stream_synchronize(self, stream: Any) -> None:
        pass

    def create_event(self) -> Any:
        return None

    def destroy_event(self, event: Any) -> None:
        pass

    def record_event(self, event: Any, stream: Any = None) -> None:
        pass

    def wait_event(self, event: Any, stream: Any = None) -> None:
        pass


__all__ = [
    "DeviceType",
    "MemoryKind",
    "DeviceInfo",
    "MemoryHandle",
    "KernelConfig",
    "HardwareBackend",
    "DummyBackend",
]
