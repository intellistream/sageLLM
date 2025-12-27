# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware backend protocols for domestic accelerator support.

This module defines the unified hardware backend abstraction layer that supports:
- NVIDIA CUDA
- Huawei Ascend (昇腾)
- Cambricon MLU (寒武纪)
- Hygon DCU (海光)

Design principles:
1. Unified interface: All backends implement the same protocol
2. Auto-discovery: Runtime automatically detects available hardware
3. Graceful degradation: Fallback when hardware unavailable
4. Extensibility: Easy to add new hardware support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import torch


class BackendType(Enum):
    """Hardware backend types."""

    CUDA = auto()  # NVIDIA CUDA
    ASCEND = auto()  # Huawei Ascend NPU
    CAMBRICON = auto()  # Cambricon MLU
    HYGON = auto()  # Hygon DCU (ROCm-based)
    CPU = auto()  # CPU fallback


@dataclass
class DeviceInfo:
    """Device information."""

    backend: BackendType
    device_id: int
    name: str

    # Compute capability
    compute_capability: Optional[str] = None  # e.g., "8.0" for A100

    # Memory
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0

    # Cores
    num_cores: int = 0

    # Driver/SDK version
    driver_version: Optional[str] = None
    sdk_version: Optional[str] = None

    # Additional properties
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelCapabilities:
    """Kernel capabilities."""

    # Supported precisions
    supports_fp32: bool = True
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_int8: bool = False
    supports_int4: bool = False

    # Sparse support
    supports_sparse_2_4: bool = False

    # Special operators
    supports_flash_attention: bool = False
    supports_paged_attention: bool = False
    supports_fused_moe: bool = False

    # Communication
    supports_nccl: bool = False
    supports_hccl: bool = False  # Huawei HCCL


class HardwareBackend(ABC):
    """Hardware backend abstract base class.

    Defines the interface that all hardware backends must implement.
    """

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return backend type."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        ...

    @abstractmethod
    def get_device_count(self) -> int:
        """Get number of available devices."""
        ...

    @abstractmethod
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get device information."""
        ...

    @abstractmethod
    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        """Get kernel capabilities."""
        ...

    @abstractmethod
    def get_device(self, device_id: int = 0) -> torch.device:
        """Get PyTorch device object."""
        ...

    @abstractmethod
    def synchronize(self, device_id: Optional[int] = None) -> None:
        """Synchronize device.

        Args:
            device_id: Device ID, None for current device
        """
        ...

    @abstractmethod
    def memory_stats(self, device_id: int = 0) -> dict[str, float]:
        """Get memory statistics.

        Returns:
            Dictionary containing total_gb, used_gb, free_gb
        """
        ...

    @abstractmethod
    def empty_cache(self, device_id: Optional[int] = None) -> None:
        """Empty cache."""
        ...

    # === Optional methods (with default implementation) ===

    def set_device(self, device_id: int) -> None:
        """Set current device."""
        torch.cuda.set_device(device_id)  # Default implementation

    def current_device(self) -> int:
        """Get current device ID."""
        return torch.cuda.current_device()  # Default implementation

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device_id: int = 0,
    ) -> torch.Tensor:
        """Allocate tensor.

        Some backends may require special memory allocation strategies.
        """
        device = self.get_device(device_id)
        return torch.empty(shape, dtype=dtype, device=device)

    def copy_to_device(
        self,
        tensor: torch.Tensor,
        device_id: int = 0,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Copy tensor to device."""
        device = self.get_device(device_id)
        return tensor.to(device, non_blocking=non_blocking)

    def copy_to_host(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Copy tensor to CPU."""
        return tensor.cpu()


class CommunicationBackend(ABC):
    """Communication backend abstraction.

    For multi-device/multi-node communication.
    """

    @abstractmethod
    def init_process_group(
        self,
        backend: str,
        world_size: int,
        rank: int,
        **kwargs: Any,
    ) -> None:
        """Initialize process group."""
        ...

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
    ) -> torch.Tensor:
        """All-reduce operation."""
        ...

    @abstractmethod
    def all_gather(
        self,
        tensor: torch.Tensor,
        world_size: int,
    ) -> list[torch.Tensor]:
        """All-gather operation."""
        ...

    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast operation."""
        ...

    @abstractmethod
    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
    ) -> None:
        """Send tensor."""
        ...

    @abstractmethod
    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src: int,
    ) -> torch.Tensor:
        """Receive tensor."""
        ...
