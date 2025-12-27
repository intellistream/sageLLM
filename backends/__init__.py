# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Backends - Abstraction layer for different accelerator platforms.

Supported backends:
- cuda: NVIDIA GPUs (CUDA)
- ascend: Huawei Ascend NPUs (CANN)
- cambricon: Cambricon MLUs (CNToolkit)
- hygon: Hygon DCUs (ROCm-compatible)

Each backend provides:
- Device management (discovery, memory, compute capabilities)
- Kernel execution interface
- Memory allocation and transfer
- Collective communication integration

Example:
    >>> from sageLLM.backends import get_backend
    >>> backend = get_backend("cuda")
    >>> backend.initialize(device_id=0)
    >>> tensor = backend.allocate(shape=(1024, 1024), dtype="float16")
    
New protocols (Task 4):
    >>> from sageLLM.backends import get_backend, BackendType
    >>> # Auto-detect default backend
    >>> backend = get_backend()
    >>> # Or specify backend type
    >>> backend = get_backend(BackendType.CUDA)
    >>> # Get device info
    >>> info = backend.get_device_info()
    >>> print(f"Using {info.name} with {info.total_memory_gb:.1f} GB")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base import HardwareBackend
    from .protocols import HardwareBackend as ProtocolBackend

__all__ = [
    "get_backend",
    "list_available_backends",
    "discover_devices",
    "HardwareBackend",
    "BackendType",
    "DeviceInfo",
    "KernelCapabilities",
    "CommunicationBackend",
    "BackendRegistry",
]


def get_backend(backend_type: Optional[BackendType] = None, **kwargs) -> ProtocolBackend:
    """Get hardware backend.
    
    Args:
        backend_type: Backend type (BackendType enum, string name, or None for auto-detect)
        **kwargs: Backend-specific configuration (unused with new protocol)
        
    Returns:
        Hardware backend instance
        
    Examples:
        >>> # New protocol API (recommended)
        >>> backend = get_backend(BackendType.CUDA)
        >>> 
        >>> # Legacy string API (for backward compatibility)
        >>> backend = get_backend("cuda")
        >>> 
        >>> # Auto-detect
        >>> backend = get_backend()
    """
    from .protocols import BackendType
    from .registry import BackendRegistry

    # Handle legacy string-based API - convert to BackendType
    if isinstance(backend_type, str):
        backend_name_upper = backend_type.upper()
        try:
            backend_type = BackendType[backend_name_upper]
        except KeyError:
            available = [bt.name.lower() for bt in BackendRegistry._backends.keys()]
            msg = f"Backend '{backend_type}' not available. Available: {available}"
            raise ValueError(msg) from None

    # New protocol-based API
    if backend_type is None:
        return BackendRegistry.get_default()

    backend = BackendRegistry.get(backend_type)
    if backend is None:
        raise RuntimeError(f"Backend {backend_type.name} not available")
    return backend


def list_available_backends() -> list[str]:
    """List all available (registered) backends.
    
    Returns:
        List of backend names (lowercase strings)
    """
    from .registry import BackendRegistry

    return [backend_type.name.lower() for backend_type in BackendRegistry._backends.keys()]


def discover_devices():
    """Discover all available devices.
    
    Returns:
        Dictionary mapping backend type to device info
    """
    from .registry import BackendRegistry

    return BackendRegistry.discover()


# Re-export base class for type hints (legacy)
# Import all backends to trigger registration
from . import ascend, cambricon, cuda, hygon  # noqa: E402, F401
from .base import HardwareBackend  # noqa: E402

# Re-export new protocols
from .protocols import (  # noqa: E402
    BackendType,
    CommunicationBackend,
    DeviceInfo,
    KernelCapabilities,
)
from .registry import BackendRegistry  # noqa: E402
