# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware backend registry.

Provides:
1. Backend registration and discovery
2. Automatic detection of available backends
3. Graceful fallback when hardware unavailable
"""

import logging
from typing import Optional

from .protocols import BackendType, DeviceInfo, HardwareBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Hardware backend registry.

    Provides:
    1. Backend registration and discovery
    2. Automatic detection of available backends
    3. Graceful degradation (fallback)
    """

    _backends: dict[BackendType, type[HardwareBackend]] = {}
    _instances: dict[BackendType, HardwareBackend] = {}
    _default_backend: Optional[BackendType] = None

    @classmethod
    def register(cls, backend_type: BackendType):
        """Decorator: register backend.

        Usage:
            @BackendRegistry.register(BackendType.CUDA)
            class CUDABackend(HardwareBackend):
                ...
        """

        def decorator(backend_cls: type[HardwareBackend]):
            cls._backends[backend_type] = backend_cls
            logger.debug(f"Registered backend: {backend_type.name}")
            return backend_cls

        return decorator

    @classmethod
    def get(cls, backend_type: BackendType) -> Optional[HardwareBackend]:
        """Get backend instance.

        Args:
            backend_type: Backend type

        Returns:
            Backend instance, or None if unavailable
        """
        # Check cache
        if backend_type in cls._instances:
            return cls._instances[backend_type]

        # Create instance
        if backend_type not in cls._backends:
            logger.warning(f"Backend {backend_type.name} not registered")
            return None

        try:
            instance = cls._backends[backend_type]()
            if instance.is_available():
                cls._instances[backend_type] = instance
                return instance
            else:
                logger.info(f"Backend {backend_type.name} not available")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize backend {backend_type.name}: {e}")
            return None

    @classmethod
    def get_default(cls) -> HardwareBackend:
        """Get default backend.

        Priority: CUDA > ASCEND > CAMBRICON > HYGON > CPU
        """
        if cls._default_backend:
            backend = cls.get(cls._default_backend)
            if backend:
                return backend

        # Try by priority
        priority = [
            BackendType.CUDA,
            BackendType.ASCEND,
            BackendType.CAMBRICON,
            BackendType.HYGON,
            BackendType.CPU,
        ]

        for bt in priority:
            backend = cls.get(bt)
            if backend:
                cls._default_backend = bt
                logger.info(f"Using default backend: {bt.name}")
                return backend

        raise RuntimeError("No available hardware backend")

    @classmethod
    def set_default(cls, backend_type: BackendType) -> None:
        """Set default backend."""
        cls._default_backend = backend_type
        logger.info(f"Set default backend to: {backend_type.name}")

    @classmethod
    def list_available(cls) -> list[BackendType]:
        """List all available backends."""
        available = []
        for bt in cls._backends:
            try:
                instance = cls._backends[bt]()
                if instance.is_available():
                    available.append(bt)
            except Exception:
                pass
        return available

    @classmethod
    def discover(cls) -> dict[BackendType, DeviceInfo]:
        """Discover all available devices.

        Returns:
            Mapping from backend type to device info
        """
        devices = {}
        for bt in cls.list_available():
            backend = cls.get(bt)
            if backend and backend.get_device_count() > 0:
                try:
                    devices[bt] = backend.get_device_info(0)
                except Exception as e:
                    logger.warning(f"Failed to get device info for {bt.name}: {e}")
        return devices

    @classmethod
    def reset(cls) -> None:
        """Reset registry (mainly for testing)."""
        cls._instances.clear()
        cls._default_backend = None
        logger.debug("Registry reset")
