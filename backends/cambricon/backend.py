# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cambricon MLU backend implementation."""

import logging
from typing import Any, Optional

import torch

from ..protocols import BackendType, DeviceInfo, HardwareBackend, KernelCapabilities
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.CAMBRICON)
class CambriconBackend(HardwareBackend):
    """Cambricon MLU backend.

    Depends on torch_mlu (catch) package.
    """

    def __init__(self) -> None:
        self._mlu: Optional[Any] = None
        self._available = False
        self._init_mlu()

    def _init_mlu(self) -> None:
        """Initialize torch_mlu."""
        try:
            import torch_mlu  # type: ignore

            self._mlu = torch_mlu
            self._available = torch_mlu.mlu.is_available()
            if self._available:
                logger.info("Cambricon MLU available")
        except ImportError:
            logger.debug("torch_mlu not installed")
            self._available = False

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CAMBRICON

    def is_available(self) -> bool:
        return self._available

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return self._mlu.mlu.device_count()

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("Cambricon MLU not available")

        try:
            name = self._mlu.mlu.get_device_name(device_id)
        except Exception:
            name = f"MLU {device_id}"

        try:
            props = self._mlu.mlu.get_device_properties(device_id)
            total_memory = props.total_memory / (1024**3)
        except Exception:
            total_memory = 32.0  # Default assume 32GB

        return DeviceInfo(
            backend=BackendType.CAMBRICON,
            device_id=device_id,
            name=name,
            total_memory_gb=total_memory,
            free_memory_gb=0,  # Need to query
        )

    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        # MLU590 typical capabilities
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=False,  # Need verification
            supports_fp8=False,
            supports_int8=True,
            supports_int4=False,
            supports_sparse_2_4=False,
            supports_flash_attention=False,  # Need verification
            supports_paged_attention=False,
            supports_fused_moe=False,
            supports_nccl=False,
            supports_hccl=False,
        )

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"mlu:{device_id}")

    def set_device(self, device_id: int) -> None:
        self._mlu.mlu.set_device(device_id)

    def current_device(self) -> int:
        return self._mlu.mlu.current_device()

    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            self._mlu.mlu.synchronize(device_id)
        else:
            self._mlu.mlu.synchronize()

    def memory_stats(self, device_id: int = 0) -> dict[str, float]:
        try:
            # API may differ
            allocated = self._mlu.mlu.memory_allocated(device_id)
            reserved = self._mlu.mlu.memory_reserved(device_id)
            total = self.get_device_info(device_id).total_memory_gb * (1024**3)
            return {
                "total_gb": total / (1024**3),
                "used_gb": allocated / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}

    def empty_cache(self, device_id: Optional[int] = None) -> None:
        self._mlu.mlu.empty_cache()
