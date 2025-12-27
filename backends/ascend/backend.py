# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Huawei Ascend NPU backend implementation."""

import logging
from typing import Any, Optional

import torch

from ..protocols import BackendType, DeviceInfo, HardwareBackend, KernelCapabilities
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.ASCEND)
class AscendBackend(HardwareBackend):
    """Huawei Ascend backend.

    Depends on torch_npu package.
    """

    def __init__(self) -> None:
        self._npu: Optional[Any] = None
        self._available = False
        self._init_npu()

    def _init_npu(self) -> None:
        """Initialize torch_npu."""
        try:
            import torch_npu  # type: ignore

            self._npu = torch_npu
            self._available = torch_npu.npu.is_available()
            if self._available:
                logger.info("Ascend NPU available")
        except ImportError:
            logger.debug("torch_npu not installed")
            self._available = False

    @property
    def backend_type(self) -> BackendType:
        return BackendType.ASCEND

    def is_available(self) -> bool:
        return self._available

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return self._npu.npu.device_count()

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("Ascend NPU not available")

        # Get device properties (API may differ from CUDA)
        try:
            props = self._npu.npu.get_device_properties(device_id)
            name = props.name if hasattr(props, "name") else f"Ascend NPU {device_id}"
            total_memory = (
                props.total_memory / (1024**3) if hasattr(props, "total_memory") else 0
            )
        except Exception:
            name = f"Ascend NPU {device_id}"
            total_memory = 64.0  # Default assume 64GB

        # Get available memory
        try:
            free, total = self._npu.npu.mem_get_info(device_id)
            free_memory = free / (1024**3)
            total_memory = total / (1024**3)
        except Exception:
            free_memory = 0.0

        return DeviceInfo(
            backend=BackendType.ASCEND,
            device_id=device_id,
            name=name,
            compute_capability=None,  # Ascend has no equivalent concept
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            num_cores=0,  # Need to query actual value
            sdk_version=self._get_cann_version(),
        )

    def _get_cann_version(self) -> Optional[str]:
        """Get CANN version."""
        try:
            return self._npu.version.cann
        except Exception:
            return None

    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        # Ascend capabilities vary by model, this shows typical 910B capabilities
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=True,  # 910B supports BF16
            supports_fp8=False,  # Not yet supported
            supports_int8=True,
            supports_int4=False,  # Need verification
            supports_sparse_2_4=False,
            supports_flash_attention=True,  # Supported via CANN
            supports_paged_attention=True,  # vLLM-Ascend supports it
            supports_fused_moe=False,  # Need verification
            supports_nccl=False,
            supports_hccl=True,
        )

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"npu:{device_id}")

    def set_device(self, device_id: int) -> None:
        self._npu.npu.set_device(device_id)

    def current_device(self) -> int:
        return self._npu.npu.current_device()

    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            self._npu.npu.synchronize(device_id)
        else:
            self._npu.npu.synchronize()

    def memory_stats(self, device_id: int = 0) -> dict[str, float]:
        try:
            free, total = self._npu.npu.mem_get_info(device_id)
            return {
                "total_gb": total / (1024**3),
                "used_gb": (total - free) / (1024**3),
                "free_gb": free / (1024**3),
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}

    def empty_cache(self, device_id: Optional[int] = None) -> None:
        self._npu.npu.empty_cache()
