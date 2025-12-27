# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hygon DCU backend implementation."""

import logging
from typing import Optional

import torch

from ..protocols import BackendType, DeviceInfo, HardwareBackend, KernelCapabilities
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.HYGON)
class HygonBackend(HardwareBackend):
    """Hygon DCU backend.

    Based on ROCm/HIP, similar API to AMD GPUs.
    Uses PyTorch's ROCm support.
    """

    def __init__(self) -> None:
        self._available = False
        self._init_dcu()

    def _init_dcu(self) -> None:
        """Initialize Hygon DCU."""
        # Hygon DCU uses ROCm, accessed via torch.cuda (if ROCm version)
        # or may have dedicated torch_dcu
        try:
            # Check if this is ROCm version of PyTorch
            if torch.version.hip is not None:
                # ROCm version, check for Hygon devices
                self._available = torch.cuda.is_available()
                if self._available:
                    # Further check if it's Hygon device
                    device_name = torch.cuda.get_device_name(0)
                    if "Hygon" in device_name or "DCU" in device_name:
                        logger.info(f"Hygon DCU available: {device_name}")
                    else:
                        # May be other ROCm devices (e.g., AMD)
                        self._available = False
            else:
                # Try importing dedicated torch_dcu
                try:
                    import torch_dcu  # type: ignore

                    self._available = torch_dcu.dcu.is_available()
                except ImportError:
                    self._available = False
        except Exception:
            self._available = False

    @property
    def backend_type(self) -> BackendType:
        return BackendType.HYGON

    def is_available(self) -> bool:
        return self._available

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return torch.cuda.device_count()  # ROCm uses cuda API

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("Hygon DCU not available")

        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / (1024**3)
        free, _ = torch.cuda.mem_get_info(device_id)
        free_memory = free / (1024**3)

        return DeviceInfo(
            backend=BackendType.HYGON,
            device_id=device_id,
            name=props.name,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            num_cores=props.multi_processor_count,
            sdk_version=torch.version.hip,
        )

    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        # Hygon DCU capabilities (based on ROCm)
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=True,  # Newer versions support it
            supports_fp8=False,
            supports_int8=True,
            supports_int4=False,
            supports_sparse_2_4=False,
            supports_flash_attention=True,  # ROCm has Flash Attention
            supports_paged_attention=True,  # vLLM supports ROCm
            supports_fused_moe=False,
            supports_nccl=True,  # ROCm NCCL (RCCL)
            supports_hccl=False,
        )

    def get_device(self, device_id: int = 0) -> torch.device:
        # ROCm uses cuda device type
        return torch.device(f"cuda:{device_id}")

    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()

    def memory_stats(self, device_id: int = 0) -> dict[str, float]:
        free, total = torch.cuda.mem_get_info(device_id)
        return {
            "total_gb": total / (1024**3),
            "used_gb": (total - free) / (1024**3),
            "free_gb": free / (1024**3),
        }

    def empty_cache(self, device_id: Optional[int] = None) -> None:
        torch.cuda.empty_cache()
