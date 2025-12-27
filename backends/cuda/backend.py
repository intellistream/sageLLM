# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA CUDA backend implementation."""

import logging
from typing import Optional

import torch

from ..protocols import BackendType, DeviceInfo, HardwareBackend, KernelCapabilities
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.CUDA)
class CUDABackend(HardwareBackend):
    """NVIDIA CUDA backend."""

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return torch.cuda.device_count()

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("CUDA not available")

        props = torch.cuda.get_device_properties(device_id)

        # Get compute capability
        compute_capability = f"{props.major}.{props.minor}"

        # Get memory info
        total_memory = props.total_memory / (1024**3)
        free_memory = torch.cuda.mem_get_info(device_id)[0] / (1024**3)

        return DeviceInfo(
            backend=BackendType.CUDA,
            device_id=device_id,
            name=props.name,
            compute_capability=compute_capability,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            num_cores=props.multi_processor_count,
            driver_version=torch.version.cuda,
            properties={
                "warp_size": getattr(props, "warp_size", 32),
            },
        )

    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        info = self.get_device_info(device_id)
        major, minor = map(int, info.compute_capability.split("."))

        # Determine supported features based on compute capability
        supports_bf16 = major >= 8  # Ampere+
        supports_fp8 = major >= 9  # Hopper+
        supports_sparse_2_4 = major >= 8  # Ampere+
        supports_flash_attention = major >= 8

        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=supports_bf16,
            supports_fp8=supports_fp8,
            supports_int8=True,
            supports_int4=True,
            supports_sparse_2_4=supports_sparse_2_4,
            supports_flash_attention=supports_flash_attention,
            supports_paged_attention=True,
            supports_fused_moe=True,
            supports_nccl=True,
            supports_hccl=False,
        )

    def get_device(self, device_id: int = 0) -> torch.device:
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
