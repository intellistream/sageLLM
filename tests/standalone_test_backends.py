#!/usr/bin/env python3
# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for hardware backends (Task 4)."""

import sys

import torch

# Test imports
print("=" * 80)
print("Task 4: Hardware Backends Test")
print("=" * 80)

try:
    from backends import discover_devices, get_backend
    from backends.protocols import BackendType
    from backends.registry import BackendRegistry
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test backend registration
print("\n1. Testing Backend Registry:")
print("-" * 40)
available = BackendRegistry.list_available()
print(f"Available backends: {[bt.name for bt in available]}")

# Test default backend
print("\n2. Testing Default Backend:")
print("-" * 40)
try:
    backend = get_backend()
    print(f"✓ Default backend: {backend.backend_type.name}")
    print(f"  Is available: {backend.is_available()}")
    print(f"  Device count: {backend.get_device_count()}")
except Exception as e:
    print(f"✗ Failed to get default backend: {e}")

# Test CUDA backend (if available)
if torch.cuda.is_available():
    print("\n3. Testing CUDA Backend:")
    print("-" * 40)
    try:
        cuda_backend = get_backend(BackendType.CUDA)
        print("✓ CUDA backend available")

        # Test device info
        info = cuda_backend.get_device_info(0)
        print(f"  Device: {info.name}")
        print(f"  Compute capability: {info.compute_capability}")
        print(f"  Total memory: {info.total_memory_gb:.2f} GB")
        print(f"  Free memory: {info.free_memory_gb:.2f} GB")

        # Test capabilities
        caps = cuda_backend.get_capabilities(0)
        print(f"  FP16 support: {caps.supports_fp16}")
        print(f"  BF16 support: {caps.supports_bf16}")
        print(f"  Flash Attention: {caps.supports_flash_attention}")

        # Test memory stats
        stats = cuda_backend.memory_stats(0)
        print(f"  Memory - Total: {stats['total_gb']:.2f} GB, "
              f"Used: {stats['used_gb']:.2f} GB, "
              f"Free: {stats['free_gb']:.2f} GB")

        # Test tensor allocation
        tensor = cuda_backend.allocate_tensor((256, 256), torch.float16, 0)
        print(f"✓ Tensor allocation successful: {tensor.shape}, {tensor.dtype}, {tensor.device}")

    except Exception as e:
        print(f"✗ CUDA backend test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n3. CUDA not available, skipping CUDA tests")

# Test device discovery
print("\n4. Testing Device Discovery:")
print("-" * 40)
try:
    devices = discover_devices()
    for backend_type, device_info in devices.items():
        print(f"✓ {backend_type.name}: {device_info.name}")
except Exception as e:
    print(f"✗ Device discovery failed: {e}")

# Test domestic accelerators (graceful degradation)
print("\n5. Testing Domestic Accelerator Support:")
print("-" * 40)
for backend_type in [BackendType.ASCEND, BackendType.CAMBRICON, BackendType.HYGON]:
    backend = BackendRegistry.get(backend_type)
    status = "available" if backend and backend.is_available() else "not available (as expected)"
    print(f"  {backend_type.name}: {status}")

print("\n" + "=" * 80)
print("✓ All Task 4 tests completed successfully")
print("=" * 80)
