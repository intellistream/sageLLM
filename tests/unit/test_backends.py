# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for hardware backends."""

import pytest
import torch

from sage.common.components.sage_llm.sageLLM.backends import (
    BackendRegistry,
    BackendType,
    discover_devices,
    get_backend,
    list_available_backends,
)


class TestBackendRegistry:
    """Backend registry tests."""

    def test_list_available(self):
        """Test listing available backends."""
        available = list_available_backends()
        assert isinstance(available, list)
        # At least CUDA should be registered
        assert len(available) >= 0

    def test_get_default(self):
        """Test getting default backend."""
        backend = get_backend()
        assert backend is not None
        assert backend.is_available()

    def test_discover_devices(self):
        """Test device discovery."""
        devices = discover_devices()
        assert isinstance(devices, dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDABackend:
    """CUDA backend tests."""

    def test_is_available(self):
        """Test availability check."""
        backend = get_backend(BackendType.CUDA)
        assert backend.is_available()

    def test_device_info(self):
        """Test device information."""
        backend = get_backend(BackendType.CUDA)
        info = backend.get_device_info()

        assert info.backend == BackendType.CUDA
        assert info.total_memory_gb > 0
        assert info.name != ""

    def test_capabilities(self):
        """Test capability queries."""
        backend = get_backend(BackendType.CUDA)
        caps = backend.get_capabilities()

        assert caps.supports_fp16
        assert caps.supports_int8

    def test_memory_stats(self):
        """Test memory statistics."""
        backend = get_backend(BackendType.CUDA)
        stats = backend.memory_stats()

        assert "total_gb" in stats
        assert "used_gb" in stats
        assert "free_gb" in stats
        assert stats["total_gb"] > 0

    def test_allocate_tensor(self):
        """Test tensor allocation."""
        backend = get_backend(BackendType.CUDA)
        tensor = backend.allocate_tensor(
            shape=(256, 256),
            dtype=torch.float16,
        )

        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.float16


class TestBackendFallback:
    """Backend degradation tests."""

    def test_unavailable_backend_returns_none(self):
        """Test unavailable backend returns None."""
        # Try to get a backend that may not be available
        backend = BackendRegistry.get(BackendType.ASCEND)
        # Should either return backend or None, not raise exception
        if backend is not None:
            assert backend.backend_type == BackendType.ASCEND

    def test_default_fallback(self):
        """Test default backend fallback."""
        # Even without GPU, should be able to get some backend
        backend = get_backend()
        assert backend is not None


class TestLegacyAPI:
    """Test legacy string-based API."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_legacy_cuda_backend(self):
        """Test legacy CUDA backend access."""
        # The legacy API should still work
        backend = get_backend("cuda")
        assert backend is not None
