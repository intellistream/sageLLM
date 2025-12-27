# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""
Third-Party Engine Backends - Optional inference engine integrations.

This module provides wrappers for third-party inference engines for:
- Baseline comparison in experiments
- Fallback when self-developed runtime is not suitable
- Compatibility with existing deployments

Supported engines:
- lmdeploy: InternLM's LMDeploy/TurboMind engine
- vllm: vLLM with PagedAttention
- vllm_ascend: vLLM fork optimized for Huawei Ascend NPUs

Note: These are OPTIONAL dependencies. The self-developed runtime in
`sageLLM.runtime` is the primary inference path.

Example:
    >>> from sageLLM.third_party_engines import get_engine, list_available_engines
    >>> engines = list_available_engines()
    >>> if "vllm" in engines:
    ...     engine = get_engine("vllm", model="meta-llama/Llama-2-7b")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseEngine

__all__ = [
    "get_engine",
    "list_available_engines",
    "register_engine",
    "BaseEngine",
    "EngineCapability",
    "EngineConfig",
]

_ENGINE_REGISTRY: dict[str, type] = {}


def register_engine(name: str):
    """Decorator to register a third-party engine."""

    def decorator(cls):
        _ENGINE_REGISTRY[name] = cls
        return cls

    return decorator


def get_engine(name: str, **kwargs):
    """
    Get a third-party engine by name.

    Args:
        name: Engine name ("lmdeploy", "vllm", "vllm_ascend")
        **kwargs: Engine-specific configuration

    Returns:
        An initialized engine instance

    Raises:
        ValueError: If engine is not available
    """
    if name not in _ENGINE_REGISTRY:
        available = list_available_engines()
        msg = f"Engine '{name}' not available. Available: {available}"
        raise ValueError(msg)

    engine_cls = _ENGINE_REGISTRY[name]
    return engine_cls(**kwargs)


def list_available_engines() -> list[str]:
    """List all available (registered) engines."""
    return list(_ENGINE_REGISTRY.keys())


# Re-export base class and types
from .base import BaseEngine, EngineCapability  # noqa: E402
from .types import EngineConfig  # noqa: E402

# Auto-register available engines
try:
    from .lmdeploy import LMDeployEngine  # noqa: F401
except ImportError:
    pass

try:
    from .vllm import VLLMEngine  # noqa: F401
except ImportError:
    pass

try:
    from .vllm_ascend import VLLMAscendEngine  # noqa: F401
except ImportError:
    pass
