# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Core module for sageLLM - streamlined Control Plane coordination.

This module provides the minimal coordination layer for sageLLM, handling:
- Configuration management (pydantic-based)
- Engine instance registration and lifecycle
- Event loop coordination
- Health check orchestration

The core module does NOT contain scheduling strategies, KV management, or
execution logic - those belong in their respective modules (scheduler_ir,
kv_runtime, kv_policy, etc.).

Example:
    >>> from sage.common.components.sage_llm.sageLLM.core import (
    ...     ControlPlaneConfig,
    ...     EngineInfo,
    ...     RequestMetadata,
    ... )
    >>> config = ControlPlaneConfig(health_check_interval=10.0)
"""

from .config import (
    ControlPlaneConfig,
    EngineConfig,
    KVCachePreset,
    SageLLMConfig,
)
from .manager import ControlPlaneManagerLite
from .types import (
    EngineInfo,
    EngineKind,
    EngineState,
    RequestMetadata,
    RequestPriority,
    RequestStatus,
    RequestType,
)

__all__ = [
    # Config
    "ControlPlaneConfig",
    "EngineConfig",
    "KVCachePreset",
    "SageLLMConfig",
    # Manager
    "ControlPlaneManagerLite",
    # Types
    "EngineInfo",
    "EngineKind",
    "EngineState",
    "RequestMetadata",
    "RequestPriority",
    "RequestStatus",
    "RequestType",
]
