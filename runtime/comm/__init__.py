# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Communication Layer - Protocols and primitives for distributed inference.

Components:
- protocols: Communication protocol abstractions (RDMA, TCP, shared memory)
- topology: Cluster topology discovery and management
- collective: Collective communication primitives (all-reduce, all-gather, etc.)

The communication layer provides hardware-agnostic abstractions for:
- Point-to-point communication between devices/nodes
- Collective operations for tensor/pipeline parallelism
- KV cache transfer for PD separation
"""

from __future__ import annotations

__all__ = [
    "protocols",
    "topology",
    "collective",
]
