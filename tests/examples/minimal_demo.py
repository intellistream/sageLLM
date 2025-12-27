# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Minimal demo for sageLLM modular architecture.

This example demonstrates:
1. Creating a ControlPlaneManagerLite
2. Registering an engine (mock or real)
3. Basic engine lifecycle management

Run:
    python -m sage.common.components.sage_llm.sageLLM.tests.examples.minimal_demo
"""

from __future__ import annotations

import asyncio
import logging

from sage.common.components.sage_llm.sageLLM.core import (
    ControlPlaneConfig,
    ControlPlaneManagerLite,
    EngineState,
)
from sage.common.components.sage_llm.sageLLM.kv_runtime.types import (
    KVCacheSchema,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_control_plane() -> None:
    """Demonstrate Control Plane Manager basic operations."""
    print("\n" + "=" * 60)
    print("sageLLM Minimal Demo - Control Plane Manager")
    print("=" * 60)

    # Create configuration
    config = ControlPlaneConfig(
        health_check_interval=5.0,
        auto_restart=False,  # Disable for demo
    )

    # Create manager
    manager = ControlPlaneManagerLite(config=config)

    # Register a mock engine
    engine_info = manager.register_engine(
        engine_id="demo-engine-1",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        host="localhost",
        port=8001,
        engine_kind="llm",
        backend_type="lmdeploy",
        metadata={"demo": True},
    )
    print(f"\nRegistered engine: {engine_info.engine_id}")
    print(f"  Model: {engine_info.model_id}")
    print(f"  State: {engine_info.state.value}")

    # Register another engine
    manager.register_engine(
        engine_id="demo-engine-2",
        model_id="BAAI/bge-m3",
        host="localhost",
        port=8090,
        engine_kind="embedding",
        backend_type="tei",
    )

    # Get all engines
    all_engines = manager.get_all_engines()
    print(f"\nTotal registered engines: {len(all_engines)}")
    for eng in all_engines:
        print(f"  - {eng.engine_id} ({eng.engine_kind}): {eng.state.value}")

    # Get stats
    stats = manager.get_stats()
    print("\nControl Plane Stats:")
    print(f"  Total engines: {stats['total_engines']}")
    print(f"  Healthy engines: {stats['healthy_engines']}")

    # Simulate state change
    engine_info.state = EngineState.READY
    healthy = manager.get_healthy_engines()
    print(f"\nHealthy engines after state change: {len(healthy)}")

    # Deregister
    manager.deregister_engine("demo-engine-2")
    print(f"\nAfter deregistration: {len(manager.get_all_engines())} engines")


async def demo_kv_schema() -> None:
    """Demonstrate KV cache schema creation."""
    print("\n" + "=" * 60)
    print("sageLLM Minimal Demo - KV Cache Schema")
    print("=" * 60)

    # Create schema for known model
    schema = KVCacheSchema.for_model("Qwen/Qwen2.5-7B-Instruct")
    print("\nKV Schema for Qwen2.5-7B:")
    print(f"  Layers: {schema.num_layers}")
    print(f"  Heads: {schema.num_heads}")
    print(f"  Head dim: {schema.head_dim}")
    print(f"  Block size: {schema.block_size}")
    print(f"  Page size: {schema.page_size_bytes:,} bytes")
    print(f"  Max seq len: {schema.max_seq_len}")

    # Serialize and deserialize
    data = schema.to_dict()
    restored = KVCacheSchema.from_dict(data)
    print(f"\nSerialization test passed: {restored.num_layers == schema.num_layers}")


async def demo_third_party_status() -> None:
    """Demonstrate third-party module status checking."""
    print("\n" + "=" * 60)
    print("sageLLM Minimal Demo - Third Party Status")
    print("=" * 60)

    from sage.common.components.sage_llm.sageLLM.third_party import (
        get_lmdeploy_version,
        get_patches_status,
        is_lmdeploy_available,
    )

    print(f"\nLMDeploy pinned version: {get_lmdeploy_version()}")
    print(f"LMDeploy available: {is_lmdeploy_available()}")

    patches = get_patches_status()
    if patches:
        print("\nPatch status:")
        for patch, applied in patches.items():
            status = "✓ Applied" if applied else "○ Not applied"
            print(f"  {patch}: {status}")
    else:
        print("\nNo patches found (this is expected before integration)")


async def main() -> None:
    """Run all demos."""
    await demo_control_plane()
    await demo_kv_schema()
    await demo_third_party_status()

    print("\n" + "=" * 60)
    print("Demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
