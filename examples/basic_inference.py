#!/usr/bin/env python3
# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sageLLM basic inference example.

Demonstrates how to use the sageLLM engine for inference.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from sageLLM import (
    BenchmarkConfig,
    GenerateRequest,
    KVCacheConfig,
    ModelConfig,
    SageLLMConfig,
    SageLLMEngine,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Run basic inference example."""
    print("=" * 80)
    print("sageLLM Basic Inference Example")
    print("=" * 80)

    # 1. Create configuration
    print("\n1. Creating configuration...")
    config = SageLLMConfig(
        model=ModelConfig(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            num_layers=32,
            num_heads=32,
            hidden_size=4096,
        ),
        kv_cache=KVCacheConfig(
            max_tokens=65536,
            block_size=16,
            enable_prefix_caching=True,
            enable_cross_request_sharing=True,
        ),
        benchmark=BenchmarkConfig(
            enable_metrics=True,
        ),
    )
    print(f"  Model: {config.model.model_id}")
    print(f"  KV cache: {config.kv_cache.max_tokens} tokens")
    print(f"  Block size: {config.kv_cache.block_size}")

    # 2. Initialize engine
    print("\n2. Initializing engine...")
    engine = SageLLMEngine(config)
    engine.initialize()

    print("\nEngine initialized:")
    stats = engine.get_stats()
    print(f"  Backend: {stats['backend']}")
    print(f"  Initialized: {stats['initialized']}")

    # 3. First request
    print("\n3. Sending first request...")
    request1 = GenerateRequest(
        request_id="test_001",
        prompt_tokens=[1, 2, 3, 4, 5],  # In practice, use actual tokenized input
        max_new_tokens=50,
        temperature=1.0,
    )

    output1 = engine.generate(request1)

    print("\nGeneration result:")
    print(f"  Request ID: {output1.request_id}")
    print(f"  Output tokens: {len(output1.output_tokens)}")
    print(f"  Finish reason: {output1.finish_reason}")

    if output1.metrics:
        print("  Metrics:")
        print(f"    Throughput: {output1.metrics['throughput_tps']:.1f} tokens/s")
        print(f"    TTFT: {output1.metrics['ttft_ms']:.2f} ms")
        print(f"    TPOT: {output1.metrics['tpot_ms']:.2f} ms")
        print(f"    Total time: {output1.metrics['total_time_s']:.3f} s")

    # 4. Test KV reuse
    print("\n4. Testing KV reuse with similar prompt...")

    # Request with same prefix
    request2 = GenerateRequest(
        request_id="test_002",
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7],  # Contains same prefix
        max_new_tokens=30,
    )

    output2 = engine.generate(request2)
    print("\nSecond request completed:")
    print(f"  Request ID: {output2.request_id}")
    print(f"  Output tokens: {len(output2.output_tokens)}")

    if output2.metrics:
        print(f"  KV reuse: {output2.metrics.get('kv_reuse_tokens', 0)} tokens reused")
        print(f"  Throughput: {output2.metrics['throughput_tps']:.1f} tokens/s")

    # 5. Multiple requests to test throughput
    print("\n5. Running multiple requests...")
    for i in range(3):
        request = GenerateRequest(
            request_id=f"batch_{i}",
            prompt_tokens=[i + 1, i + 2, i + 3],
            max_new_tokens=20,
        )
        output = engine.generate(request)
        print(f"  Request {i+1}: {len(output.output_tokens)} tokens generated")

    # 6. Show final statistics
    print("\n6. Final engine statistics:")
    final_stats = engine.get_stats()
    print(f"  Total requests: {final_stats['total_requests']}")
    print(f"  Total tokens: {final_stats['total_tokens']}")
    print(f"  Avg throughput: {final_stats['avg_throughput_tps']:.1f} tokens/s")
    print(f"  Uptime: {final_stats['uptime_s']:.2f} s")

    # 7. Shutdown
    print("\n7. Shutting down engine...")
    engine.shutdown()

    print("\n" + "=" * 80)
    print("✓ Example completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
