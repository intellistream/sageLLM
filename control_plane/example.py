#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""
Example usage of the Control Plane for sageLLM.

This example demonstrates:
1. Setting up the Control Plane
2. Registering vLLM instances
3. Submitting requests with different priorities
4. Monitoring performance
"""

import asyncio
import random

from . import ControlPlaneManager, ExecutionInstance, RequestMetadata, RequestPriority
from .types import ParallelismType


async def setup_control_plane():
    """Set up Control Plane with multiple vLLM instances."""

    print("=" * 70)
    print("Setting up Control Plane for sageLLM")
    print("=" * 70)

    # Create Control Plane with adaptive scheduling
    cp = ControlPlaneManager(
        scheduling_policy="adaptive",
        routing_strategy="load_balanced",
        enable_monitoring=True,
    )

    # Register multiple vLLM instances with different configurations

    # Instance 1: Pure Tensor Parallelism (4 GPUs)
    instance1 = ExecutionInstance(
        instance_id="vllm-tp4",
        host="localhost",
        port=8000,
        model_name="llama-3-70b",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=100,
        avg_latency_ms=50.0,
        throughput_tokens_per_sec=1000.0,
    )
    cp.register_instance(instance1)
    print(f"✓ Registered {instance1.instance_id} (TP=4)")

    # Instance 2: Hybrid Parallelism (TP=2, PP=2)
    instance2 = ExecutionInstance(
        instance_id="vllm-hybrid",
        host="localhost",
        port=8001,
        model_name="llama-3-70b",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=80,
        avg_latency_ms=60.0,
        throughput_tokens_per_sec=900.0,
    )
    cp.register_instance(instance2)
    print(f"✓ Registered {instance2.instance_id} (TP=2, PP=2)")

    # Instance 3: Data Parallelism (2 replicas)
    instance3 = ExecutionInstance(
        instance_id="vllm-dp2",
        host="localhost",
        port=8002,
        model_name="llama-3-70b",
        tensor_parallel_size=2,
        data_parallel_size=2,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=120,
        avg_latency_ms=55.0,
        throughput_tokens_per_sec=1200.0,
    )
    cp.register_instance(instance3)
    print(f"✓ Registered {instance3.instance_id} (TP=2, DP=2)")

    # Instance 4: Small instance for low-priority requests
    instance4 = ExecutionInstance(
        instance_id="vllm-small",
        host="localhost",
        port=8003,
        model_name="llama-3-8b",
        tensor_parallel_size=1,
        gpu_count=1,
        gpu_memory_gb=24.0,
        max_concurrent_requests=50,
        avg_latency_ms=30.0,
        throughput_tokens_per_sec=500.0,
    )
    cp.register_instance(instance4)
    print(f"✓ Registered {instance4.instance_id} (Single GPU)")

    return cp


async def submit_sample_requests(cp: ControlPlaneManager):
    """Submit sample requests with different priorities and requirements."""

    print("\n" + "=" * 70)
    print("Submitting Sample Requests")
    print("=" * 70)

    # Critical request with tight SLO
    critical_req = RequestMetadata(
        request_id="req-critical-001",
        user_id="premium-user-1",
        priority=RequestPriority.CRITICAL,
        slo_deadline_ms=500,  # 500ms SLO
        max_tokens=50,
        model_name="llama-3-70b",
        tags={"type": "interactive", "service": "chatbot"},
    )
    await cp.submit_request(critical_req)
    print("✓ Submitted CRITICAL request (SLO: 500ms)")

    # High priority request
    high_req = RequestMetadata(
        request_id="req-high-001",
        user_id="user-123",
        priority=RequestPriority.HIGH,
        slo_deadline_ms=1000,  # 1s SLO
        max_tokens=100,
        model_name="llama-3-70b",
        tags={"type": "generation", "customer": "enterprise"},
    )
    await cp.submit_request(high_req)
    print("✓ Submitted HIGH priority request (SLO: 1000ms)")

    # Normal requests with different configurations
    for i in range(5):
        normal_req = RequestMetadata(
            request_id=f"req-normal-{i:03d}",
            user_id=f"user-{random.randint(100, 999)}",
            priority=RequestPriority.NORMAL,
            slo_deadline_ms=2000,  # 2s SLO
            max_tokens=random.randint(50, 200),
            model_name="llama-3-70b",
        )
        await cp.submit_request(normal_req)
    print("✓ Submitted 5 NORMAL priority requests")

    # Low priority batch request
    low_req = RequestMetadata(
        request_id="req-low-001",
        user_id="batch-user",
        priority=RequestPriority.LOW,
        max_tokens=500,
        model_name="llama-3-70b",
        cost_budget=0.05,  # Cost-sensitive
        tags={"type": "batch", "urgency": "low"},
    )
    await cp.submit_request(low_req)
    print("✓ Submitted LOW priority batch request")

    # Background request with parallelism hint
    bg_req = RequestMetadata(
        request_id="req-background-001",
        user_id="analytics",
        priority=RequestPriority.BACKGROUND,
        max_tokens=1000,
        model_name="llama-3-70b",
        parallelism_hint=ParallelismType.DATA_PARALLEL,
        tags={"type": "analytics", "async": True},
    )
    await cp.submit_request(bg_req)
    print("✓ Submitted BACKGROUND request with DP hint")


async def monitor_performance(cp: ControlPlaneManager, duration_sec: int = 5):
    """Monitor Control Plane performance."""

    print("\n" + "=" * 70)
    print("Monitoring Performance")
    print("=" * 70)

    for i in range(duration_sec):
        await asyncio.sleep(1)

        # Get overall metrics
        metrics = cp.get_metrics()
        status = cp.get_status()

        print(f"\n[{i + 1}s] Status:")
        print(f"  Pending: {status['pending_requests']}")
        print(f"  Running: {status['running_requests']}")
        print(f"  Completed: {metrics.completed_requests}")
        print(f"  Failed: {metrics.failed_requests}")
        print(f"  Avg Latency: {metrics.avg_latency_ms:.2f}ms")
        print(f"  SLO Compliance: {metrics.slo_compliance_rate:.2%}")

        # Get instance metrics
        print("\n  Instance Status:")
        for instance in cp.get_instances():
            inst_metrics = cp.get_instance_metrics(instance.instance_id)
            if inst_metrics:
                print(
                    f"    {instance.instance_id}: "
                    f"Load={inst_metrics['current_load']:.2%}, "
                    f"Active={inst_metrics['active_requests']}, "
                    f"Healthy={inst_metrics['is_healthy']}"
                )


async def demonstrate_policy_switching(cp: ControlPlaneManager):
    """Demonstrate dynamic policy switching."""

    print("\n" + "=" * 70)
    print("Demonstrating Policy Switching")
    print("=" * 70)

    policies = ["fifo", "priority", "slo_aware", "cost_optimized", "adaptive"]

    for policy in policies:
        print(f"\n✓ Switching to {policy.upper()} policy")
        cp.update_policy(policy)

        # Submit a test request
        test_req = RequestMetadata(
            request_id=f"req-test-{policy}",
            priority=RequestPriority.NORMAL,
            max_tokens=50,
            model_name="llama-3-70b",
        )
        await cp.submit_request(test_req)

        await asyncio.sleep(0.5)


async def main():
    """Main function demonstrating Control Plane usage."""

    # Setup
    cp = await setup_control_plane()

    # Start Control Plane
    await cp.start()
    print("\n✓ Control Plane started\n")

    try:
        # Submit requests
        await submit_sample_requests(cp)

        # Monitor performance
        await monitor_performance(cp, duration_sec=5)

        # Demonstrate policy switching
        await demonstrate_policy_switching(cp)

        # Final status
        print("\n" + "=" * 70)
        print("Final Status")
        print("=" * 70)

        final_status = cp.get_status()
        print(f"Policy: {final_status['scheduling_policy']}")
        print(f"Routing: {final_status['routing_strategy']}")
        print(f"Total Instances: {final_status['registered_instances']}")
        print(f"Available Instances: {final_status['available_instances']}")

        metrics = cp.get_metrics()
        print(f"\nTotal Requests: {metrics.total_requests}")
        print(f"Completed: {metrics.completed_requests}")
        print(f"Failed: {metrics.failed_requests}")
        print(f"SLO Violations: {metrics.slo_violations}")
        print(f"SLO Compliance Rate: {metrics.slo_compliance_rate:.2%}")

    finally:
        # Stop Control Plane
        await cp.stop()
        print("\n✓ Control Plane stopped")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Control Plane Example for sageLLM")
    print("=" * 70)

    asyncio.run(main())

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")
