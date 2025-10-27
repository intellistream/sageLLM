# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""
HTTP Client Mode Usage Example

This example demonstrates how to use the Control Plane in HTTP client mode
to schedule requests across multiple vLLM instances (local and/or remote).
"""

import asyncio
import logging

from control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    RequestMetadata,
    RequestPriority,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_local_single_machine():
    """
    Example 1: Local single-machine deployment (4 GPUs).
    
    Prerequisites:
        Start 4 vLLM servers on localhost ports 8000-8003:
        
        CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-2-7b --port 8000
        CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-2-7b --port 8001
        CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-2-7b --port 8002
        CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-2-7b --port 8003
    """
    logger.info("=== Example 1: Local Single Machine (4 GPUs) ===")
    
    # Create Control Plane
    cp = ControlPlaneManager(
        scheduling_policy="adaptive",
        routing_strategy="load_balanced",
    )
    
    # Register 4 local GPU instances
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"local-gpu-{gpu_id}",
            host="localhost",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            tensor_parallel_size=1,
            gpu_count=1,
        )
        cp.register_instance(instance)
        logger.info(f"Registered {instance.instance_id}")
    
    # Start Control Plane
    await cp.start()
    logger.info("Control Plane started")
    
    # Submit test requests
    requests = [
        RequestMetadata(
            request_id=f"req-{i}",
            prompt=f"Request {i}: Tell me about AI and machine learning",
            max_tokens=100,
            priority=RequestPriority.NORMAL,
        )
        for i in range(10)
    ]
    
    for request in requests:
        await cp.submit_request(request)
        logger.info(f"Submitted {request.request_id}")
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Get metrics
    metrics = cp.get_metrics()
    logger.info(f"Metrics: {metrics}")
    
    # Stop Control Plane
    await cp.stop()
    logger.info("Control Plane stopped")


async def example_multi_machine():
    """
    Example 2: Multi-machine deployment.
    
    Prerequisites:
        - Machine A (192.168.1.100): 4 GPUs running vLLM on ports 8000-8003
        - Machine B (192.168.1.101): 4 GPUs running vLLM on ports 8000-8003
        - Control Plane can run on any machine (even without GPU)
    """
    logger.info("=== Example 2: Multi-Machine (8 GPUs across 2 machines) ===")
    
    cp = ControlPlaneManager(
        scheduling_policy="cost_optimized",
        routing_strategy="load_balanced",
    )
    
    # Register Machine A's GPUs
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"machine-a-gpu-{gpu_id}",
            host="192.168.1.100",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    # Register Machine B's GPUs
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"machine-b-gpu-{gpu_id}",
            host="192.168.1.101",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    logger.info("Registered 8 GPUs across 2 machines")
    
    await cp.start()
    
    # Submit requests - Control Plane will schedule across all 8 GPUs
    for i in range(20):
        request = RequestMetadata(
            request_id=f"cross-machine-req-{i}",
            prompt=f"Request {i}: Explain quantum computing",
            max_tokens=150,
            priority=RequestPriority.HIGH if i < 5 else RequestPriority.NORMAL,
        )
        await cp.submit_request(request)
    
    await asyncio.sleep(10)
    await cp.stop()


async def example_mixed_deployment():
    """
    Example 3: Mixed deployment (local + remote).
    
    Use local GPUs first (low latency), overflow to remote when busy.
    """
    logger.info("=== Example 3: Mixed Local + Remote ===")
    
    cp = ControlPlaneManager(
        scheduling_policy="slo_aware",
        routing_strategy="load_balanced",
    )
    
    # Local GPUs (prioritized)
    for gpu_id in range(2):
        instance = ExecutionInstance(
            instance_id=f"local-gpu-{gpu_id}",
            host="localhost",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    # Remote backup cluster
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"remote-gpu-{gpu_id}",
            host="192.168.1.200",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    await cp.start()
    
    # High priority requests (will use local first)
    for i in range(5):
        request = RequestMetadata(
            request_id=f"priority-req-{i}",
            prompt=f"Urgent: {i}",
            max_tokens=50,
            priority=RequestPriority.CRITICAL,
            slo_deadline_ms=500,
        )
        await cp.submit_request(request)
    
    await asyncio.sleep(3)
    await cp.stop()


async def example_custom_scheduling():
    """
    Example 4: Custom scheduling policy implementation focus.
    
    The key benefit: You can now focus purely on implementing
    scheduling strategies without worrying about vLLM integration.
    """
    logger.info("=== Example 4: Custom Scheduling Focus ===")
    
    # With HTTP client mode, you can:
    # 1. Implement custom scheduling policies in policies.py
    # 2. Optimize for different metrics (latency, cost, throughput)
    # 3. Not worry about GPU management or vLLM internals
    
    cp = ControlPlaneManager(
        scheduling_policy="adaptive",  # Or your custom policy
    )
    
    # Register instances (transparent whether local or remote)
    instances = [
        ExecutionInstance(
            instance_id=f"gpu-{i}",
            host="localhost" if i < 2 else "192.168.1.100",
            port=8000 + (i % 4),
            model_name="meta-llama/Llama-2-7b",
            gpu_count=1,
        )
        for i in range(6)
    ]
    
    for instance in instances:
        cp.register_instance(instance)
    
    await cp.start()
    
    # Your custom scheduling policy will decide how to distribute these
    requests = [
        RequestMetadata(
            request_id=f"custom-{i}",
            prompt=f"Custom scheduling request {i}",
            max_tokens=100,
            priority=RequestPriority.HIGH if i % 2 == 0 else RequestPriority.LOW,
            slo_deadline_ms=1000 if i < 5 else None,
        )
        for i in range(15)
    ]
    
    for request in requests:
        await cp.submit_request(request)
    
    await asyncio.sleep(5)
    
    # Check how scheduling performed
    all_instances = cp.executor.get_all_instances()
    for instance in all_instances:
        logger.info(
            f"{instance.instance_id}: load={instance.current_load:.2f}, "
            f"active={instance.active_requests}, "
            f"avg_latency={instance.avg_latency_ms:.2f}ms"
        )
    
    await cp.stop()


async def main():
    """Run examples."""
    print("\n" + "="*70)
    print("Control Plane HTTP Client Mode Examples")
    print("="*70 + "\n")
    
    # Uncomment the example you want to run:
    
    # await example_local_single_machine()
    # await example_multi_machine()
    # await example_mixed_deployment()
    await example_custom_scheduling()
    
    print("\n" + "="*70)
    print("Example completed! Check logs above for details.")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
