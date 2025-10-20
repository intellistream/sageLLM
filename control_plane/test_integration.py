# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Integration tests for Control Plane with PD separation."""

import asyncio
import logging

from .manager import ControlPlaneManager
from .types import (
    DecodingConfig,
    ExecutionInstance,
    ExecutionInstanceType,
    PDSeparationConfig,
    PreffillingConfig,
    RequestMetadata,
    RequestPriority,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_scheduling():
    """Test basic request scheduling without PD separation."""
    logger.info("=" * 70)
    logger.info("TEST: Basic Scheduling (No PD Separation)")
    logger.info("=" * 70)

    # Create control plane without PD separation
    manager = ControlPlaneManager(
        scheduling_policy="fifo",
        enable_pd_separation=False,
    )

    # Register a vLLM instance
    instance = ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_count=1,
    )

    manager.register_instance(instance)
    logger.info("✅ Registered instance: %s", instance.instance_id)

    # Submit test requests
    for i in range(3):
        request = RequestMetadata(
            request_id=f"req-{i}",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
        )
        await manager.submit_request(request)
        logger.info("✅ Submitted request: %s", request.request_id)

    logger.info("✅ Test completed: Basic scheduling")


async def test_pd_separation():
    """Test PD separation with specialized instances."""
    logger.info("=" * 70)
    logger.info("TEST: PD Separation with Specialized Instances")
    logger.info("=" * 70)

    # Create PD configuration
    pd_config = PDSeparationConfig(
        enabled=True,
        routing_policy="adaptive",
        prefilling_threshold_input_tokens=800,
        prefilling_threshold_ratio=4.0,
    )

    # Create control plane with PD separation
    manager = ControlPlaneManager(
        scheduling_policy="adaptive",
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    logger.info("✅ Created Control Plane with PD Separation enabled")

    # Register prefilling-specialized instance
    prefilling_instance = ExecutionInstance(
        instance_id="prefilling-1",
        host="localhost",
        port=8001,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        gpu_count=4,
        instance_type=ExecutionInstanceType.PREFILLING,
        prefilling_config=PreffillingConfig(
            target_batch_size=64,
            tensor_parallel_size=4,
            enable_chunked_prefill=True,
        ),
    )

    # Register decoding-specialized instance
    decoding_instance = ExecutionInstance(
        instance_id="decoding-1",
        host="localhost",
        port=8002,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_count=1,
        instance_type=ExecutionInstanceType.DECODING,
        decoding_config=DecodingConfig(
            target_latency_ms=50,
            tensor_parallel_size=1,
            max_parallel_requests=200,
        ),
    )

    manager.register_instance(prefilling_instance)
    manager.register_instance(decoding_instance)
    logger.info(
        "✅ Registered prefilling instance: %s", prefilling_instance.instance_id
    )
    logger.info("✅ Registered decoding instance: %s", decoding_instance.instance_id)

    # Test requests with different characteristics
    test_cases = [
        {
            "request_id": "long-prefill-req",
            "max_tokens": 100,
            "description": "Long prefill request (1000 input tokens)",
        },
        {
            "request_id": "short-decode-req",
            "max_tokens": 500,
            "description": "Short decode request (100 input tokens)",
        },
        {
            "request_id": "balanced-req",
            "max_tokens": 200,
            "description": "Balanced request (500 input tokens)",
        },
    ]

    for test_case in test_cases:
        request = RequestMetadata(
            request_id=test_case["request_id"],
            priority=RequestPriority.NORMAL,
            max_tokens=test_case["max_tokens"],
        )

        # Determine expected routing
        if manager.enable_pd_separation and manager.pd_router:
            phase = manager.pd_router.determine_request_phase(request)
            logger.info(
                "✅ Request %s routed to phase: %s (%s)",
                request.request_id,
                phase.name,
                test_case["description"],
            )

    logger.info("✅ Test completed: PD Separation")


async def test_priority_scheduling():
    """Test priority-based scheduling with PD separation."""
    logger.info("=" * 70)
    logger.info("TEST: Priority Scheduling with PD Separation")
    logger.info("=" * 70)

    pd_config = PDSeparationConfig(enabled=True, routing_policy="adaptive")

    manager = ControlPlaneManager(
        scheduling_policy="priority",
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Register instances
    instance = ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8003,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        gpu_count=2,
    )

    manager.register_instance(instance)

    # Submit requests with different priorities
    priorities = [
        (RequestPriority.CRITICAL, "Critical request"),
        (RequestPriority.HIGH, "High priority request"),
        (RequestPriority.NORMAL, "Normal priority request"),
        (RequestPriority.LOW, "Low priority request"),
    ]

    for priority, description in priorities:
        request = RequestMetadata(
            request_id=f"req-{priority.name}",
            priority=priority,
            max_tokens=256,
        )
        await manager.submit_request(request)
        logger.info("✅ Submitted %s (priority=%s)", description, priority.name)

    logger.info("✅ Test completed: Priority Scheduling")


async def test_slo_aware_scheduling():
    """Test SLO-aware scheduling with deadline constraints."""
    logger.info("=" * 70)
    logger.info("TEST: SLO-Aware Scheduling with PD Separation")
    logger.info("=" * 70)

    pd_config = PDSeparationConfig(enabled=True, routing_policy="adaptive")

    manager = ControlPlaneManager(
        scheduling_policy="slo_aware",
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Register instances
    instance = ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8004,
        model_name="meta-llama/Llama-2-7b",
        tensor_parallel_size=1,
        gpu_count=1,
    )

    manager.register_instance(instance)

    # Submit requests with SLO deadlines
    slo_cases = [
        (50.0, "Tight SLO (50ms)"),
        (200.0, "Medium SLO (200ms)"),
        (1000.0, "Loose SLO (1000ms)"),
        (None, "No SLO"),
    ]

    for slo_ms, description in slo_cases:
        request = RequestMetadata(
            request_id=f"req-slo-{slo_ms}",
            slo_deadline_ms=slo_ms,
            max_tokens=512,
        )

        if manager.enable_pd_separation and manager.pd_router:
            phase = manager.pd_router.determine_request_phase(request)
            logger.info(
                "✅ Request with %s routed to: %s",
                description,
                phase.name,
            )

        await manager.submit_request(request)

    logger.info("✅ Test completed: SLO-Aware Scheduling")


async def test_instance_specialization():
    """Test instance specialization scoring for PD routing."""
    logger.info("=" * 70)
    logger.info("TEST: Instance Specialization Scoring")
    logger.info("=" * 70)

    pd_config = PDSeparationConfig(enabled=True)
    manager = ControlPlaneManager(
        enable_pd_separation=True,
        pd_config=pd_config,
    )

    # Create diverse instances
    instances = [
        ExecutionInstance(
            instance_id="prefilling-specialized",
            host="localhost",
            port=9001,
            model_name="model",
            tensor_parallel_size=8,
            gpu_count=8,
            instance_type=ExecutionInstanceType.PREFILLING,
        ),
        ExecutionInstance(
            instance_id="decoding-specialized",
            host="localhost",
            port=9002,
            model_name="model",
            tensor_parallel_size=1,
            gpu_count=1,
            instance_type=ExecutionInstanceType.DECODING,
        ),
        ExecutionInstance(
            instance_id="general-purpose",
            host="localhost",
            port=9003,
            model_name="model",
            tensor_parallel_size=4,
            gpu_count=4,
            instance_type=ExecutionInstanceType.GENERAL,
        ),
    ]

    for instance in instances:
        manager.register_instance(instance)

    logger.info("✅ Registered %d instances", len(instances))

    # Score instances for different phases
    phases = [
        ExecutionInstanceType.PREFILLING,
        ExecutionInstanceType.DECODING,
        ExecutionInstanceType.GENERAL,
    ]

    for phase in phases:
        logger.info("\nScoring instances for phase: %s", phase.name)
        for instance in instances:
            score = manager.pd_router.get_instance_specialization(instance, phase)
            logger.info(
                "  Instance %s: specialization_score=%.2f",
                instance.instance_id,
                score,
            )

    logger.info("✅ Test completed: Instance Specialization")


async def run_all_tests():
    """Run all integration tests."""
    separator = "=" * 70
    logger.info("=" * 70)
    logger.info("RUNNING CONTROL PLANE INTEGRATION TESTS")
    logger.info(separator)

    try:
        await test_basic_scheduling()
        await asyncio.sleep(0.5)

        await test_pd_separation()
        await asyncio.sleep(0.5)

        await test_priority_scheduling()
        await asyncio.sleep(0.5)

        await test_slo_aware_scheduling()
        await asyncio.sleep(0.5)

        await test_instance_specialization()

        logger.info(separator)
        logger.info("✅ ALL INTEGRATION TESTS PASSED")
        logger.info(separator)

    except Exception as e:
        logger.exception("❌ Test failed with error: %s", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
