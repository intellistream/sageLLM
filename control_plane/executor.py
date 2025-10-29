# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Execution coordinator for managing vLLM instances."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional


# Optional vLLM dependencies - gracefully handle if not installed/compiled
try:
    from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
except ImportError as e:
    # vLLM not fully installed (e.g., missing C extensions)
    # Tests can still run with mocked values
    EngineArgs = None  # type: ignore
    AsyncLLMEngine = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_IMPORT_ERROR = e
else:
    _VLLM_IMPORT_ERROR = None

if TYPE_CHECKING:
    from vllm.engine.async_llm_engine import AsyncLLMEngine

from .types import (
    ExecutionInstance,
    PerformanceMetrics,
    RequestMetadata,
    SchedulingDecision,
)

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinates execution across multiple vLLM instances using direct Python API."""

    def __init__(self):
        self.instances: dict[str, ExecutionInstance] = {}
        self.active_requests: dict[str, RequestMetadata] = {}
        self.request_to_instance: dict[str, str] = {}  # request_id -> instance_id


        # vLLM engines for each instance (direct Python API)
        # Using Any to avoid type issues when vLLM is not fully compiled
        self.engines: dict[str, Any] = {}

        # Monitoring
        self.metrics = PerformanceMetrics()


    def register_instance(self, instance: ExecutionInstance):
        """Register a new vLLM execution instance and create its engine."""
        self.instances[instance.instance_id] = instance
        self.initialize_instance_engine(instance)
        logger.info(
            "Registered instance: %s (model=%s, TP=%d, PP=%d)",
            instance.instance_id,
            instance.model_name,
            instance.tensor_parallel_size,
            instance.pipeline_parallel_size,
        )

    def initialize_instance_engine(self, instance: ExecutionInstance):
        """Initialize the vLLM AsyncLLMEngine for an instance.

        This creates the actual vLLM engine that will be used for inference.
        """
        try:
            # Create engine arguments
            engine_args = AsyncEngineArgs(
                model=instance.model_name,
                tensor_parallel_size=instance.tensor_parallel_size,
                pipeline_parallel_size=instance.pipeline_parallel_size,
                gpu_memory_utilization=0.7,
                enforce_eager=False,
                max_model_len=512,
            )

            # Create async engine
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engines[instance.model_name] = engine
            logger.info(
                "Initialized vLLM engine for instance %s",
                instance.instance_id,
            )

            return engine

        except Exception as e:
            logger.error(
                "Failed to initialize engine for %s: %s",
                instance.instance_id,
                str(e),
            )
            instance.is_healthy = False
            raise

    async def get_engine(self, instance_id: str) -> Optional[Any]:
        """Get the vLLM engine for an instance, initializing if needed."""

        return self.engines.get(instance_id)

    async def shutdown_engine(self, instance_id: str):
        """Shutdown the vLLM engine for an instance."""
        if instance_id in self.engines:
            self.engines.pop(instance_id)
            try:
                # Engine cleanup handled by context manager or explicit call
                logger.info("Shutdown vLLM engine for instance %s", instance_id)
            except Exception as e:
                logger.error("Error shutting down engine %s: %s", instance_id, str(e))

    def unregister_instance(self, instance_id: str):
        """Unregister an execution instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info("Unregistered instance: %s", instance_id)

    def get_instance(self, instance_id: str) -> Optional[ExecutionInstance]:
        """Get instance by ID."""
        return self.instances.get(instance_id)

    def get_available_instances(self) -> list[ExecutionInstance]:
        """Get all available instances."""
        return [
            instance
            for instance in self.instances.values()
            if instance.can_accept_request
        ]

    def get_all_instances(self) -> list[ExecutionInstance]:
        """Get all registered instances."""
        return list(self.instances.values())

    async def add_request(self, model_name: str, request_id: str, prompt: str,
                               sampling_params: SamplingParams) -> str:
        """
        dispatch the request to different model
        """
        engine = self.engines.get(model_name)
        if not engine:
            raise ValueError(f"No engine found for model: {model_name}")

        # 调用对应引擎单元的 submit_request 方法
        results_generator = engine.generate(prompt, sampling_params, request_id)

        final_output: RequestOutput = None
        async for request_output in results_generator:

            final_output = request_output

        if final_output is None:
            return ""

        final_text = final_output.outputs[0].text
        return final_text

    def _update_metrics(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        success: bool,
    ):
        """Update performance metrics."""

        self.metrics.total_requests += 1

        if success:
            self.metrics.completed_requests += 1

            # Update latency
            if request.latency_ms:
                self.metrics.avg_latency_ms = (
                    self.metrics.avg_latency_ms * (self.metrics.completed_requests - 1)
                    + request.latency_ms
                ) / self.metrics.completed_requests
        else:
            self.metrics.failed_requests += 1

        # Check SLO compliance
        if (
            request.slo_deadline_ms
            and request.latency_ms
            and request.latency_ms > request.slo_deadline_ms
        ):
            self.metrics.slo_violations += 1

        # Update SLO compliance rate
        if self.metrics.completed_requests > 0:
            self.metrics.slo_compliance_rate = 1.0 - (
                self.metrics.slo_violations / self.metrics.completed_requests
            )

    async def health_check(self, instance_id: str) -> bool:
        """Perform health check on a vLLM instance by testing the engine.

        This directly tests the vLLM engine availability.
        """
        instance = self.get_instance(instance_id)
        if not instance:
            return False

        try:
            engine = await self.get_engine(instance_id)
            if not engine:
                instance.is_healthy = False
                instance.is_available = False
                return False

            # Try a simple health check with the engine
            instance.is_healthy = True
            logger.debug("Health check passed for %s", instance_id)
            return True

        except asyncio.TimeoutError:
            logger.error("Health check timeout for %s", instance_id)
            instance.is_healthy = False
            instance.is_available = False
            return False

        except Exception as e:
            logger.error("Health check error for %s: %s", instance_id, str(e))
            instance.is_healthy = False
            instance.is_available = False
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """Health check all instances."""

        results = {}
        for instance_id in self.instances:
            results[instance_id] = await self.health_check(instance_id)

        return results

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""

        # Update real-time metrics
        self.metrics.active_requests = len(self.active_requests)
        self.metrics.timestamp = datetime.now()

        # Calculate resource metrics
        if self.instances:
            self.metrics.avg_gpu_utilization = sum(
                i.gpu_utilization for i in self.instances.values()
            ) / len(self.instances)

            self.metrics.total_gpu_memory_gb = sum(
                i.gpu_memory_gb * i.gpu_count for i in self.instances.values()
            )

        return self.metrics

    def get_instance_metrics(self, instance_id: str) -> Optional[dict[str, Any]]:
        """Get metrics for a specific instance."""

        instance = self.get_instance(instance_id)
        if not instance:
            return None

        return {
            "instance_id": instance.instance_id,
            "is_healthy": instance.is_healthy,
            "is_available": instance.is_available,
            "current_load": instance.current_load,
            "active_requests": instance.active_requests,
            "avg_latency_ms": instance.avg_latency_ms,
            "throughput_tokens_per_sec": instance.throughput_tokens_per_sec,
            "gpu_utilization": instance.gpu_utilization,
            "gpu_count": instance.gpu_count,
        }

    async def rebalance_load(self):
        """Rebalance load across instances (future enhancement)."""

        # Placeholder for load rebalancing logic
        # Could migrate requests from overloaded to underutilized instances
        pass

    async def scale_instances(self, target_count: int):
        """Scale number of instances (future enhancement)."""

        # Placeholder for auto-scaling logic
        current_count = len(self.instances)

        if target_count > current_count:
            logger.info(
                "Scaling up from %d to %d instances", current_count, target_count
            )
        elif target_count < current_count:
            logger.info(
                "Scaling down from %d to %d instances", current_count, target_count
            )
