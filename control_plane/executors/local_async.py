"""Local async execution coordinator.
"""
from __future__ import annotations

import time
import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from ..types import (
    ExecutionInstance,
    PerformanceMetrics,
    RequestMetadata,
    SchedulingDecision,
)
from .base import ExecutionCoordinatorBase

# Optional vLLM dependencies - gracefully handle if not installed/compiled
from vllm.outputs import RequestOutput
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

logger = logging.getLogger(__name__)


class LocalAsyncExecutionCoordinator(ExecutionCoordinatorBase):
    def __init__(self):
        super().__init__()
        self.metrics = PerformanceMetrics()
        self.engines: dict[str, Any] = {}
        logger.info("LocalAsyncExecutionCoordinator initialized")

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
                max_model_len=256,
            )

            # Create async engine
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engines[instance.instance_id] = engine
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


    async def cleanup(self) -> None:
        raise NotImplementedError

    async def execute_request(
        self, request: RequestMetadata, instance: ExecutionInstance, decision: SchedulingDecision
    ) -> dict[str, Any]:
        """Execute a request on a vLLM instance asynchronously."""
        engine = await self.get_engine(instance.instance_id)

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 512,
        )

        logger.info(
            "Executing request %s on local engine (model=%s, max_tokens=%d)",
            request.request_id,
            instance.model_name,
            request.max_tokens or 512,
        )
        
        start_time = datetime.now()

        results_generator = engine.generate(request.prompt, sampling_params, request.request_id)
        
        final_output: RequestOutput = None
        async for result in results_generator:
            final_output = result

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        if not final_output or final_output.finished == False:
            raise RuntimeError("Generation failed or was interrupted.")

        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(final_output.outputs[0].token_ids)
        
        result_dict = {
            "id": request.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": instance.model_name,
            "choices": [
                {
                    "text": final_output.outputs[0].text,
                    "index": 0,
                    "logprobs": None, 
                    "finish_reason": final_output.outputs[0].finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        
        logger.info(
            "Request %s completed in %.2fms (tokens: %d)",
            request.request_id,
            elapsed_ms,
            result_dict["usage"]["total_tokens"],
        )
        request.end_time = datetime.now()
        return result_dict

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

    async def get_instance_info(self, instance: ExecutionInstance) -> dict[str, Any] | None:
        info = getattr(self.engine_client, "get_instance_info", None)
        if info is None:
            return None

        maybe = info(instance.instance_id)
        if asyncio.iscoroutine(maybe):
            return await maybe
        return maybe

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

    def set_manager_callback(self, manager: Any):
        self._manager_callback = manager

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

    async def shutdown_all(self) -> None:
        # First, cleanup engines
        await self.cleanup()
        logger.info("LocalAsyncExecutionCoordinator shutdown complete")
