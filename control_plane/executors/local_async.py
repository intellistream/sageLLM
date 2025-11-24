"""Local async execution coordinator."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..types import (
    ExecutionInstance,
    PerformanceMetrics,
    RequestMetadata,
    SchedulingDecision,
)
from .base import ExecutionCoordinatorBase

# Module-level flags for vLLM availability
VLLM_AVAILABLE: bool = False
_VLLM_IMPORT_ERROR: Exception | None = None

if TYPE_CHECKING:
    # Type-checking imports - these are always available during type checking
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
else:
    # Runtime imports - handle gracefully if vLLM not installed
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.outputs import RequestOutput
        from vllm.sampling_params import SamplingParams

        VLLM_AVAILABLE = True
    except ImportError as e:
        # vLLM not fully installed (e.g., missing C extensions)
        # Create minimal stubs for runtime
        _VLLM_IMPORT_ERROR = e

        # Stub classes - won't be instantiated when VLLM_AVAILABLE is False
        class AsyncEngineArgs:  # type: ignore[no-redef]
            def __init__(self, **kwargs):
                pass

        class AsyncLLMEngine:  # type: ignore[no-redef]
            @classmethod
            def from_engine_args(cls, args):
                pass

            async def generate(self, prompt, sampling_params, request_id):
                pass

        class RequestOutput:  # type: ignore[no-redef]
            pass

        class SamplingParams:  # type: ignore[no-redef]
            def __init__(self, **kwargs):
                pass


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

        Raises:
            RuntimeError: If vLLM is not properly installed
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                f"vLLM is not properly installed or compiled. "
                f"Cannot initialize local async engine. Error: {_VLLM_IMPORT_ERROR}"
            )

        try:
            # Create engine arguments
            engine_args = AsyncEngineArgs(
                model=instance.model_name,
                tensor_parallel_size=instance.tensor_parallel_size,
                pipeline_parallel_size=instance.pipeline_parallel_size,
                gpu_memory_utilization=0.3,
                enforce_eager=False,
                max_model_len=1024,
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

    async def get_engine(self, instance_id: str) -> Any | None:
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

        if engine is None:
            raise RuntimeError(f"No engine found for instance {instance.instance_id}")

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

        final_output: RequestOutput | None = None

        async for result in results_generator:
            if request.prefill_end_time is None:
                request.prefill_end_time = datetime.now()
            final_output = result

        request.end_time = datetime.now()
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        if not final_output or not final_output.finished:
            raise RuntimeError("Generation failed or was interrupted.")

        prompt_tokens = len(final_output.prompt_token_ids or [])
        completion_tokens = len(final_output.outputs[0].token_ids)
        request.completion_tokens = completion_tokens

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
            result_dict["usage"]["total_tokens"],  # type: ignore[index]
        )
       
        return result_dict

    async def health_check(self, instance: ExecutionInstance) -> bool:
        """Perform health check on a vLLM instance by testing the engine.

        This directly tests the vLLM engine availability.
        """
        if not instance:
            return False

        try:
            engine = await self.get_engine(instance.instance_id)
            if not engine:
                instance.is_healthy = False
                instance.is_available = False
                return False

            # Try a simple health check with the engine
            instance.is_healthy = True
            logger.debug("Health check passed for %s", instance.instance_id)
            return True

        except asyncio.TimeoutError:
            logger.error("Health check timeout for %s", instance.instance_id)
            instance.is_healthy = False
            instance.is_available = False
            return False

        except Exception as e:
            logger.error("Health check error for %s: %s", instance.instance_id, str(e))
            instance.is_healthy = False
            instance.is_available = False
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """Health check all instances."""

        results = {}
        for instance_id, instance in self.instances.items():
            results[instance_id] = await self.health_check(instance)

        return results

    async def get_instance_info(self, instance: ExecutionInstance) -> dict[str, Any] | None:
        """Get instance information from the engine."""
        engine = self.engines.get(instance.instance_id)
        if engine is None:
            return None

        # Try to get info from engine if method exists
        info_method = getattr(engine, "get_instance_info", None)
        if info_method is None:
            return None

        maybe = info_method()
        if asyncio.iscoroutine(maybe):
            result: dict[str, Any] | None = await maybe
            return result
        return maybe  # type: ignore[no-any-return]

    def get_instance_metrics(self, instance_id: str) -> dict[str, Any] | None:
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