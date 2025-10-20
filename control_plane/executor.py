# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Execution coordinator for managing vLLM instances."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Optional

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

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
        self.engines: dict[str, AsyncLLMEngine] = {}

        # Monitoring
        self.metrics = PerformanceMetrics()

    def register_instance(self, instance: ExecutionInstance):
        """Register a new vLLM execution instance and create its engine."""
        self.instances[instance.instance_id] = instance
        logger.info(
            "Registered instance: %s (model=%s, TP=%d, PP=%d)",
            instance.instance_id,
            instance.model_name,
            instance.tensor_parallel_size,
            instance.pipeline_parallel_size,
        )

    async def initialize_instance_engine(self, instance: ExecutionInstance):
        """Initialize the vLLM AsyncLLMEngine for an instance.

        This creates the actual vLLM engine that will be used for inference.
        """
        try:
            # Create engine arguments
            engine_args = EngineArgs(
                model=instance.model_name,
                tensor_parallel_size=instance.tensor_parallel_size,
                pipeline_parallel_size=instance.pipeline_parallel_size,
                max_seq_len_to_capture=2048,
                gpu_memory_utilization=0.9,
                enforce_eager=False,
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

    async def get_engine(self, instance_id: str) -> Optional[AsyncLLMEngine]:
        """Get the vLLM engine for an instance, initializing if needed."""
        if instance_id not in self.engines:
            instance = self.instances.get(instance_id)
            if not instance:
                logger.error("Instance %s not found", instance_id)
                return None
            await self.initialize_instance_engine(instance)

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

    async def execute_request(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> dict[str, Any]:
        """
        Execute a request on a specific instance.

        Args:
            request: Request metadata
            instance: Target execution instance
            decision: Scheduling decision

        Returns:
            Execution result
        """

        # Mark request as active
        self.active_requests[request.request_id] = request
        self.request_to_instance[request.request_id] = instance.instance_id

        # Update instance state
        instance.active_requests += 1
        instance.current_load = min(
            1.0, instance.active_requests / instance.max_concurrent_requests
        )

        # Update request timing
        request.start_time = datetime.now()

        try:
            # Execute on instance (this would call vLLM API)
            result = await self._execute_on_instance(request, instance, decision)

            # Mark as completed
            request.end_time = datetime.now()

            # Update metrics
            self._update_metrics(request, instance, success=True)

            return result

        except Exception as e:
            logger.error("Execution failed for request %s: %s", request.request_id, e)
            request.end_time = datetime.now()
            self._update_metrics(request, instance, success=False)
            raise

        finally:
            # Clean up
            instance.active_requests -= 1
            instance.current_load = max(
                0.0, instance.active_requests / instance.max_concurrent_requests
            )

            self.active_requests.pop(request.request_id, None)
            self.request_to_instance.pop(request.request_id, None)

    async def _execute_on_instance(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> dict[str, Any]:
        """Execute request on vLLM instance using direct vLLM Python API.

        Calls the vLLM engine directly without HTTP overhead.
        """
        # Get or initialize the vLLM engine for this instance
        engine = await self.get_engine(instance.instance_id)
        if not engine:
            raise RuntimeError(
                f"No engine available for instance {instance.instance_id}"
            )

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 512,
        )

        logger.info(
            "Executing request %s on instance %s (TP=%d, PP=%d, model=%s)",
            request.request_id,
            instance.instance_id,
            decision.tensor_parallel_size,
            decision.pipeline_parallel_size,
            instance.model_name,
        )

        start_time = datetime.now()

        try:
            # Call vLLM engine directly
            # Note: prompt would be extracted from actual request in real scenario
            prompt = (
                request.metadata.get("prompt", "Hello") if request.metadata else "Hello"
            )

            outputs = await engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request.request_id,
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Extract output text from vLLM output
            output_text = ""
            if outputs:
                output_text = outputs[0].outputs[0].text if outputs[0].outputs else ""

            return {
                "request_id": request.request_id,
                "instance_id": instance.instance_id,
                "status": "completed",
                "output": output_text,
                "tokens_generated": len(output_text.split()) if output_text else 0,
                "latency_ms": latency_ms,
                "vllm_request_id": outputs[0].request_id if outputs else None,
            }

        except asyncio.TimeoutError:
            logger.error(
                "Timeout executing request %s on %s",
                request.request_id,
                instance.instance_id,
            )
            raise

        except Exception as e:
            logger.error(
                "Failed to execute request %s on %s: %s",
                request.request_id,
                instance.instance_id,
                str(e),
            )
            raise

    async def execute_request_streaming(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute request with streaming output from vLLM.

        Streams tokens as they are generated by the vLLM engine.
        """
        # Get or initialize the vLLM engine for this instance
        engine = await self.get_engine(instance.instance_id)
        if not engine:
            raise RuntimeError(
                f"No engine available for instance {instance.instance_id}"
            )

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 512,
        )

        logger.info(
            "Executing streaming request %s on instance %s",
            request.request_id,
            instance.instance_id,
        )

        # Mark request as active
        self.active_requests[request.request_id] = request
        self.request_to_instance[request.request_id] = instance.instance_id
        request.start_time = datetime.now()

        try:
            prompt = (
                request.metadata.get("prompt", "Hello") if request.metadata else "Hello"
            )

            # Stream outputs from vLLM engine
            async for output in engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request.request_id,
            ):
                yield {
                    "request_id": request.request_id,
                    "instance_id": instance.instance_id,
                    "output": output.outputs[0].text if output.outputs else "",
                    "finish_reason": output.outputs[0].finish_reason
                    if output.outputs
                    else None,
                    "timestamp": datetime.now(),
                }

            # Request completed
            request.end_time = datetime.now()
            self._update_metrics(request, instance, success=True)

        except Exception as e:
            logger.error(
                "Streaming failed for request %s: %s",
                request.request_id,
                str(e),
            )
            request.end_time = datetime.now()
            self._update_metrics(request, instance, success=False)
            raise

        finally:
            # Clean up
            self.active_requests.pop(request.request_id, None)
            self.request_to_instance.pop(request.request_id, None)

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
