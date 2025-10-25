# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Execution coordinator for managing vLLM instances."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Optional

import aiohttp
from vllm.control_plane.types import (ExecutionInstance, PerformanceMetrics,
                                      RequestMetadata, SchedulingDecision)

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinates execution across multiple vLLM instances."""

    def __init__(self, timeout: float = 300.0):
        self.instances: dict[str, ExecutionInstance] = {}
        self.active_requests: dict[str, RequestMetadata] = {}
        self.request_to_instance: dict[str, str] = {}  # request_id -> instance_id

        # HTTP session for vLLM API calls
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Monitoring
        self.metrics = PerformanceMetrics()

    def register_instance(self, instance: ExecutionInstance):
        """Register a new vLLM execution instance."""
        self.instances[instance.instance_id] = instance
        logger.info("Registered instance: %s at %s:%d", instance.instance_id, 
                    instance.host, instance.port)

    async def ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

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
        """Execute request on vLLM instance using actual vLLM OpenAI-compatible API.
        
        Uses the OpenAI-compatible completions endpoint provided by vLLM.
        """
        await self.ensure_session()

        url = f"http://{instance.host}:{instance.port}/v1/completions"

        # Prepare request payload
        payload = {
            "model": request.model_name or "default",
            "prompt": "",  # Would be populated from actual request
            "max_tokens": request.max_tokens or 512,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "request_id": request.request_id,
        }

        logger.info(
            "Executing request %s on %s:%d (TP=%d, PP=%d) via /v1/completions",
            request.request_id,
            instance.host,
            instance.port,
            decision.tensor_parallel_size,
            decision.pipeline_parallel_size,
        )

        start_time = datetime.now()

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"vLLM API returned {response.status}: {error_text}"
                    )

                result = await response.json()

            # Extract completion data from vLLM response
            completion = result.get("choices", [{}])[0]
            # Check if completion was truncated due to length
            _ = completion.get("finish_reason") == "length"
            
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "request_id": request.request_id,
                "instance_id": instance.instance_id,
                "status": "completed",
                "output": completion.get("text", ""),
                "tokens_generated": len(completion.get("text", "").split()),
                "latency_ms": latency_ms,
                "vllm_response": result,
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
                str(e)
            )
            raise

    async def execute_request_streaming(
        self,
        request: RequestMetadata,
        instance: ExecutionInstance,
        decision: SchedulingDecision,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute request with streaming output from vLLM.
        
        Streams tokens as they are generated by vLLM.
        """
        await self.ensure_session()

        url = f"http://{instance.host}:{instance.port}/v1/completions"

        # Prepare streaming request payload
        payload = {
            "model": request.model_name or "default",
            "prompt": "",  # Would be populated from actual request
            "max_tokens": request.max_tokens or 512,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,  # Enable streaming
            "request_id": request.request_id,
        }

        logger.info(
            "Executing streaming request %s on %s:%d",
            request.request_id,
            instance.host,
            instance.port,
        )

        # Mark request as active
        self.active_requests[request.request_id] = request
        self.request_to_instance[request.request_id] = instance.instance_id
        request.start_time = datetime.now()

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"vLLM API returned {response.status}: {error_text}"
                    )

                async for line in response.content:
                    if line:
                        line_str = line.decode().strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix
                            if data_str != "[DONE]":
                                chunk = eval(data_str)  # Parse JSON
                                yield {
                                    "request_id": request.request_id,
                                    "instance_id": instance.instance_id,
                                    "chunk": chunk,
                                    "timestamp": datetime.now(),
                                }

                # Request completed
                request.end_time = datetime.now()
                self._update_metrics(request, instance, success=True)

        except Exception as e:
            logger.error(
                "Streaming failed for request %s: %s",
                request.request_id,
                str(e)
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
        """Perform health check on a vLLM instance via /health endpoint."""

        instance = self.get_instance(instance_id)
        if not instance:
            return False

        await self.ensure_session()

        try:
            url = f"http://{instance.host}:{instance.port}/health"
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with self.session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    instance.is_healthy = True
                    logger.debug("Health check passed for %s", instance_id)
                    return True
                else:
                    instance.is_healthy = False
                    instance.is_available = False
                    logger.warning(
                        "Health check failed for %s: status %d",
                        instance_id,
                        response.status
                    )
                    return False

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
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
