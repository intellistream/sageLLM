# Copyright (c) 2024 SAGE Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sageLLM inference engine - Integration layer."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from .backends import HardwareBackend, get_backend
from .config import SageLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerateRequest:
    """Generation request."""

    request_id: str
    prompt_tokens: list[int]
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: Optional[list[str]] = None


@dataclass
class GenerateOutput:
    """Generation output."""

    request_id: str
    output_tokens: list[int]
    finish_reason: str  # "length", "stop", "error"
    metrics: Optional[dict[str, float]] = None


class SageLLMEngine:
    """sageLLM inference engine.

    Integrates all modules to provide a unified inference interface.

    Usage:
        config = SageLLMConfig(...)
        engine = SageLLMEngine(config)
        engine.initialize()

        request = GenerateRequest(
            request_id="req_1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=100,
        )

        output = engine.generate(request)
    """

    def __init__(self, config: SageLLMConfig):
        self.config = config

        # Components (lazy initialization)
        self._backend: Optional[HardwareBackend] = None
        self._scheduler: Optional[Any] = None
        self._kv_pool: Optional[Any] = None
        self._kv_cache: Optional[Any] = None
        self._kv_storage: Optional[Any] = None

        # Metrics
        self._throughput_metric: Optional[Any] = None
        self._latency_metric: Optional[Any] = None

        # State
        self._initialized = False
        self._start_time = 0.0
        self._total_tokens = 0
        self._total_requests = 0

    def initialize(self) -> None:
        """Initialize engine."""
        if self._initialized:
            logger.warning("Engine already initialized")
            return

        logger.info(f"Initializing sageLLM engine for {self.config.model.model_id}")

        # 1. Initialize hardware backend
        self._init_backend()

        # 2. Initialize KV cache
        self._init_kv_cache()

        # 3. Initialize scheduler
        self._init_scheduler()

        # 4. Initialize metrics
        if self.config.benchmark.enable_metrics:
            self._init_metrics()

        self._initialized = True
        self._start_time = time.time()
        logger.info("Engine initialization complete")

    def _init_backend(self) -> None:
        """Initialize hardware backend."""
        backend_type = self.config.backend.backend_type
        self._backend = get_backend(backend_type)

        device_info = self._backend.get_device_info()
        logger.info(f"Using backend: {device_info.name}")
        logger.info(f"  Memory: {device_info.total_memory_gb:.1f} GB")

        caps = self._backend.get_capabilities()
        logger.info(f"  FP16: {caps.supports_fp16}, BF16: {caps.supports_bf16}")
        logger.info(f"  Flash Attention: {caps.supports_flash_attention}")

    def _init_kv_cache(self) -> None:
        """Initialize KV cache."""
        kv_config = self.config.kv_cache

        # Try to import KV cache components
        try:
            from .kv_runtime.blocks.multi_granular import (
                KVPoolConfig,
                MultiGranularKVPool,
            )

            pool_config = KVPoolConfig(
                block_size=kv_config.block_size,
                default_granularity=kv_config.granularity,
                enable_sharing=kv_config.enable_cross_request_sharing,
                enable_tiering=kv_config.enable_tiering,
            )
            self._kv_pool = MultiGranularKVPool(pool_config)

            # Try to create cross-request cache
            try:
                from .kv_runtime.reuse.cross_request import CrossRequestKVCache

                self._kv_cache = CrossRequestKVCache(
                    pool=self._kv_pool,
                    enable_tenant_isolation=False,
                )
            except ImportError:
                logger.warning("CrossRequestKVCache not available, using basic pool")
                self._kv_cache = None

            logger.info(f"KV cache initialized: block_size={kv_config.block_size}")

        except ImportError as e:
            logger.warning(f"KV cache modules not available: {e}")
            self._kv_pool = None
            self._kv_cache = None

    def _init_scheduler(self) -> None:
        """Initialize scheduler."""
        sched_config = self.config.scheduler

        try:
            from .runtime.scheduler.pd_scheduler import PDScheduler, PDSchedulerConfig

            scheduler_config = PDSchedulerConfig(
                mode=sched_config.mode,
                max_prefill_batch_size=sched_config.max_prefill_batch,
                max_decode_batch_size=sched_config.max_decode_batch,
            )

            self._scheduler = PDScheduler(scheduler_config)
            logger.info(f"Scheduler initialized: mode={sched_config.mode}")

        except ImportError as e:
            logger.warning(f"Scheduler not available: {e}")
            self._scheduler = None

    def _init_metrics(self) -> None:
        """Initialize metrics."""
        try:
            from .benchmarks.metrics.latency import LatencyMetric
            from .benchmarks.metrics.throughput import ThroughputMetric

            self._throughput_metric = ThroughputMetric()
            self._latency_metric = LatencyMetric()
            logger.info("Metrics initialized")

        except ImportError as e:
            logger.warning(f"Metrics not available: {e}")
            self._throughput_metric = None
            self._latency_metric = None

    def generate(self, request: GenerateRequest) -> GenerateOutput:
        """Synchronous generation.

        Args:
            request: Generation request

        Returns:
            Generation output
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Start timing
        start_time = time.time()

        # 1. Try KV reuse
        reuse_matched = 0
        if self._kv_cache and self.config.kv_cache.enable_prefix_caching:
            try:
                reuse_result = self._kv_cache.try_reuse(
                    request_id=request.request_id,
                    token_ids=request.prompt_tokens,
                )
                if reuse_result.reused:
                    reuse_matched = reuse_result.matched_tokens
                    logger.debug(
                        f"KV reuse: {reuse_matched}/{len(request.prompt_tokens)} tokens"
                    )
            except Exception as e:
                logger.warning(f"KV reuse failed: {e}")

        # 2. Prefill phase (simulated)
        prefill_time = time.time()
        # In real implementation, this would call the model
        # For now, we simulate prefill
        time.sleep(0.001 * len(request.prompt_tokens))  # Simulate processing

        # 3. Decode phase (simulated)
        output_tokens = []
        for i in range(request.max_new_tokens):
            # Simulate token generation
            new_token = 1000 + i  # Placeholder
            output_tokens.append(new_token)

            # Simulate decoding latency
            time.sleep(0.001)

        # 4. Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time

        metrics = None
        if self.config.benchmark.enable_metrics:
            # Calculate throughput
            throughput_tps = len(output_tokens) / total_time if total_time > 0 else 0

            # Calculate latencies
            ttft_ms = (prefill_time - start_time) * 1000
            tpot_ms = (
                (end_time - prefill_time) / len(output_tokens) * 1000
                if output_tokens
                else 0
            )

            metrics = {
                "throughput_tps": throughput_tps,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "total_time_s": total_time,
                "kv_reuse_tokens": reuse_matched,
            }

        # 5. Commit KV for reuse
        if self._kv_cache and self.config.kv_cache.enable_prefix_caching:
            try:
                # In real implementation, would allocate actual KV blocks
                # For now, we simulate by tracking the request
                pass
            except Exception as e:
                logger.warning(f"KV commit failed: {e}")

        # Update stats
        self._total_tokens += len(output_tokens)
        self._total_requests += 1

        return GenerateOutput(
            request_id=request.request_id,
            output_tokens=output_tokens,
            finish_reason="length",
            metrics=metrics,
        )

    async def generate_async(self, request: GenerateRequest) -> GenerateOutput:
        """Asynchronous generation."""
        # Simplified implementation: wrap sync method
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncIterator[int]:
        """Streaming generation."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        # Simplified implementation: yield tokens one by one
        for i in range(request.max_new_tokens):
            yield 1000 + i  # Placeholder
            await asyncio.sleep(0.001)  # Simulate latency

    def get_stats(self) -> dict[str, Any]:
        """Get statistics."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0

        stats = {
            "initialized": self._initialized,
            "backend": (
                self._backend.backend_type.name if self._backend else None
            ),
            "uptime_s": uptime,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "avg_throughput_tps": (
                self._total_tokens / uptime if uptime > 0 else 0
            ),
        }

        if self._kv_pool:
            try:
                stats["kv_pool"] = self._kv_pool.get_stats()
            except Exception:
                pass

        if self._kv_cache:
            try:
                stats["kv_cache"] = self._kv_cache.get_stats()
            except Exception:
                pass

        return stats

    def shutdown(self) -> None:
        """Shutdown engine."""
        logger.info("Shutting down engine")

        # Clean up resources
        if self._backend:
            self._backend.empty_cache()

        self._initialized = False
