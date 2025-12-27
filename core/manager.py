# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Streamlined Control Plane Manager for sageLLM.

This module provides a lightweight coordination layer for sageLLM,
handling engine registration, health checks, and basic lifecycle management.

Scheduling strategies, KV management, and execution logic have been
moved to their respective modules (scheduler_ir, kv_runtime, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

import aiohttp

from .config import ControlPlaneConfig
from .types import EngineInfo, EngineState

if TYPE_CHECKING:
    from ..engines.base import BaseEngine

logger = logging.getLogger(__name__)


class ControlPlaneManagerLite:
    """Lightweight Control Plane Manager for sageLLM.

    This manager provides core coordination functionality:
    - Engine registration and deregistration
    - Health check monitoring
    - Basic lifecycle management

    Scheduling strategies and request routing are delegated to the
    scheduler_ir module for flexibility.

    Attributes:
        config: Control plane configuration.
        engines: Dictionary of registered engines.
    """

    def __init__(self, config: ControlPlaneConfig | None = None) -> None:
        """Initialize the Control Plane Manager.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ControlPlaneConfig.default()

        # Engine registry
        self._engines: dict[str, EngineInfo] = {}
        self._engine_instances: dict[str, BaseEngine] = {}
        self._registry_lock = threading.Lock()

        # Health check state
        self._health_check_task: asyncio.Task[None] | None = None
        self._restart_state: dict[str, dict[str, Any]] = {}

        # Running state
        self._is_running = False
        self._event_loop: asyncio.AbstractEventLoop | None = None

    # =========================================================================
    # Engine Registration
    # =========================================================================

    def register_engine(
        self,
        engine_id: str,
        model_id: str,
        host: str,
        port: int,
        engine_kind: str = "llm",
        backend_type: str = "lmdeploy",
        metadata: dict[str, Any] | None = None,
    ) -> EngineInfo:
        """Register an engine with the Control Plane.

        Args:
            engine_id: Unique identifier for the engine.
            model_id: Model loaded on the engine.
            host: Engine host address.
            port: Engine port number.
            engine_kind: Type of engine (llm, embedding, hybrid).
            backend_type: Backend implementation.
            metadata: Additional engine metadata.

        Returns:
            EngineInfo for the registered engine.

        Raises:
            ValueError: If engine_id is already registered.
        """
        with self._registry_lock:
            if engine_id in self._engines:
                raise ValueError(f"Engine {engine_id} is already registered")

            engine_info = EngineInfo(
                engine_id=engine_id,
                model_id=model_id,
                host=host,
                port=port,
                engine_kind=engine_kind,
                backend_type=backend_type,
                metadata=metadata or {},
            )
            self._engines[engine_id] = engine_info
            self._restart_state[engine_id] = {
                "consecutive_failures": 0,
                "restart_count": 0,
                "last_restart_time": None,
            }

            logger.info(
                "Registered engine %s (%s on %s:%d)",
                engine_id,
                model_id,
                host,
                port,
            )
            return engine_info

    def register_engine_instance(
        self,
        engine_id: str,
        engine: BaseEngine,
    ) -> None:
        """Register an engine instance for direct management.

        Args:
            engine_id: Engine identifier.
            engine: Engine instance.
        """
        with self._registry_lock:
            self._engine_instances[engine_id] = engine

    def deregister_engine(self, engine_id: str) -> bool:
        """Deregister an engine from the Control Plane.

        Args:
            engine_id: Engine identifier to remove.

        Returns:
            True if engine was found and removed.
        """
        with self._registry_lock:
            if engine_id in self._engines:
                del self._engines[engine_id]
                self._engine_instances.pop(engine_id, None)
                self._restart_state.pop(engine_id, None)
                logger.info("Deregistered engine %s", engine_id)
                return True
            return False

    def get_engine(self, engine_id: str) -> EngineInfo | None:
        """Get engine info by ID.

        Args:
            engine_id: Engine identifier.

        Returns:
            EngineInfo if found, None otherwise.
        """
        return self._engines.get(engine_id)

    def get_all_engines(self) -> list[EngineInfo]:
        """Get all registered engines.

        Returns:
            List of all EngineInfo objects.
        """
        return list(self._engines.values())

    def get_healthy_engines(self, engine_kind: str | None = None) -> list[EngineInfo]:
        """Get all healthy engines, optionally filtered by kind.

        Args:
            engine_kind: Optional filter by engine kind.

        Returns:
            List of healthy EngineInfo objects.
        """
        engines = [e for e in self._engines.values() if e.is_healthy]
        if engine_kind:
            engines = [e for e in engines if e.engine_kind == engine_kind]
        return engines

    # =========================================================================
    # Health Checking
    # =========================================================================

    async def _check_engine_health(self, engine_info: EngineInfo) -> bool:
        """Check health of a single engine.

        Args:
            engine_info: Engine to check.

        Returns:
            True if engine is healthy.
        """
        url = f"http://{engine_info.host}:{engine_info.port}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(
                        total=self.config.health_check_timeout
                    ),
                ) as response:
                    if response.status == 200:
                        return True
        except Exception as e:
            logger.debug(
                "Health check failed for %s: %s",
                engine_info.engine_id,
                e,
            )
        return False

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._is_running:
            for engine_id, engine_info in list(self._engines.items()):
                if engine_info.state == EngineState.STOPPED:
                    continue

                is_healthy = await self._check_engine_health(engine_info)
                restart_state = self._restart_state.get(engine_id, {})

                if is_healthy:
                    # Update healthy state
                    engine_info.last_heartbeat = datetime.now()
                    engine_info.consecutive_failures = 0
                    restart_state["consecutive_failures"] = 0

                    if engine_info.state == EngineState.STARTING:
                        engine_info.state = EngineState.READY
                        logger.info("Engine %s is now READY", engine_id)
                    elif engine_info.state == EngineState.ERROR:
                        engine_info.state = EngineState.READY
                        logger.info("Engine %s recovered from ERROR", engine_id)
                else:
                    # Update failure state
                    engine_info.consecutive_failures += 1
                    restart_state["consecutive_failures"] = (
                        restart_state.get("consecutive_failures", 0) + 1
                    )

                    if (
                        engine_info.consecutive_failures
                        >= self.config.consecutive_failures_threshold
                    ):
                        if engine_info.state != EngineState.ERROR:
                            engine_info.state = EngineState.ERROR
                            logger.warning(
                                "Engine %s marked as ERROR after %d failures",
                                engine_id,
                                engine_info.consecutive_failures,
                            )

                        # Attempt restart if configured
                        if self.config.auto_restart:
                            await self._attempt_restart(engine_id)

            await asyncio.sleep(self.config.health_check_interval)

    async def _attempt_restart(self, engine_id: str) -> bool:
        """Attempt to restart a failed engine.

        Args:
            engine_id: Engine to restart.

        Returns:
            True if restart was initiated.
        """
        restart_state = self._restart_state.get(engine_id, {})
        restart_count = restart_state.get("restart_count", 0)

        if restart_count >= self.config.max_restart_attempts:
            logger.error(
                "Engine %s exceeded max restart attempts (%d)",
                engine_id,
                self.config.max_restart_attempts,
            )
            return False

        # Check if we have an engine instance to restart
        engine = self._engine_instances.get(engine_id)
        if engine is None:
            logger.warning(
                "No engine instance available for %s, cannot restart",
                engine_id,
            )
            return False

        # Calculate backoff
        backoff = self.config.restart_backoff_base * (2**restart_count)
        last_restart = restart_state.get("last_restart_time")
        if last_restart:
            elapsed = (datetime.now() - last_restart).total_seconds()
            if elapsed < backoff:
                return False

        logger.info(
            "Attempting restart %d/%d for engine %s",
            restart_count + 1,
            self.config.max_restart_attempts,
            engine_id,
        )

        try:
            # Stop and restart
            await engine.stop()
            await engine.start()

            restart_state["restart_count"] = restart_count + 1
            restart_state["last_restart_time"] = datetime.now()

            # Update engine state
            engine_info = self._engines.get(engine_id)
            if engine_info:
                engine_info.state = EngineState.STARTING
                engine_info.consecutive_failures = 0

            return True
        except Exception as e:
            logger.error("Failed to restart engine %s: %s", engine_id, e)
            return False

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start the Control Plane Manager.

        Begins health check monitoring and event processing.
        """
        if self._is_running:
            return

        self._is_running = True
        self._event_loop = asyncio.get_event_loop()

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Control Plane Manager started")

    async def stop(self) -> None:
        """Stop the Control Plane Manager.

        Stops health checking and cleans up resources.
        """
        if not self._is_running:
            return

        self._is_running = False

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Stop all managed engines
        for engine_id, engine in list(self._engine_instances.items()):
            try:
                await engine.stop()
            except Exception as e:
                logger.warning("Error stopping engine %s: %s", engine_id, e)

        logger.info("Control Plane Manager stopped")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get Control Plane statistics.

        Returns:
            Dictionary with engine counts and states.
        """
        engines = list(self._engines.values())
        return {
            "total_engines": len(engines),
            "healthy_engines": sum(1 for e in engines if e.is_healthy),
            "unhealthy_engines": sum(1 for e in engines if not e.is_healthy),
            "engines_by_state": {
                state.value: sum(1 for e in engines if e.state == state)
                for state in EngineState
            },
            "engines_by_kind": {},
            "is_running": self._is_running,
        }

    async def __aenter__(self) -> ControlPlaneManagerLite:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.stop()
