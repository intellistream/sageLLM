# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Lifecycle management for vLLM engines managed by the control plane."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil
from sage.common.config.ports import SagePorts

logger = logging.getLogger(__name__)


class EngineStatus(str, Enum):
    """Lifecycle states tracked for managed vLLM engines."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class EngineRuntime(str, Enum):
    """Process runtime type managed by the lifecycle manager."""

    LLM = "llm"
    EMBEDDING = "embedding"


@dataclass
class EngineProcessInfo:
    """Bookkeeping structure for each managed engine instance."""

    engine_id: str
    model_id: str
    port: int
    gpu_ids: list[int]
    pid: int
    command: list[str]
    env_overrides: dict[str, str]
    runtime: EngineRuntime = EngineRuntime.LLM
    created_at: float = field(default_factory=time.time)
    status: EngineStatus = EngineStatus.STARTING
    stopped_at: float | None = None
    last_exit_code: int | None = None
    last_error: str | None = None


class EngineLifecycleManager:
    """Spawn, monitor, and stop vLLM engines for the control plane."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        host: str = "0.0.0.0",
        stop_timeout: float = 20.0,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.host = host
        self.stop_timeout = stop_timeout

        self._lock = threading.RLock()
        self._engines: dict[str, EngineProcessInfo] = {}
        self._reserved_ports: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def spawn_engine(
        self,
        model_id: str,
        gpu_ids: list[int],
        port: int | None = None,
        *,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        extra_args: list[str] | None = None,
        engine_kind: EngineRuntime = EngineRuntime.LLM,
    ) -> str:
        """Launch a vLLM OpenAI-compatible server and register it."""

        extra_args = list(extra_args or [])
        with self._lock:
            resolved_port = self._reserve_port(port)
            engine_id = self._generate_engine_id(model_id)
            command = self._build_engine_command(
                engine_kind,
                model_id,
                resolved_port,
                tensor_parallel_size,
                pipeline_parallel_size,
                extra_args,
            )
            env = self._build_environment(gpu_ids)

            logger.info(
                "Spawning %s engine %s (model=%s, port=%s, gpus=%s)",
                engine_kind.value,
                engine_id,
                model_id,
                resolved_port,
                ",".join(str(gpu) for gpu in gpu_ids) or "CPU",
            )

            try:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
            except Exception:
                self._reserved_ports.discard(resolved_port)
                logger.exception("Failed to spawn vLLM engine for %s", model_id)
                raise

            info = EngineProcessInfo(
                engine_id=engine_id,
                model_id=model_id,
                port=resolved_port,
                gpu_ids=list(gpu_ids),
                pid=process.pid,
                command=command,
                env_overrides=self._extract_env_overrides(env),
                runtime=engine_kind,
            )
            self._engines[engine_id] = info
            self._reserved_ports.add(resolved_port)
            logger.debug("Engine %s registered with PID %d", engine_id, process.pid)
            return engine_id

    def stop_engine(self, engine_id: str, *, timeout: float | None = None) -> bool:
        """Stop a managed engine by sending SIGTERM followed by SIGKILL if needed."""

        with self._lock:
            info = self._engines.get(engine_id)
            if info is None:
                logger.warning("Requested stop for unknown engine %s", engine_id)
                return False
            info.status = EngineStatus.STOPPING
        process = self._get_process(info.pid)
        if process is None:
            logger.info("Engine %s already stopped", engine_id)
            self._finalize_engine(info, EngineStatus.STOPPED)
            return True

        success = self._terminate_process(process, timeout or self.stop_timeout)
        with self._lock:
            new_status = EngineStatus.STOPPED if success else EngineStatus.FAILED
            self._finalize_engine(info, new_status)
        return success

    def get_engine_status(self, engine_id: str) -> dict[str, Any]:
        """Return runtime metadata for the specified engine."""

        with self._lock:
            info = self._engines.get(engine_id)
            if info is None:
                raise KeyError(f"Engine '{engine_id}' not found")
            status = self._refresh_status(info)
            return self._serialize(info, status)

    def list_engines(self) -> list[dict[str, Any]]:
        """Return status objects for all tracked engines."""

        with self._lock:
            return [
                self._serialize(info, self._refresh_status(info))
                for info in self._engines.values()
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_engine_id(self, model_id: str) -> str:
        suffix = uuid.uuid4().hex[:8]
        sanitized = model_id.split("/")[-1].replace(" ", "-").replace(".", "-")
        return f"engine-{sanitized.lower()}-{suffix}"

    def _build_engine_command(
        self,
        engine_kind: EngineRuntime,
        model_id: str,
        port: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        extra_args: list[str],
    ) -> list[str]:
        if engine_kind == EngineRuntime.EMBEDDING:
            return self._build_embedding_command(model_id, port, extra_args)
        return self._build_llm_command(
            model_id,
            port,
            tensor_parallel_size,
            pipeline_parallel_size,
            extra_args,
        )

    def _build_llm_command(
        self,
        model_id: str,
        port: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        extra_args: list[str],
    ) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--host",
            self.host,
            "--port",
            str(port),
        ]
        command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        command.extend(["--pipeline-parallel-size", str(pipeline_parallel_size)])
        command.extend(extra_args)
        return command

    def _build_embedding_command(
        self,
        model_id: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "sage.common.components.sage_embedding.embedding_server",
            "--model",
            model_id,
            "--host",
            self.host,
            "--port",
            str(port),
        ]
        command.extend(extra_args)
        return command

    def _build_environment(self, gpu_ids: list[int]) -> dict[str, str]:
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        return env

    def _extract_env_overrides(self, env: dict[str, str]) -> dict[str, str]:
        overrides = {}
        for key in ("CUDA_VISIBLE_DEVICES",):
            if key in env:
                overrides[key] = env[key]
        return overrides

    def _reserve_port(self, requested: int | None) -> int:
        if requested is not None:
            self._validate_port(requested)
            return requested

        candidates = [SagePorts.get_recommended_llm_port(), *SagePorts.get_llm_ports()]
        seen: set[int] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                self._validate_port(candidate)
                return candidate
            except ValueError:
                continue

        fallback = SagePorts.find_available_port()
        if fallback is None:
            raise RuntimeError("No available port for spawning engine")
        return fallback

    def _validate_port(self, port: int) -> None:
        if port in self._reserved_ports:
            raise ValueError(f"Port {port} already reserved by control plane")
        if not SagePorts.is_available(port):
            raise ValueError(f"Port {port} is already in use")

    def _get_process(self, pid: int) -> psutil.Process | None:
        try:
            process = psutil.Process(pid)
        except psutil.Error:
            return None
        return process

    def _terminate_process(self, process: psutil.Process, timeout: float) -> bool:
        try:
            process.terminate()
            process.wait(timeout=timeout)
            logger.info("Engine PID %d stopped gracefully", process.pid)
            return True
        except psutil.TimeoutExpired:
            logger.warning(
                "Engine PID %d did not stop in %.1fs; force killing",
                process.pid,
                timeout,
            )
            try:
                process.kill()
                process.wait(timeout=5)
                return True
            except psutil.Error as exc:
                logger.error("Failed to kill engine PID %d: %s", process.pid, exc)
                return False
        except psutil.NoSuchProcess:
            return True
        except psutil.Error as exc:
            logger.error("Error while stopping engine PID %d: %s", process.pid, exc)
            return False

    def _refresh_status(self, info: EngineProcessInfo) -> EngineStatus:
        if info.status in {EngineStatus.STOPPED, EngineStatus.FAILED}:
            return info.status

        process = self._get_process(info.pid)
        if process is None:
            info.last_exit_code = info.last_exit_code or -1
            self._finalize_engine(info, EngineStatus.FAILED)
            return info.status

        if process.is_running():
            info.status = EngineStatus.RUNNING
            return info.status

        try:
            exit_code = process.wait(timeout=0)
        except (psutil.TimeoutExpired, psutil.Error):
            exit_code = None

        info.last_exit_code = exit_code
        final_status = EngineStatus.STOPPED if (exit_code or 0) == 0 else EngineStatus.FAILED
        self._finalize_engine(info, final_status)
        return info.status

    def _finalize_engine(self, info: EngineProcessInfo, status: EngineStatus) -> None:
        info.status = status
        if info.stopped_at is None and status in {EngineStatus.STOPPED, EngineStatus.FAILED}:
            info.stopped_at = time.time()
        self._reserved_ports.discard(info.port)

    def _serialize(self, info: EngineProcessInfo, status: EngineStatus) -> dict[str, Any]:
        uptime = (info.stopped_at or time.time()) - info.created_at
        return {
            "engine_id": info.engine_id,
            "model_id": info.model_id,
            "port": info.port,
            "gpu_ids": info.gpu_ids,
            "pid": info.pid,
            "status": status.value,
            "runtime": info.runtime.value,
            "uptime_seconds": max(uptime, 0.0),
            "command": info.command,
            "env": info.env_overrides,
            "last_exit_code": info.last_exit_code,
        }


__all__ = [
    "EngineLifecycleManager",
    "EngineProcessInfo",
    "EngineRuntime",
    "EngineStatus",
]
