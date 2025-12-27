# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Core types and data structures for sageLLM modular architecture.

This module defines the fundamental types used across all sageLLM modules,
including engine state management, request metadata, and base enumerations.

These types are intentionally minimal and stable - KV-specific fields have been
moved to kv_runtime module, scheduling-specific fields to scheduler_ir, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# =============================================================================
# Engine State and Lifecycle
# =============================================================================


class EngineState(str, Enum):
    """Lifecycle states for registered engines in the Control Plane.

    State transitions:
        STARTING -> READY (after successful health check)
        READY -> DRAINING (when graceful shutdown requested)
        DRAINING -> STOPPED (after all requests complete)
        Any -> ERROR (on consecutive health check failures)
        ERROR -> STARTING (on restart attempt)
    """

    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class EngineKind(str, Enum):
    """Kind of engine for capability classification."""

    LLM = "llm"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"  # Both LLM and embedding


@dataclass
class EngineInfo:
    """Information about a registered engine in the Control Plane.

    This dataclass tracks the state and metadata of an engine that has
    registered with the Control Plane. KV-specific fields have been moved
    to kv_runtime.KVBlockInfo.

    Attributes:
        engine_id: Unique identifier for this engine.
        model_id: The model loaded on this engine.
        host: Hostname or IP address of the engine.
        port: Port number the engine is listening on.
        state: Current lifecycle state of the engine.
        engine_kind: Type of engine (llm, embedding, or hybrid).
        backend_type: Backend implementation (lmdeploy, vllm, etc.).
        created_at: When the engine was registered.
        last_heartbeat: Timestamp of last successful heartbeat.
        consecutive_failures: Number of consecutive health check failures.
        active_requests: Number of requests currently being processed.
        metadata: Additional engine metadata (labels, GPU info, etc.).
    """

    engine_id: str
    model_id: str
    host: str
    port: int
    state: EngineState = EngineState.STARTING
    engine_kind: EngineKind = EngineKind.LLM
    backend_type: str = "lmdeploy"
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime | None = None
    consecutive_failures: int = 0
    active_requests: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if the engine is in a healthy state."""
        return self.state == EngineState.READY

    @property
    def is_accepting_requests(self) -> bool:
        """Check if the engine can accept new requests."""
        return self.state == EngineState.READY

    @property
    def is_terminal(self) -> bool:
        """Check if the engine is in a terminal state (STOPPED or ERROR)."""
        return self.state in (EngineState.STOPPED, EngineState.ERROR)

    @property
    def base_url(self) -> str:
        """Get the base URL for this engine."""
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.value,
            "engine_kind": self.engine_kind.value,
            "backend_type": self.backend_type,
            "created_at": self.created_at.isoformat(),
            "last_heartbeat": (
                self.last_heartbeat.isoformat() if self.last_heartbeat else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "active_requests": self.active_requests,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineInfo:
        """Create from dictionary."""
        return cls(
            engine_id=data["engine_id"],
            model_id=data["model_id"],
            host=data["host"],
            port=data["port"],
            state=EngineState(data.get("state", "STARTING")),
            engine_kind=EngineKind(data.get("engine_kind", "llm")),
            backend_type=data.get("backend_type", "lmdeploy"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"])
            if data.get("last_heartbeat")
            else None,
            consecutive_failures=data.get("consecutive_failures", 0),
            active_requests=data.get("active_requests", 0),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Request Types and Metadata
# =============================================================================


class RequestType(str, Enum):
    """Request type for hybrid scheduling."""

    LLM_CHAT = "llm_chat"
    LLM_GENERATE = "llm_generate"
    EMBEDDING = "embedding"


class RequestPriority(int, Enum):
    """Request priority levels."""

    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


class RequestStatus(str, Enum):
    """Request lifecycle status."""

    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RequestMetadata:
    """Minimal metadata for an inference request.

    This dataclass stores essential metadata for request tracking.
    KV-related fields (kv_block_ids, prefix_hit, etc.) have been moved
    to kv_runtime module. Scheduling-specific fields are in scheduler_ir.

    Attributes:
        request_id: Unique identifier for this request.
        request_type: Type of request (LLM_CHAT, LLM_GENERATE, EMBEDDING).
        model_name: Model name/identifier.
        priority: Request priority level for scheduling.
        status: Current request lifecycle status.
        user_id: Optional user identifier for tracking/billing.

        arrival_time: When the request arrived.
        schedule_time: When the request was scheduled.
        start_time: When execution started.
        end_time: When execution completed.

        slo_deadline_ms: SLO deadline in milliseconds.
        tags: Additional metadata tags.
    """

    request_id: str
    request_type: RequestType = RequestType.LLM_CHAT
    model_name: str | None = None
    priority: RequestPriority = RequestPriority.NORMAL
    status: RequestStatus = RequestStatus.PENDING
    user_id: str | None = None

    # Timing information
    arrival_time: datetime = field(default_factory=datetime.now)
    schedule_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    # SLO
    slo_deadline_ms: float | None = None

    # Additional metadata
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float | None:
        """Calculate end-to-end latency in milliseconds."""
        if self.end_time and self.arrival_time:
            return (self.end_time - self.arrival_time).total_seconds() * 1000
        return None

    @property
    def queue_wait_ms(self) -> float | None:
        """Calculate queue waiting time in milliseconds."""
        if self.schedule_time and self.arrival_time:
            return (self.schedule_time - self.arrival_time).total_seconds() * 1000
        return None

    @property
    def is_embedding_request(self) -> bool:
        """Check if this is an embedding request."""
        return self.request_type == RequestType.EMBEDDING

    @property
    def is_llm_request(self) -> bool:
        """Check if this is an LLM request (chat or generate)."""
        return self.request_type in (RequestType.LLM_CHAT, RequestType.LLM_GENERATE)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "request_type": self.request_type.value,
            "model_name": self.model_name,
            "priority": self.priority.value,
            "status": self.status.value,
            "user_id": self.user_id,
            "arrival_time": self.arrival_time.isoformat(),
            "schedule_time": (
                self.schedule_time.isoformat() if self.schedule_time else None
            ),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "slo_deadline_ms": self.slo_deadline_ms,
            "tags": self.tags,
        }
