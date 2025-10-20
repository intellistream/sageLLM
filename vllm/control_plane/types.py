# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Common types and data structures for Control Plane."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


class RequestStatus(Enum):
    """Request lifecycle status."""
    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ParallelismType(Enum):
    """Types of parallelism strategies."""
    TENSOR_PARALLEL = "tp"
    PIPELINE_PARALLEL = "pp"
    DATA_PARALLEL = "dp"
    EXPERT_PARALLEL = "ep"
    HYBRID = "hybrid"


@dataclass
class RequestMetadata:
    """Metadata for an inference request."""
    
    request_id: str
    user_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    slo_deadline_ms: Optional[float] = None  # SLO deadline in milliseconds
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    model_name: Optional[str] = None
    
    # Timing information
    arrival_time: datetime = field(default_factory=datetime.now)
    queue_time: Optional[datetime] = None
    schedule_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Resource preferences
    preferred_instance_id: Optional[str] = None
    parallelism_hint: Optional[ParallelismType] = None
    
    # Cost and billing
    cost_budget: Optional[float] = None
    billing_tier: str = "standard"
    
    # Additional metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def latency_ms(self) -> Optional[float]:
        """Calculate end-to-end latency in milliseconds."""
        if self.end_time and self.arrival_time:
            return (self.end_time - self.arrival_time).total_seconds() * 1000
        return None
    
    @property
    def queue_wait_ms(self) -> Optional[float]:
        """Calculate queue waiting time in milliseconds."""
        if self.schedule_time and self.queue_time:
            return (self.schedule_time - self.queue_time).total_seconds() * 1000
        return None


@dataclass
class ExecutionInstance:
    """Represents a vLLM execution instance."""
    
    instance_id: str
    host: str
    port: int
    model_name: str
    
    # Parallelism configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Resource information
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    
    # Status
    is_available: bool = True
    is_healthy: bool = True
    current_load: float = 0.0  # 0.0 to 1.0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    # Active requests
    active_requests: int = 0
    max_concurrent_requests: int = 100
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_capacity(self) -> float:
        """Calculate available capacity (0.0 to 1.0)."""
        return max(0.0, 1.0 - self.current_load)
    
    @property
    def can_accept_request(self) -> bool:
        """Check if instance can accept new requests."""
        return (
            self.is_available
            and self.is_healthy
            and self.active_requests < self.max_concurrent_requests
            and self.current_load < 0.95
        )


@dataclass
class SchedulingDecision:
    """Decision made by the scheduler."""
    
    request_id: str
    target_instance_id: str
    parallelism_strategy: ParallelismType
    estimated_latency_ms: float
    estimated_cost: float
    
    # Configuration for execution
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Scheduling metadata
    decision_time: datetime = field(default_factory=datetime.now)
    reason: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Request metrics
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    active_requests: int = 0
    queued_requests: int = 0
    
    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Resource metrics
    avg_gpu_utilization: float = 0.0
    total_gpu_memory_gb: float = 0.0
    used_gpu_memory_gb: float = 0.0
    
    # Cost metrics
    total_cost: float = 0.0
    cost_per_token: float = 0.0
    
    # SLO metrics
    slo_violations: int = 0
    slo_compliance_rate: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
