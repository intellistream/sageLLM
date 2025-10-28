# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Common types and data structures for Control Plane."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


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


class ExecutionInstanceType(Enum):
    """Types/roles of execution instances for PD separation."""

    GENERAL = "general"  # General-purpose instance
    PREFILLING = "prefilling"  # Specialized for prefilling phase
    DECODING = "decoding"  # Specialized for decoding phase
    HYBRID = "hybrid"  # Handles both prefilling and decoding


@dataclass
class RequestMetadata:
    """Metadata for an inference request."""

    request_id: str
    user_id: Optional[str] = None
    prompt: str = ''
    key: int = 0
    priority: RequestPriority = RequestPriority.NORMAL
    slo_deadline_ms: Optional[float] = None  # SLO deadline in milliseconds
    max_tokens: Optional[int] = 256
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
    tags: dict[str, Any] = field(default_factory=dict)

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
class PrefillingConfig:
    """Configuration for prefilling-specialized instances."""

    # Optimization target
    target_batch_size: int = 64
    target_throughput_tokens_per_sec: float = 1000.0

    # Parallelism settings
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1

    # Performance tuning
    enable_kv_cache: bool = True
    enable_chunked_prefill: bool = True
    max_chunk_size_tokens: int = 4096

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingConfig:
    """Configuration for decoding-specialized instances."""

    # Optimization target
    target_latency_ms: float = 50.0
    target_tokens_per_sec_per_gpu: float = 100.0

    # Parallelism settings (typically minimal)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Performance tuning
    enable_prefix_caching: bool = True
    max_parallel_requests: int = 200
    kv_cache_memory_fraction: float = 0.85

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionInstance:
    """Represents a vLLM execution instance."""

    instance_id: str
    model_name: str
    host: str = '127.0.0.1'
    port: int = 5050
    # Parallelism configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    # PD Separation: Instance type and specialized configs
    instance_type: ExecutionInstanceType = ExecutionInstanceType.GENERAL
    prefilling_config: Optional[PrefillingConfig] = None
    decoding_config: Optional[DecodingConfig] = None

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

    # PD Separation: Request type tracking
    prefilling_active_requests: int = 0
    decoding_active_requests: int = 0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def can_accept_prefilling_request(self) -> bool:
        """Check if instance can accept prefilling requests."""
        if self.instance_type == ExecutionInstanceType.DECODING:
            return False
        return self.can_accept_request

    def can_accept_decoding_request(self) -> bool:
        """Check if instance can accept decoding requests."""
        if self.instance_type == ExecutionInstanceType.PREFILLING:
            return False
        return self.can_accept_request


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

    metadata: dict[str, Any] = field(default_factory=dict)


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

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PDSeparationConfig:
    """Configuration for Prefilling/Decoding separation."""

    # Enable/disable PD separation
    enabled: bool = True

    # Dynamic scaling configuration
    enable_dynamic_scaling: bool = True
    prefilling_min_instances: int = 1
    prefilling_max_instances: int = 5
    decoding_min_instances: int = 2
    decoding_max_instances: int = 8

    # Routing decision thresholds
    routing_policy: str = "adaptive"  # adaptive | threshold | ml
    prefilling_threshold_input_tokens: int = 800
    prefilling_threshold_ratio: float = 4.0  # input_tokens / output_tokens

    # KV-Cache management
    kv_cache_storage: str = "gpu"  # gpu | cpu | shared
    kv_cache_eviction_policy: str = "lru"  # lru | lfu | fifo
    kv_cache_memory_fraction: float = 0.85

    # Inter-cluster communication
    enable_kv_cache_transfer: bool = True
    kv_cache_transfer_buffer_mb: int = 2048
    max_transfer_latency_ms: float = 100.0

    # Monitoring and metrics
    collect_pd_metrics: bool = True
    pd_metrics_interval_sec: int = 10
    enable_pd_tracing: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PDMetrics:
    """Metrics specific to PD separation."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Prefilling metrics
    prefilling_throughput_tokens_per_sec: float = 0.0
    prefilling_latency_avg_ms: float = 0.0
    prefilling_latency_p99_ms: float = 0.0
    prefilling_gpu_util: float = 0.0
    prefilling_active_requests: int = 0

    # Decoding metrics
    decoding_latency_avg_ms: float = 0.0
    decoding_latency_p99_ms: float = 0.0
    decoding_gpu_util: float = 0.0
    decoding_active_requests: int = 0

    # Inter-cluster metrics
    kv_cache_hit_rate: float = 0.0  # Cache hit rate 0.0-1.0
    inter_cluster_transfer_latency_avg_ms: float = 0.0
    kv_cache_evictions_per_sec: float = 0.0

    # Cost metrics
    prefilling_cost_per_token: float = 0.0
    decoding_cost_per_token: float = 0.0
    total_cost_optimization_percent: float = 0.0  # vs mixed baseline

    # Routing statistics
    requests_routed_to_prefilling: int = 0
    requests_routed_to_decoding: int = 0
    requests_routed_to_hybrid: int = 0
    avg_routing_decision_time_us: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)
