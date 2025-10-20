# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""
Control Plane for sageLLM - Intelligent request routing and scheduling layer.

The Control Plane sits between users and vLLM execution instances, providing:
- Intelligent request routing and load balancing
- Advanced scheduling policies (priority, SLO-aware, cost-optimized)
- Dynamic parallelism strategy selection and optimization
- Multi-instance management and coordination
- Performance monitoring and adaptive optimization
"""

# Import types
from vllm.control_plane.executor import ExecutionCoordinator

# Import manager
from vllm.control_plane.manager import ControlPlaneManager

# Import parallelism strategies
from vllm.control_plane.parallelism import (
    DataParallelStrategy,
    ExpertParallelStrategy,
    HybridParallelStrategy,
    ParallelismConfig,
    ParallelismOptimizer,
    ParallelismStrategy,
    PipelineParallelStrategy,
    TensorParallelStrategy,
)

# Import PD separation routing
from vllm.control_plane.pd_routing import PDRoutingStrategy

# Import policies
from vllm.control_plane.policies import (
    AdaptivePolicy,
    CostOptimizedPolicy,
    FIFOPolicy,
    PriorityPolicy,
    SchedulingPolicy,
    SLOAwarePolicy,
)

# Import router and coordinator
from vllm.control_plane.router import LoadBalancer, RequestRouter
from vllm.control_plane.types import (
    DecodingConfig,
    ExecutionInstance,
    ExecutionInstanceType,
    ParallelismType,
    PDMetrics,
    PDSeparationConfig,
    PerformanceMetrics,
    PreffillingConfig,
    RequestMetadata,
    RequestPriority,
    RequestStatus,
    SchedulingDecision,
)

__all__ = [
    # Types - Core
    "RequestPriority",
    "RequestStatus",
    "ParallelismType",
    "RequestMetadata",
    "ExecutionInstance",
    "SchedulingDecision",
    "PerformanceMetrics",
    # Types - PD Separation
    "ExecutionInstanceType",
    "PreffillingConfig",
    "DecodingConfig",
    "PDSeparationConfig",
    "PDMetrics",
    # Policies
    "SchedulingPolicy",
    "FIFOPolicy",
    "PriorityPolicy",
    "SLOAwarePolicy",
    "CostOptimizedPolicy",
    "AdaptivePolicy",
    # Parallelism
    "ParallelismConfig",
    "ParallelismStrategy",
    "TensorParallelStrategy",
    "PipelineParallelStrategy",
    "DataParallelStrategy",
    "ExpertParallelStrategy",
    "HybridParallelStrategy",
    "ParallelismOptimizer",
    # Router and Executor
    "RequestRouter",
    "LoadBalancer",
    "ExecutionCoordinator",
    # PD Routing
    "PDRoutingStrategy",
    # Manager
    "ControlPlaneManager",
]
