# sageLLM Control Plane - Design Documentation

## 📋 Overview

A comprehensive **Control Plane** component has been designed for sageLLM, serving as an intelligent middleware layer between user requests and vLLM execution instances.

## 🎯 Core Objectives

- **Intelligent Request Scheduling** - Multiple scheduling algorithms for different scenarios
- **Dynamic Parallelism Optimization** - Automatic selection of optimal model partitioning strategies
- **Efficient Load Balancing** - Multiple routing algorithms ensure resource utilization
- **Performance Monitoring** - Real-time metrics collection and SLO tracking
- **Production Ready** - Async architecture, error handling, and health checks

## 🏗️ Architecture

### System Stack

```
┌─────────────────────────────────────────────────┐
│            User Requests                        │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │ Request Submission  │
    └──────────┬──────────┘
               │
┌──────────────▼─────────────────────────────────────┐
│         CONTROL PLANE LAYER                        │
│ ┌─────────────────────────────────────────────┐  │
│ │ Control Plane Manager                       │  │
│ │ - Request queuing                           │  │
│ │ - Scheduling loop                           │  │
│ │ - Health checks                             │  │
│ │ - Performance monitoring                    │  │
│ └─────────────────────────────────────────────┘  │
│  │          │                    │               │
│  ▼          ▼                    ▼               │
│ ┌──────────┐ ┌──────────────┐ ┌────────────┐  │
│ │Scheduling│ │Parallelism   │ │Router &    │  │
│ │Policies  │ │Optimizer     │ │LoadBal.   │  │
│ └──────────┘ └──────────────┘ └────────────┘  │
│                                                │
│ ┌──────────────────────────────────────────┐  │
│ │ Execution Coordinator                    │  │
│ │ - Instance management                    │  │
│ │ - Async execution                        │  │
│ │ - Metrics collection                     │  │
│ └──────────────────────────────────────────┘  │
└──────────────┬─────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────┐
│          EXECUTION LAYER (vLLM)                    │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │
│ │vLLM-1  │ │vLLM-2  │ │vLLM-3  │ │vLLM-N  │      │
│ │TP=4    │ │PP=2    │ │Hybrid  │ │DP=2    │      │
│ │4 GPUs  │ │4 GPUs  │ │8 GPUs  │ │2 GPUs  │      │
│ └────────┘ └────────┘ └────────┘ └────────┘      │
└─────────────────────────────────────────────────────┘
```

## 📦 Components

### 1. **types.py** - Core Data Structures

Defines all essential data types and enums:

```python
# Priority levels
class RequestPriority(Enum):
    CRITICAL = 0      # Highest
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4    # Lowest

# Parallelism types
class ParallelismType(Enum):
    TENSOR_PARALLEL = "tp"      # Weight sharding
    PIPELINE_PARALLEL = "pp"    # Layer sharding
    DATA_PARALLEL = "dp"        # Model replication
    EXPERT_PARALLEL = "ep"      # For MoE
    HYBRID = "hybrid"           # Combined

# Key data classes
@dataclass
class RequestMetadata:
    request_id: str
    priority: RequestPriority
    slo_deadline_ms: Optional[float]    # SLO deadline
    cost_budget: Optional[float]         # Cost limit
    parallelism_hint: Optional[ParallelismType]
    # ... other fields

@dataclass
class ExecutionInstance:
    instance_id: str
    model_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_count: int
    current_load: float
    # ... other fields

@dataclass
class PerformanceMetrics:
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    slo_compliance_rate: float
    # ... other metrics
```

### 2. **policies.py** - Scheduling Strategies

Five scheduling policies for different scenarios:

#### **FIFOPolicy**
- First-In-First-Out
- Sort by arrival time
- Low overhead
- Use case: Simple scenarios

#### **PriorityPolicy**
- Priority-based scheduling
- High-priority requests get better instances
- Use case: Mixed-priority workloads

#### **SLOAwarePolicy** ⭐
- Considers deadline urgency
- Calculates urgency score: `urgency = min(1.0, elapsed / deadline)`
- Urgent requests→fast instances
- Use case: Time-sensitive applications

#### **CostOptimizedPolicy**
- Minimizes cost while meeting SLO
- Estimates cost: `cost = gpu_hours * price_per_hour`
- Use case: Cost-sensitive batch processing

#### **AdaptivePolicy** 🚀 **Recommended**
- Automatically switches policies based on conditions:
  - High priority requests detected → PriorityPolicy
  - SLO requests under high load → SLOAwarePolicy
  - Low load → CostOptimizedPolicy
  - Otherwise → SLOAwarePolicy
- Use case: Production systems

### 3. **parallelism.py** - Parallelism Strategies

Automatic optimization of model partitioning across GPUs:

#### Five Parallelism Strategies

| Strategy | Use Case | GPU Distribution | Overhead |
|----------|----------|-----------------|----------|
| **TP (Tensor Parallel)** | Model too large for single GPU | Weight sharding across GPUs | Communication overhead ≈ 0.1 * log₂(tp_size) |
| **PP (Pipeline Parallel)** | Ultra-large models | Layer sharding | Bubble overhead ≈ 0.15 * pp_size |
| **DP (Data Parallel)** | High throughput | Model replication | Low communication |
| **EP (Expert Parallel)** | MoE models | Expert sharding | Routing overhead |
| **Hybrid** | Large-scale deployment | TP + PP + DP combination | Optimal efficiency |

#### Automatic Configuration Selection

```python
def select_best_config(gpu_count):
    if gpu_count >= 16:
        return TP=4, PP=2, DP=auto  # Large scale
    elif gpu_count >= 8:
        return TP=4, DP=auto         # Medium scale
    elif gpu_count >= 4:
        return TP=4                  # Small scale
    else:
        return TP=min(gpu_count, 2)  # Minimal
```

#### Performance Estimation

Each strategy can estimate:
- `latency_ms` - Expected inference latency
- `throughput_tokens_per_sec` - Token generation speed
- `gpu_utilization` - GPU usage percentage

### 4. **router.py** - Request Routing

#### RequestRouter - 5 Routing Strategies

| Strategy | Behavior | Best For |
|----------|----------|----------|
| **load_balanced** | Route to least loaded instance | General use, **default** |
| **round_robin** | Cycle through instances | Simple load distribution |
| **random** | Random selection | Distributed load |
| **affinity** | User session affinity | Stateful applications |
| **locality** | Hash-based routing | High cache hit rate |

#### LoadBalancer - 4 Algorithms

- **weighted_round_robin** - Weight by available capacity
- **least_connections** - Prefer instance with fewest active requests
- **least_response_time** - Prefer low-latency instance
- **power_of_two** - Random choice between two random instances (high performance)

### 5. **executor.py** - Execution Coordinator

Manages vLLM instances and request execution:

```python
class ExecutionCoordinator:
    
    async def execute_request(request, instance, decision):
        """Execute request on instance with specified parallelism config"""
        
    async def health_check(instance_id):
        """Check instance health"""
        
    async def health_check_all():
        """Health check all instances"""
        
    def get_metrics():
        """Get current performance metrics"""
        
    def get_instance_metrics(instance_id):
        """Get metrics for specific instance"""
```

### 6. **manager.py** - Control Plane Manager

Main orchestrator tying everything together:

```python
class ControlPlaneManager:
    
    def __init__(
        self,
        scheduling_policy: str = "adaptive",
        routing_strategy: str = "load_balanced",
        enable_monitoring: bool = True,
    ):
        # Initialize all components
        
    async def start():
        """Start background tasks"""
        # - Scheduling loop (100ms interval)
        # - Health check loop (10s interval)
        # - Monitoring loop (5s interval)
        
    async def submit_request(request: RequestMetadata) -> str:
        """Submit new inference request"""
        
    async def _scheduling_loop():
        """Main scheduling loop - continuously schedules pending requests"""
        
    async def _health_check_loop():
        """Periodic health checks"""
        
    async def _monitoring_loop():
        """Collect and log metrics"""
```

## 🚀 Usage Example

### Basic Setup

```python
import asyncio
from vllm.control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    RequestMetadata,
    RequestPriority,
)

async def main():
    # 1. Create Control Plane
    cp = ControlPlaneManager(
        scheduling_policy="adaptive",
        routing_strategy="load_balanced",
    )
    
    # 2. Register vLLM instances
    instance1 = ExecutionInstance(
        instance_id="vllm-1",
        host="localhost",
        port=8000,
        model_name="llama-3-70b",
        tensor_parallel_size=4,
        gpu_count=4,
        gpu_memory_gb=80.0,
        max_concurrent_requests=100,
    )
    cp.register_instance(instance1)
    
    # 3. Start Control Plane
    await cp.start()
    
    # 4. Submit requests
    request = RequestMetadata(
        request_id="req-001",
        user_id="user-123",
        priority=RequestPriority.HIGH,
        slo_deadline_ms=1000,  # 1 second
        max_tokens=100,
        model_name="llama-3-70b",
    )
    await cp.submit_request(request)
    
    # 5. Monitor
    metrics = cp.get_metrics()
    print(f"Avg Latency: {metrics.avg_latency_ms}ms")
    print(f"SLO Compliance: {metrics.slo_compliance_rate:.2%}")
    
    # 6. Stop
    await cp.stop()

asyncio.run(main())
```

### Advanced Configuration

```python
from vllm.control_plane.types import ParallelismType

# Parallelism hint
request = RequestMetadata(
    request_id="req-002",
    parallelism_hint=ParallelismType.HYBRID,  # Use hybrid parallelism
)

# Cost-aware request
request = RequestMetadata(
    request_id="req-003",
    cost_budget=0.01,  # $0.01 budget
)

# Dynamic policy switching
cp.update_policy("slo_aware")  # Switch to SLO-aware scheduling
```

## 📊 Monitoring and Metrics

### Available Metrics

```python
metrics = cp.get_metrics()

# Request metrics
print(metrics.total_requests)          # Total submitted
print(metrics.completed_requests)      # Successfully completed
print(metrics.failed_requests)         # Failed
print(metrics.active_requests)         # Currently running

# Latency metrics
print(metrics.avg_latency_ms)          # Average latency
print(metrics.p95_latency_ms)          # 95th percentile
print(metrics.p99_latency_ms)          # 99th percentile

# Throughput
print(metrics.tokens_per_second)       # Generation speed
print(metrics.requests_per_second)     # Request throughput

# SLO compliance
print(metrics.slo_violations)          # Number of violations
print(metrics.slo_compliance_rate)     # Compliance percentage

# Resource utilization
print(metrics.avg_gpu_utilization)     # GPU usage
print(metrics.used_gpu_memory_gb)      # Memory used
```

## 🎯 Scenario-Based Configuration

### Scenario 1: Enterprise SaaS
```python
cp = ControlPlaneManager(
    scheduling_policy="slo_aware",
    routing_strategy="affinity",
)
# ✓ Strict SLO guarantees
# ✓ Consistent user experience
# ✓ Session state preservation
```

### Scenario 2: Cost-Optimized Batch
```python
cp = ControlPlaneManager(
    scheduling_policy="cost_optimized",
    routing_strategy="load_balanced",
)
# ✓ Minimal cost
# ✓ Sufficient throughput
# ✓ Flexible latency
```

### Scenario 3: Mixed Production
```python
cp = ControlPlaneManager(
    scheduling_policy="adaptive",
    routing_strategy="load_balanced",
)
# ✓ Auto-adapts to load conditions
# ✓ Balances cost and performance
# ✓ **Recommended for production**
```

### Scenario 4: Real-Time Interactive
```python
cp = ControlPlaneManager(
    scheduling_policy="priority",
    routing_strategy="locality",
)
# ✓ Priority queue handling
# ✓ High cache hit rate
# ✓ Low latency for interactive requests
```

## 💡 Key Design Features

### 1. Multi-Level Scheduling
- **Global Level** - Which instance?
- **Instance Level** - How to parallelize?
- **Routing Level** - How to route?

### 2. Async-First Architecture
- Non-blocking request submission
- Async scheduling and execution
- Background monitoring loops

### 3. Performance-Aware
- Learning from historical data
- Predictive cost and latency estimation
- Dynamic policy adjustment

### 4. Highly Configurable
- 5 scheduling policies
- 5 parallelism strategies
- 5 routing algorithms
- Runtime policy switching

### 5. Observable
- Rich performance metrics
- SLO compliance tracking
- Per-instance health status

## 🔮 Future Enhancements

1. **Auto-Scaling** - Dynamic instance count adjustment
2. **Smart Caching** - KV cache sharing and management
3. **Multi-Model Support** - Managing multiple different models
4. **Cost Prediction** - Historical data-based forecasting
5. **Fault Recovery** - Auto-detection and failover
6. **Model Hot-Swapping** - Zero-downtime model updates
7. **User Quotas** - Fine-grained resource control
8. **A/B Testing** - Multi-version comparison

## 📁 File Structure

```
control_plane/
├── __init__.py                  # Unified exports
├── types.py                     # Data structures (7 classes, 4 enums)
├── policies.py                  # 5 scheduling policies (~500 lines)
├── parallelism.py               # 5 parallelism strategies + optimizer (~600 lines)
├── router.py                    # Routing and load balancing (~300 lines)
├── executor.py                  # Execution coordinator (~250 lines)
├── manager.py                   # Main manager (~450 lines)
├── example.py                   # Complete usage example (~400 lines)
├── README.md                    # Detailed documentation
├── DESIGN_SUMMARY_CN.md         # Chinese design summary
└── DESIGN.md                    # This file
```

## ✨ Summary

The Control Plane provides sageLLM with:

✅ **Intelligent Scheduling** - 5 strategies for different scenarios
✅ **Dynamic Parallelism** - Auto-optimized model partitioning
✅ **Efficient Routing** - 5 algorithms + 4 load balancing methods
✅ **Performance Monitoring** - Comprehensive metrics and SLO tracking
✅ **Production-Ready** - Async, error-handled, health-checked
✅ **Highly Configurable** - Runtime policy switching
✅ **Observable** - Detailed monitoring and diagnostics

Ready to integrate into sageLLM!
