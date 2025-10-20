# SAGE Control Plane - Integration Architecture

## Overview

Control Plane acts as an intelligent request coordinator between SAGE applications and multiple vLLM instances. It provides:

1. **Request Scheduling**: FIFO, Priority, SLO-Aware, Cost-Optimized, Adaptive policies
2. **PD Separation (Prefilling/Decoding)**: Optimize throughput by routing requests to specialized instances
3. **Direct vLLM Integration**: Uses AsyncLLMEngine Python API (zero HTTP overhead)
4. **Instance Management**: Register, monitor, and health-check vLLM instances

## Architecture

### How SAGE Communicates with vLLM through Control Plane

```
┌──────────────────────────────────┐
│     SAGE Applications            │
│  (chat, embedding, batch, etc)   │
└────────────┬─────────────────────┘
             │ submit_request()
             ▼
┌──────────────────────────────────┐
│   Control Plane Manager          │
│  - Scheduling Policies           │
│  - PD Separation Routing         │
│  - Request Queue Management      │
└────────────┬─────────────────────┘
             │ route_to_instance()
             ▼
┌──────────────────────────────────┐
│   Execution Coordinator          │
│  - Instance Registry             │
│  - Engine Management             │
│  - Performance Metrics           │
└────────────┬─────────────────────┘
             │ execute via Python API
             ▼
┌──────────────────────────────────┐
│   vLLM Instances (Multiple)      │
│  - AsyncLLMEngine                │
│  - Direct Python API calls       │
│  - Prefilling/Decoding/Hybrid    │
└──────────────────────────────────┘
```

### Request Flow Example

```python
# 1. SAGE App submits request to Control Plane
request = RequestMetadata(
    request_id="req-123",
    priority=RequestPriority.HIGH,
    max_tokens=512,
    prompt="Explain quantum computing..."
)

manager.submit_request(request)

# 2. Control Plane routes based on scheduling policy
scheduling_decision = manager.get_next_request()

# 3. Control Plane determines request phase (prefilling/decoding)
phase = manager.pd_router.determine_request_phase(request)

# 4. Control Plane selects appropriate instance
instance = manager.select_instance_for_phase(phase)

# 5. Execution Coordinator executes via vLLM Python API
engine = coordinator.get_engine(instance.instance_id)
outputs = await engine.generate(
    request.prompt,
    sampling_params=SamplingParams(max_tokens=512)
)

# 6. Result returns to SAGE
return outputs
```

## Reference: SAGE's Previous vLLM Integration

See `service.py` for how SAGE originally integrated vLLM:

- **Design Pattern**: Try-except for optional vLLM imports
- **Model Management**: Automatic download and version control
- **Sampling Configuration**: Default params with runtime overrides
- **Thread Safety**: RLock for multi-threaded access

Control Plane extends this pattern with:
- **Multiple Instances**: Manage multiple vLLM models simultaneously
- **Request Routing**: Intelligent routing based on request characteristics
- **PD Separation**: Specialized prefilling and decoding instances
- **Async Operations**: Full async/await support for concurrent requests

## Running Tests

### Local Development

```bash
# Navigate to control_plane test directory
cd tests/control_plane

# Run all tests
python -m pytest -v

# Run specific test module
python -m pytest test_scheduling.py -v

# Run specific test
python -m pytest test_integration.py::test_sage_control_plane_vllm_integration -v
```

### CI/CD

The GitHub Actions workflow runs:

```bash
cd tests/control_plane && python -m pytest -v --tb=short
```

**Why this approach?**

- Avoids loading `tests/conftest.py` which has heavy vLLM dependencies
- `tests/control_plane/pytest.ini` provides isolated configuration
- Each test directory has its own conftest.py with minimal dependencies
- Works with or without compiled vLLM C extensions

## Test Coverage

- **test_scheduling.py** (5 tests): Scheduling policy validation
- **test_pd_separation.py** (5 tests): PD routing and instance specialization
- **test_executor.py** (5 tests): Executor lifecycle and instance management
- **test_integration.py** (5 tests): Full SAGE ↔ Control Plane ↔ vLLM flow

**Total: 20 tests, all passing ✅**

## Key Components

### Types (types.py)

```python
# Request metadata
RequestMetadata(request_id, priority, max_tokens, prompt, ...)

# Execution instance
ExecutionInstance(instance_id, model_name, tensor_parallel_size, ...)

# Scheduling decisions
SchedulingDecision(instance_id, request_id, phase, parallelism_config)
```

### Manager (manager.py)

```python
manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    enable_pd_separation=True
)
```

### Executor (executor.py)

```python
# Register and initialize vLLM instances
coordinator.register_instance(instance)
await coordinator.initialize_instance_engine(instance)

# Execute requests
result = await coordinator.execute_request(request, instance, decision)
```

### PD Routing (pd_routing.py)

```python
# Determine request phase
phase = router.determine_request_phase(request)

# Get instance specialization scores
scores = router.get_instance_specialization(instance)
```

## Integration Points

1. **SAGE Apps**: Submit RequestMetadata to manager
2. **Model Registry**: Download and cache models
3. **vLLM**: Direct AsyncLLMEngine Python API
4. **Monitoring**: Collect metrics and health status

## Configuration Examples

### Basic Setup

```python
from control_plane import ControlPlaneManager, ExecutionInstance

manager = ControlPlaneManager(
    scheduling_policy="fifo",
    enable_pd_separation=False
)

instance = ExecutionInstance(
    instance_id="vllm-1",
    model_name="meta-llama/Llama-2-7b",
    tensor_parallel_size=1,
    gpu_count=1
)

manager.register_instance(instance)
```

### Advanced Setup with PD Separation

```python
manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    enable_pd_separation=True,
    pd_separation_config=PDSeparationConfig(
        enabled=True,
        routing_policy="adaptive",
        prefilling_threshold_input_tokens=2048
    )
)

# Prefilling-optimized instance
prefilling_instance = ExecutionInstance(
    instance_id="prefilling-1",
    model_name="meta-llama/Llama-2-70b",
    instance_type=ExecutionInstanceType.PREFILLING,
    tensor_parallel_size=8,
    gpu_count=8
)

# Decoding-optimized instance  
decoding_instance = ExecutionInstance(
    instance_id="decoding-1",
    model_name="meta-llama/Llama-2-70b",
    instance_type=ExecutionInstanceType.DECODING,
    tensor_parallel_size=2,
    gpu_count=2
)

manager.register_instance(prefilling_instance)
manager.register_instance(decoding_instance)
```

## FAQ

**Q: How does this compare to vLLM's built-in request queue?**

A: Control Plane adds high-level routing and scheduling strategies above vLLM's queues. It's designed to coordinate multiple instances and apply domain-specific policies (like PD separation).

**Q: What about dynamic batching?**

A: Control Plane works with vLLM's dynamic batching. The scheduler decides which instance handles each request, then vLLM batches on that instance.

**Q: Can I use this for inference-only or training+inference?**

A: Currently designed for inference. The architecture supports extension to fine-tuning workflows.

**Q: How does failure recovery work?**

A: Instances have health status. Requests fail over to healthy instances if an instance becomes unhealthy.

## Next Steps

1. Deploy to production with real vLLM instances
2. Add metrics export (Prometheus, CloudWatch)
3. Implement distributed coordination for multi-node setups
4. Add request-level tracing and observability
