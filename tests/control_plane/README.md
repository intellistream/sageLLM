# Control Plane Tests

Comprehensive pytest test suite for the sageLLM Control Plane component.

## Quick Start

```bash
# Run all tests
cd tests/control_plane
python -m pytest -v

# Run specific test module
python -m pytest test_scheduling.py -v

# Run with coverage
python -m pytest --cov=../../control_plane --cov-report=html
```

## Test Modules

### test_scheduling.py (5 tests)

Tests scheduling policies for request queuing and prioritization:

- `test_basic_scheduling`: FIFO scheduling without PD separation
- `test_priority_scheduling`: Priority-based request ordering
- `test_slo_aware_scheduling`: SLO deadline handling
- `test_instance_registration`: Instance lifecycle management

**Key Assertions**:
- Requests are scheduled in correct order for FIFO
- Priority queue respects priority levels (CRITICAL > HIGH > NORMAL > LOW)
- SLO deadlines trigger priority boost
- Instances register/unregister correctly

### test_pd_separation.py (5 tests)

Tests Prefilling/Decoding (PD) separation routing:

- `test_pd_separation_routing`: Route requests to specialized instances
- `test_instance_specialization_scoring`: Specialization metrics
- `test_pd_routing_policy_threshold`: Threshold-based routing decisions
- `test_pd_routing_policy_adaptive`: Adaptive multi-factor routing

**Key Assertions**:
- Long-input requests route to prefilling instances
- Short-input requests route to decoding instances
- Specialization scores reflect instance efficiency
- Hybrid instances intelligently balance load

### test_executor.py (5 tests)

Tests the execution coordinator and instance management:

- `test_executor_initialization`: Executor setup
- `test_instance_registration`: Instance lifecycle
- `test_instance_unregistration`: Instance cleanup
- `test_get_available_instances`: Health-based filtering
- `test_metrics_collection`: Performance metric tracking

**Key Assertions**:
- Instances register with correct metadata
- Only healthy instances returned as available
- Metrics accumulate during execution
- Cleanup removes all instance state

### test_integration.py (5 tests)

End-to-end integration tests showing SAGE ↔ Control Plane ↔ vLLM flow:

- `test_sage_control_plane_vllm_integration`: Main integration flow
  - SAGE app submits request → Control Plane routes → vLLM executes → result returned
- `test_control_plane_request_flow`: Multiple SAGE components
  - Different SAGE apps (chat, embedding, batch) share Control Plane
- `test_multi_model_deployment`: Multiple model sizes
  - 7B, 13B, 70B models with appropriate parallelism
- `test_control_plane_health_monitoring`: Instance health tracking
  - Healthy and degraded instance handling

**Key Assertions**:
- Complete request flow executes successfully
- Multiple components can share Control Plane
- Proper instance selection based on model size
- Health status affects routing decisions

## Architecture Validation

These tests validate that:

✅ SAGE can submit requests to Control Plane via `manager.submit_request(request)`

✅ Control Plane routes requests based on scheduling policy (FIFO, Priority, SLO-Aware)

✅ Control Plane determines request phases (Prefilling vs Decoding) via `pd_router.determine_request_phase()`

✅ Control Plane selects appropriate vLLM instances via `manager.select_instance_for_phase()`

✅ Execution Coordinator manages vLLM AsyncLLMEngine instances

✅ Full communication flow works: SAGE → Control Plane → vLLM → SAGE

## Running Tests Locally

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# From repository root
cd packages/sage-common/src/sage/common/components/sage_vllm/sageLLM
```

### Run All Tests

```bash
cd tests/control_plane
python -m pytest -v
```

### Run Specific Test

```bash
cd tests/control_plane
python -m pytest test_scheduling.py::test_basic_scheduling -v
```

### Run with Logging

```bash
cd tests/control_plane
python -m pytest -v --log-cli-level=DEBUG
```

### Run with Markers

```bash
cd tests/control_plane
# Run only integration tests
python -m pytest -m integration -v

# Run only scheduling tests  
python -m pytest -m scheduling -v
```

## CI/CD Integration

The GitHub Actions workflow runs these tests automatically:

```yaml
- name: Run Control Plane pytest tests
  run: |
    cd tests/control_plane && python -m pytest -v --tb=short
```

**Why `cd tests/control_plane`?**

- Avoids loading the main `tests/conftest.py` which imports heavy vLLM modules
- Uses `tests/control_plane/pytest.ini` for isolated configuration
- Ensures tests run even when vLLM C extensions aren't compiled
- Faster test discovery and execution

## Test Configuration

### pytest.ini

Located in `tests/control_plane/pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests/control_plane
markers =
    asyncio: async test
    control_plane: control plane component tests
    integration: end-to-end integration tests
    scheduling: scheduling policy tests
    pd_separation: PD separation tests
```

### conftest.py

Located in `tests/control_plane/conftest.py`:

Provides pytest fixtures:
- `event_loop`: Async event loop for tests
- `control_plane_manager`: ControlPlaneManager instance
- `sample_execution_instance`: ExecutionInstance for testing
- `sample_request_metadata`: RequestMetadata for testing

## Expected Output

Successful test run shows:

```
============================= test session starts ==============================
collected 20 items

test_scheduling.py::test_basic_scheduling PASSED                        [ 5%]
test_scheduling.py::test_priority_scheduling PASSED                     [10%]
test_scheduling.py::test_slo_aware_scheduling PASSED                    [15%]
test_scheduling.py::test_instance_registration PASSED                   [20%]
test_executor.py::test_executor_initialization PASSED                   [25%]
...
======================== 20 passed in ~2.0s ========================
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'control_plane'"

**Solution**: Run tests from `tests/control_plane` directory:

```bash
cd tests/control_plane
python -m pytest -v
```

### Issue: "ModuleNotFoundError: No module named 'vllm._C'"

**Expected**: This warning is normal when vLLM C extensions aren't compiled.

The tests still pass because:
1. executor.py has try-except for optional vLLM imports
2. Tests use fixtures that gracefully skip on import errors
3. Core routing and scheduling logic doesn't depend on compiled C

**If tests fail with this error**:
```bash
# Ensure you're in the right directory
cd tests/control_plane

# Run tests
python -m pytest -v
```

### Issue: Tests timeout

**Solution**: Tests have short default timeouts. Run with explicit timeout:

```bash
python -m pytest -v --timeout=10
```

## Test Statistics

- **Total Tests**: 20
- **Modules**: 4
- **Async Tests**: All (use @pytest.mark.asyncio)
- **Fixtures**: 4
- **Average Runtime**: ~2.0 seconds
- **Coverage**: All Control Plane components

## Development Workflow

1. **Make changes to control_plane code**
2. **Run tests locally**:
   ```bash
   cd tests/control_plane && python -m pytest -v
   ```
3. **Check code quality**:
   ```bash
   ruff check control_plane/
   ruff format control_plane/
   ```
4. **Push changes** - CI/CD runs tests automatically

## References

- [sageLLM Main README](../../README.md)
- [Integration Guide](../../docs/INTEGRATION.md)
- [Deployment Guide](../../docs/DEPLOYMENT.md)
