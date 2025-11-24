# sageLLM Testing Guide

This guide explains how to run and contribute to the sageLLM test suite.

## Installation

### Install with Test Dependencies

```bash
cd packages/sage-common/src/sage/common/components/sage_llm/sageLLM

# Install in editable mode with test dependencies
pip install -e ".[test]"

# Or install all development dependencies
pip install -e ".[dev]"
```

### Install for Development

```bash
# Install with all optional dependencies
pip install -e ".[all]"
```

## Test Structure

The test suite is organized as follows:

```
tests/
├── control_plane/
│   ├── conftest.py              # Shared fixtures and configuration
│   ├── test_executor.py         # Executor tests
│   ├── test_fault_tolerance.py  # Fault tolerance tests
│   ├── test_integration.py      # Integration tests
│   ├── test_monitoring.py       # Monitoring tests
│   ├── test_parallelism.py      # NEW: Parallelism optimizer tests
│   ├── test_pd_separation.py    # Prefilling/Decoding separation tests
│   ├── test_router.py           # NEW: Router and load balancer tests
│   ├── test_scheduling.py       # Scheduling tests
│   ├── test_strategies.py       # NEW: Scheduling strategy tests
│   ├── test_topology.py         # Topology-aware tests
│   └── test_types.py            # NEW: Type and dataclass tests
```

## Running Tests

### Run All Tests

```bash
cd packages/sage-common/src/sage/common/components/sage_llm/sageLLM
pytest
```

### Run with Coverage

```bash
pytest --cov=control_plane --cov-report=html --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Test routing functionality
pytest tests/control_plane/test_router.py -v

# Test scheduling strategies
pytest tests/control_plane/test_strategies.py -v

# Test parallelism optimization
pytest tests/control_plane/test_parallelism.py -v

# Test data types
pytest tests/control_plane/test_types.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/control_plane/test_router.py::TestRequestRouter -v

# Run a specific test method
pytest tests/control_plane/test_router.py::TestRequestRouter::test_load_balanced_routing -v
```

### Run Tests by Marker

```bash
# Run only async tests
pytest -m asyncio

# Run integration tests
pytest -m integration

# Run scheduling tests
pytest -m scheduling
```

### Run Tests in Parallel

```bash
# Use multiple CPUs for faster execution
pytest -n auto
```

## CI/CD Integration

The test suite automatically runs on:

- **Pull Requests** to `main` or `main-dev` branches
- **Push** events to `main` or `main-dev` branches
- **Manual workflow dispatch**

### Workflow Features

The GitHub Actions workflow (`.github/workflows/sagellm-tests.yml`) includes:

1. **Unit Tests** - Full test suite with coverage reporting
1. **Code Quality** - Black, isort, Ruff, and MyPy checks
1. **Integration Tests** - End-to-end integration tests
1. **Multi-Python Support** - Tests run on Python 3.10 and 3.11

### Coverage Requirements

- **Minimum coverage**: 70%
- Coverage reports are uploaded to Codecov
- HTML coverage reports are available as artifacts

## Test Coverage

### Current Test Coverage

The new test files provide comprehensive coverage for:

#### 1. Router Module (`test_router.py`)

- ✅ All routing strategies (load_balanced, round_robin, random, affinity, locality, topology_aware)
- ✅ LoadBalancer algorithms (weighted_round_robin, least_connections, least_response_time,
  power_of_two)
- ✅ Request routing with no available instances
- ✅ Affinity management and clearing
- ✅ Latency and request tracking

#### 2. Parallelism Module (`test_parallelism.py`)

- ✅ ParallelismConfig validation
- ✅ Tensor Parallel Strategy
- ✅ Pipeline Parallel Strategy
- ✅ Data Parallel Strategy
- ✅ Expert Parallel Strategy
- ✅ Hybrid Parallel Strategy
- ✅ ParallelismOptimizer selection and comparison
- ✅ Different GPU count scenarios

#### 3. Types Module (`test_types.py`)

- ✅ All Enum types (RequestPriority, RequestStatus, ParallelismType, ExecutionInstanceType)
- ✅ RequestMetadata with latency calculations
- ✅ ExecutionInstance with capacity and affinity calculations
- ✅ SchedulingDecision creation
- ✅ PerformanceMetrics
- ✅ PrefillingConfig and DecodingConfig

#### 4. Strategies Module (`test_strategies.py`)

- ✅ FIFO Policy
- ✅ Priority Policy
- ✅ SLO-Aware Policy
- ✅ Cost-Optimized Policy
- ✅ Adaptive Policy
- ✅ Integration tests for all policies

## Writing New Tests

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_is_being_tested>`

### Example Test

```python
import pytest
from control_plane import ExecutionInstance

def test_instance_capacity():
    """Test instance capacity calculation."""
    instance = ExecutionInstance(
        instance_id="test-1",
        host="localhost",
        port=8000,
        model_name="llama-7b",
        current_load=0.3,
    )

    assert instance.available_capacity == 0.7
```

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used across tests:

```python
def test_with_fixture(sample_instances):
    """Test using fixture."""
    assert len(sample_instances) > 0
```

### Async Tests

For async tests, use the `@pytest.mark.asyncio` decorator:

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await some_async_function()
    assert result is not None
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're in the correct directory and the package is installed:

```bash
cd packages/sage-common/src/sage/common/components/sage_llm/sageLLM
pip install -e .
```

### Missing Dependencies

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Async Test Failures

Make sure `pytest-asyncio` is installed and configured:

```bash
pip install pytest-asyncio>=0.21.0
```

## Code Quality Checks

### Run All Quality Checks

```bash
# Format code
black control_plane/ tests/

# Sort imports
isort --profile=black control_plane/ tests/

# Lint code
ruff check control_plane/ tests/

# Type check
mypy control_plane/ --ignore-missing-imports
```

### Auto-fix Issues

```bash
# Auto-format
black control_plane/ tests/
isort --profile=black control_plane/ tests/

# Auto-fix linting issues
ruff check --fix control_plane/ tests/
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach recommended)
1. **Ensure >70% coverage** for new code
1. **Add docstrings** to test functions explaining what is being tested
1. **Use descriptive test names** that clearly indicate the test purpose
1. **Run all tests** before submitting PR
1. **Check code quality** with Black, isort, and Ruff

## Test Performance

### Benchmark Tests

To run tests with timing information:

```bash
pytest --durations=10
```

### Profile Tests

To profile slow tests:

```bash
pytest --profile
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [sageLLM Control Plane Documentation](docs/)

## Support

For issues or questions about testing:

- Check existing test files for examples
- Review pytest documentation
- Open an issue in the SAGE repository
