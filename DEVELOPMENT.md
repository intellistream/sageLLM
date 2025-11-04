# sageLLM Control Plane - Development Setup Guide

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd packages/sage-common/src/sage/common/components/sage_vllm/sageLLM

# Install with all development dependencies
pip install -e ".[all]"
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test the installation
pre-commit run --all-files
```

This will automatically run code quality checks before every commit.

## 📦 Dependency Groups

Install only what you need:

```bash
# Testing only
pip install -e ".[test]"

# Code quality tools only
pip install -e ".[quality]"

# Development (includes testing + quality + pre-commit)
pip install -e ".[dev]"

# Documentation tools
pip install -e ".[docs]"

# Everything
pip install -e ".[all]"
```

## 🔍 Code Quality Checks

### Pre-commit (Recommended)

Pre-commit runs automatically before each commit. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files

# Skip hooks temporarily
git commit --no-verify

# Update hook versions
pre-commit autoupdate
```

### Manual Checks

If you prefer to run tools manually:

```bash
# Format code with Ruff
ruff format control_plane/ tests/

# Lint with Ruff (with auto-fix)
ruff check --fix control_plane/ tests/

# Format with Black (alternative)
black control_plane/ tests/

# Sort imports with isort
isort control_plane/ tests/

# Type check with MyPy
mypy control_plane/ --ignore-missing-imports
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=control_plane --cov-report=html --cov-report=term

# Run specific test file
pytest tests/control_plane/test_router.py

# Run specific test class
pytest tests/control_plane/test_router.py::TestRequestRouter

# Run specific test method
pytest tests/control_plane/test_router.py::TestRequestRouter::test_basic_routing

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto

# Run with specific timeout
pytest --timeout=60
```

### Test Coverage

```bash
# Generate HTML coverage report
pytest --cov=control_plane --cov-report=html
# Open htmlcov/index.html in browser

# Terminal coverage report
pytest --cov=control_plane --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=control_plane --cov-fail-under=70
```

### Integration Tests

```bash
# Run only integration tests (requires GPU/vLLM)
pytest -m integration

# Skip integration tests
pytest -m "not integration"
```

## 🔧 Pre-commit Hook Details

### What Gets Checked

1. **File Quality**

   - Trailing whitespace
   - End-of-file fixer
   - YAML/JSON/TOML syntax
   - Large files (>1MB)
   - Merge conflicts
   - Private keys/secrets

1. **Python Code Quality**

   - Ruff linting and formatting (replaces Black + isort + flake8)
   - MyPy type checking (warnings only)
   - No print() statements in control_plane code
   - Import verification

1. **Shell Scripts**

   - Shellcheck validation

1. **Documentation**

   - YAML formatting
   - Markdown formatting (mdformat)

### Excluded from Checks

- `vendors/` directory (vLLM submodule)
- `build/` directory
- `__pycache__/` directories
- Generated files

### Custom sageLLM Checks

1. **Test Naming Convention**: Test files must match `tests/**/test_*.py`
1. **No Print Statements**: Use logging instead of print() in control_plane
1. **Import Validation**: Verify all Python files can be imported

## 🛠️ Development Workflow

### Typical Development Cycle

```bash
# 1. Make your changes
vim control_plane/router.py

# 2. Run tests to verify
pytest tests/control_plane/test_router.py -v

# 3. Check code quality (optional - pre-commit will do this)
ruff check --fix control_plane/
ruff format control_plane/

# 4. Commit changes (pre-commit runs automatically)
git add control_plane/router.py tests/
git commit -m "feat: add new routing strategy"

# If pre-commit fails:
# - Review the errors
# - Fix them or let auto-fix handle it
# - Stage the fixes: git add .
# - Commit again
```

### Skipping Hooks

Sometimes you need to skip hooks (use sparingly):

```bash
# Skip all hooks
git commit --no-verify -m "WIP: debugging"

# Skip specific hook via environment
SKIP=mypy git commit -m "commit without type checking"

# Skip multiple hooks
SKIP=mypy,ruff git commit -m "quick fix"
```

## 📊 CI/CD Integration

The CI workflow (`.github/workflows/tests.yml`) runs the same checks:

1. **Code Quality Job**

   - Runs pre-commit on all files
   - Runs Black, isort, Ruff, MyPy

1. **Test Job** (Python 3.10, 3.11, 3.12)

   - Runs pytest with coverage
   - Uploads coverage to Codecov

1. **Integration Tests** (on push only)

   - Tests with real vLLM instances

### Local Pre-flight Check

Before pushing, run what CI will run:

```bash
# Code quality checks
pre-commit run --all-files

# Tests with coverage
pytest --cov=control_plane --cov-fail-under=70

# Verify on all Python versions (requires pyenv)
tox  # If tox.ini is configured
```

## 🐛 Troubleshooting

### Pre-commit Issues

**Hook fails with "command not found"**

```bash
# Reinstall pre-commit environment
pre-commit clean
pre-commit install --install-hooks
```

**Hooks run on vendors directory**

```bash
# Update .pre-commit-config.yaml to exclude vendors/
# This is already configured, but check if it was modified
```

**MyPy errors**

```bash
# MyPy is in warning mode and won't block commits
# To fix type errors:
pip install types-aiofiles types-requests
mypy control_plane/ --ignore-missing-imports
```

### Test Issues

**Import errors in tests**

```bash
# Reinstall in editable mode
pip uninstall sagellm
pip install -e ".[test]"
```

**Timeout errors**

```bash
# Increase timeout
pytest --timeout=300

# Or disable timeout
pytest --timeout=0
```

**Coverage too low**

```bash
# See which lines are not covered
pytest --cov=control_plane --cov-report=html
open htmlcov/index.html

# Add tests for uncovered code
```

### Ruff vs Black Conflicts

We use Ruff format as primary formatter (Black is legacy):

```bash
# Use Ruff format
ruff format control_plane/

# If you must use Black
black control_plane/

# They should be compatible, but Ruff is preferred
```

## 📚 Additional Resources

- [Pre-commit Documentation](https://pre-commit.com)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org)
- [MyPy Documentation](https://mypy.readthedocs.io)

## 🔗 Related Files

- `.pre-commit-config.yaml` - Pre-commit configuration
- `.github/workflows/tests.yml` - CI/CD workflow
- `pyproject.toml` - Project configuration and dependencies
- `tests/README.md` - Testing guide
- `.github/README.md` - CI/CD documentation
