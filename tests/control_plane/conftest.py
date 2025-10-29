# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""pytest configuration for control_plane tests."""

import asyncio
import logging
import sys
from pathlib import Path

import pytest
from aioresponses import aioresponses

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up path for control_plane imports
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


@pytest.fixture
def event_loop():
    """Create an async event loop for tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_aiohttp():
    """Fixture providing mocked aiohttp responses for vLLM API."""
    with aioresponses() as m:
        yield m


@pytest.fixture
def mock_vllm_completion_response():
    """Fixture providing a sample vLLM completion response."""
    return {
        "id": "cmpl-test-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "meta-llama/Llama-2-7b",
        "choices": [
            {
                "text": " This is a test completion response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
async def control_plane_manager():  # noqa: E501
    """Fixture providing a control plane manager instance."""
    try:
        from control_plane import ControlPlaneManager  # noqa: F401, E402

        manager = ControlPlaneManager(
            scheduling_policy="adaptive",
            enable_pd_separation=True,
        )
        yield manager
        # Cleanup
        if hasattr(manager, "_running") and manager._running:
            await manager.stop()
    except ImportError as e:
        pytest.skip(f"Control plane not available: {e}")


@pytest.fixture
def sample_execution_instance():  # noqa: E501
    """Fixture providing a sample vLLM execution instance."""
    try:
        from control_plane import ExecutionInstance  # noqa: F401, E402

        return ExecutionInstance(
            instance_id="test-instance-1",
            host="localhost",
            port=8000,
            model_name="meta-llama/Llama-2-7b",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_count=1,
        )
    except ImportError as e:
        pytest.skip(f"Control plane not available: {e}")


@pytest.fixture
def sample_request_metadata():  # noqa: E501
    """Fixture providing a sample request metadata."""
    try:
        from control_plane import (
            RequestMetadata,  # noqa: F401, E402
            RequestPriority,
        )

        return RequestMetadata(
            request_id="test-req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
        )
    except ImportError as e:
        pytest.skip(f"Control plane not available: {e}")
