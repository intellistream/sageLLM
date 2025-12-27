# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""LMDeploy engine backend for sageLLM.

This module provides integration with LMDeploy/TurboMind:
- LMDeployEngine: Main engine wrapper with dependency injection
- KVManager: Extended SequenceManager with sageLLM hooks
- Scheduler: IR/strategy injection support

Example:
    >>> from sageLLM.third_party_engines.lmdeploy import LMDeployEngine
    >>> engine = LMDeployEngine(config=EngineConfig(
    ...     model_id="Qwen/Qwen2.5-7B-Instruct",
    ... ))
    >>> await engine.start()
    >>> response = await engine.generate("Hello, world")
"""

from .. import register_engine
from .engine import LMDeployEngine

# Register the engine
register_engine("lmdeploy")(LMDeployEngine)

__all__ = [
    "LMDeployEngine",
]
