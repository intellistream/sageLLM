"""
sageLLM Control Plane
A high-performance request scheduling and routing layer for vLLM
"""

from setuptools import find_packages, setup

setup(
    name="sage-control-plane",
    version="0.1.0",
    description="SAGE Control Plane for vLLM orchestration",
    author="SAGE Project",
    license="Apache License 2.0",
    packages=find_packages(include=["control_plane"]),
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.4.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
            "tblib>=1.7.0",
        ],
    },
)
