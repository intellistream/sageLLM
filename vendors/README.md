# Vendor Libraries

This directory contains third-party dependencies that are bundled with sageLLM.

## vllm/

Original vLLM source code from [vLLM Project](https://github.com/vllm-project/vllm).

- **License**: Apache 2.0
- **Purpose**: Provides the core LLM inference engine with CUDA optimizations
- **Status**: As-is copy from upstream vLLM repository

### Note

sageLLM treats vLLM as a black-box dependency. While we include the source code here for reference and CUDA compilation purposes, in production you should consider:

1. Using the pre-built vLLM package: `pip install vllm`
2. Separating this to an external dependency
3. Using a Docker image with pre-compiled vLLM

See the main README.md for more details on installation options.
