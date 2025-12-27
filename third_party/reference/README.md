# Third Party Reference Code

This directory contains reference implementations and documentation for studying existing inference engine designs. These are **NOT** direct dependencies - they serve as learning resources for our self-developed runtime.

## Reference Materials

### Inference Engines

| Project | Focus Areas | Status |
|---------|-------------|--------|
| vLLM | PagedAttention, continuous batching | Reference for KV cache management |
| LMDeploy | TurboMind engine, W4A16 quantization | Reference for kernel optimization |
| TensorRT-LLM | NVIDIA optimizations, FP8 | Reference for CUDA kernels |
| MLC-LLM | Multi-backend compilation | Reference for hardware abstraction |

### Research Papers

Key papers informing our design:

1. **PagedAttention** (vLLM) - KV cache memory management
2. **FlashAttention** - Memory-efficient attention
3. **Splitwise/DistServe** - PD disaggregation
4. **Mooncake** - KV cache-centric scheduling

## Usage Guidelines

1. **Do NOT copy code directly** - Study the design patterns and algorithms
2. **Document inspirations** - If implementing a similar approach, cite the source
3. **Maintain independence** - Our runtime should work without these codebases

## Directory Structure

```
reference/
├── README.md           # This file
├── notes/              # Design notes and analysis
│   ├── vllm-analysis.md
│   ├── lmdeploy-analysis.md
│   └── ...
└── papers/             # Paper summaries (not full papers due to copyright)
    ├── pagedattention.md
    └── ...
```

## Adding Reference Material

When adding new reference materials:

1. Create a summary document, not a code copy
2. Note the license of the original project
3. Focus on architectural insights, not implementation details
4. Link to original sources
