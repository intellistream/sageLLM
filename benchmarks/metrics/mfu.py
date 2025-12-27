# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""MFU (Model FLOPs Utilization) metric."""

from __future__ import annotations

from dataclasses import dataclass

from . import Metric, MetricRegistry, MetricType


@dataclass
class MFUResult:
    """MFU measurement result.

    Attributes:
        mfu: Model FLOPs Utilization (0-1)
        achieved_tflops: Achieved TFLOPS
        theoretical_tflops: Theoretical peak TFLOPS
        total_flops: Total FLOPs for the operation
        duration_s: Operation duration in seconds
    """

    mfu: float
    achieved_tflops: float
    theoretical_tflops: float
    total_flops: int
    duration_s: float


@MetricRegistry.register("mfu")
class MFUMetric(Metric[MFUResult]):
    """Measures Model FLOPs Utilization.

    MFU = Achieved TFLOPS / Theoretical Peak TFLOPS

    Reference: PaLM paper (https://arxiv.org/abs/2204.02311)

    Example:
        >>> metric = MFUMetric()
        >>> result = metric.compute(
        ...     num_tokens=1024,
        ...     num_layers=32,
        ...     hidden_size=4096,
        ...     duration_s=0.1,
        ...     gpu_tflops=312.0  # A100 80GB
        ... )
        >>> print(f"MFU: {result.mfu:.2%}")
    """

    @property
    def name(self) -> str:
        return "mfu"

    @property
    def unit(self) -> str:
        return "ratio"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COMPUTE

    def compute(
        self,
        num_tokens: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int | None = None,
        seq_len: int | None = None,
        duration_s: float = 1.0,
        gpu_tflops: float = 312.0,  # A100 80GB FP16
    ) -> MFUResult:
        """Compute MFU.

        Args:
            num_tokens: Number of tokens processed
            seq_len: Sequence length (for attention FLOPs, optional)
            num_layers: Number of transformer layers
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate size (default 4*hidden_size)
            duration_s: Operation duration in seconds
            gpu_tflops: GPU theoretical peak TFLOPS

        Returns:
            MFUResult with MFU and TFLOPS metrics
        """
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        # Attention FLOPs per token per layer
        # If seq_len is provided, include softmax + matmul cost; otherwise fall back to simplified formula
        if seq_len is not None and seq_len > 0:
            # QKV projections + output projection: ~8 * hidden_size^2
            proj_flops = 8 * hidden_size * hidden_size
            # Attention scores and weighted values: ~4 * seq_len * hidden_size
            attn_flops = 4 * seq_len * hidden_size
            attention_flops = proj_flops + attn_flops
        else:
            # Backward compatible simplified attention cost
            attention_flops = 4 * hidden_size * hidden_size

        # MLP FLOPs
        mlp_flops = 2 * hidden_size * intermediate_size

        flops_per_token_per_layer = attention_flops + mlp_flops

        # Total FLOPs
        total_flops = num_tokens * num_layers * flops_per_token_per_layer

        # Achieved TFLOPS
        achieved_tflops = (total_flops / duration_s) / 1e12 if duration_s > 0 else 0.0

        # MFU
        mfu = achieved_tflops / gpu_tflops if gpu_tflops > 0 else 0.0

        return MFUResult(
            mfu=mfu,
            achieved_tflops=achieved_tflops,
            theoretical_tflops=gpu_tflops,
            total_flops=total_flops,
            duration_s=duration_s,
        )

    @staticmethod
    def get_gpu_peak_tflops(gpu_name: str | None = None) -> float:
        """Get theoretical peak TFLOPS for common GPUs.

        Args:
            gpu_name: GPU name (auto-detect if None)

        Returns:
            Peak TFLOPS for FP16/BF16
        """
        # Common GPU peak TFLOPS (FP16/BF16)
        gpu_specs = {
            "A100": 312.0,  # A100 80GB
            "H100": 989.0,  # H100 80GB (Tensor Core)
            "V100": 125.0,  # V100 32GB
            "A40": 150.0,  # A40 48GB
            "L40": 181.0,  # L40 48GB
            "4090": 165.0,  # RTX 4090 24GB
        }

        if gpu_name:
            for key, tflops in gpu_specs.items():
                if key.lower() in gpu_name.lower():
                    return tflops

        # Try to auto-detect
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                for key, tflops in gpu_specs.items():
                    if key in gpu_name:
                        return tflops
        except ImportError:
            pass

        # Default to A100
        return 312.0
