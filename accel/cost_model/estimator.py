# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Cost estimation for model inference.

Provides analytical cost modeling for:
- Latency: Compute time per token
- Memory: KV cache + activations + weights
- Throughput: Tokens/second under different batch sizes
"""

from __future__ import annotations

from dataclasses import dataclass

from ..quantize.base import QuantizationType


@dataclass
class ModelProfile:
    """Model architecture profile."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_length: int = 4096


@dataclass
class CostMetrics:
    """Inference cost metrics."""

    # Latency (ms)
    prefill_latency_per_token: float  # Time per token in prefill phase
    decode_latency_per_token: float  # Time per token in decode phase

    # Memory (GB)
    weight_memory: float  # Model weights
    kv_cache_memory: float  # KV cache for max_batch_size
    activation_memory: float  # Peak activation memory

    # Throughput
    max_throughput_tokens_per_sec: float  # Theoretical max throughput


class CostEstimator:
    """Analytical cost estimator for LLM inference.

    Example:
        >>> profile = ModelProfile(
        ...     num_layers=32, hidden_size=4096,
        ...     num_attention_heads=32, intermediate_size=11008,
        ...     vocab_size=32000
        ... )
        >>> estimator = CostEstimator(profile)
        >>> metrics = estimator.estimate(
        ...     batch_size=8, seq_length=2048,
        ...     quant_type=QuantizationType.INT8
        ... )
        >>> print(f"Weight memory: {metrics.weight_memory:.2f} GB")
    """

    def __init__(self, profile: ModelProfile):
        """Initialize.

        Args:
            profile: Model architecture profile
        """
        self.profile = profile

    def estimate(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        quant_type: QuantizationType = QuantizationType.FP16,
    ) -> CostMetrics:
        """Estimate inference cost.

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            quant_type: Weight quantization type

        Returns:
            Cost metrics
        """
        p = self.profile

        # Bytes per parameter
        bytes_per_param = self._get_bytes_per_param(quant_type)

        # 1. Weight memory
        num_params = self._count_params()
        weight_memory_gb = num_params * bytes_per_param / 1e9

        # 2. KV cache memory
        # Each layer stores K and V: [batch, num_heads, seq_len, head_dim]
        head_dim = p.hidden_size // p.num_attention_heads
        kv_size_per_layer = 2 * batch_size * p.num_attention_heads * seq_length * head_dim
        kv_cache_memory_gb = kv_size_per_layer * p.num_layers * 2 / 1e9  # FP16 (2 bytes)

        # 3. Activation memory (rough estimate)
        activation_memory_gb = batch_size * seq_length * p.hidden_size * 4 * 2 / 1e9  # 4 layers, FP16

        # 4. Latency
        # Simplified: latency ∝ FLOPs / (GPU compute throughput)
        # Assume A100 80GB: ~312 TFLOPS (FP16)
        gpu_tflops = 312.0

        # Prefill: full attention + MLP
        prefill_flops_per_token = (
            2 * p.num_layers * seq_length * p.hidden_size * p.hidden_size * 4  # Attention QKV + Output
            + 2 * p.num_layers * p.hidden_size * p.intermediate_size * 2  # MLP up + down
        )
        prefill_latency_ms = (prefill_flops_per_token / (gpu_tflops * 1e12)) * 1000

        # Decode: incremental attention + MLP
        decode_flops_per_token = (
            2 * p.num_layers * seq_length * p.hidden_size  # Attention (QK^T for cached K)
            + 2 * p.num_layers * p.hidden_size * p.intermediate_size * 2  # MLP
        )
        decode_latency_ms = (decode_flops_per_token / (gpu_tflops * 1e12)) * 1000

        # 5. Throughput (memory-bound in decode)
        # Simplified: limited by memory bandwidth
        # A100 80GB: ~2 TB/s memory bandwidth
        memory_bandwidth_gb_per_sec = 2000.0
        tokens_per_sec = memory_bandwidth_gb_per_sec / (weight_memory_gb / batch_size)

        return CostMetrics(
            prefill_latency_per_token=prefill_latency_ms,
            decode_latency_per_token=decode_latency_ms,
            weight_memory=weight_memory_gb,
            kv_cache_memory=kv_cache_memory_gb,
            activation_memory=activation_memory_gb,
            max_throughput_tokens_per_sec=tokens_per_sec,
        )

    def _count_params(self) -> int:
        """Count total parameters."""
        p = self.profile

        # Embedding
        embed_params = p.vocab_size * p.hidden_size

        # Transformer layers
        layer_params = (
            # Attention: QKV + Output
            4 * p.hidden_size * p.hidden_size
            # MLP: Up + Down
            + 2 * p.hidden_size * p.intermediate_size
            # LayerNorm
            + 2 * p.hidden_size
        )
        transformer_params = p.num_layers * layer_params

        # LM head
        lm_head_params = p.hidden_size * p.vocab_size

        return embed_params + transformer_params + lm_head_params

    def _get_bytes_per_param(self, quant_type: QuantizationType) -> float:
        """Get bytes per parameter for quantization type."""
        mapping = {
            QuantizationType.FP32: 4.0,
            QuantizationType.FP16: 2.0,
            QuantizationType.BF16: 2.0,
            QuantizationType.FP8_E4M3: 1.0,
            QuantizationType.FP8_E5M2: 1.0,
            QuantizationType.INT8: 1.0,
            QuantizationType.INT4: 0.5,
            QuantizationType.NF4: 0.5,
        }
        return mapping.get(quant_type, 2.0)
