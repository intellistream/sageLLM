# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for Control Plane parallelism strategies."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane import (  # noqa: E402
    ExecutionInstance,
    ParallelismType,
    RequestMetadata,
    RequestPriority,
)
from control_plane.parallelism import (  # noqa: E402
    DataParallelStrategy,
    ExpertParallelStrategy,
    HybridParallelStrategy,
    ParallelismConfig,
    ParallelismOptimizer,
    PipelineParallelStrategy,
    TensorParallelStrategy,
)


@pytest.fixture
def sample_instance():
    """Create a sample execution instance."""
    return ExecutionInstance(
        instance_id="instance-1",
        host="localhost",
        port=8000,
        model_name="llama-70b",
        tensor_parallel_size=1,
        gpu_count=8,
        gpu_memory_gb=80.0,
    )


@pytest.fixture
def sample_request():
    """Create a sample request."""
    return RequestMetadata(
        request_id="req-1",
        priority=RequestPriority.NORMAL,
        max_tokens=512,
        model_name="llama-70b",
    )


class TestParallelismConfig:
    """Tests for ParallelismConfig."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            total_gpus=8,
        )
        assert config.validate() is True

    def test_validate_invalid_config(self):
        """Test validation of invalid configuration."""
        config = ParallelismConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=4,
            data_parallel_size=2,
            total_gpus=8,  # 4*4*2 = 32 > 8
        )
        assert config.validate() is False

    def test_total_parallel_size(self):
        """Test total parallel size calculation."""
        config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            expert_parallel_size=2,
        )
        assert config.total_parallel_size == 16  # 2*2*2*2

    def test_minimal_config(self):
        """Test minimal configuration."""
        config = ParallelismConfig(total_gpus=1)
        assert config.validate() is True
        assert config.total_parallel_size == 1


class TestTensorParallelStrategy:
    """Tests for TensorParallelStrategy."""

    def test_optimize_single_gpu(self, sample_request, sample_instance):
        """Test optimization for single GPU."""
        strategy = TensorParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=1)

        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.data_parallel_size == 1

    def test_optimize_power_of_two(self, sample_request, sample_instance):
        """Test optimization uses power of 2 for TP size."""
        strategy = TensorParallelStrategy()

        # 8 GPUs -> TP=8
        config = strategy.optimize(sample_request, sample_instance, available_gpus=8)
        assert config.tensor_parallel_size == 8

        # 7 GPUs -> TP=4 (largest power of 2)
        config = strategy.optimize(sample_request, sample_instance, available_gpus=7)
        assert config.tensor_parallel_size == 4

        # 16 GPUs -> TP=16
        config = strategy.optimize(sample_request, sample_instance, available_gpus=16)
        assert config.tensor_parallel_size == 16

    def test_estimate_performance(self, sample_request):
        """Test performance estimation."""
        strategy = TensorParallelStrategy()
        config = ParallelismConfig(tensor_parallel_size=8, total_gpus=8)

        perf = strategy.estimate_performance(config, sample_request)

        assert "latency_ms" in perf
        assert "throughput_tokens_per_sec" in perf
        assert "gpu_utilization" in perf
        assert perf["latency_ms"] > 0
        assert perf["throughput_tokens_per_sec"] > 0
        assert 0 < perf["gpu_utilization"] <= 1


class TestPipelineParallelStrategy:
    """Tests for PipelineParallelStrategy."""

    def test_optimize_small_scale(self, sample_request, sample_instance):
        """Test optimization for small scale."""
        strategy = PipelineParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=2)

        assert config.pipeline_parallel_size == 2
        assert config.tensor_parallel_size == 1

    def test_optimize_capped_at_four(self, sample_request, sample_instance):
        """Test pipeline parallel size is capped at 4."""
        strategy = PipelineParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=16)

        assert config.pipeline_parallel_size == 4

    def test_estimate_performance(self, sample_request):
        """Test performance estimation with bubble overhead."""
        strategy = PipelineParallelStrategy()
        config = ParallelismConfig(pipeline_parallel_size=4, total_gpus=4)

        perf = strategy.estimate_performance(config, sample_request)

        assert "latency_ms" in perf
        assert perf["latency_ms"] > 100  # Should have overhead


class TestDataParallelStrategy:
    """Tests for DataParallelStrategy."""

    def test_optimize_uses_all_gpus(self, sample_request, sample_instance):
        """Test data parallelism uses all available GPUs."""
        strategy = DataParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=8)

        assert config.data_parallel_size == 8
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1

    def test_estimate_performance_scales(self, sample_request):
        """Test performance scales with data parallelism."""
        strategy = DataParallelStrategy()

        config_1 = ParallelismConfig(data_parallel_size=1, total_gpus=1)
        perf_1 = strategy.estimate_performance(config_1, sample_request)

        config_8 = ParallelismConfig(data_parallel_size=8, total_gpus=8)
        perf_8 = strategy.estimate_performance(config_8, sample_request)

        # Throughput should scale (not perfectly linear due to 0.95 factor)
        assert perf_8["throughput_tokens_per_sec"] > perf_1["throughput_tokens_per_sec"]


class TestExpertParallelStrategy:
    """Tests for ExpertParallelStrategy."""

    def test_optimize_moe_model(self, sample_request, sample_instance):
        """Test optimization for MoE models."""
        strategy = ExpertParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=8)

        assert config.expert_parallel_size == 8
        assert config.tensor_parallel_size == 1

    def test_optimize_capped_at_eight(self, sample_request, sample_instance):
        """Test expert parallel size is capped at 8."""
        strategy = ExpertParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=16)

        assert config.expert_parallel_size == 8

    def test_estimate_performance(self, sample_request):
        """Test performance estimation."""
        strategy = ExpertParallelStrategy()
        config = ParallelismConfig(expert_parallel_size=8, total_gpus=8)

        perf = strategy.estimate_performance(config, sample_request)

        assert perf["latency_ms"] > 100  # Has routing overhead


class TestHybridParallelStrategy:
    """Tests for HybridParallelStrategy."""

    def test_optimize_large_scale(self, sample_request, sample_instance):
        """Test hybrid strategy for large scale (16+ GPUs)."""
        strategy = HybridParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=16)

        # Should use TP + PP + DP
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 2
        assert config.data_parallel_size == 2
        assert config.total_parallel_size == 16

    def test_optimize_medium_scale(self, sample_request, sample_instance):
        """Test hybrid strategy for medium scale (8-15 GPUs)."""
        strategy = HybridParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=8)

        # Should use TP + DP
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 1
        assert config.data_parallel_size == 2
        assert config.total_parallel_size == 8

    def test_optimize_small_scale(self, sample_request, sample_instance):
        """Test hybrid strategy for small scale (4-7 GPUs)."""
        strategy = HybridParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=4)

        # Should use TP only
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 1
        assert config.data_parallel_size == 1

    def test_optimize_minimal_scale(self, sample_request, sample_instance):
        """Test hybrid strategy for minimal scale (1-3 GPUs)."""
        strategy = HybridParallelStrategy()
        config = strategy.optimize(sample_request, sample_instance, available_gpus=2)

        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 1
        assert config.data_parallel_size == 1

    def test_estimate_performance_combined_overhead(self, sample_request):
        """Test performance estimation combines overheads."""
        strategy = HybridParallelStrategy()
        config = ParallelismConfig(
            tensor_parallel_size=4, pipeline_parallel_size=2, data_parallel_size=2
        )

        perf = strategy.estimate_performance(config, sample_request)

        # Should have combined overhead from TP and PP
        assert perf["latency_ms"] > 100


class TestParallelismOptimizer:
    """Tests for ParallelismOptimizer."""

    def test_select_strategy_with_hint(self, sample_request, sample_instance):
        """Test strategy selection with parallelism hint."""
        optimizer = ParallelismOptimizer()

        request = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.NORMAL,
            max_tokens=512,
            parallelism_hint=ParallelismType.TENSOR_PARALLEL,
        )

        strategy, config = optimizer.select_strategy(request, sample_instance, 8)

        assert isinstance(strategy, TensorParallelStrategy)
        assert config.tensor_parallel_size > 1

    def test_select_strategy_large_scale(self, sample_request, sample_instance):
        """Test automatic strategy selection for large scale."""
        optimizer = ParallelismOptimizer()

        strategy, config = optimizer.select_strategy(sample_request, sample_instance, 16)

        assert isinstance(strategy, HybridParallelStrategy)

    def test_select_strategy_medium_scale(self, sample_request, sample_instance):
        """Test automatic strategy selection for medium scale."""
        optimizer = ParallelismOptimizer()

        strategy, config = optimizer.select_strategy(sample_request, sample_instance, 4)

        assert isinstance(strategy, TensorParallelStrategy)

    def test_select_strategy_small_scale(self, sample_request, sample_instance):
        """Test automatic strategy selection for small scale."""
        optimizer = ParallelismOptimizer()

        strategy, config = optimizer.select_strategy(sample_request, sample_instance, 2)

        assert isinstance(strategy, DataParallelStrategy)

    def test_select_strategy_single_gpu(self, sample_request, sample_instance):
        """Test automatic strategy selection for single GPU."""
        optimizer = ParallelismOptimizer()

        strategy, config = optimizer.select_strategy(sample_request, sample_instance, 1)

        assert isinstance(strategy, TensorParallelStrategy)
        assert config.tensor_parallel_size == 1

    def test_compare_strategies(self, sample_request, sample_instance):
        """Test comparing all strategies."""
        optimizer = ParallelismOptimizer()

        results = optimizer.compare_strategies(sample_request, sample_instance, 8)

        # Should have results for all strategy types
        assert ParallelismType.TENSOR_PARALLEL in results
        assert ParallelismType.PIPELINE_PARALLEL in results
        assert ParallelismType.DATA_PARALLEL in results
        assert ParallelismType.EXPERT_PARALLEL in results
        assert ParallelismType.HYBRID in results

        # Each result should have performance metrics
        for _strategy_type, perf in results.items():
            assert "latency_ms" in perf
            assert "throughput_tokens_per_sec" in perf
            assert "gpu_utilization" in perf
            assert "config" in perf

    def test_all_strategies_available(self):
        """Test all strategy types are available."""
        optimizer = ParallelismOptimizer()

        expected_strategies = [
            ParallelismType.TENSOR_PARALLEL,
            ParallelismType.PIPELINE_PARALLEL,
            ParallelismType.DATA_PARALLEL,
            ParallelismType.EXPERT_PARALLEL,
            ParallelismType.HYBRID,
        ]

        for strategy_type in expected_strategies:
            assert strategy_type in optimizer.strategies


class TestParallelismIntegration:
    """Integration tests for parallelism optimization."""

    def test_end_to_end_optimization(self, sample_instance):
        """Test complete optimization flow."""
        optimizer = ParallelismOptimizer()

        # Create request with different requirements
        request = RequestMetadata(
            request_id="req-1",
            priority=RequestPriority.CRITICAL,
            max_tokens=2048,
            model_name="llama-70b",
        )

        # Optimize for 8 GPUs
        strategy, config = optimizer.select_strategy(request, sample_instance, 8)

        # Verify configuration is valid
        assert config.validate() is True
        assert config.total_gpus == 8

        # Estimate performance
        perf = strategy.estimate_performance(config, request)
        assert perf["latency_ms"] > 0
        assert perf["throughput_tokens_per_sec"] > 0

    def test_different_gpu_counts(self, sample_request, sample_instance):
        """Test optimization across different GPU counts."""
        optimizer = ParallelismOptimizer()

        gpu_counts = [1, 2, 4, 8, 16, 32]

        for gpu_count in gpu_counts:
            strategy, config = optimizer.select_strategy(sample_request, sample_instance, gpu_count)

            # Configuration should be valid
            assert config.validate() is True

            # Should not exceed available GPUs
            assert config.total_parallel_size <= gpu_count

            # Performance should be estimable
            perf = strategy.estimate_performance(config, sample_request)
            assert all(v > 0 for v in perf.values() if isinstance(v, (int, float)))
