# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for autoscaler module."""

import pytest

from control_plane import (
    Autoscaler,
    AutoscalerConfig,
    LoadPredictor,
    MetricsCollector,
    SystemMetrics,
)


def test_load_predictor_constant():
    """Test constant load predictor."""
    predictor = LoadPredictor(predictor_type="constant", window_size=5)

    # Add some data points
    predictor.add_data_point(num_requests=10.0, avg_isl=1024.0, avg_osl=256.0)
    predictor.add_data_point(num_requests=12.0, avg_isl=1100.0, avg_osl=280.0)
    predictor.add_data_point(num_requests=11.0, avg_isl=1050.0, avg_osl=270.0)

    # Predict (should return most recent)
    num_req, isl, osl = predictor.predict()

    assert num_req == 11.0
    assert isl == 1050.0
    assert osl == 270.0


def test_load_predictor_moving_average():
    """Test moving average load predictor."""
    predictor = LoadPredictor(predictor_type="moving_average", window_size=3)

    # Add data points
    predictor.add_data_point(num_requests=10.0, avg_isl=1000.0, avg_osl=250.0)
    predictor.add_data_point(num_requests=12.0, avg_isl=1200.0, avg_osl=300.0)
    predictor.add_data_point(num_requests=14.0, avg_isl=1400.0, avg_osl=350.0)

    # Predict (should return average)
    num_req, isl, osl = predictor.predict()

    assert num_req == pytest.approx(12.0, rel=0.01)
    assert isl == pytest.approx(1200.0, rel=0.01)
    assert osl == pytest.approx(300.0, rel=0.01)


def test_system_metrics_is_valid():
    """Test SystemMetrics.is_valid()."""

    # Valid metrics
    metrics = SystemMetrics(
        num_requests=10.0,
        avg_isl=1024.0,
        avg_osl=256.0,
        avg_ttft_ms=150.0,
        avg_itl_ms=45.0,
    )
    assert metrics.is_valid()

    # Invalid: no requests
    metrics = SystemMetrics(
        num_requests=0.0,
        avg_isl=1024.0,
        avg_osl=256.0,
        avg_ttft_ms=150.0,
        avg_itl_ms=45.0,
    )
    assert not metrics.is_valid()

    # Invalid: missing TTFT
    metrics = SystemMetrics(
        num_requests=10.0,
        avg_isl=1024.0,
        avg_osl=256.0,
        avg_ttft_ms=None,
        avg_itl_ms=45.0,
    )
    assert not metrics.is_valid()


@pytest.mark.asyncio
async def test_autoscaler_initialization():
    """Test autoscaler initialization."""

    config = AutoscalerConfig(
        target_ttft_ms=200.0,
        target_itl_ms=50.0,
        adjustment_interval_sec=60,
        load_predictor_type="constant",
    )

    metrics_collector = MetricsCollector()

    # Mock callback
    scaling_decisions = []

    async def callback(decision):
        scaling_decisions.append(decision)

    autoscaler = Autoscaler(
        config=config,
        metrics_collector=metrics_collector,
        executor_callback=callback,
    )

    assert autoscaler.config.target_ttft_ms == 200.0
    assert autoscaler.config.target_itl_ms == 50.0
    assert not autoscaler.running

    # Check status
    status = autoscaler.get_status()
    assert "running" in status
    assert "config" in status
    assert "correction_factors" in status


@pytest.mark.asyncio
async def test_autoscaler_compute_scaling_decision():
    """Test scaling decision computation."""

    config = AutoscalerConfig(
        target_ttft_ms=200.0,
        target_itl_ms=50.0,
        min_prefill_instances=1,
        max_prefill_instances=5,
        min_decode_instances=1,
        max_decode_instances=10,
        max_gpu_budget=20,
        prefill_gpus_per_instance=4,
        decode_gpus_per_instance=1,
    )

    metrics_collector = MetricsCollector()
    scaling_decisions = []

    async def callback(decision):
        scaling_decisions.append(decision)

    autoscaler = Autoscaler(
        config=config,
        metrics_collector=metrics_collector,
        executor_callback=callback,
    )

    # Test scaling decision computation
    decision = autoscaler._compute_scaling_decision(
        next_num_req=20.0,  # 20 requests
        next_isl=1000.0,  # 1000 token input
        next_osl=200.0,  # 200 token output
    )

    assert decision.num_prefill_instances >= config.min_prefill_instances
    assert decision.num_prefill_instances <= config.max_prefill_instances
    assert decision.num_decode_instances >= config.min_decode_instances
    assert decision.num_decode_instances <= config.max_decode_instances

    # Check GPU budget constraint
    total_gpus = (
        decision.num_prefill_instances * config.prefill_gpus_per_instance
        + decision.num_decode_instances * config.decode_gpus_per_instance
    )
    assert total_gpus <= config.max_gpu_budget


@pytest.mark.asyncio
async def test_autoscaler_apply_constraints():
    """Test constraint application."""

    config = AutoscalerConfig(
        min_prefill_instances=1,
        max_prefill_instances=3,
        min_decode_instances=2,
        max_decode_instances=8,
        max_gpu_budget=12,
        prefill_gpus_per_instance=4,
        decode_gpus_per_instance=1,
    )

    metrics_collector = MetricsCollector()

    async def callback(decision):
        pass

    autoscaler = Autoscaler(
        config=config,
        metrics_collector=metrics_collector,
        executor_callback=callback,
    )

    # Test min constraints
    num_prefill, num_decode = autoscaler._apply_constraints(0, 0)
    assert num_prefill == 1  # min
    assert num_decode == 2  # min

    # Test max constraints
    num_prefill, num_decode = autoscaler._apply_constraints(10, 20)
    assert num_prefill == 3  # max
    assert num_decode == 8  # max

    # Test GPU budget constraint
    # Request 3 prefill (12 GPUs) + 5 decode (5 GPUs) = 17 GPUs > 12 budget
    num_prefill, num_decode = autoscaler._apply_constraints(3, 5)
    total_gpus = num_prefill * 4 + num_decode * 1
    assert total_gpus <= 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
