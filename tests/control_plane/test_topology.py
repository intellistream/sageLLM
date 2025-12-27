# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for topology detection and awareness."""

import pytest
from control_plane import ExecutionInstance, TopologyDetector


class TestTopologyDetector:
    """Test topology detection functionality."""

    def test_get_machine_id(self):
        """Test machine ID detection."""
        machine_id = TopologyDetector.get_machine_id()
        assert machine_id is not None
        assert isinstance(machine_id, str)
        assert len(machine_id) > 0

    def test_create_instance_with_topology(self):
        """Test creating instance with topology info (mock mode)."""
        instance = ExecutionInstance(
            instance_id="test-gpu-0",
            host="localhost",
            port=8000,
            model_name="test-model",
            machine_id="test-machine",
            gpu_device_id=0,
            nvlink_peers=["test-gpu-1", "test-gpu-2"],
            numa_node=0,
        )

        assert instance.machine_id == "test-machine"
        assert instance.gpu_device_id == 0
        assert len(instance.nvlink_peers) == 2
        assert instance.numa_node == 0

    def test_affinity_score_same_machine_nvlink(self):
        """Test affinity score for NVLINK connected GPUs."""
        instance1 = ExecutionInstance(
            instance_id="gpu-0",
            host="localhost",
            port=8000,
            model_name="test",
            machine_id="machine-1",
            nvlink_peers=["gpu-1"],
        )

        instance2 = ExecutionInstance(
            instance_id="gpu-1",
            host="localhost",
            port=8001,
            model_name="test",
            machine_id="machine-1",
        )

        # Same machine with NVLINK should score 1.0
        score = instance1.get_affinity_score(instance2)
        assert score == 1.0

    def test_affinity_score_same_machine_no_nvlink(self):
        """Test affinity score for same machine without NVLINK."""
        instance1 = ExecutionInstance(
            instance_id="gpu-0",
            host="localhost",
            port=8000,
            model_name="test",
            machine_id="machine-1",
        )

        instance2 = ExecutionInstance(
            instance_id="gpu-1",
            host="localhost",
            port=8001,
            model_name="test",
            machine_id="machine-1",
        )

        # Same machine without NVLINK should score 0.5
        score = instance1.get_affinity_score(instance2)
        assert score == 0.5

    def test_affinity_score_different_machines_same_rack(self):
        """Test affinity score for different machines, same rack."""
        instance1 = ExecutionInstance(
            instance_id="gpu-0",
            host="192.168.1.100",
            port=8000,
            model_name="test",
            machine_id="machine-1",
            rack_id="rack-1",
        )

        instance2 = ExecutionInstance(
            instance_id="gpu-1",
            host="192.168.1.101",
            port=8000,
            model_name="test",
            machine_id="machine-2",
            rack_id="rack-1",
        )

        # Same rack, different machines should score 0.1
        score = instance1.get_affinity_score(instance2)
        assert score == 0.1

    def test_affinity_score_different_racks(self):
        """Test affinity score for different racks."""
        instance1 = ExecutionInstance(
            instance_id="gpu-0",
            host="192.168.1.100",
            port=8000,
            model_name="test",
            machine_id="machine-1",
            rack_id="rack-1",
        )

        instance2 = ExecutionInstance(
            instance_id="gpu-1",
            host="192.168.2.100",
            port=8000,
            model_name="test",
            machine_id="machine-2",
            rack_id="rack-2",
        )

        # Different racks should score 0.01
        score = instance1.get_affinity_score(instance2)
        assert score == 0.01

    def test_is_local_to(self):
        """Test local instance detection."""
        instance1 = ExecutionInstance(
            instance_id="gpu-0",
            host="localhost",
            port=8000,
            model_name="test",
            machine_id="machine-1",
        )

        instance2 = ExecutionInstance(
            instance_id="gpu-1",
            host="localhost",
            port=8001,
            model_name="test",
            machine_id="machine-1",
        )

        instance3 = ExecutionInstance(
            instance_id="gpu-2",
            host="192.168.1.100",
            port=8000,
            model_name="test",
            machine_id="machine-2",
        )

        assert instance1.is_local_to(instance2) is True
        assert instance1.is_local_to(instance3) is False


@pytest.mark.skipif(True, reason="Requires nvidia-smi - enable for real GPU testing")
class TestTopologyDetectionReal:
    """Tests that require real GPU hardware (optional)."""

    def test_detect_nvlink_topology_real(self):
        """Test real NVLINK topology detection."""
        topology = TopologyDetector.detect_nvlink_topology()
        # This will only pass on systems with nvidia-smi
        assert isinstance(topology, dict)

    def test_detect_numa_nodes_real(self):
        """Test real NUMA detection."""
        numa_map = TopologyDetector.detect_numa_nodes()
        assert isinstance(numa_map, dict)

    def test_auto_detect_local_instances_real(self):
        """Test auto-detection of local instances."""
        instances = TopologyDetector.auto_detect_local_instances(
            base_instance_id="test-gpu",
            host="localhost",
            base_port=8000,
            model_name="test-model",
        )
        # This will only work on systems with GPUs
        assert isinstance(instances, list)
