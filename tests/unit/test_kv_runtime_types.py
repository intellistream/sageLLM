# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Unit tests for sageLLM kv_runtime types."""


from sage.common.components.sage_llm.sageLLM.kv_runtime.types import (
    AllocationResult,
    KVBlockInfo,
    KVCacheSchema,
    KVDataType,
    KVTier,
    MigrationPlan,
)


class TestKVTier:
    """Tests for KVTier enum."""

    def test_tier_values(self) -> None:
        """Test that all expected tiers exist."""
        assert KVTier.GPU.value == "gpu"
        assert KVTier.CPU.value == "cpu"
        assert KVTier.NVME.value == "nvme"


class TestKVCacheSchema:
    """Tests for KVCacheSchema dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating schema with minimal parameters."""
        schema = KVCacheSchema(
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        assert schema.num_layers == 32
        assert schema.block_size == 16
        assert schema.dtype == KVDataType.FP16
        assert schema.page_size_bytes is not None

    def test_page_size_calculation(self) -> None:
        """Test automatic page size calculation."""
        schema = KVCacheSchema(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
            dtype=KVDataType.FP16,
        )
        # 2 (K+V) * 32 layers * 32 heads * 128 dim * 16 tokens * 2 bytes
        expected = 2 * 32 * 32 * 128 * 16 * 2
        assert schema.page_size_bytes == expected

    def test_tokens_per_block(self) -> None:
        """Test tokens_per_block property."""
        schema = KVCacheSchema(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=32,
        )
        assert schema.tokens_per_block == 32

    def test_to_dict(self) -> None:
        """Test serialization."""
        schema = KVCacheSchema(
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        data = schema.to_dict()
        assert data["num_layers"] == 32
        assert data["dtype"] == "fp16"

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "num_layers": 40,
            "num_heads": 40,
            "head_dim": 128,
            "dtype": "fp8_e4m3",
        }
        schema = KVCacheSchema.from_dict(data)
        assert schema.num_layers == 40
        assert schema.dtype == KVDataType.FP8_E4M3

    def test_for_model_qwen(self) -> None:
        """Test creating schema for Qwen model."""
        schema = KVCacheSchema.for_model("Qwen/Qwen2.5-7B-Instruct")
        assert schema.num_layers == 28
        assert schema.num_heads == 28

    def test_for_model_llama(self) -> None:
        """Test creating schema for LLaMA model."""
        schema = KVCacheSchema.for_model("meta-llama/llama3-8b")
        assert schema.num_layers == 32
        assert schema.num_heads == 32

    def test_for_model_unknown(self) -> None:
        """Test creating schema for unknown model uses defaults."""
        schema = KVCacheSchema.for_model("unknown/model")
        assert schema.num_layers == 32


class TestKVBlockInfo:
    """Tests for KVBlockInfo dataclass."""

    def test_create(self) -> None:
        """Test creating block info."""
        info = KVBlockInfo(
            block_id=42,
            tier=KVTier.GPU,
            size_bytes=1024,
        )
        assert info.block_id == 42
        assert info.tier == KVTier.GPU
        assert info.ref_count == 1

    def test_to_dict(self) -> None:
        """Test serialization."""
        info = KVBlockInfo(
            block_id=1,
            tier=KVTier.CPU,
            is_prefix=True,
        )
        data = info.to_dict()
        assert data["block_id"] == 1
        assert data["tier"] == "cpu"
        assert data["is_prefix"] is True


class TestAllocationResult:
    """Tests for AllocationResult dataclass."""

    def test_success(self) -> None:
        """Test successful allocation."""
        result = AllocationResult(
            success=True,
            block_ids=[1, 2, 3],
            tier=KVTier.GPU,
            total_bytes=3072,
        )
        assert result.success is True
        assert len(result.block_ids) == 3

    def test_failure(self) -> None:
        """Test failed allocation."""
        result = AllocationResult(
            success=False,
            error_message="Out of memory",
        )
        assert result.success is False
        assert result.error_message == "Out of memory"


class TestMigrationPlan:
    """Tests for MigrationPlan dataclass."""

    def test_create(self) -> None:
        """Test creating migration plan."""
        plan = MigrationPlan(
            plan_id="migrate-1",
            src_tier=KVTier.GPU,
            dst_tier=KVTier.CPU,
            block_ids=[1, 2, 3],
            total_bytes=3072,
        )
        assert plan.src_tier == KVTier.GPU
        assert plan.dst_tier == KVTier.CPU
        assert len(plan.block_ids) == 3

    def test_to_dict(self) -> None:
        """Test serialization."""
        plan = MigrationPlan(
            plan_id="migrate-1",
            src_tier=KVTier.GPU,
            dst_tier=KVTier.CPU,
            block_ids=[1],
        )
        data = plan.to_dict()
        assert data["plan_id"] == "migrate-1"
        assert data["src_tier"] == "gpu"
        assert data["dst_tier"] == "cpu"
