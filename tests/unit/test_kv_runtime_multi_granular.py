# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Unit tests for kv_runtime multi-granularity KV management.

Tests cover:
- MultiGranularKVPool: Block/Token/Head/Layer granularity allocation
- TieredKVStorage: HBM/DDR/NVMe three-tier storage
- HotColdClassifier: Temperature-based block classification
- KVMigrator: Cross-tier migration
- CrossRequestKVCache: Prefix-based KV reuse
"""

from __future__ import annotations

import os
import tempfile
import time

# Import modules using absolute paths
from sage.common.components.sage_llm.sageLLM.kv_runtime.blocks.multi_granular import (
    KVBlockDescriptor,
    KVGranularity,
    KVPoolConfig,
    MultiGranularKVPool,
    StorageTier,
)
from sage.common.components.sage_llm.sageLLM.kv_runtime.hierarchy.tiered_storage import (
    TierConfig,
    TieredKVStorage,
)
from sage.common.components.sage_llm.sageLLM.kv_runtime.migration.hot_cold import (
    HotColdClassifier,
    KVMigrator,
    MigrationPlan,
)
from sage.common.components.sage_llm.sageLLM.kv_runtime.reuse.cross_request import (
    CrossRequestKVCache,
)


class TestMultiGranularKVPool:
    """Tests for MultiGranularKVPool."""

    def test_allocate_block_granularity(self):
        """Test block-level granularity allocation."""
        config = KVPoolConfig(block_size=16)
        pool = MultiGranularKVPool(config)

        blocks = pool.allocate(
            sequence_id=1,
            request_id="req_1",
            num_tokens=64,
            layer_ids=[0, 1, 2],
        )

        # 64 tokens / 16 tokens per block = 4 blocks
        assert len(blocks) == 4
        assert all(b.granularity.name == "BLOCK" for b in blocks)

    def test_allocate_token_granularity(self):
        """Test token-level granularity allocation."""
        config = KVPoolConfig()
        pool = MultiGranularKVPool(config)

        blocks = pool.allocate(
            sequence_id=1,
            request_id="req_1",
            num_tokens=10,
            layer_ids=[0],
            granularity=KVGranularity.TOKEN,
        )

        # Token granularity: 1 block per token
        assert len(blocks) == 10
        assert all(b.granularity == KVGranularity.TOKEN for b in blocks)

    def test_allocate_head_granularity(self):
        """Test head-level granularity allocation."""
        config = KVPoolConfig()
        pool = MultiGranularKVPool(config)

        blocks = pool.allocate(
            sequence_id=1,
            request_id="req_1",
            num_tokens=64,
            layer_ids=[0, 1],
            head_ids=[0, 1, 2, 3],
            granularity=KVGranularity.HEAD,
        )

        # HEAD granularity: 1 block total
        assert len(blocks) == 1
        assert blocks[0].head_ids == [0, 1, 2, 3]

    def test_deallocate(self):
        """Test block deallocation."""
        pool = MultiGranularKVPool(KVPoolConfig())
        blocks = pool.allocate(1, "req_1", 16, [0])

        assert pool.get_stats()["total_blocks"] == 1

        pool.deallocate(blocks)

        assert pool.get_stats()["total_blocks"] == 0

    def test_get_blocks_by_sequence(self):
        """Test retrieving blocks by sequence ID."""
        pool = MultiGranularKVPool(KVPoolConfig(block_size=16))

        # Allocate for two sequences
        pool.allocate(1, "req_1", 32, [0])  # 2 blocks
        pool.allocate(2, "req_2", 48, [0])  # 3 blocks

        seq1_blocks = pool.get_blocks_by_sequence(1)
        seq2_blocks = pool.get_blocks_by_sequence(2)

        assert len(seq1_blocks) == 2
        assert len(seq2_blocks) == 3

    def test_update_block_tier(self):
        """Test updating block storage tier."""
        pool = MultiGranularKVPool(KVPoolConfig())
        blocks = pool.allocate(1, "req_1", 16, [0])
        block = blocks[0]

        assert block.tier == StorageTier.HBM

        result = pool.update_block_tier(block.block_id, StorageTier.DDR)

        assert result is True
        assert block.tier == StorageTier.DDR

    def test_tier_usage_tracking(self):
        """Test tier usage statistics."""
        pool = MultiGranularKVPool(KVPoolConfig())
        pool.allocate(1, "req_1", 16, [0])

        usage = pool.get_tier_usage(StorageTier.HBM)

        assert usage["num_blocks"] == 1
        assert usage["tier"] == "HBM"


class TestKVBlockDescriptor:
    """Tests for KVBlockDescriptor."""

    def test_update_access(self):
        """Test access statistics update."""
        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )

        initial_count = block.access_count
        block.update_access()

        assert block.access_count == initial_count + 1
        assert block.last_access_time > 0

    def test_num_tokens_property(self):
        """Test num_tokens calculation."""
        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0, 1, 2],
            head_ids=[0, 1],
            token_range=(10, 26),
            sequence_id=1,
            request_id="req_1",
        )

        assert block.num_tokens == 16
        assert block.num_layers == 3
        assert block.num_heads == 2


class TestTieredKVStorage:
    """Tests for TieredKVStorage."""

    def test_initialization(self):
        """Test storage initialization."""
        storage = TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,  # 1 MB
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,  # 4 MB
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )

        assert StorageTier.HBM in storage.backends
        assert StorageTier.DDR in storage.backends

        storage.close()

    def test_tier_usage(self):
        """Test tier usage statistics."""
        storage = TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )

        usage = storage.get_tier_usage(StorageTier.HBM)

        assert usage.capacity_bytes == 1024 * 1024
        assert usage.utilization == 0.0

        storage.close()

    def test_write_and_read_block(self):
        """Test writing and reading a block."""
        storage = TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
            tier=StorageTier.HBM,
        )

        # Write data
        test_data = b"test_kv_data_1234567890"
        storage.write_block(block, test_data)

        # Read data back
        read_data = storage.read_block(block)

        assert read_data == test_data

        storage.close()

    def test_migrate_block(self):
        """Test migrating a block between tiers."""
        storage = TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
            tier=StorageTier.HBM,
        )

        # Write initial data
        test_data = b"migration_test_data"
        storage.write_block(block, test_data)

        # Migrate to DDR
        result = storage.migrate_block(block, StorageTier.DDR)

        assert result is True
        assert block.tier == StorageTier.DDR

        # Verify data is still accessible
        read_data = storage.read_block(block)
        assert read_data == test_data

        storage.close()

    def test_nvme_backend(self):
        """Test NVMe storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nvme_path = os.path.join(tmpdir, "kv_cache.bin")

            storage = TieredKVStorage(
                hbm_config=TierConfig(
                    tier=StorageTier.HBM,
                    capacity_bytes=1024 * 1024,
                    bandwidth_gbps=900.0,
                    latency_us=1.0,
                    device_id=0,
                ),
                ddr_config=TierConfig(
                    tier=StorageTier.DDR,
                    capacity_bytes=4 * 1024 * 1024,
                    bandwidth_gbps=50.0,
                    latency_us=100.0,
                ),
                nvme_config=TierConfig(
                    tier=StorageTier.NVME,
                    capacity_bytes=16 * 1024 * 1024,
                    bandwidth_gbps=3.0,
                    latency_us=10000.0,
                    path=nvme_path,
                ),
            )

            assert StorageTier.NVME in storage.backends

            storage.close()


class TestHotColdClassifier:
    """Tests for HotColdClassifier."""

    def test_classify_hot(self):
        """Test hot block classification."""
        classifier = HotColdClassifier()

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        # High frequency = hot
        block.access_frequency = 2.0

        assert classifier.classify(block) == "hot"

    def test_classify_cold_by_timeout(self):
        """Test cold classification by timeout."""
        classifier = HotColdClassifier(cold_timeout_s=1.0)

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        # Old access time
        block.last_access_time = time.time() - 100
        block.access_frequency = 0.01

        assert classifier.classify(block) == "cold"

    def test_classify_warm(self):
        """Test warm block classification."""
        classifier = HotColdClassifier(
            hot_frequency_threshold=1.0,
            cold_timeout_s=60.0,
            warm_frequency_threshold=0.1,
        )

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        # Moderate frequency and recent access = warm
        block.access_frequency = 0.5
        block.last_access_time = time.time() - 10

        assert classifier.classify(block) == "warm"

    def test_classify_stale_high_freq_becomes_warm(self):
        """High historical frequency should decay if stale."""
        classifier = HotColdClassifier(
            hot_frequency_threshold=1.0,
            cold_timeout_s=300.0,
            warm_frequency_threshold=0.1,
            decay_half_life_s=30.0,
        )

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        # Historically hot but stale for 120s → frequency should decay
        block.access_frequency = 5.0
        block.last_access_time = time.time() - 120

        assert classifier.classify(block) == "warm"

    def test_priority_score(self):
        """Test migration priority score calculation."""
        classifier = HotColdClassifier()

        hot_block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        hot_block.access_frequency = 2.0

        cold_block = KVBlockDescriptor(
            block_id=2,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(16, 32),
            sequence_id=1,
            request_id="req_1",
        )
        cold_block.access_frequency = 0.01
        cold_block.last_access_time = time.time() - 120

        # Cold block should have higher priority (should migrate first)
        hot_priority = classifier.get_priority_score(hot_block)
        cold_priority = classifier.get_priority_score(cold_block)

        assert cold_priority > hot_priority


class TestKVMigrator:
    """Tests for KVMigrator."""

    def test_plan_migration_under_pressure(self):
        """Test migration planning under tier pressure."""
        # Setup storage
        storage = TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )

        classifier = HotColdClassifier()
        migrator = KVMigrator(storage, classifier)

        # Create cold blocks
        blocks = []
        for i in range(5):
            block = KVBlockDescriptor(
                block_id=i,
                granularity=KVGranularity.BLOCK,
                layer_ids=[0],
                head_ids=[],
                token_range=(i * 16, (i + 1) * 16),
                sequence_id=1,
                request_id="req_1",
                tier=StorageTier.HBM,
                size_bytes=1024,
            )
            block.access_frequency = 0.01
            block.last_access_time = time.time() - 120
            blocks.append(block)

        # High HBM pressure
        pressure = {
            StorageTier.HBM: 0.95,
            StorageTier.DDR: 0.5,
        }

        plans = migrator.plan_migration(blocks, pressure)

        # Should plan to migrate cold blocks from HBM to DDR
        assert len(plans) > 0
        assert all(p.from_tier == StorageTier.HBM for p in plans)
        assert all(p.to_tier == StorageTier.DDR for p in plans)

        storage.close()

    def test_execute_migration(self):
        """Test migration execution."""
        storage = TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )

        classifier = HotColdClassifier()
        migrator = KVMigrator(storage, classifier)

        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
            tier=StorageTier.HBM,
        )

        # Write data first
        test_data = b"test_migration_data"
        storage.write_block(block, test_data)

        plan = MigrationPlan(
            block_id=1,
            from_tier=StorageTier.HBM,
            to_tier=StorageTier.DDR,
        )

        result = migrator.execute_migration(plan, block)

        assert result.success is True
        assert result.from_tier == StorageTier.HBM
        assert result.to_tier == StorageTier.DDR
        assert block.tier == StorageTier.DDR

        storage.close()


class TestCrossRequestKVCache:
    """Tests for CrossRequestKVCache."""

    def test_commit_and_reuse_exact_match(self):
        """Test exact prefix match reuse."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)

        # First request commits KV
        token_ids = [1, 2, 3, 4, 5]
        blocks = pool.allocate(1, "req_1", len(token_ids), [0])
        cache.commit("req_1", token_ids, blocks)

        # Second request tries to reuse
        result = cache.try_reuse("req_2", token_ids)

        assert result.reused is True
        assert result.matched_tokens == 5
        assert result.reuse_ratio == 1.0

    def test_reuse_prefix_match(self):
        """Test prefix matching reuse."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)

        # Commit a prefix
        prefix = [1, 2, 3]
        blocks = pool.allocate(1, "req_1", len(prefix), [0])
        cache.commit("req_1", prefix, blocks)

        # Query with longer sequence that starts with prefix
        result = cache.try_reuse("req_2", [1, 2, 3, 4, 5])

        assert result.reused is True
        assert result.matched_tokens == 3
        assert result.reuse_ratio == 0.6
        assert result.new_tokens_needed == 2

    def test_reuse_no_match(self):
        """Test when no matching prefix exists."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)

        # Query without any committed prefixes
        result = cache.try_reuse("req_1", [1, 2, 3])

        assert result.reused is False
        assert result.matched_tokens == 0

    def test_release_reference(self):
        """Test reference count management."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)

        token_ids = [1, 2, 3]
        blocks = pool.allocate(1, "req_1", len(token_ids), [0])
        cache.commit("req_1", token_ids, blocks)

        # Reuse increases ref count
        cache.try_reuse("req_2", token_ids)

        entry = cache.get_entry(token_ids)
        assert entry is not None
        assert entry.ref_count == 2

        # Release decreases ref count
        cache.release("req_2", token_ids)

        entry = cache.get_entry(token_ids)
        assert entry.ref_count == 1

    def test_tenant_isolation(self):
        """Test multi-tenant isolation."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool, enable_tenant_isolation=True)

        token_ids = [1, 2, 3]

        # Tenant A commits
        blocks = pool.allocate(1, "req_1", len(token_ids), [0])
        cache.commit("req_1", token_ids, blocks, tenant_id="tenant_a")

        # Tenant B cannot reuse
        result = cache.try_reuse("req_2", token_ids, tenant_id="tenant_b")
        assert result.reused is False

        # Tenant A can reuse
        result = cache.try_reuse("req_3", token_ids, tenant_id="tenant_a")
        assert result.reused is True

    def test_cache_stats(self):
        """Test cache statistics."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)

        # Generate some activity
        token_ids = [1, 2, 3]
        blocks = pool.allocate(1, "req_1", len(token_ids), [0])
        cache.commit("req_1", token_ids, blocks)

        cache.try_reuse("req_2", token_ids)  # hit
        cache.try_reuse("req_3", [4, 5, 6])  # miss

        stats = cache.get_stats()

        assert stats["total_queries"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_evict_by_age(self):
        """Test evicting old entries."""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)

        token_ids = [1, 2, 3]
        blocks = pool.allocate(1, "req_1", len(token_ids), [0])
        cache.commit("req_1", token_ids, blocks)

        # Evict entries older than 0 seconds (all entries)
        time.sleep(0.01)
        evicted = cache.evict_by_age(0.001)

        assert evicted == 1
        assert cache.get_entry(token_ids) is None
