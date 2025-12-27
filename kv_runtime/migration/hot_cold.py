# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Hot/cold KV block classification and migration.

This module provides:
- HotColdClassifier: Classifies KV blocks based on access patterns
- KVMigrator: Executes migrations between storage tiers
- MigrationPlan/MigrationResult: Migration data structures

The migration strategy follows the tiered storage principle:
- Hot data (frequently accessed) stays in HBM
- Warm data migrates to DDR
- Cold data (rarely accessed) moves to NVMe
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..blocks.multi_granular import KVBlockDescriptor
    from ..hierarchy.tiered_storage import TieredKVStorage


def _get_storage_tier() -> type:
    """Get StorageTier enum to avoid circular imports."""
    from ..blocks.multi_granular import StorageTier

    return StorageTier


@dataclass
class MigrationPlan:
    """Migration plan for a KV block.

    Attributes:
        block_id: Block to migrate.
        from_tier: Source storage tier.
        to_tier: Destination storage tier.
        priority: Migration priority (higher = more urgent).
        deadline_ms: Optional deadline in milliseconds.
    """

    block_id: int
    from_tier: Any  # StorageTier
    to_tier: Any  # StorageTier
    priority: int = 0
    deadline_ms: float | None = None


@dataclass
class MigrationResult:
    """Result of a migration operation.

    Attributes:
        success: Whether migration completed successfully.
        block_id: Migrated block ID.
        from_tier: Source tier.
        to_tier: Destination tier.
        duration_ms: Time taken in milliseconds.
        size_bytes: Amount of data migrated.
        error: Error message if failed.
    """

    success: bool
    block_id: int
    from_tier: Any  # StorageTier
    to_tier: Any  # StorageTier
    duration_ms: float
    size_bytes: int
    error: str | None = None


class HotColdClassifier:
    """KV block hot/cold classifier.

    Classifies blocks based on access frequency and recency:
    - hot: Frequently accessed, should stay in HBM
    - warm: Moderate access, can be in DDR
    - cold: Rarely accessed, can migrate to NVMe

    Example:
        >>> classifier = HotColdClassifier(
        ...     hot_frequency_threshold=1.0,  # >1 access/sec = hot
        ...     cold_timeout_s=60.0,          # >60s since last access = cold
        ... )
        >>> classification = classifier.classify(block)
        >>> print(f"Block is {classification}")
    """

    def __init__(
        self,
        hot_frequency_threshold: float = 1.0,
        cold_timeout_s: float = 60.0,
        warm_frequency_threshold: float = 0.1,
        decay_half_life_s: float = 30.0,
    ):
        """Initialize the classifier.

        Args:
            hot_frequency_threshold: Access frequency (per second) above which
                a block is considered hot.
            cold_timeout_s: Time (seconds) since last access after which
                a block is considered cold.
            warm_frequency_threshold: Access frequency below which a block
                is considered cold (if not already cold by timeout).
        """
        self.hot_frequency_threshold = hot_frequency_threshold
        self.cold_timeout_s = cold_timeout_s
        self.warm_frequency_threshold = warm_frequency_threshold
        self.decay_half_life_s = decay_half_life_s

    def _effective_frequency(self, block: KVBlockDescriptor, now: float) -> float:
        """Compute decay-aware access frequency.

        Uses exponential decay so that stale high-frequency blocks don't stay
        "hot" purely due to historic activity.
        """
        time_since_access = now - block.last_access_time
        if time_since_access <= 0:
            return block.access_frequency

        decay_factor = 0.5 ** (time_since_access / max(self.decay_half_life_s, 1e-3))
        return block.access_frequency * decay_factor

    def classify(
        self,
        block: KVBlockDescriptor,
    ) -> Literal["hot", "warm", "cold"]:
        """Classify a KV block by temperature with clear priority rules.

        Classification priority:
        1. Very recent access (<1s) → hot (regardless of frequency)
        2. High frequency (>= hot_threshold) → hot
        3. Very old access (> cold_timeout) → cold (regardless of frequency)
        4. Low frequency and moderate age → cold
        5. Default → warm

        Args:
            block: KV block descriptor with access statistics.

        Returns:
            "hot", "warm", or "cold" classification.
        """
        now = time.time()
        time_since_access = now - block.last_access_time
        effective_freq = self._effective_frequency(block, now)

        # Priority 1: Very recent access = hot
        if time_since_access < 1.0:
            return "hot"

        # Priority 2: High frequency = hot
        if effective_freq >= self.hot_frequency_threshold:
            return "hot"

        # Priority 3: Very old = cold (regardless of frequency)
        if time_since_access > self.cold_timeout_s:
            return "cold"

        # Priority 4: Low frequency and moderate age = cold
        if (
            effective_freq < self.warm_frequency_threshold
            and time_since_access > self.cold_timeout_s / 2
        ):
            return "cold"

        # Default: warm
        return "warm"

    def predict_lifetime(self, block: KVBlockDescriptor) -> float:
        """Predict remaining lifetime of a KV block.

        Estimates how long the block will continue to be accessed,
        used to decide if migration is worthwhile.

        Args:
            block: KV block descriptor.

        Returns:
            Predicted remaining lifetime in seconds.
        """
        if block.access_frequency > 0:
            # Higher frequency = longer expected lifetime
            # Cap at 1 hour for stability
            return min(10.0 / block.access_frequency, 3600.0)
        else:
            # No access history = likely short lifetime
            return 0.0

    def get_priority_score(self, block: KVBlockDescriptor) -> float:
        """Calculate migration priority score.

        Higher score = should be migrated to lower tier sooner.

        Args:
            block: KV block descriptor.

        Returns:
            Priority score between 0.0 and 1.0.
        """
        classification = self.classify(block)
        base_score = {"hot": 0.0, "warm": 0.5, "cold": 1.0}[classification]

        # Time factor: longer since access = higher priority
        time_since_access = time.time() - block.last_access_time
        time_factor = min(time_since_access / self.cold_timeout_s, 1.0)

        # Frequency factor: lower frequency = higher priority
        effective_freq = self._effective_frequency(block, time.time())
        frequency_factor = 1.0 - min(
            effective_freq / self.hot_frequency_threshold, 1.0
        )

        # Weighted combination
        return base_score * 0.4 + time_factor * 0.3 + frequency_factor * 0.3

    def should_promote(self, block: KVBlockDescriptor) -> bool:
        """Check if a block should be promoted to a higher tier.

        Args:
            block: KV block descriptor.

        Returns:
            True if block should be promoted.
        """
        return self.classify(block) == "hot"

    def should_demote(self, block: KVBlockDescriptor) -> bool:
        """Check if a block should be demoted to a lower tier.

        Args:
            block: KV block descriptor.

        Returns:
            True if block should be demoted.
        """
        return self.classify(block) == "cold"


class KVMigrator:
    """KV block migrator.

    Handles migration of KV blocks between storage tiers:
    - Plans migrations based on classifier output and tier pressure
    - Executes migrations with optional overlap with computation
    - Tracks migration statistics

    Example:
        >>> migrator = KVMigrator(storage, classifier)
        >>> plans = migrator.plan_migration(blocks, pressure)
        >>> for plan in plans:
        ...     result = migrator.execute_migration(plan, blocks[plan.block_id])
    """

    def __init__(
        self,
        storage: TieredKVStorage,
        classifier: HotColdClassifier,
    ):
        """Initialize the migrator.

        Args:
            storage: Tiered storage manager.
            classifier: Hot/cold classifier.
        """
        self.storage = storage
        self.classifier = classifier

        # Statistics
        self._stats: dict[str, int] = {
            "total_migrations": 0,
            "hbm_to_ddr": 0,
            "ddr_to_nvme": 0,
            "ddr_to_hbm": 0,
            "nvme_to_ddr": 0,
            "total_bytes_migrated": 0,
            "failed_migrations": 0,
        }

    def plan_migration(
        self,
        blocks: list[KVBlockDescriptor],
        pressure: dict[Any, float],  # StorageTier -> pressure (0.0-1.0)
    ) -> list[MigrationPlan]:
        """Plan migrations based on tier pressure and block temperature.

        Args:
            blocks: All KV blocks to consider.
            pressure: Pressure per tier (0.0-1.0), high pressure needs space.

        Returns:
            List of migration plans, sorted by priority.
        """
        plans: list[MigrationPlan] = []

        # 1. Handle high-pressure tiers: demote cold blocks
        for tier, p in pressure.items():
            if p > 0.9:  # >90% utilization needs migration
                tier_blocks = [b for b in blocks if b.tier == tier]

                # Sort by priority (higher = should migrate first)
                tier_blocks.sort(
                    key=lambda b: self.classifier.get_priority_score(b),
                    reverse=True,
                )

                # Find target tier (lower in hierarchy)
                target_tier = self._get_lower_tier(tier)
                if target_tier is None:
                    continue

                # Calculate how many bytes to free (target: 80% utilization)
                tier_config = self.storage.configs[tier]
                bytes_to_free = int((p - 0.8) * tier_config.capacity_bytes)
                bytes_planned = 0

                for block in tier_blocks:
                    if bytes_planned >= bytes_to_free:
                        break

                    classification = self.classifier.classify(block)
                    if classification in ("cold", "warm"):
                        plans.append(
                            MigrationPlan(
                                block_id=block.block_id,
                                from_tier=tier,
                                to_tier=target_tier,
                                priority=int(
                                    self.classifier.get_priority_score(block) * 100
                                ),
                            )
                        )
                        bytes_planned += block.size_bytes

        # 2. Handle low-pressure tiers: promote hot blocks from lower tiers
        for tier, p in pressure.items():
            if p < 0.5:  # <50% utilization has room
                lower_tier = self._get_lower_tier(tier)
                if lower_tier is None:
                    continue

                # Check if lower tier exists in storage
                if lower_tier not in self.storage.backends:
                    continue

                lower_blocks = [b for b in blocks if b.tier == lower_tier]
                for block in lower_blocks:
                    if self.classifier.classify(block) == "hot":
                        plans.append(
                            MigrationPlan(
                                block_id=block.block_id,
                                from_tier=lower_tier,
                                to_tier=tier,
                                priority=90,  # Hot promotion is high priority
                            )
                        )

        # Sort by priority (descending)
        plans.sort(key=lambda p: p.priority, reverse=True)
        return plans

    def _get_lower_tier(self, tier: Any) -> Any | None:
        """Get the next lower tier in hierarchy.

        Args:
            tier: Current storage tier.

        Returns:
            Lower tier or None if at bottom.
        """
        StorageTier = _get_storage_tier()

        if tier == StorageTier.HBM:
            return StorageTier.DDR
        elif tier == StorageTier.DDR:
            if StorageTier.NVME in self.storage.backends:
                return StorageTier.NVME
        return None

    def _get_higher_tier(self, tier: Any) -> Any | None:
        """Get the next higher tier in hierarchy.

        Args:
            tier: Current storage tier.

        Returns:
            Higher tier or None if at top.
        """
        StorageTier = _get_storage_tier()

        if tier == StorageTier.NVME:
            return StorageTier.DDR
        elif tier == StorageTier.DDR:
            return StorageTier.HBM
        return None

    def execute_migration(
        self,
        plan: MigrationPlan,
        block: KVBlockDescriptor,
    ) -> MigrationResult:
        """Execute a single migration.

        Args:
            plan: Migration plan.
            block: KV block descriptor.

        Returns:
            Migration result.
        """
        start_time = time.time()
        error_msg = None

        try:
            success = self.storage.migrate_block(block, plan.to_tier)
        except Exception as e:
            success = False
            error_msg = str(e)

        duration_ms = (time.time() - start_time) * 1000

        # Update statistics
        if success:
            self._stats["total_migrations"] += 1
            self._stats["total_bytes_migrated"] += block.size_bytes

            # Track direction-specific stats
            key = f"{plan.from_tier.name.lower()}_to_{plan.to_tier.name.lower()}"
            if key in self._stats:
                self._stats[key] += 1
        else:
            self._stats["failed_migrations"] += 1

        return MigrationResult(
            success=success,
            block_id=plan.block_id,
            from_tier=plan.from_tier,
            to_tier=plan.to_tier,
            duration_ms=duration_ms,
            size_bytes=block.size_bytes,
            error=error_msg,
        )

    async def execute_migration_async(
        self,
        plans: list[MigrationPlan],
        blocks: dict[int, KVBlockDescriptor],
        overlap_compute: bool = True,
    ) -> list[MigrationResult]:
        """Asynchronously execute multiple migrations.

        Args:
            plans: List of migration plans.
            blocks: Mapping of block_id to descriptor.
            overlap_compute: Whether to overlap with computation.

        Returns:
            List of migration results.
        """
        results = []
        for plan in plans:
            block = blocks.get(plan.block_id)
            if block:
                result = self.execute_migration(plan, block)
                results.append(result)
        return results

    def execute_batch_migration(
        self,
        plans: list[MigrationPlan],
        blocks: dict[int, KVBlockDescriptor],
    ) -> list[MigrationResult]:
        """Execute multiple migrations synchronously.

        Args:
            plans: List of migration plans.
            blocks: Mapping of block_id to descriptor.

        Returns:
            List of migration results.
        """
        results = []
        for plan in plans:
            block = blocks.get(plan.block_id)
            if block:
                result = self.execute_migration(plan, block)
                results.append(result)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get migration statistics.

        Returns:
            Dictionary of migration statistics.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset migration statistics."""
        for key in self._stats:
            self._stats[key] = 0
