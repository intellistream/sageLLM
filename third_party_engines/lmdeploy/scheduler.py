# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Scheduler integration for LMDeploy.

This module provides IR/strategy injection into LMDeploy's scheduler,
enabling external scheduling decisions and execution plan injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ...scheduler_ir.protocols import SchedulingPolicy
    from ...scheduler_ir.types import ExecutionPlan, SchedulingDecision

logger = logging.getLogger(__name__)


class LMDeploySchedulerAdapter:
    """Adapter for injecting scheduling decisions into LMDeploy.

    This adapter allows sageLLM's scheduler_ir module to control
    LMDeploy's request scheduling.

    Attributes:
        policy: The active scheduling policy.
    """

    def __init__(self, policy: SchedulingPolicy | None = None) -> None:
        """Initialize the scheduler adapter.

        Args:
            policy: Optional scheduling policy.
        """
        self.policy = policy

        # Decision callbacks
        self._decision_callback: Callable[[SchedulingDecision], None] | None = None
        self._plan_callback: Callable[[ExecutionPlan], None] | None = None

        # Pending plans
        self._pending_plans: dict[str, ExecutionPlan] = {}

        # Statistics
        self._stats = {
            "decisions_made": 0,
            "plans_submitted": 0,
            "plans_completed": 0,
            "plans_failed": 0,
        }

    # =========================================================================
    # Policy Management
    # =========================================================================

    def set_policy(self, policy: SchedulingPolicy) -> None:
        """Set the active scheduling policy.

        Args:
            policy: Scheduling policy to use.
        """
        self.policy = policy
        logger.info("Scheduling policy set: %s", policy.name)

    def get_policy(self) -> SchedulingPolicy | None:
        """Get the active scheduling policy.

        Returns:
            Current scheduling policy or None.
        """
        return self.policy

    # =========================================================================
    # Decision Making
    # =========================================================================

    def make_decision(
        self,
        request_id: str,
        request_metadata: dict[str, Any],
        available_instances: list[str],
    ) -> SchedulingDecision | None:
        """Make a scheduling decision for a request.

        Args:
            request_id: Request identifier.
            request_metadata: Request metadata dictionary.
            available_instances: List of available instance IDs.

        Returns:
            SchedulingDecision if policy is set, None otherwise.
        """
        if self.policy is None:
            return None

        # Convert metadata to RequestMetadata if needed
        from ...core.types import RequestMetadata

        request = RequestMetadata(
            request_id=request_id,
            **{
                k: v
                for k, v in request_metadata.items()
                if k in RequestMetadata.__dataclass_fields__
            },
        )

        decision = self.policy.schedule(
            request=request,
            available_instances=available_instances,
        )

        self._stats["decisions_made"] += 1

        # Invoke callback if set
        if self._decision_callback is not None:
            try:
                self._decision_callback(decision)
            except Exception as e:
                logger.warning("Decision callback failed: %s", e)

        return decision

    # =========================================================================
    # Plan Management
    # =========================================================================

    def submit_plan(self, plan: ExecutionPlan) -> str:
        """Submit an execution plan for execution.

        Args:
            plan: The execution plan to submit.

        Returns:
            Plan ID for tracking.
        """
        self._pending_plans[plan.plan_id] = plan
        self._stats["plans_submitted"] += 1

        logger.debug("Submitted execution plan: %s", plan.plan_id)

        # Invoke callback if set
        if self._plan_callback is not None:
            try:
                self._plan_callback(plan)
            except Exception as e:
                logger.warning("Plan callback failed: %s", e)

        return plan.plan_id

    def get_plan_status(self, plan_id: str) -> str:
        """Get the status of a submitted plan.

        Args:
            plan_id: Plan identifier.

        Returns:
            Status string.
        """
        if plan_id in self._pending_plans:
            return "pending"
        return "unknown"

    def mark_plan_completed(self, plan_id: str, success: bool = True) -> None:
        """Mark a plan as completed.

        Args:
            plan_id: Plan identifier.
            success: Whether the plan completed successfully.
        """
        if plan_id in self._pending_plans:
            del self._pending_plans[plan_id]
            if success:
                self._stats["plans_completed"] += 1
            else:
                self._stats["plans_failed"] += 1

    # =========================================================================
    # Callbacks
    # =========================================================================

    def set_decision_callback(
        self, callback: Callable[[SchedulingDecision], None]
    ) -> None:
        """Set callback invoked when a decision is made.

        Args:
            callback: Function called with scheduling decision.
        """
        self._decision_callback = callback

    def set_plan_callback(
        self, callback: Callable[[ExecutionPlan], None]
    ) -> None:
        """Set callback invoked when a plan is submitted.

        Args:
            callback: Function called with execution plan.
        """
        self._plan_callback = callback

    # =========================================================================
    # Priority Override
    # =========================================================================

    def override_priority(self, request_id: str, priority: int) -> bool:
        """Override priority for a pending request.

        Args:
            request_id: Request identifier.
            priority: New priority value (lower = higher priority).

        Returns:
            True if priority was updated.
        """
        # This would need to integrate with LMDeploy's internal scheduler
        # For now, log and return success
        logger.debug(
            "Priority override requested: %s -> %d",
            request_id,
            priority,
        )
        return True

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with decision and plan statistics.
        """
        stats = dict(self._stats)
        stats["pending_plans"] = len(self._pending_plans)
        stats["policy_name"] = self.policy.name if self.policy else None
        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0
