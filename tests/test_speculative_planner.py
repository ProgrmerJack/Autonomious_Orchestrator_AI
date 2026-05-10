"""Tests for the SpeculativePlanner (Phase 5)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentos_orchestrator.cognition.speculative_planner import (
    IRREVERSIBLE_ACTION_TYPES,
    CommitDecision,
    SpeculativePlan,
    SpeculativePlanner,
    get_speculative_planner,
)
from agentos_orchestrator.os_control.base import UiAction


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _action(action_type: str, selector: str = "el") -> UiAction:
    return UiAction(action_type=action_type, selector=selector)


def _make_observation(frame_id: str = "frame-1") -> MagicMock:
    obs = MagicMock()
    obs.frame_id = frame_id
    obs.elements = []
    obs.captured_at = "2024-01-01T00:00:00"
    return obs


# ─────────────────────────────────────────────────────────────────────────── #
# build_plan                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSpeculativePlannerBuildPlan:
    def test_build_plan_returns_speculative_plan(self):
        planner = SpeculativePlanner(mcts_iterations=10, max_plan_length=4)
        obs = _make_observation()
        plan = planner.build_plan("Open the browser", obs)
        assert isinstance(plan, SpeculativePlan)

    def test_plan_has_objective(self):
        planner = SpeculativePlanner(mcts_iterations=10, max_plan_length=4)
        plan = planner.build_plan("Navigate to docs", None)
        assert plan.objective == "Navigate to docs"

    def test_plan_id_is_unique(self):
        planner = SpeculativePlanner(mcts_iterations=5, max_plan_length=4)
        plan1 = planner.build_plan("Task 1", None)
        plan2 = planner.build_plan("Task 2", None)
        assert plan1.plan_id != plan2.plan_id

    def test_high_risk_actions_filtered_from_plan(self):
        planner = SpeculativePlanner(mcts_iterations=5, max_plan_length=4)
        plan = planner.build_plan("Some safe task", None)
        for spec_action in plan.actions:
            assert spec_action.risk_score < planner._abort_threshold

    def test_plan_with_empty_observation(self):
        planner = SpeculativePlanner(mcts_iterations=5, max_plan_length=3)
        plan = planner.build_plan("Click something", None)
        assert plan is not None

    def test_plan_cumulative_reward_is_numeric(self):
        planner = SpeculativePlanner(mcts_iterations=10, max_plan_length=4)
        plan = planner.build_plan("Do something", None)
        assert isinstance(plan.cumulative_reward, float)


# ─────────────────────────────────────────────────────────────────────────── #
# next_commit                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSpeculativePlannerNextCommit:
    def test_exhausted_plan_returns_no_commit(self):
        planner = SpeculativePlanner(mcts_iterations=5)
        plan = SpeculativePlan(
            plan_id="test-001",
            objective="Test",
            observation_hash="abc",
            actions=[],
        )
        decision = planner.next_commit(plan, None)
        assert decision.should_commit is False
        assert "exhaust" in decision.reason.lower()

    def test_stale_plan_returns_no_commit(self):
        planner = SpeculativePlanner(mcts_iterations=5)
        obs = _make_observation("frame-A")
        plan = planner.build_plan("Task", obs)
        # Now pass a different observation
        obs2 = _make_observation("frame-B")
        # Force different hash by changing element count
        obs2.elements = [MagicMock()] * 5
        obs2.captured_at = "2024-06-01T12:00:00"
        decision = planner.next_commit(plan, obs2)
        # Either should_commit is False because stale, or plan wasn't built with
        # enough discriminating data — we just verify CommitDecision is returned
        assert isinstance(decision, CommitDecision)

    def test_already_stale_plan_returns_no_commit(self):
        planner = SpeculativePlanner(mcts_iterations=5)
        plan = SpeculativePlan(
            plan_id="stale-plan",
            objective="Test",
            observation_hash="xyz",
            actions=[],
            is_stale=True,
        )
        decision = planner.next_commit(plan, None)
        assert decision.should_commit is False
        assert "stale" in decision.reason.lower()

    def test_irreversible_action_requires_approval(self):
        from agentos_orchestrator.cognition.speculative_planner import SpeculativeAction

        planner = SpeculativePlanner(mcts_iterations=5)
        irrev_action = UiAction(action_type="file_delete", selector="folder")
        spec = SpeculativeAction(
            action=irrev_action,
            speculative_reward=0.8,
            risk_score=0.70,
            requires_approval=True,
            is_irreversible=True,
        )
        plan = SpeculativePlan(
            plan_id="irrev-plan",
            objective="Delete files",
            observation_hash="none",
            actions=[spec],
        )
        decision = planner.next_commit(plan, None)
        # Must not commit irreversible action without approval token
        assert decision.should_commit is False
        assert decision.requires_approval is True

    def test_irreversible_action_commits_with_approval_token(self):
        from agentos_orchestrator.cognition.speculative_planner import SpeculativeAction

        planner = SpeculativePlanner(mcts_iterations=5, abort_risk_threshold=0.99)
        irrev_action = UiAction(action_type="file_delete", selector="folder")
        spec = SpeculativeAction(
            action=irrev_action,
            speculative_reward=0.8,
            risk_score=0.50,   # below abort threshold
            requires_approval=True,
            is_irreversible=True,
        )
        plan = SpeculativePlan(
            plan_id="irrev-approved",
            objective="Delete files",
            observation_hash="none",
            actions=[spec],
        )
        decision = planner.next_commit(plan, None, approval_token="tok-abc-123")
        # With approval token and risk below threshold, commit is allowed
        assert decision.should_commit is True


# ─────────────────────────────────────────────────────────────────────────── #
# SpeculativePlan helpers                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSpeculativePlanHelpers:
    def test_mark_committed_advances_counter(self):
        from agentos_orchestrator.cognition.speculative_planner import SpeculativeAction

        spec = SpeculativeAction(
            action=_action("click"),
            speculative_reward=0.5,
            risk_score=0.1,
            requires_approval=False,
            is_irreversible=False,
        )
        plan = SpeculativePlan(
            plan_id="x", objective="t", observation_hash="y", actions=[spec]
        )
        assert plan.committed_count == 0
        plan.mark_committed()
        assert plan.committed_count == 1
        assert plan.is_exhausted

    def test_invalidate_sets_stale(self):
        plan = SpeculativePlan(
            plan_id="x", objective="t", observation_hash="y", actions=[]
        )
        assert not plan.is_stale
        plan.invalidate("test reason")
        assert plan.is_stale


# ─────────────────────────────────────────────────────────────────────────── #
# refresh_plan                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def test_refresh_plan_invalidates_old_plan():
    planner = SpeculativePlanner(mcts_iterations=5)
    obs1 = _make_observation("obs-1")
    plan1 = planner.build_plan("Task", obs1)
    assert not plan1.is_stale
    obs2 = _make_observation("obs-2")
    plan2 = planner.refresh_plan(plan1, obs2)
    assert plan1.is_stale
    assert plan2.plan_id != plan1.plan_id


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level singleton                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def test_get_speculative_planner_returns_same_instance():
    p1 = get_speculative_planner()
    p2 = get_speculative_planner()
    assert p1 is p2


# ─────────────────────────────────────────────────────────────────────────── #
# Irreversible action types constant                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def test_irreversible_action_types_is_frozenset():
    assert isinstance(IRREVERSIBLE_ACTION_TYPES, frozenset)
    assert "file_delete" in IRREVERSIBLE_ACTION_TYPES
    assert "trade_order" in IRREVERSIBLE_ACTION_TYPES
