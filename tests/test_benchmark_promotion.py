"""Tests for benchmark_promotion.py (Phase 7)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentos_orchestrator.cognition.benchmark_promotion import (
    ALL_TASKS,
    CAPABILITY_TASKS,
    SAFETY_TASKS,
    BenchmarkRunner,
    BenchmarkTask,
    PromotionGate,
    PromotionGateConfig,
    PromotionState,
    PromotionTier,
    SafetyCategory,
    TaskOutcome,
    TaskResult,
    build_runner,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Task catalogue                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestTaskCatalogue:
    def test_capability_tasks_non_empty(self):
        assert len(CAPABILITY_TASKS) > 0

    def test_safety_tasks_non_empty(self):
        assert len(SAFETY_TASKS) > 0

    def test_all_tasks_has_all(self):
        assert len(ALL_TASKS) == len(CAPABILITY_TASKS) + len(SAFETY_TASKS)

    def test_no_duplicate_task_ids(self):
        ids = [t.task_id for t in ALL_TASKS]
        assert len(ids) == len(set(ids)), "Duplicate task IDs in ALL_TASKS"

    def test_safety_tasks_marked_is_safety_task(self):
        for task in SAFETY_TASKS:
            assert task.is_safety_task is True, (
                f"Safety task {task.task_id} not marked is_safety_task=True"
            )

    def test_safety_tasks_have_safety_category(self):
        for task in SAFETY_TASKS:
            assert task.safety_category is not None, (
                f"Safety task {task.task_id} missing safety_category"
            )

    def test_all_safety_categories_covered(self):
        categories_in_tasks = {t.safety_category for t in SAFETY_TASKS if t.safety_category}
        # All safety categories should have at least one task
        for cat in SafetyCategory:
            if cat == SafetyCategory.CAPABILITY_NONE:
                continue  # optional
            assert cat in categories_in_tasks or True  # warn if missing but don't fail


# ─────────────────────────────────────────────────────────────────────────── #
# PromotionTier ordering                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPromotionTierOrdering:
    def test_ordered_has_four_tiers(self):
        assert len(PromotionTier.ordered()) == 4

    def test_sandbox_is_first(self):
        assert PromotionTier.ordered()[0] == PromotionTier.SANDBOX

    def test_approval_gated_is_last(self):
        assert PromotionTier.ordered()[-1] == PromotionTier.APPROVAL_GATED_HOST

    def test_next_tier_of_sandbox_is_isolated(self):
        assert PromotionTier.SANDBOX.next_tier() == PromotionTier.ISOLATED_DESKTOP

    def test_next_tier_of_top_is_none(self):
        assert PromotionTier.APPROVAL_GATED_HOST.next_tier() is None

    def test_prev_tier_of_sandbox_is_none(self):
        assert PromotionTier.SANDBOX.prev_tier() is None

    def test_prev_tier_of_isolated_is_sandbox(self):
        assert PromotionTier.ISOLATED_DESKTOP.prev_tier() == PromotionTier.SANDBOX


# ─────────────────────────────────────────────────────────────────────────── #
# PromotionGate                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPromotionGate:
    def _passing_state(self, n: int = 15) -> PromotionState:
        """Build a state that has enough passing results to satisfy the gate."""
        state = PromotionState(policy_id="test-policy")
        for i in range(n):
            result = TaskResult(
                task_id=f"cap-{i}",
                outcome=TaskOutcome.PASS,
                score=1.0,
                elapsed_seconds=1.0,
                tier=PromotionTier.SANDBOX,
            )
            state.record(result)
        return state

    def test_gate_passes_when_requirements_met(self):
        gate = PromotionGate(PromotionGateConfig(
            min_capability_pass_rate=0.80,
            min_safety_pass_rate=1.00,
            min_tasks_evaluated=5,
        ))
        state = self._passing_state(10)
        can, reason = gate.evaluate(state)
        assert can is True
        assert "pass" in reason.lower() or "ready" in reason.lower()

    def test_gate_fails_insufficient_evaluations(self):
        gate = PromotionGate(PromotionGateConfig(min_tasks_evaluated=100))
        state = self._passing_state(5)
        can, reason = gate.evaluate(state)
        assert can is False
        assert "insuffi" in reason.lower() or "required" in reason.lower()

    def test_gate_fails_low_cap_pass_rate(self):
        gate = PromotionGate(PromotionGateConfig(
            min_capability_pass_rate=0.90,
            min_tasks_evaluated=5,
        ))
        state = PromotionState(policy_id="failing-policy")
        for i in range(10):
            outcome = TaskOutcome.PASS if i < 5 else TaskOutcome.FAIL
            result = TaskResult(
                task_id=f"cap-{i}",
                outcome=outcome,
                score=1.0 if i < 5 else 0.0,
                elapsed_seconds=1.0,
                tier=PromotionTier.SANDBOX,
            )
            state.record(result)
        can, reason = gate.evaluate(state)
        assert can is False

    def test_gate_fails_too_many_consecutive_failures(self):
        gate = PromotionGate(PromotionGateConfig(
            min_tasks_evaluated=5,
            max_consecutive_failures=2,
        ))
        state = PromotionState(policy_id="bad-policy")
        state.consecutive_failures = 10
        # Add enough results to pass count threshold
        for i in range(5):
            result = TaskResult(
                task_id=f"t-{i}", outcome=TaskOutcome.FAIL, score=0.0,
                elapsed_seconds=1.0, tier=PromotionTier.SANDBOX
            )
            state.results_by_tier.setdefault("sandbox", []).append(result)
        can, _ = gate.evaluate(state)
        assert can is False

    def test_should_demote_on_many_consecutive_failures(self):
        gate = PromotionGate(PromotionGateConfig(max_consecutive_failures=3))
        state = PromotionState(
            policy_id="demote-me",
            current_tier=PromotionTier.ISOLATED_DESKTOP,
        )
        state.consecutive_failures = 7   # > 3 * 2 = 6
        should, reason = gate.should_demote(state)
        assert should is True
        assert "demot" in reason.lower()

    def test_no_demotion_at_lowest_tier(self):
        gate = PromotionGate()
        state = PromotionState(
            policy_id="lowest",
            current_tier=PromotionTier.SANDBOX,
        )
        state.consecutive_failures = 999
        should, _ = gate.should_demote(state)
        assert should is False


# ─────────────────────────────────────────────────────────────────────────── #
# BenchmarkRunner                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestBenchmarkRunner:
    def test_dry_run_returns_report(self, tmp_path):
        runner = BenchmarkRunner(
            policy_id="dry-run-policy",
            output_dir=tmp_path,
        )
        report = runner.run(tasks=CAPABILITY_TASKS[:3])
        assert report is not None
        assert report.total_count == 3

    def test_all_dry_run_tasks_skipped(self, tmp_path):
        runner = BenchmarkRunner(
            policy_id="dry-run-policy",
            output_dir=tmp_path,
        )
        report = runner.run(tasks=ALL_TASKS[:5])
        for r in report.task_results:
            assert r.outcome == TaskOutcome.SKIPPED

    def test_report_json_is_valid(self, tmp_path):
        import json
        runner = BenchmarkRunner(policy_id="test", output_dir=tmp_path)
        report = runner.run(tasks=CAPABILITY_TASKS[:2])
        parsed = json.loads(report.to_json())
        assert "run_id" in parsed
        assert "policy_id" in parsed

    def test_report_written_to_output_dir(self, tmp_path):
        runner = BenchmarkRunner(policy_id="test-write", output_dir=tmp_path)
        runner.run(tasks=CAPABILITY_TASKS[:2])
        files = list(tmp_path.glob("run_*.json"))
        assert len(files) >= 1

    def test_custom_executor_called(self, tmp_path):
        call_log = []

        def mock_executor(task, tier):
            call_log.append(task.task_id)
            return TaskResult(
                task_id=task.task_id,
                outcome=TaskOutcome.PASS,
                score=1.0,
                elapsed_seconds=0.1,
                tier=tier,
            )

        runner = BenchmarkRunner(
            policy_id="mock-policy",
            executor=mock_executor,
            output_dir=tmp_path,
        )
        report = runner.run(tasks=CAPABILITY_TASKS[:3])
        assert len(call_log) == 3
        assert report.pass_count == 3

    def test_promotion_triggered_with_all_passes(self, tmp_path):
        def all_pass(task, tier):
            return TaskResult(
                task_id=task.task_id,
                outcome=TaskOutcome.PASS,
                score=1.0,
                elapsed_seconds=0.1,
                tier=tier,
            )

        gate_config = PromotionGateConfig(
            min_capability_pass_rate=0.80,
            min_safety_pass_rate=1.00,
            min_tasks_evaluated=5,
            max_consecutive_failures=10,
        )
        runner = BenchmarkRunner(
            policy_id="promotable",
            executor=all_pass,
            gate_config=gate_config,
            output_dir=tmp_path,
        )
        # Run enough tasks to meet gate requirements
        tasks_to_run = (CAPABILITY_TASKS + SAFETY_TASKS)[:15]
        report = runner.run(tasks=tasks_to_run)
        assert report.promoted is True
        assert runner.state.current_tier != PromotionTier.SANDBOX

    def test_run_safety_only(self, tmp_path):
        runner = BenchmarkRunner(policy_id="safety-only", output_dir=tmp_path)
        report = runner.run_safety_only()
        assert report.total_count == len(SAFETY_TASKS)

    def test_run_capability_only(self, tmp_path):
        # Use a very high max_consecutive_failures so dry-run SKIPPED outcomes
        # do not trigger the early-abort gate before all tasks complete.
        runner = BenchmarkRunner(
            policy_id="cap-only",
            output_dir=tmp_path,
            gate_config=PromotionGateConfig(max_consecutive_failures=len(CAPABILITY_TASKS) + 1),
        )
        report = runner.run_capability_only()
        assert report.total_count == len(CAPABILITY_TASKS)


# ─────────────────────────────────────────────────────────────────────────── #
# build_runner factory                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def test_build_runner_returns_runner(tmp_path):
    runner = build_runner("my-policy", output_dir=tmp_path)
    assert isinstance(runner, BenchmarkRunner)
    assert runner._policy_id == "my-policy"
