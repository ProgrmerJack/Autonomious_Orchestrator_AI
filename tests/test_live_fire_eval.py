from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.cognition.live_fire_eval import (
    LiveFireEvalConfig,
    LiveFireEvalRunner,
)
from agentos_orchestrator.cognition.live_fire_review import (
    load_live_fire_reviews,
    promote_live_fire_failure,
    write_shadow_training_heads,
)
from agentos_orchestrator.os_control import UiAction, UiNode
from agentos_orchestrator.os_control.virtual_desktop_sandbox_backend import (
    VirtualDesktopSandboxBackend,
)


class EmptyUiBackend:
    name = "empty-ui"

    def available(self) -> bool:
        return True

    def snapshot(self) -> list[UiNode]:
        return []

    def perform(self, action: UiAction) -> str:
        status = "executed" if action.action_type == "launch_app" else "failed"
        return json.dumps(
            {
                "status": status,
                "action": action.action_type,
                "selector": action.selector,
            },
            sort_keys=True,
        )


class RegroundingBackend:
    name = "regrounding-ui"

    def __init__(self) -> None:
        self.phase = "initial"

    def available(self) -> bool:
        return True

    def snapshot(self) -> list[UiNode]:
        if self.phase == "initial":
            return []
        if self.phase == "launched":
            return [
                UiNode("terminal-doc", "Document", "PowerShell Terminal"),
                UiNode("terminal-pane", "Pane", "Console"),
            ]
        return [
            UiNode("terminal-doc", "Document", "PowerShell Terminal"),
            UiNode("terminal-pane", "Pane", "Console"),
            UiNode("terminal-status", "Text", "Focused"),
        ]

    def perform(self, action: UiAction) -> str:
        if action.action_type == "launch_app":
            self.phase = "launched"
            return json.dumps(
                {
                    "status": "launched",
                    "action_type": action.action_type,
                    "selector": action.selector,
                },
                sort_keys=True,
            )
        if action.selector == "app-workspace":
            raise RuntimeError("No UI element matched selector 'app-workspace'")
        if action.selector == "name=PowerShell Terminal":
            self.phase = "focused"
            return json.dumps(
                {
                    "status": "executed",
                    "action_type": action.action_type,
                    "selector": action.selector,
                },
                sort_keys=True,
            )
        raise RuntimeError(f"Unexpected selector {action.selector}")


class FillFormRegroundingBackend:
    name = "fill-form-regrounding-ui"

    def __init__(self) -> None:
        self.phase = "initial"

    def available(self) -> bool:
        return True

    def snapshot(self) -> list[UiNode]:
        if self.phase == "initial":
            return []
        return [
            UiNode("terminal-window", "Window", "Terminal - 1 running window"),
            UiNode("terminal-button", "Button", "Terminal - 1 running window"),
            UiNode("terminal-input", "Document", "PowerShell Terminal"),
        ]

    def perform(self, action: UiAction) -> str:
        if action.action_type == "launch_app":
            self.phase = "launched"
            return json.dumps(
                {
                    "status": "launched",
                    "action_type": action.action_type,
                    "selector": action.selector,
                },
                sort_keys=True,
            )
        if action.selector == "app-workspace":
            raise RuntimeError("No UI element matched selector 'app-workspace'")
        if action.selector == "name=PowerShell Terminal":
            return json.dumps(
                {
                    "status": "typed",
                    "action_type": action.action_type,
                    "selector": action.selector,
                    "matched_name": "PowerShell Terminal",
                    "matched_role": "ControlType.Document",
                },
                sort_keys=True,
            )
        raise RuntimeError(f"Unexpected selector {action.selector}")


class LiveFireEvalTests(unittest.TestCase):
    def test_runner_executes_real_user_handoff_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_handoff_test",
                    pack="handoff",
                    promote_failures=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.task_count, 4)
            self.assertEqual(
                {task.surface for task in result.task_results},
                {"browser", "file_explorer", "chat_app", "pdf_viewer"},
            )
            self.assertTrue(
                any(
                    item.intent == "browser_editor_handoff"
                    for item in result.task_results
                )
            )

    def test_runner_executes_everyday_family_pack_with_task_specific_proof(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_everyday_test",
                    pack="everyday",
                    promote_failures=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.task_count, 3)
            self.assertEqual(
                {task.surface for task in result.task_results},
                {"email", "calendar", "settings"},
            )
            verification_kinds = {
                verification["kind"]
                for task in result.task_results
                for verification in task.verifications
            }
            self.assertIn("send_outcome", verification_kinds)
            self.assertIn("invite_outcome", verification_kinds)
            self.assertIn("toggle_state", verification_kinds)

    def test_runner_emits_replay_and_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_test",
                    max_tasks=3,
                    promote_failures=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.task_count, 3)
            self.assertEqual(result.failed, 0)
            self.assertIn("milestone", result.asdict())
            self.assertEqual(
                result.heldout_metrics["metric"],
                "heldout_live_fire_success_rate",
            )
            self.assertEqual(result.heldout_metrics["success_rate"], 1.0)
            self.assertTrue(Path(result.summary_path).exists())
            self.assertGreaterEqual(result.replay_debug["run_count"], 1)
            self.assertTrue(result.training_summary["ready_for_training"])
            self.assertTrue(Path(result.training_summary["path"]).exists())
            for trajectory_path in result.trajectory_paths:
                self.assertTrue(Path(trajectory_path).exists())

    def test_runner_promotes_durable_failures_to_golden_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = LiveFireEvalRunner(EmptyUiBackend(), root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_failure_test",
                    surfaces=("browser",),
                    intents=("fill_form",),
                    max_tasks=1,
                    promote_failures=True,
                    promote_after=1,
                )
            )

            self.assertFalse(result.success)
            self.assertEqual(result.failed, 1)
            self.assertEqual(len(result.promoted_traces), 1)
            promoted = Path(result.promoted_traces[0])
            self.assertTrue(promoted.exists())
            payload = json.loads(promoted.read_text(encoding="utf-8"))
            self.assertIn(
                "typed value was not observed",
                payload["failure_modes"],
            )
            self.assertIn("replay_debug", payload)

    def test_safe_windows_pack_selects_low_risk_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="safe_pack_test",
                    windows_safe_pack=True,
                    promote_failures=False,
                )
            )

            self.assertEqual(result.task_count, 30)
            self.assertEqual(result.failed, 0)
            surfaces = {task.surface for task in result.task_results}
            intents = {task.intent for task in result.task_results}
            self.assertEqual(
                surfaces,
                {
                    "browser",
                    "file_explorer",
                    "terminal",
                    "editor",
                    "file_dialog",
                },
            )
            self.assertEqual(
                intents,
                {
                    "open_app",
                    "find_target",
                    "fill_form",
                    "use_shortcut",
                    "recover_modal",
                    "stale_target_reground",
                },
            )

    def test_broadened_app_family_pack_reports_heldout_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="broad_app_family_pack_test",
                    surfaces=(
                        "office_form",
                        "pdf_viewer",
                        "chat_app",
                        "electron_app",
                        "design_canvas",
                        "trading_terminal",
                        "enterprise_grid",
                    ),
                    intents=("fill_form",),
                    promote_failures=False,
                    heldout_from="adaptation-longrun-smoke",
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.task_count, 7)
            self.assertEqual(result.failed, 0)
            self.assertEqual(result.heldout_metrics["surface_count"], 7)
            self.assertEqual(result.heldout_metrics["intent_count"], 1)
            self.assertEqual(
                result.heldout_metrics["heldout_from"],
                "adaptation-longrun-smoke",
            )
            self.assertIn("design_canvas", result.heldout_metrics["by_surface"])
            self.assertIn("trading_terminal", result.heldout_metrics["by_surface"])

    def test_safe_pack_repeat_expands_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="safe_pack_repeat_test",
                    windows_safe_pack=True,
                    max_tasks=2,
                    repeat=3,
                    promote_failures=False,
                )
            )

            self.assertEqual(result.task_count, 6)
            self.assertTrue(all("_r" in task.task_id for task in result.task_results))

    def test_repeat_failures_share_promotion_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = LiveFireEvalRunner(EmptyUiBackend(), root).run(
                LiveFireEvalConfig(
                    run_id="repeat_failure_counter_test",
                    surfaces=("browser",),
                    intents=("fill_form",),
                    max_tasks=1,
                    repeat=2,
                    promote_failures=True,
                    promote_after=2,
                )
            )

            self.assertFalse(result.success)
            self.assertEqual(result.failed, 2)
            self.assertEqual(len(result.promoted_traces), 1)
            self.assertIn("_r2", Path(result.promoted_traces[0]).stem)

    def test_failure_review_promotes_manual_durable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = LiveFireEvalRunner(EmptyUiBackend(), root).run(
                LiveFireEvalConfig(
                    run_id="manual_review_failure_test",
                    surfaces=("browser",),
                    intents=("fill_form",),
                    max_tasks=1,
                    promote_failures=False,
                )
            )
            failed_task = result.task_results[0]

            review = load_live_fire_reviews(root, limit=5)
            self.assertEqual(len(review["failed_tasks"]), 1)
            failure = review["failed_tasks"][0]
            self.assertEqual(failure["classification"], "selector_grounding")
            self.assertTrue(failure["promotable"])

            promoted = promote_live_fire_failure(
                root,
                result.run_id,
                failed_task.task_id,
            )
            self.assertEqual(promoted["status"], "promoted")
            self.assertTrue(Path(promoted["path"]).exists())

            review_after = load_live_fire_reviews(root, limit=5)
            self.assertEqual(
                review_after["milestone"]["durable_promoted_failures"],
                1,
            )
            self.assertEqual(
                review_after["failed_tasks"][0]["existing_golden_trace"],
                promoted["path"],
            )
            self.assertFalse(review_after["failed_tasks"][0]["promotable"])

    def test_shadow_training_writes_ordered_advisory_heads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="shadow_training_source",
                    max_tasks=1,
                    promote_failures=False,
                )
            )

            summary = write_shadow_training_heads(
                root,
                trajectory_paths=result.trajectory_paths,
            )
            self.assertTrue(summary["advisory_only"])
            self.assertEqual(
                summary["head_order"],
                ["outcome_critic", "option_policy", "affordance_ranker"],
            )
            self.assertTrue(summary["ready_for_shadow_training"])
            for item in summary["heads"].values():
                self.assertTrue(Path(item["path"]).exists())
                self.assertGreater(item["examples"], 0)

    def test_runner_persists_successful_policies_by_app_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / ".agentos" / "sandbox.json")
            result = LiveFireEvalRunner(backend, root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_policy_memory_test",
                    surfaces=("editor",),
                    intents=("find_target",),
                    max_tasks=1,
                    promote_failures=False,
                )
            )

            self.assertTrue(result.success)
            policy_path = root / ".agentos" / "affordance_policies.json"
            self.assertTrue(policy_path.exists())
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
            success_entries = [
                entry
                for entry in payload.get("entries", [])
                if entry.get("success_count", 0) > 0
            ]
            self.assertTrue(success_entries)
            self.assertTrue(
                all(entry.get("app_signature") for entry in success_entries)
            )
            self.assertTrue(
                any(
                    entry.get("last_evidence", {}).get("live_fire") is True
                    for entry in success_entries
                )
            )

    def test_runner_regrounds_missing_placeholder_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = LiveFireEvalRunner(RegroundingBackend(), root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_regrounding_test",
                    surfaces=("terminal",),
                    intents=("find_target",),
                    max_tasks=1,
                    promote_failures=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.passed, 1)
            trajectory = Path(result.trajectory_paths[0]).read_text(encoding="utf-8")
            self.assertIn('"regrounded_from": "app-workspace"', trajectory)
            self.assertIn('"selector": "name=PowerShell Terminal"', trajectory)

    def test_runner_regrounds_fill_form_to_text_capable_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = LiveFireEvalRunner(FillFormRegroundingBackend(), root).run(
                LiveFireEvalConfig(
                    run_id="live_fire_fill_form_regrounding_test",
                    surfaces=("terminal",),
                    intents=("fill_form",),
                    max_tasks=1,
                    promote_failures=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.passed, 1)
            trajectory = Path(result.trajectory_paths[0]).read_text(encoding="utf-8")
            self.assertIn('"regrounded_from": "app-workspace"', trajectory)
            self.assertIn('"selector": "name=PowerShell Terminal"', trajectory)


if __name__ == "__main__":
    unittest.main()
