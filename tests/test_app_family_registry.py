from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.app_family_registry import (
    app_family_names,
    app_family_specs,
    eval_surface_families,
    launch_target_for_family,
    primary_selector_for_family,
)
from agentos_orchestrator.cognition.adaptation_readiness import (
    collect_adaptation_readiness,
)
from agentos_orchestrator.cognition.app_adapters import adapter_families
from agentos_orchestrator.cognition.capability_profile import known_app_families
from agentos_orchestrator.cognition.live_fire_eval_recipes import actions_for_task
from agentos_orchestrator.cognition.os_eval_packs import EvalTask, SURFACE_FAMILIES
from agentos_orchestrator.os_control import UiAction, VirtualDesktopSandboxBackend
from agentos_orchestrator.product.status import benchmark_status


class AppFamilyRegistryTests(unittest.TestCase):
    def test_registry_is_shared_across_adapters_profiling_and_eval_pack(self) -> None:
        self.assertEqual(SURFACE_FAMILIES, eval_surface_families())
        self.assertEqual(set(adapter_families()), set(app_family_names()))
        self.assertEqual(set(known_app_families()), set(app_family_names()))

    def test_live_fire_recipes_use_registry_launch_targets_and_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for family in eval_surface_families():
                task = EvalTask(
                    task_id=f"{family}_fill_form",
                    surface=family,
                    intent="fill_form",
                    objective="fill form",
                    expected_verifications=["field_contains"],
                )
                setup, action = actions_for_task(task, root, "registry-test")
                self.assertEqual(setup.value, launch_target_for_family(family))
                self.assertEqual(action.selector, primary_selector_for_family(family))

    def test_virtual_sandbox_launches_registry_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = VirtualDesktopSandboxBackend(Path(temp_dir) / "sandbox.json")
            for spec in app_family_specs(include_unknown=False):
                backend.perform(
                    UiAction("launch_app", spec.launch_target, spec.launch_target)
                )
            selectors = {node.node_id for node in backend.snapshot()}
            for spec in app_family_specs(include_unknown=False):
                self.assertIn(spec.primary_selector, selectors)
            self.assertIn("office-ribbon", selectors)
            self.assertIn("layers-panel", selectors)
            self.assertIn("enterprise-detail-panel", selectors)

    def test_virtual_sandbox_preserves_action_history_on_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "sandbox.json"
            backend = VirtualDesktopSandboxBackend(state_path)
            capabilities = backend.capabilities()["capabilities"]
            self.assertIn("history-preserving-reset", capabilities)
            self.assertIn("multi-panel-app-surfaces", capabilities)

            backend.perform(UiAction("focus", "browser-address-bar"))
            backend.reset()
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload.get("agent_history") or []), 1)

    def test_virtual_sandbox_confines_full_system_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "sandbox.json"
            backend = VirtualDesktopSandboxBackend(state_path)
            capabilities = backend.capabilities()["capabilities"]
            self.assertIn("virtual-filesystem", capabilities)
            self.assertIn("virtual-processes", capabilities)
            self.assertIn("clipboard", capabilities)

            write_receipt = json.loads(
                backend.perform(
                    UiAction(
                        "write_file",
                        "",
                        metadata={
                            "path": "artifacts/workflows/new.txt",
                            "content": "hello sandbox",
                        },
                    )
                )
            )
            self.assertEqual(write_receipt.get("status"), "file-op-executed")

            read_receipt = json.loads(
                backend.perform(
                    UiAction(
                        "read_file",
                        "",
                        metadata={"path": "artifacts/workflows/new.txt"},
                    )
                )
            )
            self.assertEqual(
                read_receipt.get("file_op", {}).get("content"),
                "hello sandbox",
            )

            backend.perform(UiAction("set_clipboard", "", "copied text"))
            backend.perform(UiAction("execute_command", "", "python -V"))
            backend.perform(UiAction("open_modal", "", "Sandbox Dialog"))
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("clipboard"), "copied text")
            self.assertGreaterEqual(len(payload.get("virtual_processes") or []), 1)
            self.assertGreaterEqual(len(payload.get("modals") or []), 1)
            self.assertFalse((state_path.parent / "new.txt").exists())

    def test_adaptation_readiness_connects_training_and_live_fire_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_dir = root / ".agentos" / "adaptation_longrun" / "run"
            training_dir.mkdir(parents=True)
            (training_dir / "long_run_result.json").write_text(
                json.dumps(
                    {
                        "scale_report": {
                            "grounding_examples": 100_000,
                            "world_model_transitions": 100_000,
                            "meets_minimum_scale": True,
                        },
                        "underfill": {"underfilled": False, "missing_total": 0},
                    }
                ),
                encoding="utf-8",
            )
            live_fire_dir = root / ".agentos" / "live_fire_eval"
            live_fire_dir.mkdir(parents=True)
            (live_fire_dir / "heldout.json").write_text(
                json.dumps(
                    {
                        "heldout_metrics": {
                            "success_rate": 0.92,
                            "task_count": 50,
                        }
                    }
                ),
                encoding="utf-8",
            )

            readiness = collect_adaptation_readiness(root)
            self.assertEqual(readiness.status, "ready")
            self.assertTrue(readiness.connected)

            benchmarks = benchmark_status([], [], [], root, {"passed": True})
            self.assertEqual(benchmarks["adaptation_readiness"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
