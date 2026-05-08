from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.os_control.workflow.planner import (
    DesktopWorkflowPlanner,
)
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
)
from agentos_orchestrator.os_control.workflow.service import (
    DesktopWorkflowService,
    WorkflowVerificationError,
)


class FakeSelectorRecoveryBackend:
    def __init__(self) -> None:
        self._nodes = [
            UiNode(
                node_id="doc-canvas",
                role="Document",
                name="Document Canvas",
            ),
            UiNode(node_id="app-window", role="Window", name="App Window"),
        ]

    def snapshot(self) -> list[UiNode]:
        return list(self._nodes)

    def perform(self, action) -> str:
        if action.action_type == "type" and action.selector == "document-canvas":
            return json.dumps({"status": "selector-not-found"})
        return json.dumps(
            {
                "status": "executed",
                "action": action.action_type,
                "selector": action.selector,
            }
        )


class AdaptiveOnlyBackend:
    def __init__(self) -> None:
        self._nodes = [
            UiNode(
                node_id="app-window",
                role="Window",
                name="Foobar",
                focused=True,
            ),
            UiNode(
                node_id="workspace-1",
                role="Document",
                name="Main Workspace",
                focused=True,
            ),
        ]

    def snapshot(self) -> list[UiNode]:
        return list(self._nodes)

    def perform(self, action) -> str:
        if action.action_type in {"type", "set_text", "set_value"}:
            return json.dumps(
                {
                    "status": "selector-not-found",
                    "action": action.action_type,
                    "selector": action.selector,
                    "value": action.value,
                }
            )
        payload = {
            "status": "executed",
            "action": action.action_type,
            "selector": action.selector,
            "value": action.value,
        }
        return json.dumps(payload)


class VerificationBackend:
    def __init__(
        self,
        *,
        reflect_value_in_snapshot: bool,
        include_value_in_receipt: bool,
    ) -> None:
        self.field_value = "Document Canvas"
        self.reflect_value_in_snapshot = reflect_value_in_snapshot
        self.include_value_in_receipt = include_value_in_receipt

    def snapshot(self) -> list[UiNode]:
        value = (
            self.field_value if self.reflect_value_in_snapshot else "Document Canvas"
        )
        return [UiNode(node_id="doc-canvas", role="Document", name=value)]

    def perform(self, action) -> str:
        if action.action_type in {"type", "set_text", "set_value"}:
            self.field_value = action.value or ""
            payload = {
                "status": (
                    "value-set" if self.include_value_in_receipt else "executed"
                ),
                "action": action.action_type,
                "selector": action.selector,
            }
            if self.include_value_in_receipt:
                payload["value"] = action.value
            return json.dumps(payload)
        return json.dumps({"status": "executed", "selector": action.selector})


class RecordingVerificationBackend(VerificationBackend):
    def __init__(self) -> None:
        super().__init__(
            reflect_value_in_snapshot=False,
            include_value_in_receipt=False,
        )
        self.actions: list[tuple[str, str]] = []

    def perform(self, action) -> str:
        self.actions.append((action.action_type, action.selector))
        return super().perform(action)


class WorkflowServiceTests(unittest.TestCase):
    def test_selector_recovery_uses_ranked_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = FakeSelectorRecoveryBackend()
            result = service.execute(
                "write a report about orchestrators",
                backend,
            )

            type_receipts = [
                item for item in result["receipts"] if item["action_type"] == "type"
            ]
            self.assertTrue(type_receipts)
            self.assertTrue(type_receipts[0]["recovery"]["applied"])
            self.assertEqual(
                type_receipts[0]["selector"],
                "name=Document Canvas",
            )

    def test_task_pack_multi_surface_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state.json")
            objective = (
                "search for autonomous systems and write a report, "
                "then create slides, then draft a python script"
            )
            result = service.execute(objective, backend)

            self.assertEqual(result["plan"]["mode"], "multi-app")
            launched = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]
            self.assertIn("winword.exe", launched)
            self.assertIn("powerpnt.exe", launched)
            self.assertIn("code", launched)

    def test_task_pack_spreadsheet_and_generic_app_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state2.json")
            objective = (
                "create a spreadsheet for benchmark scores and open file "
                "explorer to verify outputs"
            )
            result = service.execute(objective, backend)

            launched = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]
            self.assertIn("excel.exe", launched)
            self.assertIn("explorer.exe", launched)
            self.assertTrue(
                any(
                    item["action_type"] == "type"
                    and item["selector"] == "spreadsheet-grid"
                    for item in result["receipts"]
                )
            )

    def test_spreadsheet_cell_and_explorer_file_ops_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state3.json")
            objective = (
                "set B2 to benchmark score and open file explorer to "
                "rename report.md to report_final.md"
            )
            result = service.execute(objective, backend)

            self.assertTrue(
                any(
                    item["action_type"] == "type"
                    and item["selector"] == "spreadsheet-grid"
                    and isinstance(item["receipt"], dict)
                    and item["receipt"].get("cell_edit", {}).get("cell") == "B2"
                    for item in result["receipts"]
                )
            )
            self.assertTrue(
                any(
                    item["action_type"] == "rename_file"
                    and isinstance(item["receipt"], dict)
                    and item["receipt"].get("file_op", {}).get("operation") == "rename"
                    for item in result["receipts"]
                )
            )

    def test_clarification_loop_then_mixed_handoff_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state4.json")

            clarification = service.execute("do it", backend)
            self.assertEqual(clarification["status"], "clarification_required")

            objective = (
                "search for desktop control benchmarks and open file explorer "
                "to move notes.txt to archive, then draft a python script"
            )
            result = service.execute(objective, backend)
            launches = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]

            self.assertIn("msedge.exe", launches)
            self.assertIn("explorer.exe", launches)
            self.assertIn("code", launches)
            self.assertTrue(
                any(item["action_type"] == "open_url" for item in result["receipts"])
            )
            self.assertTrue(
                any(
                    item["action_type"] == "move_file"
                    and isinstance(item["receipt"], dict)
                    and item["receipt"].get("file_op", {}).get("operation") == "move"
                    for item in result["receipts"]
                )
            )

    def test_explorer_copy_operation_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state5.json")
            objective = (
                "open file explorer to copy artifacts/workflows/report.md "
                "to artifacts/archive/report.md"
            )
            result = service.execute(objective, backend)

            copy_receipts = [
                item["receipt"].get("file_op")
                for item in result["receipts"]
                if item["action_type"] == "copy_file"
                and isinstance(item.get("receipt"), dict)
                and item["receipt"].get("file_op")
            ]
            self.assertTrue(copy_receipts)
            self.assertEqual(copy_receipts[0]["operation"], "copy")

    def test_planner_prefers_explorer_target_for_file_ops(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("open file explorer to rename report.md to report_final.md")
        self.assertIn(plan.app_target, {"explorer.exe", None})
        self.assertTrue(any(step.action_type == "rename_file" for step in plan.steps))

    def test_open_paint_and_draw_generic_scene_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state_paint.json")
            result = service.execute(
                "open paint app and draw a city skyline",
                backend,
            )

            launches = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]
            self.assertIn("mspaint.exe", launches)

            draw_receipts = [
                item
                for item in result["receipts"]
                if item["action_type"] == "draw_path"
            ]
            self.assertGreaterEqual(len(draw_receipts), 1)
            self.assertIn(
                "drawing canvas",
                draw_receipts[-1]["selector"].lower().replace("=", " "),
            )

    def test_open_unseen_app_infers_generic_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(
                root / "sandbox_state_unseen_app.json"
            )
            result = service.execute(
                "open foobar app and summarize current task progress",
                backend,
            )

            launches = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]
            self.assertIn("foobar.exe", launches)
            self.assertTrue(
                any(
                    item["action_type"] == "type"
                    and "operator intent"
                    in (item.get("receipt", {}) or {}).get("value", "").lower()
                    for item in result["receipts"]
                    if isinstance(item.get("receipt"), dict)
                )
            )

    def test_find_stock_and_analyze_routes_to_browser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state_stock.json")
            result = service.execute(
                "find tesla stock and analyze it",
                backend,
            )

            launches = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]
            self.assertIn("msedge.exe", launches)
            open_urls = [
                item
                for item in result["receipts"]
                if item["action_type"] == "open_url"
                and isinstance(item.get("receipt"), dict)
            ]
            self.assertTrue(open_urls)
            query_url = str(open_urls[0]["receipt"].get("value", "")).lower()
            self.assertIn("bing.com/search", query_url)
            self.assertIn("tesla", query_url)

    def test_adaptive_reasoner_applies_generic_surface_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            service.max_adaptive_steps = 2
            backend = AdaptiveOnlyBackend()
            result = service.execute(
                "open explorer and inspect files",
                backend,
            )

            adaptive_receipts = [
                item for item in result["receipts"] if item.get("adaptive")
            ]
            self.assertTrue(adaptive_receipts)
            self.assertTrue(
                any(
                    item["action_type"] in {"type", "set_text"}
                    for item in adaptive_receipts
                )
            )
            self.assertTrue(
                all(
                    item.get("reasoner") in {"heuristic", "gemini"}
                    for item in adaptive_receipts
                )
            )

    def test_perform_with_recovery_records_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = VerificationBackend(
                reflect_value_in_snapshot=True,
                include_value_in_receipt=True,
            )
            step = DesktopWorkflowStep(
                action_type="type",
                selector="name=Document Canvas",
                value="AgentOS verified value",
                description="Type into the document",
                metadata={
                    "verification_contract": {
                        "kind": "field_contains",
                        "expected": "The focused field contains the typed value.",
                        "target": "name=Document Canvas",
                        "value": "AgentOS verified value",
                        "required": True,
                    }
                },
            )

            receipt, selector, recovery, verification = service._perform_with_recovery(
                backend,
                step,
            )

            self.assertEqual(selector, "name=Document Canvas")
            self.assertEqual(json.loads(receipt)["status"], "value-set")
            self.assertFalse(recovery.get("verification_failed", False))
            self.assertTrue(verification["required"])
            self.assertTrue(verification["matched"])

    def test_perform_with_recovery_flags_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = VerificationBackend(
                reflect_value_in_snapshot=False,
                include_value_in_receipt=False,
            )
            step = DesktopWorkflowStep(
                action_type="type",
                selector="automation_id=1001&&class_name=Edit",
                value="AgentOS missing value",
                description="Type into the document",
                metadata={
                    "verification_contract": {
                        "kind": "field_contains",
                        "expected": "The focused field contains the typed value.",
                        "target": "automation_id=1001&&class_name=Edit",
                        "value": "AgentOS missing value",
                        "required": True,
                    }
                },
            )

            with self.assertRaises(WorkflowVerificationError) as caught:
                service._perform_with_recovery(
                    backend,
                    step,
                )

            failure = caught.exception.asdict()
            self.assertTrue(failure["verification"]["required"])
            self.assertFalse(failure["verification"]["matched"])
            self.assertTrue(failure["recovery"]["verification_failed"])
            self.assertIn(
                "typed value",
                failure["recovery"]["verification_reason"],
            )

    def test_execute_hard_stops_on_required_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = RecordingVerificationBackend()
            service.planner.plan = lambda objective: DesktopWorkflowPlan(
                objective=objective,
                mode="report",
                app_target=None,
                summary="verification-sensitive plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="automation_id=1001&&class_name=Edit",
                        value="fail verification",
                        description="First step must verify",
                        metadata={
                            "verification_contract": {
                                "kind": "field_contains",
                                "expected": "The focused field contains the typed value.",
                                "target": "automation_id=1001&&class_name=Edit",
                                "value": "fail verification",
                                "required": True,
                            }
                        },
                    ),
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="automation_id=1001&&class_name=Edit",
                        value="should not execute",
                        description="Must never run",
                    ),
                ],
                artifacts=[],
            )

            with self.assertRaises(WorkflowVerificationError):
                service.execute("write a report", backend)

            self.assertEqual(len(backend.actions), 1)
            self.assertEqual(backend.actions[0][1], "name=Document Canvas")


if __name__ == "__main__":
    unittest.main()
