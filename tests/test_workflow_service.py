from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.os_control.workflow.planner import (
    DesktopWorkflowPlanner,
)
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
    WorkflowArtifact,
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


class MetadataRecordingBackend(VerificationBackend):
    def __init__(self) -> None:
        super().__init__(
            reflect_value_in_snapshot=True,
            include_value_in_receipt=True,
        )
        self.action_metadata: list[dict[str, Any]] = []

    def perform(self, action) -> str:
        self.action_metadata.append(dict(action.metadata or {}))
        return super().perform(action)


class PreActionBlockingBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="file-list",
                role="List",
                name="Protected Files",
                focused=True,
            )
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class GoalLockBlockingBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="doc-canvas",
                role="Document",
                name="Document Canvas",
                focused=True,
            )
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class StaticDesktopWorkflowPlanner:
    def __init__(self, plan: DesktopWorkflowPlan) -> None:
        self._plan = plan

    def plan(self, objective: str) -> DesktopWorkflowPlan:
        self._plan.objective = objective
        return self._plan


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
            plan = DesktopWorkflowPlan(
                objective="",
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
                                "expected": (
                                    "The focused field contains the "
                                    "typed value."
                                ),
                                "target": (
                                    "automation_id=1001&&class_name=Edit"
                                ),
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
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            with self.assertRaises(WorkflowVerificationError):
                service.execute("write a report", backend)

            self.assertEqual(len(backend.actions), 1)
            self.assertEqual(backend.actions[0][1], "name=Document Canvas")

    def test_workflow_records_control_metadata_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = MetadataRecordingBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="control metadata plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="name=Document Canvas",
                        value="ledger-backed value",
                        description="Type a ledger-backed value",
                        metadata={
                            "verification_contract": {
                                "kind": "field_contains",
                                "expected": (
                                    "The field contains the value."
                                ),
                                "target": "name=Document Canvas",
                                "value": "ledger-backed value",
                                "required": True,
                            }
                        },
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            result = service.execute("write a safe report note", backend)

            first_receipt = result["receipts"][0]
            control = first_receipt["control"]
            self.assertEqual(control["control_route"], "structured_ui")
            self.assertIn("observation_fingerprint", control)
            self.assertIn("ledger_entry_id", control)
            self.assertIn("app_agent", control)
            self.assertIn("goal_lock", control)
            self.assertIn("speculation", control)
            self.assertIn("isolation", control)
            self.assertEqual(
                backend.action_metadata[0]["control"]["ledger_entry_id"],
                control["ledger_entry_id"],
            )
            app_signature = control["app_agent"]["app_signature"]
            self.assertTrue(app_signature)
            self.assertTrue(
                service.app_agent_runtime.policy_memory.preferred_channels(
                    app_signature,
                )
            )
            events = service.control_ledger.recent_events(limit=10)
            stages = [event["stage"] for event in events]
            self.assertIn("proposal", stages)
            self.assertIn("completion", stages)
            proposal = next(
                event for event in events if event["stage"] == "proposal"
            )
            self.assertEqual(
                proposal["payload"]["candidate_route"],
                "structured_ui",
            )
            self.assertIn(
                "stable_fingerprint",
                proposal["payload"]["observation_frame"],
            )

    def test_programmer_lane_generates_artifact_before_ui_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = MetadataRecordingBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="programmer lane plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="name=Document Canvas",
                        value="handoff note",
                        description="Type a handoff note",
                    )
                ],
                artifacts=[
                    WorkflowArtifact(
                        path="artifacts/workflows/report.md",
                        kind="report",
                        description="Workflow report",
                    )
                ],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            result = service.execute("write a report about orchestrators", backend)

            self.assertEqual(result["receipts"][0]["action_type"], "tool")
            self.assertEqual(len(backend.action_metadata), 1)
            report_path = Path(temp_dir) / "artifacts" / "workflows" / "report.md"
            self.assertTrue(report_path.exists())
            self.assertIn("orchestrators", report_path.read_text(encoding="utf-8"))
            self.assertTrue(
                any(
                    artifact["path"] == "artifacts/workflows/report.md"
                    for artifact in result["artifacts"]
                )
            )

    def test_goal_lock_blocks_unrelated_external_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = GoalLockBlockingBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="goal lock plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="open_url",
                        selector="browser-address-bar",
                        value="https://example.com",
                        description="Navigate away from the current editor task",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute(
                    "summarize current task progress in the document",
                    backend,
                )

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn("goal lock", failure["verification"]["reason"].lower())

    def test_pre_action_verifier_blocks_delete_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = PreActionBlockingBackend()
            step = DesktopWorkflowStep(
                action_type="delete_file",
                selector="protected-file",
                description="Delete a protected file",
                metadata={"path": "artifacts/workflows/report.md"},
            )
            plan = DesktopWorkflowPlan(
                objective="",
                mode="file-ops",
                app_target=None,
                summary="blocked delete plan",
                steps=[step],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute(
                    "summarize the protected files",
                    backend,
                )

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertEqual(
                failure["verification"]["kind"],
                "pre_action_verification",
            )
            self.assertIn(
                "requires approval",
                failure["verification"]["reason"],
            )
            events = service.control_ledger.recent_events(limit=10)
            stages = [event["stage"] for event in events]
            self.assertIn("proposal", stages)
            self.assertIn("failure", stages)
            golden_candidates = list(
                (Path(temp_dir) / "benchmarks" / "golden_traces").glob(
                    "workflow_*.json"
                )
            )
            self.assertTrue(golden_candidates)


if __name__ == "__main__":
    unittest.main()
