from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from agentos_orchestrator.app_family_registry import (
    spec_for_family,
)
from agentos_orchestrator.cognition.app_agent_runtime import (
    AppAgentSession,
    AppAgentSkillPack,
)
from agentos_orchestrator.cognition.live_fire_eval_recipes import (
    abstract_state,
)
from agentos_orchestrator.research.models import ResearchSource
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.os_control.workflow.planner import (
    DesktopWorkflowPlanner,
)
from agentos_orchestrator.os_control.workflow.reasoner import (
    DesktopWorkflowReasoner,
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


def _sample_research_sources() -> list[ResearchSource]:
    return [
        ResearchSource(
            provider="web-search",
            title="Tesla investor update highlights delivery outlook",
            url="https://example.com/tesla-update",
            year=2026,
            abstract=(
                "Tesla reiterated its delivery guidance and highlighted a "
                "new cost-reduction program in its latest investor update."
            ),
        ),
        ResearchSource(
            provider="google-news-rss",
            title="Analysts compare Tesla margin trends with EV peers",
            url="https://example.com/tesla-margins",
            year=2026,
            abstract=(
                "Analysts compared Tesla margins with major EV peers and "
                "noted continued pricing pressure alongside software upside."
            ),
        ),
    ]


def _sample_research_brief_markdown() -> str:
    return "\n".join(
        [
            "# Workflow Research Brief",
            "",
            "## Objective",
            "Universal OS control agents benchmark comparison",
            "",
            "## Query",
            "universal os control agents benchmark comparison",
            "",
            "## Coverage",
            "Collected 2 provider-backed sources before any browser-first UI handoff.",
            "",
            "## Sources",
            "1. [OSWorld-Verified benchmark audit](https://example.com/osworld)",
            "   Provider: web-search | Year: 2026",
            "   Evidence: OSWorld-Verified documents repaired benchmark tasks and centralized verification for cross-app desktop evaluation.",
            "2. [OpenCUA capability update](https://example.com/opencua)",
            "   Provider: google-news-rss | Year: 2026",
            "   Evidence: OpenCUA reports open computer-use baselines and reflective action supervision for general desktop tasks.",
            "",
            "## Next Step",
            "Use this brief as the evidence-backed handoff input for the final deliverable.",
            "",
        ]
    )


class ToolOnlyBackend:
    def snapshot(self) -> list[UiNode]:
        return []

    def perform(self, action) -> str:
        del action
        return json.dumps({"status": "executed"})


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


class CanvasRoutingBackend:
    def __init__(self) -> None:
        self.action_metadata: list[dict[str, Any]] = []

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="drawing-canvas",
                role="Canvas",
                name="Design Canvas",
                focused=True,
            ),
            UiNode(node_id="layers", role="Pane", name="Layers"),
        ]

    def perform(self, action) -> str:
        self.action_metadata.append(dict(action.metadata or {}))
        return json.dumps(
            {
                "status": "executed",
                "action": action.action_type,
                "selector": action.selector,
            }
        )


class TerminalRoutingBackend:
    def __init__(self) -> None:
        self.action_metadata: list[dict[str, Any]] = []

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="terminal-doc",
                role="Document",
                name="PowerShell Terminal",
                focused=True,
            ),
            UiNode(node_id="terminal-pane", role="Pane", name="Console"),
        ]

    def perform(self, action) -> str:
        self.action_metadata.append(dict(action.metadata or {}))
        return json.dumps(
            {
                "status": "executed",
                "action": action.action_type,
                "selector": action.selector,
                "value": action.value,
            }
        )


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


class CodeToolBypassBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="explorer-file-list",
                role="List",
                name="Explorer File List",
                focused=True,
            )
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class ApiPromotionBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="api-dashboard",
                role="Pane",
                name="API Dashboard",
                focused=True,
                metadata={"api": "http"},
            ),
            UiNode(node_id="refresh-button", role="Button", name="Refresh"),
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class TradingTerminalPolicyBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="order-ticket",
                role="Edit",
                name="Order Ticket",
                focused=True,
            ),
            UiNode(node_id="watchlist", role="Table", name="Watchlist"),
            UiNode(node_id="confirm-order", role="Button", name="Confirm"),
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class ChatPolicyBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="conversation-list",
                role="List",
                name="Conversation",
            ),
            UiNode(
                node_id="chat-composer",
                role="Edit",
                name="Chat Composer",
                focused=True,
            ),
            UiNode(
                node_id="channel-broadcast",
                role="Button",
                name="Channel Broadcast",
            ),
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class BrowserPolicyBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="browser-address",
                role="Edit",
                name="Address and search bar",
                focused=True,
            ),
            UiNode(node_id="checkout", role="Button", name="Checkout"),
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class FileDialogPolicyBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="file-name",
                role="Edit",
                name="File Name",
                focused=True,
            ),
            UiNode(node_id="save-button", role="Button", name="Save"),
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class EnterpriseGridPolicyBackend:
    def __init__(self) -> None:
        self.perform_calls = 0

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="record-grid",
                role="Table",
                name="Enterprise Record Grid",
                focused=True,
            ),
            UiNode(node_id="bulk-delete", role="Button", name="Bulk Delete"),
        ]

    def perform(self, action) -> str:
        del action
        self.perform_calls += 1
        return json.dumps({"status": "executed"})


class _LocalApiHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"allow": ["GET", "OPTIONS"]}')

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "source": "local-api"}')

    def log_message(self, fmt: str, *args: Any) -> None:
        del fmt, args


def _start_local_api_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LocalApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/health"


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


class ClipboardHandoffBackend:
    def __init__(self) -> None:
        self.field_value = "Document Canvas"
        self.clipboard = ""
        self.actions: list[tuple[str, str, str | None]] = []
        self.action_metadata: list[dict[str, Any]] = []

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="doc-canvas",
                role="Document",
                name=self.field_value,
                focused=True,
            )
        ]

    def perform(self, action) -> str:
        self.actions.append((action.action_type, action.selector, action.value))
        self.action_metadata.append(dict(action.metadata or {}))
        if action.action_type in {"set_clipboard", "clipboard_copy"}:
            self.clipboard = str(action.value or "")
            return json.dumps(
                {"status": "clipboard-updated", "clipboard": self.clipboard}
            )
        if action.action_type in {"type", "set_text", "set_value"}:
            self.field_value = str(action.value or "")
            return json.dumps(
                {
                    "status": "value-set",
                    "selector": action.selector,
                    "value": action.value,
                }
            )
        return json.dumps(
            {
                "status": "executed",
                "action": action.action_type,
                "selector": action.selector,
                "value": action.value,
            }
        )


class _StubUniversalRun:
    def __init__(self) -> None:
        self.run_id = "stub-universal-run"
        self.success = True
        self.steps: list[Any] = []
        self.exploration_probes_used = 0
        self.mcts_simulations_run = 0
        self.final_state = {"status": "stubbed"}


class RepairingUniversalAgentStub:
    def repair_bootstrap_plan(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        *,
        backend: Any | None = None,
    ) -> DesktopWorkflowPlan:
        del plan, backend
        return DesktopWorkflowPlanner().plan(objective)

    def run_with_planned_bootstrap(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
    ) -> _StubUniversalRun:
        del objective, plan
        return _StubUniversalRun()


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
            with patch(
                "agentos_orchestrator.research.deep_research.DeepResearchEngine._search_query_across_providers",
                return_value=_sample_research_sources(),
            ):
                result = service.execute(objective, backend)
            launches = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]

            self.assertIn("explorer.exe", launches)
            self.assertIn("code", launches)
            self.assertTrue(
                any(
                    item["selector"] == "tool_executor:workflow_research"
                    for item in result["receipts"]
                )
            )
            self.assertFalse(
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

    def test_planner_routes_local_file_search_to_explorer_not_browser(self) -> None:
        planner = DesktopWorkflowPlanner()

        plan = planner.plan(
            "find the latest invoice PDF in Downloads and rename it to april-invoice.pdf"
        )

        self.assertEqual(plan.intent.get("search_scope"), "local_files")
        self.assertEqual(plan.intent.get("source_surface"), "file_explorer")
        self.assertEqual(plan.app_target, "explorer.exe")
        self.assertTrue(any(step.action_type == "rename_file" for step in plan.steps))
        self.assertFalse(any(step.action_type == "open_url" for step in plan.steps))

    def test_planner_distinguishes_clipboard_copy_from_file_copy(self) -> None:
        planner = DesktopWorkflowPlanner()

        plan = planner.plan(
            "search Chrome for the nearest UPS store and copy the address into Notepad"
        )

        self.assertEqual(plan.intent.get("copy_semantics"), "text")
        self.assertTrue(plan.intent.get("cross_app"))
        self.assertEqual(plan.intent.get("source_surface"), "browser")
        self.assertEqual(plan.intent.get("destination_surface"), "editor")
        self.assertTrue(any(step.action_type == "set_clipboard" for step in plan.steps))
        self.assertTrue(any(step.selector == "document-canvas" for step in plan.steps))
        self.assertFalse(any(step.action_type == "copy_file" for step in plan.steps))

    def test_planner_routes_settings_toggle_to_settings_surface(self) -> None:
        planner = DesktopWorkflowPlanner()

        plan = planner.plan("Open Settings and turn on Night Light.")

        self.assertEqual(plan.intent.get("source_surface"), "settings")
        self.assertEqual(plan.app_target, "settings.exe")
        self.assertIn("search_settings", plan.intent.get("operations") or [])
        self.assertIn("toggle_setting", plan.intent.get("operations") or [])
        self.assertTrue(
            any(step.selector == "settings-search-box" for step in plan.steps)
        )
        self.assertTrue(any(step.selector == "settings-toggle" for step in plan.steps))
        self.assertFalse(any(step.selector == "app-workspace" for step in plan.steps))

    def test_planner_routes_attachment_workflow_into_email_family(self) -> None:
        planner = DesktopWorkflowPlanner()

        plan = planner.plan("Attach the PDF from Downloads to an email draft for Alex.")

        self.assertTrue(plan.intent.get("cross_app"))
        self.assertEqual(plan.intent.get("source_surface"), "file_explorer")
        self.assertEqual(plan.intent.get("destination_surface"), "email")
        self.assertIn("attach_file", plan.intent.get("operations") or [])
        self.assertTrue(any(step.selector == "email-to-field" for step in plan.steps))
        self.assertTrue(
            any(step.selector == "email-attachment-field" for step in plan.steps)
        )
        self.assertFalse(any(step.selector == "app-workspace" for step in plan.steps))

    def test_planner_routes_email_invite_into_calendar_family(self) -> None:
        planner = DesktopWorkflowPlanner()

        plan = planner.plan(
            "Find the Zoom invite in my email and put it on my calendar."
        )

        self.assertTrue(plan.intent.get("cross_app"))
        self.assertEqual(plan.intent.get("source_surface"), "email")
        self.assertEqual(plan.intent.get("destination_surface"), "calendar")
        self.assertIn("search_email", plan.intent.get("operations") or [])
        self.assertIn("create_calendar_event", plan.intent.get("operations") or [])
        self.assertTrue(any(step.selector == "email-search-box" for step in plan.steps))
        self.assertTrue(
            any(step.selector == "calendar-event-editor" for step in plan.steps)
        )

    def test_planner_emits_explicit_programmer_tool_step(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("write a workflow handoff report from current receipts")

        self.assertTrue(plan.steps)
        self.assertEqual(plan.steps[0].action_type, "tool")
        self.assertEqual(plan.steps[0].selector, "tool_executor:workflow_programmer")
        tool_request = plan.steps[0].metadata.get("tool_request") or {}
        self.assertEqual(tool_request.get("mode"), "report")
        self.assertTrue(tool_request.get("outputs"))

    def test_planner_emits_research_tool_for_research_report_objective(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("write a report about universal OS control agents")

        self.assertTrue(plan.steps)
        self.assertEqual(plan.steps[0].action_type, "tool")
        self.assertEqual(plan.steps[0].selector, "tool_executor:workflow_research")
        self.assertTrue(
            any(artifact.kind == "research-brief" for artifact in plan.artifacts)
        )
        self.assertTrue(any(artifact.kind == "report" for artifact in plan.artifacts))
        self.assertTrue(
            any(
                step.selector == "tool_executor:workflow_programmer"
                for step in plan.steps
            )
        )

    def test_planner_emits_research_tool_for_presentation_objective(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("create a presentation about universal OS control agents")

        self.assertTrue(plan.steps)
        self.assertEqual(plan.steps[0].selector, "tool_executor:workflow_research")
        self.assertTrue(
            any(artifact.kind == "research-brief" for artifact in plan.artifacts)
        )
        self.assertTrue(
            any(artifact.kind == "presentation-outline" for artifact in plan.artifacts)
        )
        presentation_steps = [
            step
            for step in plan.steps
            if step.action_type == "type" and step.selector == "presentation-canvas"
        ]
        self.assertTrue(presentation_steps)
        self.assertIn("presentation_outline.md", presentation_steps[0].value or "")
        self.assertNotIn("research_brief.md", presentation_steps[0].value or "")

    def test_planner_emits_api_call_for_explicit_local_endpoint(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("probe http://127.0.0.1:8765/api/health")

        api_steps = [step for step in plan.steps if step.action_type == "api_call"]
        self.assertTrue(api_steps)
        self.assertEqual(api_steps[0].selector, "http://127.0.0.1:8765/api/health")
        self.assertIsNone(plan.app_target)

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

    def test_settings_task_executes_without_generic_save_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state_settings.json")
            result = service.execute("Open Settings and turn on Night Light.", backend)

            launches = [
                item["receipt"].get("launched")
                for item in result["receipts"]
                if isinstance(item.get("receipt"), dict)
                and item["receipt"].get("launched")
            ]
            selectors = [item["selector"] for item in result["receipts"]]

            self.assertIn("settings.exe", launches)
            self.assertIn("settings-search-box", selectors)
            self.assertIn("settings-toggle", selectors)
            self.assertFalse(
                any(item["action_type"] == "hotkey" for item in result["receipts"])
            )

    def test_find_stock_and_analyze_prefers_research_tool_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state_stock.json")
            with patch(
                "agentos_orchestrator.research.deep_research.DeepResearchEngine._search_query_across_providers",
                return_value=_sample_research_sources(),
            ):
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
            self.assertNotIn("msedge.exe", launches)
            self.assertFalse(
                any(item["action_type"] == "open_url" for item in result["receipts"])
            )
            self.assertEqual(result["receipts"][0]["action_type"], "tool")
            self.assertEqual(
                result["receipts"][0]["selector"],
                "tool_executor:workflow_research",
            )
            brief_path = (
                root
                / "artifacts"
                / "workflows"
                / "find-tesla-stock-and-analyze-it"
                / "research_brief.md"
            )
            self.assertTrue(brief_path.exists())
            self.assertIn("Tesla", brief_path.read_text(encoding="utf-8"))

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

    def test_reasoner_emits_programmer_tool_before_ui_handoff(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("write a workflow handoff report from current receipts")
        reasoner = DesktopWorkflowReasoner()

        decision = reasoner.next_decision(
            "write a report about adaptive desktop workflows",
            plan,
            [UiNode("doc", "Document", "Document Canvas", focused=True)],
            [],
        )

        self.assertIsNotNone(decision.step)
        assert decision.step is not None
        self.assertEqual(decision.step.action_type, "tool")
        self.assertEqual(decision.step.selector, "tool_executor:workflow_programmer")

    def test_reasoner_emits_research_tool_before_ui_handoff(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan("find tesla stock and analyze it")
        reasoner = DesktopWorkflowReasoner()

        decision = reasoner.next_decision(
            "find tesla stock and analyze it",
            plan,
            [UiNode("doc", "Document", "Document Canvas", focused=True)],
            [],
        )

        self.assertIsNotNone(decision.step)
        assert decision.step is not None
        self.assertEqual(decision.step.action_type, "tool")
        self.assertEqual(decision.step.selector, "tool_executor:workflow_research")

    def test_reasoner_emits_api_call_for_api_surface_node(self) -> None:
        endpoint = "http://127.0.0.1:9000/api/status"
        reasoner = DesktopWorkflowReasoner()
        plan = DesktopWorkflowPlan(
            objective="refresh api dashboard",
            mode="app-task",
            app_target=None,
            summary="api surface plan",
            steps=[],
            artifacts=[],
        )

        decision = reasoner.next_decision(
            "refresh the local api dashboard",
            plan,
            [
                UiNode(
                    "api-dashboard",
                    "Pane",
                    "API Dashboard",
                    focused=True,
                    metadata={"endpoint": endpoint, "api": "http"},
                )
            ],
            [],
        )

        self.assertIsNotNone(decision.step)
        assert decision.step is not None
        self.assertEqual(decision.step.action_type, "api_call")
        self.assertEqual(decision.step.selector, endpoint)

    def test_reasoner_prefers_editor_surface_for_clipboard_handoff(self) -> None:
        planner = DesktopWorkflowPlanner()
        plan = planner.plan(
            "search Chrome for the nearest UPS store and copy the address into Notepad"
        )
        reasoner = DesktopWorkflowReasoner()

        decision = reasoner.next_decision(
            "search Chrome for the nearest UPS store and copy the address into Notepad",
            plan,
            [
                UiNode("address", "Edit", "Address and search bar"),
                UiNode("doc", "Document", "Document Canvas", focused=True),
            ],
            [],
        )

        self.assertIsNotNone(decision.step)
        assert decision.step is not None
        self.assertEqual(decision.step.selector, "name=Document Canvas")
        self.assertIn("Address result for:", decision.step.value or "")

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
                                    "The focused field contains the typed value."
                                ),
                                "target": ("automation_id=1001&&class_name=Edit"),
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
                                "expected": ("The field contains the value."),
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
            proposal = next(event for event in events if event["stage"] == "proposal")
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

    def test_programmer_lane_synthesizes_report_from_research_brief(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path = (
                root
                / "artifacts"
                / "workflows"
                / "benchmark-report"
                / "research_brief.md"
            )
            brief_path.parent.mkdir(parents=True, exist_ok=True)
            brief_path.write_text(
                _sample_research_brief_markdown(),
                encoding="utf-8",
            )
            service = DesktopWorkflowService(root)
            backend = ToolOnlyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="research-backed report plan",
                steps=[],
                artifacts=[
                    WorkflowArtifact(
                        path=("artifacts/workflows/benchmark-report/research_brief.md"),
                        kind="research-brief",
                        description="Evidence-backed brief",
                    ),
                    WorkflowArtifact(
                        path="artifacts/workflows/benchmark-report/report.md",
                        kind="report",
                        description="Workflow report",
                    ),
                ],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            result = service.execute(
                "write a report about universal OS control agents",
                backend,
            )

            report_path = (
                root / "artifacts" / "workflows" / "benchmark-report" / "report.md"
            )
            self.assertEqual(
                result["receipts"][0]["selector"], "tool_executor:workflow_programmer"
            )
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn(
                "synthesized directly from the workflow research brief", report_text
            )
            self.assertIn("OSWorld-Verified benchmark audit", report_text)
            self.assertIn("OpenCUA capability update", report_text)

    def test_programmer_lane_synthesizes_presentation_from_research_brief(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path = (
                root
                / "artifacts"
                / "workflows"
                / "benchmark-deck"
                / "research_brief.md"
            )
            brief_path.parent.mkdir(parents=True, exist_ok=True)
            brief_path.write_text(
                _sample_research_brief_markdown(),
                encoding="utf-8",
            )
            service = DesktopWorkflowService(root)
            backend = ToolOnlyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="presentation",
                app_target=None,
                summary="research-backed presentation plan",
                steps=[],
                artifacts=[
                    WorkflowArtifact(
                        path=("artifacts/workflows/benchmark-deck/research_brief.md"),
                        kind="research-brief",
                        description="Evidence-backed brief",
                    ),
                    WorkflowArtifact(
                        path=(
                            "artifacts/workflows/benchmark-deck/presentation_outline.md"
                        ),
                        kind="presentation-outline",
                        description="Slide outline",
                    ),
                ],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            result = service.execute(
                "create a presentation about universal OS control agents",
                backend,
            )

            outline_path = (
                root
                / "artifacts"
                / "workflows"
                / "benchmark-deck"
                / "presentation_outline.md"
            )
            self.assertEqual(
                result["receipts"][0]["selector"], "tool_executor:workflow_programmer"
            )
            self.assertTrue(outline_path.exists())
            outline_text = outline_path.read_text(encoding="utf-8")
            self.assertIn("## Slide 3: Key Evidence", outline_text)
            self.assertIn("OSWorld-Verified benchmark audit", outline_text)
            self.assertIn("OpenCUA capability update", outline_text)

    def test_visual_heavy_canvas_click_prefers_native_vision_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = CanvasRoutingBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="canvas route plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="click",
                        selector="drawing-canvas",
                        description="Click the design canvas",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            result = service.execute("edit the design canvas", backend)

            control = result["receipts"][0]["control"]
            self.assertEqual(control["control_route"], "native_vision")
            self.assertEqual(control["app_agent"]["family"], "design_canvas")

    def test_terminal_text_entry_keeps_structured_ui_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = TerminalRoutingBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="terminal route plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="name=PowerShell Terminal",
                        value="dir",
                        description="Type a terminal command",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            result = service.execute("run a shell command in terminal", backend)

            control = result["receipts"][0]["control"]
            self.assertEqual(control["control_route"], "structured_ui")
            self.assertEqual(control["app_agent"]["family"], "terminal")

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

    def test_goal_lock_blocks_file_copy_when_intent_requires_clipboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = GoalLockBlockingBackend()
            intent = (
                DesktopWorkflowPlanner()
                .plan(
                    "search Chrome for the nearest UPS store and copy the address into Notepad"
                )
                .intent
            )
            plan = DesktopWorkflowPlan(
                objective="",
                mode="app-task",
                app_target=None,
                summary="clipboard safety plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="copy_file",
                        selector="explorer-file-list",
                        value="ups.txt -> notes.txt",
                        description="Misrouted file copy",
                        metadata={
                            "operation": "copy",
                            "source": "ups.txt",
                            "destination": "notes.txt",
                        },
                    )
                ],
                artifacts=[],
                intent=intent,
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute(
                    "search Chrome for the nearest UPS store and copy the address into Notepad",
                    backend,
                )

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn(
                "clipboard transfer",
                failure["verification"]["reason"],
            )

    def test_execute_resolves_cross_app_handoff_blackboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = ClipboardHandoffBackend()
            service._execute_adaptive_steps = cast(
                Any,
                lambda *args, **kwargs: None,
            )

            result = service.execute(
                "search Chrome for the nearest UPS store and copy the address into Notepad",
                backend,
            )

            self.assertTrue(any(item[0] == "set_clipboard" for item in backend.actions))
            self.assertEqual(backend.field_value, backend.clipboard)
            self.assertTrue(
                any(
                    meta.get("workflow_handoff", {}).get("copied_text")
                    for meta in backend.action_metadata
                )
            )
            self.assertFalse(
                any(item["action_type"] == "copy_file" for item in result["receipts"])
            )

    def test_universal_agent_repairs_bootstrap_plan_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            from agentos_orchestrator.os_control import (
                VirtualDesktopSandboxBackend,
            )

            service = DesktopWorkflowService(root)
            backend = VirtualDesktopSandboxBackend(root / "sandbox_state_repair.json")
            stale_plan = DesktopWorkflowPlan(
                objective="",
                mode="app-task",
                app_target="msedge.exe",
                summary="stale browser bootstrap",
                steps=[
                    DesktopWorkflowStep(
                        action_type="launch_app",
                        selector="msedge.exe",
                        value="msedge.exe",
                        description="Launch a stale browser route",
                    ),
                    DesktopWorkflowStep(
                        action_type="open_url",
                        selector="browser-address-bar",
                        value="https://www.bing.com/search?q=invoice+pdf",
                        description="Navigate to a stale browser search",
                    ),
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(stale_plan))
            service._universal_agent = RepairingUniversalAgentStub()
            service._universal_backend_id = id(backend)
            service._execute_adaptive_steps = cast(
                Any,
                lambda *args, **kwargs: None,
            )

            result = service.execute(
                "find the latest invoice PDF in Downloads and rename it to april-invoice.pdf",
                backend,
            )

            self.assertFalse(
                any(item["action_type"] == "open_url" for item in result["receipts"])
            )
            self.assertTrue(
                any(item["action_type"] == "rename_file" for item in result["receipts"])
            )

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

    def test_code_tool_route_avoids_backend_for_file_op(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "notes.txt"
            source.write_text("route replacement", encoding="utf-8")
            service = DesktopWorkflowService(root)
            backend = CodeToolBypassBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="file-ops",
                app_target=None,
                summary="code tool file op plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="copy_file",
                        selector="explorer-file-list",
                        value="notes.txt -> archive/notes_copy.txt",
                        description="Copy the note through the code tool lane",
                        metadata={
                            "operation": "copy",
                            "source": "notes.txt",
                            "destination": "archive/notes_copy.txt",
                        },
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))
            service._execute_adaptive_steps = cast(
                Any,
                lambda *args, **kwargs: None,
            )

            result = service.execute(
                "copy notes.txt to archive/notes_copy.txt",
                backend,
            )

            copied = root / "archive" / "notes_copy.txt"
            self.assertTrue(copied.exists())
            self.assertEqual(backend.perform_calls, 0)
            control = result["receipts"][0]["control"]
            self.assertEqual(control["control_route"], "code_tool")
            self.assertEqual(control["materialized_action_type"], "tool")

    def test_api_route_uses_policy_memory_probe_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = DesktopWorkflowService(root)
            backend = ApiPromotionBackend()
            nodes = backend.snapshot()
            profile = service.capability_profiler.profile(
                abstract_state("unknown", nodes),
                nodes,
            )
            server, thread, endpoint = _start_local_api_server()
            try:
                service.app_agent_runtime.policy_memory.record(
                    profile.app_signature,
                    "refresh api dashboard",
                    UiAction(
                        "api_call",
                        endpoint,
                        metadata={"control_channel": "api"},
                    ),
                    True,
                    control_channel="api",
                )
                plan = DesktopWorkflowPlan(
                    objective="",
                    mode="report",
                    app_target=None,
                    summary="api promotion plan",
                    steps=[
                        DesktopWorkflowStep(
                            action_type="click",
                            selector="refresh-button",
                            description="Refresh the dashboard data",
                        )
                    ],
                    artifacts=[],
                )
                service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

                result = service.execute("refresh api dashboard", backend)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            self.assertEqual(backend.perform_calls, 0)
            control = result["receipts"][0]["control"]
            self.assertEqual(control["control_route"], "api_mcp")
            self.assertEqual(control["materialized_action_type"], "tool")
            receipt = result["receipts"][0]["receipt"]
            self.assertTrue(receipt["success"])
            self.assertEqual(receipt["kind"], "http_probe")

    def test_explicit_api_call_plan_materializes_direct_endpoint_without_backend(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = DesktopWorkflowService(root)
            backend = ApiPromotionBackend()
            server, thread, endpoint = _start_local_api_server()
            try:
                plan = DesktopWorkflowPlan(
                    objective="",
                    mode="app-task",
                    app_target=None,
                    summary="direct api plan",
                    steps=[
                        DesktopWorkflowStep(
                            action_type="api_call",
                            selector=endpoint,
                            description="Probe the local API directly",
                            metadata={"method": "GET"},
                        )
                    ],
                    artifacts=[],
                )
                service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

                result = service.execute("probe the local api surface", backend)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            self.assertEqual(backend.perform_calls, 0)
            control = result["receipts"][0]["control"]
            self.assertEqual(control["control_route"], "api_mcp")
            self.assertEqual(control["materialized_action_type"], "tool")
            receipt = result["receipts"][0]["receipt"]
            self.assertTrue(receipt["success"])
            self.assertEqual(receipt["kind"], "http_probe")

    def test_trading_terminal_policy_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = TradingTerminalPolicyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="trading approval plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="click",
                        selector="name=Confirm",
                        description="Confirm the order ticket",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))
            trading_skill_pack = AppAgentSkillPack(
                skill_pack_id="trading_terminal:test",
                family="trading_terminal",
                app_context="trading_terminal",
                app_signature="trading-terminal-test",
                preferred_channels=["api", "accessibility"],
                affordance_hints=[],
                verification_contracts=[],
                repair_recipes=[],
                action_policy={
                    "require_approval_selectors": ["confirm"],
                },
            )
            service.app_agent_runtime.resolve = cast(
                Any,
                lambda profile, objective, nodes: AppAgentSession(
                    skill_pack=trading_skill_pack,
                    adapter_context={},
                    objective=objective,
                    nodes_seen=len(nodes or []),
                ),
            )

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute(
                    "review the watchlist and current positions",
                    backend,
                )

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn(
                "trading_terminal policy",
                failure["verification"]["reason"],
            )
            self.assertIn(
                "requires approval",
                failure["verification"]["reason"],
            )

    def test_chat_policy_blocks_channel_broadcast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = ChatPolicyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="report",
                app_target=None,
                summary="chat forbid plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="click",
                        selector="name=Channel Broadcast",
                        description="Broadcast a message to the full channel",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute(
                    "review the current conversation thread",
                    backend,
                )

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn(
                "chat_app policy",
                failure["verification"]["reason"],
            )
            self.assertIn("forbids", failure["verification"]["reason"])

    def test_browser_policy_requires_approval_for_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = BrowserPolicyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="app-task",
                app_target=None,
                summary="browser policy plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="click",
                        selector="name=Checkout",
                        description="Checkout the current cart",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))
            browser_spec = spec_for_family("browser")
            browser_skill_pack = AppAgentSkillPack(
                skill_pack_id="browser:test",
                family=browser_spec.family,
                app_context=browser_spec.app_context,
                app_signature="browser-test",
                preferred_channels=list(browser_spec.preferred_channels),
                affordance_hints=list(browser_spec.affordance_hints),
                verification_contracts=list(browser_spec.verification_contracts),
                repair_recipes=list(browser_spec.repair_recipes),
                action_policy=dict(browser_spec.action_policy),
            )
            service.app_agent_runtime.resolve = cast(
                Any,
                lambda profile, objective, nodes: AppAgentSession(
                    skill_pack=browser_skill_pack,
                    adapter_context={},
                    objective=objective,
                    nodes_seen=len(nodes or []),
                ),
            )

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute("review the current cart and page state", backend)

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn("browser policy", failure["verification"]["reason"])
            self.assertIn("requires approval", failure["verification"]["reason"])

    def test_file_dialog_policy_forbids_system_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = FileDialogPolicyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="app-task",
                app_target=None,
                summary="file dialog policy plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="name=File Name",
                        value=r"C:\Windows\System32\drivers\etc\hosts",
                        description="Attempt to save into a protected system path",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))
            file_dialog_spec = spec_for_family("file_dialog")
            file_dialog_skill_pack = AppAgentSkillPack(
                skill_pack_id="file_dialog:test",
                family=file_dialog_spec.family,
                app_context=file_dialog_spec.app_context,
                app_signature="file-dialog-test",
                preferred_channels=list(file_dialog_spec.preferred_channels),
                affordance_hints=list(file_dialog_spec.affordance_hints),
                verification_contracts=list(file_dialog_spec.verification_contracts),
                repair_recipes=list(file_dialog_spec.repair_recipes),
                action_policy=dict(file_dialog_spec.action_policy),
            )
            service.app_agent_runtime.resolve = cast(
                Any,
                lambda profile, objective, nodes: AppAgentSession(
                    skill_pack=file_dialog_skill_pack,
                    adapter_context={},
                    objective=objective,
                    nodes_seen=len(nodes or []),
                ),
            )

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute("save the draft safely", backend)

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn("file_dialog policy", failure["verification"]["reason"])
            self.assertIn("forbids", failure["verification"]["reason"])

    def test_enterprise_grid_policy_requires_approval_for_bulk_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DesktopWorkflowService(Path(temp_dir))
            backend = EnterpriseGridPolicyBackend()
            plan = DesktopWorkflowPlan(
                objective="",
                mode="app-task",
                app_target=None,
                summary="enterprise grid policy plan",
                steps=[
                    DesktopWorkflowStep(
                        action_type="click",
                        selector="name=Bulk Delete",
                        description="Bulk delete the selected enterprise records",
                    )
                ],
                artifacts=[],
            )
            service.planner = cast(Any, StaticDesktopWorkflowPlanner(plan))
            enterprise_spec = spec_for_family("enterprise_grid")
            enterprise_skill_pack = AppAgentSkillPack(
                skill_pack_id="enterprise_grid:test",
                family=enterprise_spec.family,
                app_context=enterprise_spec.app_context,
                app_signature="enterprise-grid-test",
                preferred_channels=list(enterprise_spec.preferred_channels),
                affordance_hints=list(enterprise_spec.affordance_hints),
                verification_contracts=list(enterprise_spec.verification_contracts),
                repair_recipes=list(enterprise_spec.repair_recipes),
                action_policy=dict(enterprise_spec.action_policy),
            )
            service.app_agent_runtime.resolve = cast(
                Any,
                lambda profile, objective, nodes: AppAgentSession(
                    skill_pack=enterprise_skill_pack,
                    adapter_context={},
                    objective=objective,
                    nodes_seen=len(nodes or []),
                ),
            )

            with self.assertRaises(WorkflowVerificationError) as caught:
                service.execute("review the enterprise queue", backend)

            failure = caught.exception.asdict()
            self.assertEqual(backend.perform_calls, 0)
            self.assertIn(
                "enterprise_grid policy",
                failure["verification"]["reason"],
            )
            self.assertIn("requires approval", failure["verification"]["reason"])


if __name__ == "__main__":
    unittest.main()
