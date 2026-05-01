"""Tests for universal OS-agent bridge features.

Covers Set-of-Mark grounding, frontier API decisions, documentation context,
deterministic safety gates, and UniversalDesktopAgentV2 integration hooks.
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from agentos_orchestrator.cognition.abstract_world_model import (
    AbstractUIState,
    UIElementState,
)
from agentos_orchestrator.cognition.affordance_policy_memory import (
    PersistentAffordancePolicyMemory,
)
from agentos_orchestrator.cognition.app_adapters import AdapterRegistry
from agentos_orchestrator.cognition.benchmark_scenarios import (
    REQUIRED_FAILURE_MODES,
    replay_golden_traces,
)
from agentos_orchestrator.cognition.blocker_repair import BlockerRepairPlanner
from agentos_orchestrator.cognition.capability_profile import CapabilityProfiler
from agentos_orchestrator.cognition.control_surface_discovery import (
    GenericControlSurfaceDiscoverer,
)
from agentos_orchestrator.cognition.frontier_api import (
    FrontierDecision,
    FrontierPrompt,
    StaticFrontierClient,
    extract_json_object,
    normalize_decision,
)
from agentos_orchestrator.cognition.local_vla import (
    DetectedElement,
    LocalFastVLA,
    MarkedElement,
    SetOfMarkFrame,
)
from agentos_orchestrator.cognition.mode_arbitration import ModeArbiter, ModeContext
from agentos_orchestrator.cognition.os_eval_packs import (
    SURFACE_FAMILIES,
    build_universal_app_eval_pack,
)
from agentos_orchestrator.cognition.replay_debug import load_replay_debug
from agentos_orchestrator.cognition.runtime_state import (
    AgentRuntimeState,
    Blocker,
    OutcomeEvaluation,
)
from agentos_orchestrator.cognition.safety_gates import (
    FormalSafetyVerifier,
    SafetyPolicy,
    default_safety_verifier,
)
from agentos_orchestrator.cognition.self_documentation import SelfDocumentationLoop
from agentos_orchestrator.cognition.tool_executor import ToolResult
from agentos_orchestrator.cognition.trajectory_recorder import (
    TrajectoryRecorder,
)
from agentos_orchestrator.cognition.trajectory_training import (
    TRAINING_HEADS,
    TrajectoryTrainingBuilder,
)
from agentos_orchestrator.cognition.verification_contracts import (
    VerificationContract,
    ensure_verification_contract,
    verify_action_contract,
)
from agentos_orchestrator.cognition.universal_agent_v2 import (
    UniversalAgentRun,
    UniversalDesktopAgentV2,
)
from agentos_orchestrator.os_control.base import UiAction, UiNode


def _screenshot_bytes() -> bytes:
    img = Image.new("RGB", (320, 240), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 80, 140, 120], fill=(30, 120, 220))
    draw.text((65, 92), "OK", fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _frame() -> SetOfMarkFrame:
    return SetOfMarkFrame(
        annotated_png=_screenshot_bytes(),
        width=320,
        height=240,
        elements=[
            MarkedElement(
                mark_id=1,
                x=40,
                y=80,
                width=100,
                height=40,
                affordance_type="button",
                confidence=0.91,
            )
        ],
    )


@contextmanager
def _loopback_server(responder: Any):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond()

        def do_POST(self) -> None:  # noqa: N802
            self._respond()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            del format, args

        def _respond(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            status, headers, payload = responder(self.command, self.path, body)
            if isinstance(payload, (dict, list)):
                response_body = json.dumps(payload).encode("utf-8")
                headers = {"Content-Type": "application/json", **headers}
            elif isinstance(payload, bytes):
                response_body = payload
            else:
                response_body = str(payload).encode("utf-8")
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.performed: list[UiAction] = []

    def available(self) -> bool:
        return True

    def capture(self) -> bytes:
        return _screenshot_bytes()

    def snapshot(self) -> list[UiNode]:
        return [UiNode(node_id="1", role="Button", name="OK")]

    def perform(self, action: UiAction) -> str:
        self.performed.append(action)
        return "executed"


class FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str, timeout_seconds: int = 15) -> str:
        self.calls.append(url)
        return "<html><title>Official Figma Docs</title><body>Use frames, layers, and the toolbar.</body></html>"


class FakeSearch:
    def search(self, query: str, limit: int = 5) -> list[str]:
        return [
            "https://help.figma.com/hc/en-us/articles/tools",
            "https://reddit.com/r/figma/comments/example",
        ][:limit]


class SetOfMarkTests(unittest.TestCase):
    def test_render_set_of_mark_assigns_ids_and_png(self) -> None:
        vla = LocalFastVLA()
        vla.detect_elements = lambda _screenshot: [  # type: ignore[method-assign]
            DetectedElement(
                x=40,
                y=80,
                width=100,
                height=40,
                aspect_ratio=2.5,
                solidity=0.9,
                edge_density=0.2,
                color_variance=10.0,
                text_like=False,
                affordance_type="button",
                confidence=0.9,
            )
        ]
        frame = vla.render_set_of_mark(_screenshot_bytes())
        self.assertTrue(frame.annotated_png.startswith(b"\x89PNG"))
        self.assertEqual(frame.elements[0].mark_id, 1)
        self.assertEqual(frame.elements[0].cx, 90)

    def test_resolve_action_maps_target_id_to_coordinates(self) -> None:
        action = _frame().resolve_action({"action": "click", "target_id": 1})
        self.assertEqual(action.action_type, "click")
        self.assertEqual(action.metadata["x"], 90)
        self.assertEqual(action.metadata["y"], 100)
        self.assertEqual(action.metadata["source"], "set_of_mark")

    def test_resolve_action_rejects_unknown_id(self) -> None:
        with self.assertRaises(ValueError):
            _frame().resolve_action({"action": "click", "target_id": 999})


class FrontierApiTests(unittest.TestCase):
    def test_extract_json_object_from_model_text(self) -> None:
        parsed = extract_json_object(
            'Grounding {ignore me} JSON: {"action":"click","target_id":3}'
        )
        self.assertEqual(parsed["target_id"], 3)

    def test_normalize_decision(self) -> None:
        decision = normalize_decision(
            {"action": "type", "target_id": "2", "text": "hello", "confidence": 0.7}
        )
        self.assertEqual(decision.action, "type")
        self.assertEqual(decision.target_id, 2)
        self.assertEqual(decision.text, "hello")

    def test_prompt_contract_includes_state_schema_and_escape(self) -> None:
        prompt = FrontierPrompt(
            objective="cancel order",
            annotated_png=b"png",
            mark_payload={"marks": [{"id": 42}]},
            state_context={
                "element_count": 14,
                "focus_region": "SearchBar",
                "modal_open": False,
            },
        )
        text = prompt.instruction_text()
        self.assertIn("UI-control state machine", text)
        self.assertIn("Current State JSON", text)
        self.assertIn("Valid target IDs: [42]", text)
        self.assertIn("grounding.target_mapping", text)
        self.assertIn("orientation", text)
        self.assertIn("expected_observation", text)
        self.assertIn('"action": "explore"', text)

    def test_normalize_preserves_orientation_and_hypothesis(self) -> None:
        decision = normalize_decision(
            {
                "orientation": {
                    "what_changed": "Save dialog opened",
                    "current_blocker": "filename missing",
                    "relevant_history": ["pressed Ctrl+S"],
                },
                "hypothesis": {
                    "claim": "Typing into the filename field names the file",
                    "expected_observation": "Filename field contains report.txt",
                    "risk": "low",
                },
                "action": "type",
                "target_id": 1,
                "text": "report.txt",
                "confidence": 0.91,
            },
            mark_payload={"marks": [{"id": 1}]},
        )
        self.assertEqual(
            decision.metadata["frontier_orientation"]["what_changed"],
            "Save dialog opened",
        )
        self.assertEqual(
            decision.metadata["frontier_hypothesis"]["risk"],
            "low",
        )
        self.assertEqual(
            decision.metadata["expected_observation"],
            "Filename field contains report.txt",
        )

    def test_normalize_rejects_unknown_mark_id(self) -> None:
        with self.assertRaises(ValueError):
            normalize_decision(
                {"action": "click", "target_id": 99, "confidence": 0.9},
                mark_payload={"marks": [{"id": 1}]},
            )

    def test_low_confidence_coerces_to_explore(self) -> None:
        decision = normalize_decision(
            {"action": "click", "target_id": 1, "confidence": 0.2},
            mark_payload={"marks": [{"id": 1}]},
            confidence_floor=0.55,
        )
        self.assertEqual(decision.action, "explore")
        self.assertIsNone(decision.target_id)
        self.assertEqual(decision.metadata["frontier_original_target_id"], 1)

    def test_static_frontier_client_records_prompt(self) -> None:
        client = StaticFrontierClient(FrontierDecision(action="click", target_id=1))
        prompt = FrontierPrompt("click OK", b"png", {"marks": [{"id": 1}]})
        decision = client.choose_action(prompt)
        self.assertEqual(decision.target_id, 1)
        self.assertEqual(client.calls[0], prompt)


class SelfDocumentationTests(unittest.TestCase):
    def test_prepare_context_fetches_and_caches_docs(self) -> None:
        tmpdir = tempfile.mkdtemp()
        fetcher = FakeFetcher()
        loop = SelfDocumentationLoop(
            workspace_root=tmpdir,
            search_provider=FakeSearch(),
            fetcher=fetcher,
        )
        bundle = loop.prepare_context("draw a frame", app_hint="Figma")
        self.assertIn("official Figma documentation", bundle.query)
        self.assertTrue(bundle.sources)
        self.assertIn("Official Figma Docs", bundle.context)

        cached = loop.prepare_context("draw a frame", app_hint="Figma")
        self.assertTrue(cached.cache_hit)
        self.assertEqual(len(fetcher.calls), 1)

    def test_official_score_prefers_docs_over_social(self) -> None:
        docs = SelfDocumentationLoop._official_score(
            "https://help.figma.com/docs", "Figma"
        )
        social = SelfDocumentationLoop._official_score(
            "https://reddit.com/r/figma", "Figma"
        )
        self.assertGreater(docs, social)


class SafetyGateTests(unittest.TestCase):
    def test_allows_safe_click(self) -> None:
        verifier = default_safety_verifier(tempfile.mkdtemp())
        decision = verifier.verify_action(UiAction("click", "som_1_button"))
        self.assertTrue(decision.allowed)

    def test_blocks_destructive_without_approval(self) -> None:
        verifier = default_safety_verifier(tempfile.mkdtemp())
        decision = verifier.verify_action(UiAction("click", "delete_project_button"))
        self.assertFalse(decision.allowed)
        self.assertIn("approval", decision.reason)

    def test_allows_safe_app_family_names_with_destructive_substrings(self) -> None:
        verifier = default_safety_verifier(tempfile.mkdtemp())
        decision = verifier.verify_action(
            UiAction(
                "launch_app",
                "excel.exe",
                metadata={"app_context": "office_form"},
            )
        )
        self.assertTrue(decision.allowed)

        trading_decision = verifier.verify_action(
            UiAction(
                "set_text",
                "order-ticket",
                value="read-only watchlist filter",
                metadata={"app_context": "trading_terminal"},
            )
        )
        self.assertTrue(trading_decision.allowed)

    def test_approval_allows_destructive_action(self) -> None:
        verifier = default_safety_verifier(tempfile.mkdtemp())
        decision = verifier.verify_action(
            UiAction("click", "delete_project_button"),
            approval_token="approved",
        )
        self.assertTrue(decision.allowed)

    def test_blocks_paths_outside_allowed_root(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        allowed = tmpdir / "allowed"
        allowed.mkdir()
        outside = tmpdir / "outside" / "file.txt"
        verifier = FormalSafetyVerifier(SafetyPolicy(allowed_roots=[allowed]))
        decision = verifier.verify_action(
            UiAction("write", "file", metadata={"target_path": str(outside)})
        )
        self.assertFalse(decision.allowed)
        self.assertIn("outside allowed roots", decision.reason)

    def test_blocks_dangerous_shell_payload(self) -> None:
        verifier = default_safety_verifier(tempfile.mkdtemp())
        decision = verifier.verify_action(
            UiAction("tool", "shell", metadata={"command": "rm -rf ."})
        )
        self.assertFalse(decision.allowed)
        self.assertIn("Dangerous shell", decision.reason)


class RuntimeStateTests(unittest.TestCase):
    def test_temporal_trace_actions_and_reflections_enter_prompt_context(self) -> None:
        runtime = AgentRuntimeState(objective="save a report")
        before = AbstractUIState(
            app_context="text_editor",
            elements=[UIElementState("button", "main", 0.1, 0.1, True, "Save")],
        )
        after = AbstractUIState(
            app_context="text_editor",
            layout_mode="modal_open",
            active_modal="Save As",
            focus_region="modal",
            elements=[
                UIElementState("text_field", "modal", 0.5, 0.4, True, "File name")
            ],
        )

        runtime.update_observation(
            before,
            b"before",
            {"marks": [{"id": 1}]},
        )
        runtime.record_action(
            UiAction("hotkey", "ctrl+s"),
            expected_observation="Save As dialog opens",
            receipt="executed",
        )
        runtime.update_observation(
            after,
            b"after",
            {"marks": [{"id": 2}]},
        )
        evaluation = runtime.evaluate_outcome(
            UiAction("hotkey", "ctrl+s"),
            before,
            after,
            "executed",
            expected_observation="Save As dialog opens",
        )
        context = runtime.frontier_context({"focus_region": "modal"})

        self.assertTrue(evaluation.matched)
        self.assertEqual(context["focus_region"], "modal")
        self.assertEqual(context["temporal_trace"][-1]["mark_ids"], [2])
        self.assertEqual(
            context["last_actions"][-1]["expected_observation"],
            "Save As dialog opens",
        )
        self.assertIn("runtime", context)
        self.assertTrue(context["runtime"]["blockers"])
        self.assertEqual(
            context["recent_reflections"][-1]["expected"],
            "Save As dialog opens",
        )

    def test_outcome_evaluation_types_and_resolves_repairs(self) -> None:
        runtime = AgentRuntimeState(objective="click OK")
        state = AbstractUIState(app_context="other")
        failed = runtime.evaluate_outcome(
            UiAction("click", "missing_button"),
            state,
            state,
            json.dumps({"status": "selector-not-found"}),
            expected_observation="The OK button opens the dialog",
        )

        self.assertFalse(failed.matched)
        self.assertIn("Target selector", failed.failure_reason or "")
        self.assertTrue(runtime.blocker_stack[-1].active)

        repaired = runtime.evaluate_outcome(
            UiAction(
                "explore",
                "repair",
                metadata={"repair_kind": "selector_or_stale_target"},
            ),
            state,
            state,
            json.dumps({"status": "success"}),
            expected_observation="Fresh exploration produces grounded targets",
        )

        self.assertTrue(repaired.matched)
        self.assertFalse(any(blocker.active for blocker in runtime.blocker_stack))


class RepairPlannerTests(unittest.TestCase):
    def test_selector_blocker_becomes_explore_repair(self) -> None:
        planner = BlockerRepairPlanner()
        plan = planner.propose(
            [
                Blocker(
                    kind="runtime",
                    description=("Target selector or required resource was not found."),
                    repair_hint=("Re-snapshot the UI and choose a grounded target."),
                )
            ],
            AbstractUIState(app_context="other"),
            "click OK",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        assert plan.action is not None
        self.assertEqual(plan.mode, "explore")
        self.assertEqual(plan.action.action_type, "explore")
        self.assertEqual(
            plan.action.metadata["repair_kind"],
            "selector_or_stale_target",
        )

    def test_policy_blocker_requires_operator_approval(self) -> None:
        planner = BlockerRepairPlanner()
        plan = planner.propose(
            [
                Blocker(
                    kind="runtime",
                    description="Action was blocked by policy or backend.",
                )
            ],
            AbstractUIState(app_context="other"),
            "delete a protected file",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan.can_execute)
        self.assertEqual(plan.mode, "approval")

    def test_unexpected_modal_becomes_escape_repair(self) -> None:
        planner = BlockerRepairPlanner()
        plan = planner.propose(
            [
                Blocker(
                    kind="modal",
                    description="Modal active: newsletter signup",
                )
            ],
            AbstractUIState(
                app_context="browser",
                layout_mode="modal_open",
                active_modal="newsletter signup",
            ),
            "continue reading the article",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        assert plan.action is not None
        self.assertEqual(plan.kind, "unexpected_modal")
        self.assertEqual(plan.action.action_type, "hotkey")
        self.assertEqual(plan.action.value, "{ESC}")


class CapabilityAndAdapterTests(unittest.TestCase):
    def test_profiler_classifies_browser_and_channels(self) -> None:
        profile = CapabilityProfiler().profile(
            AbstractUIState(app_context="unknown"),
            nodes=[
                UiNode("1", "TabItem", "Tab 1"),
                UiNode("2", "Edit", "Address and search bar"),
                UiNode("3", "Hyperlink", "Docs"),
            ],
            screenshot_available=True,
        )

        self.assertEqual(profile.app_family, "browser")
        self.assertIn("accessibility", profile.control_channels)
        self.assertIn("dom", profile.control_channels)

    def test_profiler_preserves_file_dialog_app_context(self) -> None:
        profile = CapabilityProfiler().profile(
            AbstractUIState(app_context="file_dialog"),
            nodes=[
                UiNode("1", "Edit", "Address and search bar"),
                UiNode("2", "TabItem", "Tab 1"),
                UiNode("3", "Button", "Open"),
            ],
            screenshot_available=True,
        )

        self.assertEqual(profile.app_family, "file_dialog")
        self.assertTrue(profile.app_signature.startswith("file_dialog:"))

    def test_profiler_generates_stable_app_signature(self) -> None:
        nodes = [
            UiNode("1", "TabItem", "Tab 1"),
            UiNode("2", "Edit", "Address and search bar"),
            UiNode("3", "Hyperlink", "Docs"),
        ]
        profiler = CapabilityProfiler()
        profile_a = profiler.profile(
            AbstractUIState(app_context="unknown"),
            nodes=nodes,
            screenshot_available=True,
        )
        profile_b = profiler.profile(
            AbstractUIState(app_context="unknown"),
            nodes=list(nodes),
            screenshot_available=True,
        )

        self.assertEqual(profile_a.app_signature, profile_b.app_signature)
        self.assertTrue(profile_a.app_signature.startswith("browser:"))

    def test_registry_enriches_action_with_adapter_context(self) -> None:
        profile = CapabilityProfiler().profile(
            AbstractUIState(app_context="browser"),
            nodes=[UiNode("1", "Button", "Open")],
            screenshot_available=True,
        )
        action = AdapterRegistry().enrich_action(
            UiAction("click", "name=Open"),
            profile,
            "open docs",
        )

        self.assertEqual(action.metadata["adapter_family"], "browser")
        self.assertIn("verification_contract", action.metadata)
        self.assertIn("adapter_context", action.metadata)

    def test_mode_arbiter_uses_low_confidence_profile_for_explore(self) -> None:
        profile = CapabilityProfiler().profile(
            AbstractUIState(app_context="unknown"),
            nodes=[],
            screenshot_available=True,
        )
        decision = ModeArbiter().choose(
            "click the visible action",
            object(),
            AbstractUIState(app_context="unknown"),
            AgentRuntimeState(objective="click the visible action"),
            ModeContext(capability_profile=profile),
        )

        self.assertEqual(decision.mode, "explore")
        self.assertIn("capability_profile", decision.evidence)

    def test_mode_arbiter_does_not_treat_editor_find_target_as_research(self) -> None:
        profile = CapabilityProfiler().profile(
            AbstractUIState(app_context="text_editor"),
            nodes=[UiNode("1", "Document", "Text Editor")],
            screenshot_available=True,
        )
        decision = ModeArbiter().choose(
            "On an editor surface, find target and verify the result.",
            type("Opt", (), {"name": "find_target"})(),
            AbstractUIState(app_context="text_editor"),
            AgentRuntimeState(
                objective="On an editor surface, find target and verify the result."
            ),
            ModeContext(capability_profile=profile),
        )

        self.assertNotEqual(decision.mode, "research")


class AffordancePolicyMemoryTests(unittest.TestCase):
    def test_policy_memory_persists_by_app_signature(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = PersistentAffordancePolicyMemory(td)
            action = UiAction(
                "click",
                "name=Special Action",
                metadata={"control_channel": "accessibility"},
            )
            store.record(
                "unknown:abc123",
                "open the special action",
                action,
                success=True,
                control_channel="accessibility",
                observed="button opened panel",
            )

            reloaded = PersistentAffordancePolicyMemory(td)
            recommendation = reloaded.recommend_action(
                "unknown:abc123",
                "open the special action again",
                nodes=[UiNode("1", "Button", "Special Action")],
            )

            self.assertIsNotNone(recommendation)
            assert recommendation is not None
            self.assertEqual(recommendation.selector, "name=Special Action")
            self.assertEqual(recommendation.metadata["source"], "policy_memory")


class VerificationContractTests(unittest.TestCase):
    def test_field_contains_contract_matches_observed_label(self) -> None:
        action = UiAction(
            "type",
            "name=Search",
            value="AgentOS",
            metadata={
                "verification_contract": VerificationContract(
                    kind="field_contains",
                    expected="Field contains AgentOS",
                    value="AgentOS",
                ).asdict()
            },
        )
        result = verify_action_contract(
            action,
            AbstractUIState(app_context="browser"),
            AbstractUIState(
                app_context="browser",
                elements=[
                    UIElementState("text_field", "main", 0.5, 0.5, True, "AgentOS")
                ],
            ),
            json.dumps({"status": "typed"}),
        )

        self.assertTrue(result.matched)

    def test_file_exists_contract_checks_path(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        target = tmpdir / "artifact.txt"
        target.write_text("ok", encoding="utf-8")
        action = UiAction("click", "save", metadata={"path": str(target)})
        contract = ensure_verification_contract(action)
        result = verify_action_contract(
            action,
            AbstractUIState(),
            AbstractUIState(),
            json.dumps({"status": "executed"}),
        )

        self.assertEqual(contract.kind, "file_exists")
        self.assertTrue(result.matched)

    def test_modal_closed_contract_detects_remaining_modal(self) -> None:
        action = UiAction("hotkey", "app-window", value="{ESC}")
        result = verify_action_contract(
            action,
            AbstractUIState(active_modal="Save As", layout_mode="modal_open"),
            AbstractUIState(active_modal="Save As", layout_mode="modal_open"),
            "executed",
        )

        self.assertFalse(result.matched)
        self.assertEqual(result.kind, "modal_closed")

    def test_receipt_success_contract_accepts_hotkey_sent(self) -> None:
        action = UiAction(
            "hotkey",
            "app-window",
            value="%{TAB}",
            metadata={
                "verification_contract": VerificationContract(
                    kind="receipt_success",
                    expected="Shortcut receipt reports progress.",
                ).asdict()
            },
        )
        result = verify_action_contract(
            action,
            AbstractUIState(app_context="browser"),
            AbstractUIState(app_context="browser"),
            json.dumps({"status": "hotkey-sent", "value": "%{TAB}"}),
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.kind, "receipt_success")

    def test_state_changed_accepts_regrounded_focus_match_without_diff(self) -> None:
        action = UiAction(
            "focus",
            "name=Address and search bar",
            metadata={
                "regrounded": True,
                "verification_contract": VerificationContract(
                    kind="state_changed",
                    expected="Focus resolves a grounded target.",
                ).asdict(),
            },
        )
        result = verify_action_contract(
            action,
            AbstractUIState(app_context="file_dialog"),
            AbstractUIState(app_context="file_dialog"),
            json.dumps(
                {
                    "status": "matched",
                    "selector": "name=Address and search bar",
                    "matched_name": "Address and search bar",
                    "matched_role": "ControlType.Edit",
                    "focus_error": "Target element cannot receive focus.",
                }
            ),
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.kind, "state_changed")

    def test_field_contains_accepts_edit_receipt_when_snapshot_is_stale(self) -> None:
        action = UiAction(
            "set_text",
            "name=Address and search bar",
            value="AgentOS live-fire value",
            metadata={
                "verification_contract": VerificationContract(
                    kind="field_contains",
                    expected="The target field contains the live-fire value.",
                    value="AgentOS live-fire value",
                ).asdict()
            },
        )
        result = verify_action_contract(
            action,
            AbstractUIState(app_context="browser"),
            AbstractUIState(app_context="browser"),
            json.dumps(
                {
                    "status": "value-set",
                    "selector": "name=Address and search bar",
                    "matched_name": "Address and search bar",
                    "matched_role": "ControlType.Edit",
                }
            ),
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.kind, "field_contains")


class ModeArbitrationTests(unittest.TestCase):
    def test_chooses_tool_for_analysis_tasks(self) -> None:
        decision = ModeArbiter().choose(
            "analyze SPY volatility with code",
            type("Option", (), {"name": "run_analysis_code"})(),
            AbstractUIState(app_context="other"),
            AgentRuntimeState(objective="analyze SPY volatility"),
            ModeContext(perceived_element_count=3),
        )
        self.assertEqual(decision.mode, "tool")

    def test_active_selector_blocker_forces_explore(self) -> None:
        runtime = AgentRuntimeState(objective="click OK")
        runtime.add_blocker(
            "runtime",
            "Target selector or required resource was not found.",
            repair_hint="Re-snapshot the UI and choose a grounded target.",
        )
        decision = ModeArbiter().choose(
            "click OK",
            object(),
            AbstractUIState(app_context="other"),
            runtime,
            ModeContext(perceived_element_count=0),
        )
        self.assertEqual(decision.mode, "explore")

    def test_visible_structured_ui_uses_ui_mode(self) -> None:
        decision = ModeArbiter().choose(
            "click OK",
            object(),
            AbstractUIState(
                app_context="other",
                elements=[UIElementState("button", "main", 0.5, 0.5, True, "OK")],
            ),
            AgentRuntimeState(objective="click OK"),
            ModeContext(perceived_element_count=1),
        )
        self.assertEqual(decision.mode, "ui")

    def test_file_dialog_find_target_does_not_force_filesystem_mode(self) -> None:
        decision = ModeArbiter().choose(
            "On a file dialog surface, find target and verify the result.",
            type("Option", (), {"name": "find_target"})(),
            AbstractUIState(app_context="file_dialog"),
            AgentRuntimeState(objective="find target"),
            ModeContext(perceived_element_count=10),
        )

        self.assertNotEqual(decision.mode, "filesystem")


class TrajectoryRecorderTests(unittest.TestCase):
    def test_records_training_ready_jsonl_step(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        recorder = TrajectoryRecorder(tmpdir)
        path = recorder.start_run("run/1", "click OK")
        assert path is not None
        before = AbstractUIState(app_context="other")
        after = AbstractUIState(
            app_context="other",
            elements=[UIElementState("button", "main", 0.5, 0.5, True, "OK")],
        )
        recorder.record_step(
            run_id="run/1",
            objective="click OK",
            option_name="press_ok",
            before=before,
            after=after,
            action=UiAction("click", "som_1_button"),
            expected_observation="Dialog closes",
            receipt=json.dumps({"status": "executed"}),
            outcome=OutcomeEvaluation(
                expected="Dialog closes",
                observed="receipt ok",
                matched=True,
            ),
            capability_profile={"app_family": "browser"},
            adapter_context={"family": "browser"},
            verification_contract={"kind": "state_changed"},
            verification_result={"matched": True},
        )
        recorder.finish_run("run/1", {"success": True})

        lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(lines[0]["event"], "run_started")
        self.assertEqual(lines[1]["event"], "step")
        self.assertEqual(lines[1]["action"]["selector"], "som_1_button")
        self.assertEqual(lines[1]["capability_profile"]["app_family"], "browser")
        self.assertTrue(lines[1]["verification_result"]["matched"])
        self.assertEqual(lines[-1]["event"], "run_finished")

    def test_replay_debug_payload_exposes_action_reasoning(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        recorder = TrajectoryRecorder(tmpdir)
        path = recorder.start_run("run-debug", "click OK")
        assert path is not None
        recorder.record_step(
            run_id="run-debug",
            objective="click OK",
            option_name="press_ok",
            before=AbstractUIState(app_context="browser"),
            after=AbstractUIState(app_context="browser", task_progress={"done": 1.0}),
            action=UiAction("click", "name=OK"),
            expected_observation="State changes",
            receipt="executed",
            outcome=OutcomeEvaluation("State changes", "changed", True),
            capability_profile={"app_family": "browser"},
            adapter_context={"family": "browser"},
            verification_contract={"kind": "state_changed"},
            verification_result={"matched": True},
        )
        recorder.finish_run("run-debug", {"success": True})

        payload = load_replay_debug(tmpdir, run_id="run-debug")

        self.assertEqual(payload["run_count"], 1)
        step = payload["runs"][0]["steps"][0]
        self.assertEqual(step["capability_profile"]["app_family"], "browser")
        self.assertEqual(step["verification_contract"]["kind"], "state_changed")

    def test_training_builder_creates_all_model_head_examples(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        recorder = TrajectoryRecorder(tmpdir)
        path = recorder.start_run("run-training", "analyze with code")
        assert path is not None
        before = AbstractUIState(
            app_context="other",
            elements=[UIElementState("button", "main", 0.5, 0.5, True, "Run")],
        )
        after = AbstractUIState(
            app_context="other",
            task_progress={"verified": 1.0},
        )
        recorder.record_step(
            run_id="run-training",
            objective="analyze with code",
            option_name="run_analysis_code",
            before=before,
            after=after,
            action=UiAction("tool", "tool_executor:quant_analysis"),
            expected_observation="The tool returns structured output.",
            receipt=json.dumps({"status": "success"}),
            outcome=OutcomeEvaluation(
                expected="The tool returns structured output.",
                observed="receipt ok",
                matched=True,
            ),
            mode_decision=ModeArbiter().choose(
                "analyze with code",
                type("Option", (), {"name": "run_analysis_code"})(),
                before,
                AgentRuntimeState(objective="analyze with code"),
            ),
        )

        builder = TrajectoryTrainingBuilder(tmpdir)
        bundle = builder.build([path])
        summary = bundle.summary()
        self.assertTrue(summary["ready_for_training"])
        self.assertEqual(summary["total_examples"], 5)
        self.assertEqual(set(summary["heads"]), set(TRAINING_HEADS))
        self.assertTrue(all(count == 1 for count in summary["heads"].values()))


class BenchmarkScenarioTests(unittest.TestCase):
    def test_universal_os_golden_traces_cover_failure_modes(self) -> None:
        replay = replay_golden_traces(Path.cwd())
        self.assertTrue(replay["passed"])
        self.assertEqual(set(replay["missing_failure_modes"]), set())
        self.assertTrue(
            REQUIRED_FAILURE_MODES.issubset(set(replay["covered_failure_modes"]))
        )

    def test_eval_pack_covers_expanded_surface_families(self) -> None:
        pack = build_universal_app_eval_pack()
        summary = pack.summary()

        self.assertEqual(set(summary["surface_counts"]), set(SURFACE_FAMILIES))
        self.assertEqual(summary["task_count"], len(SURFACE_FAMILIES) * 20)
        self.assertIn("design_canvas", summary["surface_counts"])
        self.assertIn("trading_terminal", summary["surface_counts"])
        self.assertIn("enterprise_grid", summary["surface_counts"])
        self.assertTrue(summary["ready_for_live_fire"])


class UniversalAgentFrontierIntegrationTests(unittest.TestCase):
    def _agent(
        self, decision: dict[str, Any] | FrontierDecision
    ) -> UniversalDesktopAgentV2:
        backend = FakeBackend()
        agent = UniversalDesktopAgentV2(
            backend,
            workspace_root=tempfile.mkdtemp(),
            frontier_client=StaticFrontierClient(decision),
            use_self_documentation=False,
            max_steps=1,
        )
        agent._latest_screenshot = _screenshot_bytes()
        agent.local_vla.render_set_of_mark = lambda _screenshot: _frame()  # type: ignore[method-assign]
        return agent

    def test_frontier_som_action_selected_before_local_fallback(self) -> None:
        agent = self._agent(
            {
                "action": "click",
                "target_id": 1,
                "rationale": "OK",
                "hypothesis": {
                    "claim": "Clicking OK confirms the dialog",
                    "expected_observation": "The dialog closes",
                    "risk": "low",
                },
            }
        )
        run = UniversalAgentRun(run_id="test", objective="click OK")
        action = agent._select_action(
            option=object(),
            perceived_elements=[],
            current_state=AbstractUIState(app_context="other"),
            run=run,
            similar_failures=[],
        )
        self.assertEqual(action.selector, "som_1_button")
        self.assertEqual(action.metadata["source"], "frontier_som")
        self.assertEqual(action.metadata["x"], 90)
        self.assertEqual(action.metadata["expected_observation"], "The dialog closes")

    def test_frontier_prompt_receives_state_context(self) -> None:
        agent = self._agent({"action": "click", "target_id": 1, "confidence": 0.8})
        run = UniversalAgentRun(run_id="test", objective="click OK")
        agent._select_frontier_som_action(
            AbstractUIState(
                app_context="trading_dashboard",
                focus_region="SearchBar",
            ),
            run,
        )
        prompt = agent.frontier_client.calls[0]
        self.assertEqual(prompt.state_context["focus_region"], "SearchBar")
        self.assertIn("element_count", prompt.state_context)
        self.assertIn("runtime", prompt.state_context)
        self.assertEqual(prompt.state_context["temporal_trace"][-1]["mark_ids"], [1])

    def test_frontier_explore_escape_runs_local_probe(self) -> None:
        agent = self._agent(
            {
                "action": "explore",
                "target_id": None,
                "confidence": 0.3,
                "grounding": {"target_mapping": "no stable tag"},
                "rationale": "ambiguous UI",
            }
        )
        run = UniversalAgentRun(run_id="test", objective="discover OK")
        action = agent._select_frontier_som_action(
            AbstractUIState(app_context="other"),
            run,
        )
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, "explore")
        self.assertEqual(action.metadata["source"], "frontier_escape_hatch")

    def test_select_action_prefers_discovered_api_probe(self) -> None:
        class ApiBackend(FakeBackend):
            def snapshot(self) -> list[UiNode]:
                return [
                    UiNode(
                        node_id="1",
                        role="Edit",
                        name="API Endpoint",
                        metadata={"endpoint": "http://127.0.0.1:8123/api"},
                    )
                ]

        agent = UniversalDesktopAgentV2(
            ApiBackend(),
            workspace_root=tempfile.mkdtemp(),
            use_frontier_api=False,
            use_self_documentation=False,
            allow_network_tools=True,
            max_steps=1,
        )
        agent.tool_executor.run = lambda _request: ToolResult(  # type: ignore[method-assign]
            success=True,
            stdout="{}",
            parsed_results={
                "control_probe": {
                    "http://127.0.0.1:8123/api": {"OPTIONS": {"status": 200}}
                }
            },
        )
        action = agent._select_action(
            option=object(),
            perceived_elements=[],
            current_state=AbstractUIState(app_context="unknown"),
            run=UniversalAgentRun(run_id="test", objective="inspect the API"),
            similar_failures=[],
        )

        self.assertEqual(action.action_type, "tool")
        self.assertTrue(action.selector.startswith("control_surface_api_probe:"))
        self.assertEqual(action.metadata["source"], "control_surface_discovery")
        self.assertEqual(action.metadata["control_channel"], "api")

    def test_select_action_reuses_policy_memory_before_generic_fallback(self) -> None:
        class PolicyBackend(FakeBackend):
            def snapshot(self) -> list[UiNode]:
                return [UiNode(node_id="7", role="Button", name="Special Action")]

        with tempfile.TemporaryDirectory() as td:
            store = PersistentAffordancePolicyMemory(td)
            store.record(
                "unknown:seeded",
                "open the special action",
                UiAction(
                    "click",
                    "name=Special Action",
                    metadata={"control_channel": "accessibility"},
                ),
                success=True,
                control_channel="accessibility",
                observed="opened the special action panel",
            )

            agent = UniversalDesktopAgentV2(
                PolicyBackend(),
                workspace_root=td,
                use_frontier_api=False,
                use_self_documentation=False,
                max_steps=1,
            )
            capability_profile = agent.capability_profiler.profile(
                AbstractUIState(app_context="unknown"),
                nodes=agent._safe_snapshot_nodes(),
                screenshot_available=False,
            )
            seeded_signature = capability_profile.app_signature
            seeded_path = Path(td) / ".agentos" / "affordance_policies.json"
            payload = json.loads(seeded_path.read_text(encoding="utf-8"))
            payload["entries"][0]["app_signature"] = seeded_signature
            seeded_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            agent.affordance_policies = PersistentAffordancePolicyMemory(td)

            action = agent._select_action(
                option=object(),
                perceived_elements=[],
                current_state=AbstractUIState(app_context="unknown"),
                run=UniversalAgentRun(
                    run_id="test", objective="open the special action again"
                ),
                similar_failures=[],
                capability_profile=capability_profile,
            )

            self.assertEqual(action.selector, "name=Special Action")
            self.assertEqual(action.metadata["source"], "policy_memory")

    def test_select_action_synthesizes_api_workflow_from_docs_without_visible_endpoint(
        self,
    ) -> None:
        class DocsOnlyBackend(FakeBackend):
            def snapshot(self) -> list[UiNode]:
                return [UiNode(node_id="1", role="Pane", name="Workspace")]

        class ApiDocsSearch:
            def search(self, query: str, limit: int = 5) -> list[str]:
                return ["https://docs.example.com/api"][:limit]

        class ApiDocsFetcher:
            def fetch(self, url: str, timeout_seconds: int = 15) -> str:
                del timeout_seconds
                return (
                    "<html><title>Developer API</title><body>"
                    'POST /v1/actions with JSON {"query": "<query>"}. '
                    "Then GET /v1/actions to verify the result. "
                    "Authorization: Bearer token required."
                    "</body></html>"
                )

        agent = UniversalDesktopAgentV2(
            DocsOnlyBackend(),
            workspace_root=tempfile.mkdtemp(),
            use_frontier_api=False,
            allow_network_tools=True,
            max_steps=1,
        )
        agent.self_documentation = SelfDocumentationLoop(
            workspace_root=agent.workspace_root,
            search_provider=ApiDocsSearch(),
            fetcher=ApiDocsFetcher(),
        )
        agent.tool_executor.run = lambda _request: ToolResult(  # type: ignore[method-assign]
            success=True,
            stdout="[]",
            parsed_results={
                "api_workflow": [
                    {"name": "probe_surface", "status": 200},
                    {"name": "execute_objective", "status": 201},
                    {"name": "verify_result", "status": 200},
                ]
            },
        )

        action = agent._select_action(
            option=object(),
            perceived_elements=[],
            current_state=AbstractUIState(app_context="unknown"),
            run=UniversalAgentRun(
                run_id="test", objective="search customer records via API"
            ),
            similar_failures=[],
        )

        self.assertEqual(action.action_type, "tool")
        self.assertTrue(action.selector.startswith("synthesized_api_workflow:"))
        self.assertEqual(action.metadata["source"], "control_surface_discovery")
        self.assertEqual(action.metadata["control_surface_kind"], "api_workflow")
        self.assertGreaterEqual(action.metadata["workflow_step_count"], 3)

    def test_local_openapi_artifact_synthesizes_workflow_without_docs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "openapi.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "openapi": "3.0.0",
                        "servers": [{"url": "http://127.0.0.1:9000"}],
                        "paths": {
                            "/v1/search": {
                                "post": {
                                    "requestBody": {
                                        "content": {
                                            "application/json": {
                                                "example": {"query": "<query>"}
                                            }
                                        }
                                    }
                                },
                                "get": {},
                            }
                        },
                        "components": {
                            "securitySchemes": {
                                "BearerAuth": {
                                    "type": "http",
                                    "scheme": "bearer",
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            discoverer = GenericControlSurfaceDiscoverer(td)
            profile = CapabilityProfiler().profile(
                AbstractUIState(app_context="unknown"),
                nodes=[],
                screenshot_available=False,
            )

            candidates = discoverer.discover(
                profile,
                [],
                "search customer records via API",
            )

            self.assertTrue(candidates)
            best = candidates[0]
            self.assertEqual(best.kind, "api_workflow")
            self.assertEqual(best.metadata["discovery_source"], "workspace_artifact")
            self.assertEqual(best.metadata["artifact_path"], str(spec_path))
            self.assertGreaterEqual(len(best.workflow), 3)
            self.assertEqual(best.workflow[1]["method"], "POST")
            self.assertEqual(
                best.workflow[1]["json_body"]["query"],
                "search customer records via API",
            )

    def test_local_openapi_artifact_resolves_refs_and_response_examples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "openapi.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "openapi": "3.0.0",
                        "servers": [{"url": "http://127.0.0.1:9012"}],
                        "paths": {
                            "/v1/search": {
                                "post": {
                                    "requestBody": {
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                    "allOf": [
                                                        {
                                                            "$ref": "#/components/schemas/SearchBase"
                                                        },
                                                        {
                                                            "type": "object",
                                                            "properties": {
                                                                "filters": {
                                                                    "$ref": "#/components/schemas/SearchFilters"
                                                                },
                                                                "includeArchived": {
                                                                    "type": "boolean",
                                                                    "default": False,
                                                                },
                                                            },
                                                        },
                                                    ]
                                                }
                                            }
                                        }
                                    },
                                    "responses": {
                                        "200": {
                                            "content": {
                                                "application/json": {
                                                    "schema": {
                                                        "$ref": "#/components/schemas/SearchResponse"
                                                    }
                                                }
                                            }
                                        }
                                    },
                                }
                            }
                        },
                        "components": {
                            "schemas": {
                                "SearchBase": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "example": "<query>",
                                        },
                                        "limit": {"type": "integer", "default": 25},
                                    },
                                },
                                "SearchFilters": {
                                    "type": "object",
                                    "properties": {
                                        "status": {
                                            "type": "string",
                                            "enum": ["active", "archived"],
                                        }
                                    },
                                },
                                "Customer": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                    },
                                },
                                "SearchResponse": {
                                    "type": "object",
                                    "properties": {
                                        "results": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/Customer"
                                            },
                                        },
                                        "total": {"type": "integer"},
                                    },
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            discoverer = GenericControlSurfaceDiscoverer(td)
            profile = CapabilityProfiler().profile(
                AbstractUIState(app_context="unknown"),
                nodes=[],
                screenshot_available=False,
            )

            candidates = discoverer.discover(
                profile,
                [],
                "search customer records via API",
            )

            best = candidates[0]
            self.assertEqual(best.metadata["artifact_path"], str(spec_path))
            self.assertEqual(
                best.workflow[1]["json_body"]["query"],
                "search customer records via API",
            )
            self.assertEqual(best.workflow[1]["json_body"]["limit"], 25)
            self.assertEqual(
                best.workflow[1]["json_body"]["filters"]["status"], "active"
            )
            self.assertFalse(best.workflow[1]["json_body"]["includeArchived"])
            self.assertEqual(best.metadata["response_hint"]["results"][0]["id"], "<id>")
            self.assertEqual(best.metadata["response_hint"]["total"], 0)

    def test_active_loopback_openapi_service_synthesizes_workflow_without_local_spec(
        self,
    ) -> None:
        def responder(
            method: str, path: str, _body: bytes
        ) -> tuple[int, dict[str, str], Any]:
            del method
            if path == "/openapi.json":
                return (
                    200,
                    {},
                    {
                        "openapi": "3.0.0",
                        "servers": [{"url": f"http://127.0.0.1:{port}"}],
                        "paths": {
                            "/v1/search": {
                                "post": {
                                    "requestBody": {
                                        "content": {
                                            "application/json": {
                                                "example": {"query": "<query>"}
                                            }
                                        }
                                    }
                                },
                                "get": {},
                            }
                        },
                    },
                )
            return (
                200,
                {"Content-Type": "text/html"},
                "<html>Local API service</html>",
            )

        with tempfile.TemporaryDirectory() as td, _loopback_server(responder) as port:
            discoverer = GenericControlSurfaceDiscoverer(td, loopback_ports=[port])
            profile = CapabilityProfiler().profile(
                AbstractUIState(app_context="unknown"),
                nodes=[],
                screenshot_available=False,
            )
            candidates = discoverer.discover(
                profile,
                [],
                "search customer records via API",
                active_fingerprinting=True,
            )

            self.assertTrue(candidates)
            best = candidates[0]
            self.assertEqual(best.kind, "api_workflow")
            self.assertEqual(best.metadata["discovery_source"], "loopback_service")
            self.assertEqual(
                best.workflow[1]["url"], f"http://127.0.0.1:{port}/v1/search"
            )
            self.assertEqual(
                best.workflow[1]["json_body"]["query"],
                "search customer records via API",
            )

    def test_select_action_synthesizes_api_workflow_from_active_loopback_graphql_without_docs(
        self,
    ) -> None:
        class LoopbackBackend(FakeBackend):
            def snapshot(self) -> list[UiNode]:
                return [UiNode(node_id="1", role="Pane", name="Workspace")]

        def responder(
            _method: str, path: str, body: bytes
        ) -> tuple[int, dict[str, str], Any]:
            if path == "/graphql":
                payload = json.loads(body.decode("utf-8")) if body else {}
                query_text = str(payload.get("query") or "")
                if "__schema" in query_text:
                    return (
                        200,
                        {},
                        {
                            "data": {
                                "__schema": {
                                    "queryType": {"name": "Query"},
                                    "mutationType": {"name": "Mutation"},
                                    "types": [
                                        {
                                            "kind": "OBJECT",
                                            "name": "Query",
                                            "fields": [
                                                {
                                                    "name": "customerRecords",
                                                    "args": [
                                                        {
                                                            "name": "query",
                                                            "defaultValue": None,
                                                            "type": {
                                                                "kind": "NON_NULL",
                                                                "name": None,
                                                                "ofType": {
                                                                    "kind": "SCALAR",
                                                                    "name": "String",
                                                                    "ofType": None,
                                                                },
                                                            },
                                                        }
                                                    ],
                                                    "type": {
                                                        "kind": "LIST",
                                                        "name": None,
                                                        "ofType": {
                                                            "kind": "OBJECT",
                                                            "name": "Customer",
                                                            "ofType": None,
                                                        },
                                                    },
                                                }
                                            ],
                                            "inputFields": None,
                                        },
                                        {
                                            "kind": "OBJECT",
                                            "name": "Mutation",
                                            "fields": [
                                                {
                                                    "name": "createCustomer",
                                                    "args": [
                                                        {
                                                            "name": "name",
                                                            "defaultValue": None,
                                                            "type": {
                                                                "kind": "SCALAR",
                                                                "name": "String",
                                                                "ofType": None,
                                                            },
                                                        }
                                                    ],
                                                    "type": {
                                                        "kind": "OBJECT",
                                                        "name": "Customer",
                                                        "ofType": None,
                                                    },
                                                }
                                            ],
                                            "inputFields": None,
                                        },
                                        {
                                            "kind": "OBJECT",
                                            "name": "Customer",
                                            "fields": [
                                                {
                                                    "name": "id",
                                                    "args": [],
                                                    "type": {
                                                        "kind": "SCALAR",
                                                        "name": "ID",
                                                        "ofType": None,
                                                    },
                                                },
                                                {
                                                    "name": "name",
                                                    "args": [],
                                                    "type": {
                                                        "kind": "SCALAR",
                                                        "name": "String",
                                                        "ofType": None,
                                                    },
                                                },
                                            ],
                                            "inputFields": None,
                                        },
                                    ],
                                }
                            }
                        },
                    )
                return (
                    200,
                    {},
                    {"data": {"customerRecords": [{"id": "1", "name": "Alice"}]}},
                )
            return (200, {"Content-Type": "text/html"}, "<html>GraphQL service</html>")

        with tempfile.TemporaryDirectory() as td, _loopback_server(responder) as port:
            agent = UniversalDesktopAgentV2(
                LoopbackBackend(),
                workspace_root=td,
                use_frontier_api=False,
                use_self_documentation=False,
                allow_network_tools=True,
                max_steps=1,
            )
            agent.control_surface_discoverer = GenericControlSurfaceDiscoverer(
                td,
                loopback_ports=[port],
            )
            agent.tool_executor.run = lambda _request: ToolResult(  # type: ignore[method-assign]
                success=True,
                stdout="[]",
                parsed_results={
                    "api_workflow": [
                        {"name": "probe_surface", "status": 200},
                        {"name": "execute_objective", "status": 200},
                        {"name": "verify_result", "status": 200},
                    ]
                },
            )

            action = agent._select_action(
                option=object(),
                perceived_elements=[],
                current_state=AbstractUIState(app_context="unknown"),
                run=UniversalAgentRun(
                    run_id="test", objective="find customer records via graphql api"
                ),
                similar_failures=[],
            )

            self.assertEqual(action.action_type, "tool")
            self.assertTrue(action.selector.startswith("synthesized_api_workflow:"))
            self.assertEqual(action.metadata["source"], "control_surface_discovery")
            self.assertEqual(action.metadata["control_surface_kind"], "api_workflow")
            self.assertEqual(
                action.metadata["workflow"][1]["url"],
                f"http://127.0.0.1:{port}/graphql",
            )
            self.assertIn(
                "customerRecords", action.metadata["workflow"][1]["json_body"]["query"]
            )
            self.assertEqual(
                action.metadata["workflow"][1]["json_body"]["variables"]["query"],
                "find customer records via graphql api",
            )
            self.assertEqual(action.metadata["workflow"][2]["method"], "POST")

    def test_select_action_synthesizes_api_workflow_from_local_spec_without_docs(
        self,
    ) -> None:
        class LocalSpecBackend(FakeBackend):
            def snapshot(self) -> list[UiNode]:
                return [UiNode(node_id="1", role="Pane", name="Workspace")]

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "swagger.json").write_text(
                json.dumps(
                    {
                        "swagger": "2.0",
                        "host": "127.0.0.1:9010",
                        "schemes": ["http"],
                        "paths": {
                            "/v1/actions": {
                                "post": {
                                    "parameters": [
                                        {
                                            "in": "body",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "query": {"type": "string"}
                                                },
                                            },
                                        }
                                    ]
                                },
                                "get": {},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            agent = UniversalDesktopAgentV2(
                LocalSpecBackend(),
                workspace_root=td,
                use_frontier_api=False,
                use_self_documentation=False,
                allow_network_tools=True,
                max_steps=1,
            )
            agent.tool_executor.run = lambda _request: ToolResult(  # type: ignore[method-assign]
                success=True,
                stdout="[]",
                parsed_results={
                    "api_workflow": [
                        {"name": "probe_surface", "status": 200},
                        {"name": "execute_objective", "status": 201},
                        {"name": "verify_result", "status": 200},
                    ]
                },
            )

            action = agent._select_action(
                option=object(),
                perceived_elements=[],
                current_state=AbstractUIState(app_context="unknown"),
                run=UniversalAgentRun(
                    run_id="test", objective="search customer records via API"
                ),
                similar_failures=[],
            )

            self.assertEqual(action.action_type, "tool")
            self.assertTrue(action.selector.startswith("synthesized_api_workflow:"))
            self.assertEqual(action.metadata["source"], "control_surface_discovery")
            self.assertEqual(action.metadata["control_surface_kind"], "api_workflow")
            self.assertEqual(
                action.metadata["workflow"][1]["url"],
                "http://127.0.0.1:9010/v1/actions",
            )
            self.assertEqual(
                action.metadata["workflow"][1]["json_body"]["query"],
                "search customer records via API",
            )

    def test_frontier_tool_request_runs_sandbox(self) -> None:
        agent = self._agent(
            {
                "action": "tool",
                "tool": "python",
                "code": "print('RESULT: answer=42')",
                "rationale": "calculate directly",
            }
        )
        run = UniversalAgentRun(run_id="test", objective="calculate answer")
        action = agent._select_frontier_som_action(
            AbstractUIState(app_context="other"),
            run,
        )
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, "tool")
        self.assertTrue(action.metadata["tool_success"])
        self.assertEqual(action.metadata["tool_result"]["parsed_results"]["answer"], 42)

    def test_tool_fast_path_wraps_result_as_tool_action(self) -> None:
        agent = self._agent({"action": "click", "target_id": 1})
        option = type("Option", (), {"name": "run_analysis_code"})()
        run = UniversalAgentRun(run_id="test", objective="analyse SPY volatility")
        action = agent._tool_action_for_option(option, run)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, "tool")
        self.assertIn("tool_result", action.metadata)


if __name__ == "__main__":
    unittest.main()
