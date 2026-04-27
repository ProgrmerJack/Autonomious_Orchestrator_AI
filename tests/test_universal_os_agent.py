"""Tests for universal OS-agent bridge features.

Covers Set-of-Mark grounding, frontier API decisions, documentation context,
deterministic safety gates, and UniversalDesktopAgentV2 integration hooks.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from agentos_orchestrator.cognition.abstract_world_model import (
    AbstractUIState,
    UIElementState,
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
from agentos_orchestrator.cognition.runtime_state import AgentRuntimeState
from agentos_orchestrator.cognition.safety_gates import (
    FormalSafetyVerifier,
    SafetyPolicy,
    default_safety_verifier,
)
from agentos_orchestrator.cognition.self_documentation import SelfDocumentationLoop
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
        docs = SelfDocumentationLoop._official_score("https://help.figma.com/docs", "Figma")
        social = SelfDocumentationLoop._official_score("https://reddit.com/r/figma", "Figma")
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
            elements=[
                UIElementState("button", "main", 0.1, 0.1, True, "Save")
            ],
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


class UniversalAgentFrontierIntegrationTests(unittest.TestCase):
    def _agent(self, decision: dict[str, Any] | FrontierDecision) -> UniversalDesktopAgentV2:
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
        agent = self._agent(
            {"action": "click", "target_id": 1, "confidence": 0.8}
        )
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
        self.assertGreaterEqual(run.exploration_probes_used, 1)
        self.assertGreaterEqual(len(agent.backend.performed), 1)

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