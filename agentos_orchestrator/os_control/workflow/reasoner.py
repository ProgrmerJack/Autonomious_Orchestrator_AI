from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
)


@dataclass(slots=True)
class ActionDecision:
    step: DesktopWorkflowStep | None
    done: bool = False
    rationale: str = ""
    engine: str = "heuristic"


class DesktopWorkflowReasoner:
    """Adaptive next-action controller for general desktop workflows.

    The static planner still provides a bootstrap sequence, but this reasoner
    can continue execution based on the live UI snapshot. When a Gemini API key
    is available it can ask the model for the next action; otherwise it falls
    back to deterministic heuristics.
    """

    _MODEL_NAME = "gemini-2.0-flash"

    def next_decision(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> ActionDecision:
        model_decision = self._model_decision(objective, plan, nodes, receipts)
        if model_decision is not None:
            return model_decision
        return self._heuristic_decision(objective, plan, nodes, receipts)

    def _heuristic_decision(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> ActionDecision:
        if not nodes:
            return ActionDecision(None, done=True, rationale="No UI nodes available.")

        if self._is_browser_plan(plan) and self._has_action(receipts, "open_url"):
            return ActionDecision(
                None,
                done=True,
                rationale=(
                    "Browser navigation is complete; deeper reasoning requires "
                    "a model-backed controller or a downstream analysis worker."
                ),
            )

        primary_surface = self._best_surface_node(nodes)
        if primary_surface is None:
            return ActionDecision(
                None,
                done=True,
                rationale="No editable or actionable UI surface was found.",
            )

        surface_selector = self._selector_for_node(primary_surface)
        if not self._has_semantic_surface_write(receipts):
            return ActionDecision(
                DesktopWorkflowStep(
                    action_type="type",
                    selector=surface_selector,
                    value=self._intent_prompt(objective),
                    description="Apply operator intent to the best live UI surface.",
                ),
                rationale=(
                    f"Selected {primary_surface.role} '{primary_surface.name}' as the "
                    "best live workspace for a generic task handoff."
                ),
            )

        if plan.app_target and not self._has_action(receipts, "hotkey"):
            return ActionDecision(
                DesktopWorkflowStep(
                    action_type="hotkey",
                    selector="app-window",
                    value="^s",
                    description="Save progress after adaptive execution.",
                ),
                rationale="An editable surface was updated, so issue a save hotkey.",
            )

        return ActionDecision(
            None, done=True, rationale="No further adaptive action needed."
        )

    def _model_decision(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> ActionDecision | None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        prompt = self._model_prompt(objective, plan, nodes, receipts)
        try:
            payload = self._call_gemini(prompt, api_key)
        except (OSError, ValueError, urllib.error.URLError):
            return None
        decision = self._parse_model_payload(payload)
        if decision is None:
            return None
        return decision

    def _call_gemini(self, prompt: str, api_key: str) -> dict[str, Any]:
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._MODEL_NAME}:generateContent?key={urllib.parse.quote(api_key)}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def _parse_model_payload(self, payload: dict[str, Any]) -> ActionDecision | None:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        text = str(parts[0].get("text") or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if parsed.get("done"):
            return ActionDecision(
                None,
                done=True,
                rationale=str(
                    parsed.get("rationale") or "Model marked task as complete."
                ),
                engine="gemini",
            )
        action_type = str(parsed.get("action_type") or "").strip()
        selector = str(parsed.get("selector") or "").strip()
        if not action_type or not selector:
            return None
        return ActionDecision(
            DesktopWorkflowStep(
                action_type=action_type,
                selector=selector,
                value=parsed.get("value"),
                description=str(
                    parsed.get("description") or "Model-proposed adaptive action."
                ),
            ),
            rationale=str(parsed.get("rationale") or "Model-selected next action."),
            engine="gemini",
        )

    @staticmethod
    def _model_prompt(
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> str:
        node_lines = []
        for node in nodes[:40]:
            node_lines.append(
                f"- role={node.role}; name={node.name}; focused={node.focused}; enabled={node.enabled}"
            )
        recent_receipts = []
        for item in receipts[-8:]:
            recent_receipts.append(
                {
                    "action_type": item.get("action_type"),
                    "selector": item.get("selector"),
                    "receipt": item.get("receipt"),
                }
            )
        return (
            "You are a desktop action controller. Choose the SINGLE best next UI action "
            "for the objective below. Prefer using the current UI snapshot, avoid repeating "
            "actions that already happened, and only emit JSON.\n\n"
            f"Objective: {objective}\n"
            f"Plan mode: {plan.mode}\n"
            f"App target: {plan.app_target}\n\n"
            "Recent receipts:\n"
            f"{json.dumps(recent_receipts, indent=2)}\n\n"
            "Visible nodes:\n"
            f"{chr(10).join(node_lines)}\n\n"
            "Allowed action types: focus, click, invoke, type, set_text, hotkey.\n"
            "Return JSON with keys: done, action_type, selector, value, description, rationale.\n"
            'If no more useful progress can be made, return {"done": true, "rationale": "..."}.'
        )

    @staticmethod
    def _is_browser_plan(plan: DesktopWorkflowPlan) -> bool:
        return plan.app_target in {"msedge.exe", "chrome.exe"}

    @staticmethod
    def _has_action(receipts: list[dict[str, Any]], action_type: str) -> bool:
        return any(item.get("action_type") == action_type for item in receipts)

    @staticmethod
    def _has_semantic_surface_write(receipts: list[dict[str, Any]]) -> bool:
        writable_actions = {"type", "set_text", "set_value", "cell_edit", "draw_path"}
        ignored_selectors = {"browser-address-bar", "app-window"}
        for item in receipts:
            if item.get("action_type") not in writable_actions:
                continue
            if str(item.get("selector") or "") in ignored_selectors:
                continue
            payload = item.get("receipt")
            if (
                isinstance(payload, dict)
                and str(payload.get("status") or "").lower() == "selector-not-found"
            ):
                continue
            return True
        return False

    @classmethod
    def _best_surface_node(cls, nodes: list[UiNode]) -> UiNode | None:
        ranked = sorted(nodes, key=cls._surface_score, reverse=True)
        for node in ranked:
            if cls._surface_score(node) > 0:
                return node
        return None

    @staticmethod
    def _surface_score(node: UiNode) -> int:
        role_scores = {
            "Edit": 100,
            "Document": 95,
            "Canvas": 90,
            "Table": 88,
            "Pane": 75,
            "List": 65,
        }
        score = role_scores.get(node.role, 0)
        name = node.name.lower()
        if any(token in name for token in ("address", "search bar", "toolbar", "menu")):
            score -= 80
        if any(
            token in name
            for token in (
                "workspace",
                "canvas",
                "document",
                "editor",
                "sheet",
                "grid",
                "board",
            )
        ):
            score += 20
        if node.focused:
            score += 10
        if not node.enabled:
            score -= 100
        return score

    @staticmethod
    def _selector_for_node(node: UiNode) -> str:
        if node.name:
            return f"name={node.name}"
        return node.node_id

    @staticmethod
    def _intent_prompt(objective: str) -> str:
        return (
            "Operator intent:\n"
            f"{objective.strip()}\n\n"
            "Work on this task in the active application and preserve progress."
        )
