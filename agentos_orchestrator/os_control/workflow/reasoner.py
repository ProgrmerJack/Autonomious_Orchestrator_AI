from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from agentos_orchestrator.config import gemini_workflow_model
from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
)
from agentos_orchestrator.os_control.workflow.programmer import (
    build_programmer_tool_step,
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

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or gemini_workflow_model()

    def next_decision(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> ActionDecision:
        direct_decision = self._direct_lane_decision(
            objective,
            plan,
            nodes,
            receipts,
        )
        if direct_decision is not None:
            return direct_decision
        model_decision = self._model_decision(objective, plan, nodes, receipts)
        if model_decision is not None:
            return model_decision
        return self._heuristic_decision(objective, plan, nodes, receipts)

    def _direct_lane_decision(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> ActionDecision | None:
        research_step = self._research_tool_step(plan, receipts)
        tool_step = self._programmer_tool_step(plan, receipts)
        if self._has_api_receipt(receipts):
            return ActionDecision(
                None,
                done=True,
                rationale=(
                    "A direct API action already produced structured output, "
                    "so no further UI handoff is needed."
                ),
            )
        if self._has_research_receipt(receipts) and tool_step is None:
            return ActionDecision(
                None,
                done=True,
                rationale=(
                    "The workflow research brief already produced provider-backed "
                    "evidence, so no further UI handoff is needed."
                ),
            )
        if not nodes:
            if research_step is not None:
                return ActionDecision(
                    research_step,
                    rationale=(
                        "The plan already has a provider-backed research step, "
                        "so emit it before requiring any UI."
                    ),
                )
            if tool_step is not None:
                return ActionDecision(
                    tool_step,
                    rationale=(
                        "The plan already has artifact outputs, so emit the "
                        "programmer lane tool step before requiring any UI."
                    ),
                )
            return None

        api_step = self._api_surface_step(objective, nodes, receipts)
        if api_step is not None:
            return ActionDecision(
                api_step,
                rationale=(
                    "A live endpoint is exposed on the current surface, so "
                    "prefer a direct API action before brittle UI interaction."
                ),
            )

        if research_step is not None:
            return ActionDecision(
                research_step,
                rationale=(
                    "The plan already has a provider-backed research step, so "
                    "emit it before typing into the UI."
                ),
            )

        if tool_step is not None:
            return ActionDecision(
                tool_step,
                rationale=(
                    "The plan already has artifact outputs, so emit the "
                    "programmer lane tool step before typing into the UI."
                ),
            )
        return None

    def _heuristic_decision(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> ActionDecision:
        if not nodes:
            return ActionDecision(None, done=True, rationale="No UI nodes available.")

        if self._is_browser_plan(plan) and self._browser_navigation_complete(
            objective,
            receipts,
        ):
            return ActionDecision(
                None,
                done=True,
                rationale=(
                    "Browser navigation is complete; deeper reasoning requires "
                    "a model-backed controller or a downstream analysis worker."
                ),
            )

        primary_surface = self._best_surface_node(nodes, objective, plan.mode)
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
            f"{self.model_name}:generateContent?key={urllib.parse.quote(api_key)}"
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
        metadata = parsed.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return ActionDecision(
            DesktopWorkflowStep(
                action_type=action_type,
                selector=selector,
                value=parsed.get("value"),
                description=str(
                    parsed.get("description") or "Model-proposed adaptive action."
                ),
                metadata=metadata,
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
            "Allowed action types: focus, click, invoke, type, set_text, hotkey, api_call, tool.\n"
            "If action_type is api_call, selector must be the endpoint URL.\n"
            "If action_type is tool, selector must be tool_executor:workflow_programmer or tool_executor:workflow_research and metadata.tool_request must be present.\n"
            "Return JSON with keys: done, action_type, selector, value, description, rationale, metadata.\n"
            'If no more useful progress can be made, return {"done": true, "rationale": "..."}.'
        )

    @classmethod
    def _research_tool_step(
        cls,
        plan: DesktopWorkflowPlan,
        receipts: list[dict[str, Any]],
    ) -> DesktopWorkflowStep | None:
        if cls._has_tool_selector(receipts, "tool_executor:workflow_research"):
            return None
        for step in plan.steps:
            if (
                step.action_type == "tool"
                and step.selector == "tool_executor:workflow_research"
            ):
                return step
        return None

    @classmethod
    def _programmer_tool_step(
        cls,
        plan: DesktopWorkflowPlan,
        receipts: list[dict[str, Any]],
    ) -> DesktopWorkflowStep | None:
        if cls._has_tool_selector(
            receipts, "tool_executor:workflow_programmer"
        ) or cls._has_semantic_surface_write(receipts):
            return None
        for step in plan.steps:
            if (
                step.action_type == "tool"
                and step.selector == "tool_executor:workflow_programmer"
            ):
                return step
        if any(
            step.action_type == "tool"
            and step.selector == "tool_executor:workflow_research"
            for step in plan.steps
        ):
            return None
        return build_programmer_tool_step(plan)

    @classmethod
    def _api_surface_step(
        cls,
        objective: str,
        nodes: list[UiNode],
        receipts: list[dict[str, Any]],
    ) -> DesktopWorkflowStep | None:
        if cls._has_api_action(receipts):
            return None
        objective_prefers_api = cls._objective_prefers_api(objective)
        candidates: list[tuple[int, UiNode, str]] = []
        for node in nodes:
            endpoint = cls._node_endpoint(node)
            if not endpoint:
                continue
            name_lower = node.name.lower()
            metadata = dict(node.metadata or {})
            if not objective_prefers_api and not (
                "api" in name_lower or metadata.get("api")
            ):
                continue
            score = 100 if node.focused else 0
            score += 40 if metadata.get("workflow") else 0
            score += (
                25 if metadata.get("endpoint") or metadata.get("api_endpoint") else 0
            )
            score += 20 if node.role in {"Pane", "Document", "Table"} else 0
            score += 10 if "dashboard" in name_lower or "api" in name_lower else 0
            candidates.append((score, node, endpoint))
        if not candidates:
            return None
        _, node, endpoint = max(candidates, key=lambda item: item[0])
        metadata = dict(node.metadata or {})
        step_metadata: dict[str, Any] = {
            "control_channel": "api",
            "method": cls._api_method(objective, metadata),
            "expected_observation": (
                "The direct API action returns a structured response."
            ),
        }
        workflow = metadata.get("workflow")
        if isinstance(workflow, list) and workflow:
            step_metadata["workflow"] = workflow
        auth_env_keys = metadata.get("auth_env_keys") or []
        if isinstance(auth_env_keys, (list, tuple)):
            step_metadata["auth_env_keys"] = [str(item) for item in auth_env_keys]
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            step_metadata["headers"] = dict(headers)
        return DesktopWorkflowStep(
            action_type="api_call",
            selector=endpoint,
            description="Invoke the surfaced API directly before manipulating the UI.",
            metadata=step_metadata,
        )

    @staticmethod
    def _has_api_action(receipts: list[dict[str, Any]]) -> bool:
        return any(
            item.get("action_type") in {"api_call", "mcp_call", "http_request"}
            for item in receipts
        )

    @staticmethod
    def _has_tool_selector(receipts: list[dict[str, Any]], selector: str) -> bool:
        return any(
            item.get("action_type") == "tool" and item.get("selector") == selector
            for item in receipts
        )

    @staticmethod
    def _has_api_receipt(receipts: list[dict[str, Any]]) -> bool:
        for item in receipts:
            if item.get("action_type") in {"api_call", "mcp_call", "http_request"}:
                return True
            selector = str(item.get("selector") or "")
            receipt = item.get("receipt")
            if (
                "tool_executor:workflow_api" in selector
                or "control_surface_probe" in selector
            ):
                return True
            if isinstance(receipt, dict) and str(receipt.get("kind") or "") in {
                "api_workflow",
                "http_probe",
            }:
                return True
        return False

    @staticmethod
    def _has_research_receipt(receipts: list[dict[str, Any]]) -> bool:
        for item in receipts:
            if (
                item.get("action_type") == "tool"
                and item.get("selector") == "tool_executor:workflow_research"
            ):
                return True
            receipt = item.get("receipt")
            if (
                isinstance(receipt, dict)
                and str(receipt.get("kind") or "") == "workflow_research_brief"
            ):
                return True
        return False

    @staticmethod
    def _objective_prefers_api(objective: str) -> bool:
        lower = objective.lower()
        return any(
            cue in lower
            for cue in (
                " api",
                "api ",
                "endpoint",
                "graphql",
                "openapi",
                "swagger",
                "webhook",
                "refresh api",
                "health",
                "probe",
                "localhost",
                "127.0.0.1",
            )
        )

    @staticmethod
    def _node_endpoint(node: UiNode) -> str:
        metadata = dict(node.metadata or {})
        for key in ("endpoint", "api_endpoint", "url"):
            value = str(metadata.get(key) or "").strip()
            if value.startswith(("http://", "https://")):
                return value
        workflow = metadata.get("workflow")
        if isinstance(workflow, list) and workflow:
            first = workflow[0]
            if isinstance(first, dict):
                endpoint = str(first.get("url") or "").strip()
                if endpoint.startswith(("http://", "https://")):
                    return endpoint
        return ""

    @staticmethod
    def _api_method(objective: str, metadata: dict[str, Any]) -> str:
        declared = str(metadata.get("method") or "").strip().upper()
        if declared:
            return declared
        lower = objective.lower()
        if any(token in lower for token in (" delete ", " remove ")):
            return "DELETE"
        if any(token in lower for token in (" patch ", " update ")):
            return "PATCH"
        if " put " in lower:
            return "PUT"
        if any(
            token in lower for token in (" post ", " create ", " submit ", " send ")
        ):
            return "POST"
        return "GET"

    @staticmethod
    def _is_browser_plan(plan: DesktopWorkflowPlan) -> bool:
        return plan.app_target in {"msedge.exe", "chrome.exe"}

    @staticmethod
    def _has_action(receipts: list[dict[str, Any]], action_type: str) -> bool:
        return any(item.get("action_type") == action_type for item in receipts)

    @classmethod
    def _browser_navigation_complete(
        cls,
        objective: str,
        receipts: list[dict[str, Any]],
    ) -> bool:
        if not cls._has_action(receipts, "open_url"):
            return False
        lower = objective.lower()
        navigation_only = any(
            cue in lower for cue in ("open", "go to", "navigate", "visit")
        ) and not any(
            cue in lower
            for cue in (
                "analyze",
                "analyse",
                "compare",
                "research",
                "write",
                "draft",
                "extract",
                "summarize",
            )
        )
        if navigation_only:
            return True
        # If semantic write-like progress already happened, navigation phase is done.
        return cls._has_semantic_surface_write(receipts)

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
    def _best_surface_node(
        cls,
        nodes: list[UiNode],
        objective: str,
        mode: str,
    ) -> UiNode | None:
        objective_terms = cls._objective_terms(objective)
        ranked = sorted(
            nodes,
            key=lambda item: cls._surface_score(item, objective_terms, mode),
            reverse=True,
        )
        for node in ranked:
            if cls._surface_score(node, objective_terms, mode) > 0:
                return node
        return None

    @staticmethod
    def _surface_score(node: UiNode, objective_terms: set[str], mode: str) -> int:
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

        # Objective relevance: prefer nodes whose labels overlap with query terms.
        score += sum(8 for term in objective_terms if term in name)

        mode_lower = (mode or "").lower()
        if mode_lower in {"spreadsheet", "report"} and node.role in {
            "Table",
            "Document",
        }:
            score += 12
        if mode_lower in {"script", "app-task"} and node.role in {"Edit", "Pane"}:
            score += 10
        if mode_lower == "drawing" and node.role in {"Canvas", "Pane"}:
            score += 12

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
    def _objective_terms(objective: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9_\-]+", objective.lower())
            if len(term) >= 4
        }

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
