"""Persistent runtime state for the continuous deliberative agent.

This module is the memory spine between perception, frontier reasoning,
execution, and self-correction. It keeps compact temporal state instead of
passing one isolated screenshot at a time.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiAction

from .abstract_world_model import AbstractUIState


@dataclass(slots=True)
class GoalFrame:
    name: str
    intent: str
    success_criteria: list[str] = field(default_factory=list)
    status: str = "active"
    retry_budget: int = 3

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "intent": self.intent,
            "success_criteria": list(self.success_criteria),
            "status": self.status,
            "retry_budget": self.retry_budget,
        }


@dataclass(slots=True)
class Blocker:
    kind: str
    description: str
    evidence: str = ""
    repair_hint: str = ""
    active: bool = True

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "evidence": self.evidence,
            "repair_hint": self.repair_hint,
            "active": self.active,
        }


@dataclass(slots=True)
class TemporalFrame:
    timestamp: float
    screenshot_hash: str
    ui_summary: dict[str, Any]
    mark_ids: list[int] = field(default_factory=list)
    focus_target: str | None = None
    active_modal: str | None = None
    diff_from_previous: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "screenshot_hash": self.screenshot_hash[:16],
            "ui_summary": dict(self.ui_summary),
            "mark_ids": list(self.mark_ids),
            "focus_target": self.focus_target,
            "active_modal": self.active_modal,
            "diff_from_previous": dict(self.diff_from_previous),
        }


@dataclass(slots=True)
class ActionRecord:
    action_type: str
    selector: str
    value_preview: str = ""
    expected_observation: str = ""
    receipt_preview: str = ""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_action(
        cls,
        action: UiAction,
        expected_observation: str = "",
        receipt: str = "",
    ) -> "ActionRecord":
        return cls(
            action_type=action.action_type,
            selector=action.selector,
            value_preview=str(action.value or "")[:200],
            expected_observation=expected_observation[:500],
            receipt_preview=str(receipt or "")[:500],
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "selector": self.selector,
            "value_preview": self.value_preview,
            "expected_observation": self.expected_observation,
            "receipt_preview": self.receipt_preview,
            "timestamp": round(self.timestamp, 3),
        }


@dataclass(slots=True)
class Hypothesis:
    claim: str
    expected_observation: str
    risk: str = "unknown"

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "expected_observation": self.expected_observation,
            "risk": self.risk,
        }


@dataclass(slots=True)
class OutcomeEvaluation:
    expected: str
    observed: str
    matched: bool
    failure_reason: str | None = None
    new_blocker: str | None = None
    suggested_repair: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected,
            "observed": self.observed,
            "matched": self.matched,
            "failure_reason": self.failure_reason,
            "new_blocker": self.new_blocker,
            "suggested_repair": self.suggested_repair,
        }


@dataclass
class AgentRuntimeState:
    objective: str = ""
    goal_stack: list[GoalFrame] = field(default_factory=list)
    blocker_stack: list[Blocker] = field(default_factory=list)
    temporal_trace: deque[TemporalFrame] = field(
        default_factory=lambda: deque(maxlen=3),
    )
    last_actions: deque[ActionRecord] = field(
        default_factory=lambda: deque(maxlen=6),
    )
    recent_reflections: deque[OutcomeEvaluation] = field(
        default_factory=lambda: deque(maxlen=6),
    )
    current_ui: AbstractUIState | None = None
    previous_ui: AbstractUIState | None = None
    active_hypothesis: Hypothesis | None = None
    uncertainty: float = 0.0

    def reset(self, objective: str) -> None:
        self.objective = objective
        self.goal_stack.clear()
        self.blocker_stack.clear()
        self.temporal_trace.clear()
        self.last_actions.clear()
        self.recent_reflections.clear()
        self.current_ui = None
        self.previous_ui = None
        self.active_hypothesis = None
        self.uncertainty = 0.0

    def push_goal(
        self,
        name: str,
        intent: str,
        success_criteria: list[str] | None = None,
    ) -> None:
        self.goal_stack.append(
            GoalFrame(
                name=name,
                intent=intent,
                success_criteria=list(success_criteria or []),
            )
        )

    def update_observation(
        self,
        state: AbstractUIState,
        screenshot: bytes | None = None,
        mark_payload: dict[str, Any] | None = None,
    ) -> TemporalFrame:
        diff = diff_ui_states(self.current_ui, state)
        self.previous_ui = self.current_ui
        self.current_ui = state
        frame = TemporalFrame(
            timestamp=time.time(),
            screenshot_hash=_sha256_bytes(screenshot or b""),
            ui_summary=summarize_ui_state(state),
            mark_ids=_mark_ids(mark_payload or {}),
            focus_target=state.focus_region,
            active_modal=state.active_modal or None,
            diff_from_previous=diff,
        )
        self.temporal_trace.append(frame)
        if frame.active_modal and diff.get("active_modal"):
            self.add_blocker(
                kind="modal",
                description=f"Modal active: {frame.active_modal}",
                evidence=json.dumps(diff, sort_keys=True),
                repair_hint="Resolve the modal before continuing the parent goal.",
            )
        return frame

    def add_blocker(
        self,
        kind: str,
        description: str,
        evidence: str = "",
        repair_hint: str = "",
    ) -> None:
        signature = (kind, description)
        for blocker in self.blocker_stack:
            if (blocker.kind, blocker.description) == signature:
                blocker.active = True
                return
        self.blocker_stack.append(
            Blocker(
                kind=kind,
                description=description,
                evidence=evidence,
                repair_hint=repair_hint,
            )
        )

    def record_action(
        self,
        action: UiAction,
        expected_observation: str = "",
        receipt: str = "",
    ) -> ActionRecord:
        record = ActionRecord.from_action(action, expected_observation, receipt)
        self.last_actions.append(record)
        return record

    def evaluate_outcome(
        self,
        action: UiAction,
        before: AbstractUIState | None,
        after: AbstractUIState,
        receipt: str,
        expected_observation: str = "",
    ) -> OutcomeEvaluation:
        diff = diff_ui_states(before, after)
        observed = _observed_summary(receipt, diff)
        failure_reason = _failure_reason(receipt)
        new_blocker = _new_blocker(before, after, receipt)
        repair = _repair_hint(failure_reason, new_blocker)
        matched = failure_reason is None and (
            bool(diff)
            or action.action_type in {"tool", "explore", "hotkey", "type"}
            or _receipt_indicates_success(receipt)
        )
        evaluation = OutcomeEvaluation(
            expected=expected_observation or _default_expected(action),
            observed=observed,
            matched=matched,
            failure_reason=failure_reason,
            new_blocker=new_blocker,
            suggested_repair=repair,
        )
        self.recent_reflections.append(evaluation)
        if new_blocker:
            self.add_blocker(
                kind="runtime",
                description=new_blocker,
                evidence=observed,
                repair_hint=repair or "Replan around the blocker.",
            )
        return evaluation

    def frontier_context(
        self,
        current_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(current_summary or {})
        payload["runtime"] = {
            "objective": self.objective,
            "goal_stack": [goal.to_prompt_dict() for goal in self.goal_stack[-4:]],
            "blockers": [
                blocker.to_prompt_dict()
                for blocker in self.blocker_stack[-6:]
                if blocker.active
            ],
            "active_hypothesis": (
                self.active_hypothesis.to_prompt_dict()
                if self.active_hypothesis
                else None
            ),
            "uncertainty": round(float(self.uncertainty), 3),
        }
        payload["temporal_trace"] = [
            frame.to_prompt_dict() for frame in self.temporal_trace
        ]
        payload["last_actions"] = [
            action.to_prompt_dict() for action in self.last_actions
        ]
        payload["recent_reflections"] = [
            item.to_prompt_dict() for item in self.recent_reflections
        ]
        return payload


def summarize_ui_state(state: AbstractUIState) -> dict[str, Any]:
    type_counts = Counter(element.element_type for element in state.elements)
    return {
        "app_context": state.app_context,
        "layout_mode": state.layout_mode,
        "element_count": len(state.elements),
        "interactive_count": sum(1 for e in state.elements if e.is_interactive),
        "active_modal": state.active_modal,
        "focus_region": state.focus_region,
        "task_progress": dict(state.task_progress),
        "element_types": dict(sorted(type_counts.items())),
    }


def diff_ui_states(
    before: AbstractUIState | None,
    after: AbstractUIState,
) -> dict[str, Any]:
    if before is None:
        return {"initial": True, "element_count": len(after.elements)}
    diff: dict[str, Any] = {}
    for field_name in ("app_context", "layout_mode", "active_modal", "focus_region"):
        old = getattr(before, field_name)
        new = getattr(after, field_name)
        if old != new:
            diff[field_name] = {"before": old, "after": new}
    if len(before.elements) != len(after.elements):
        diff["element_count"] = {
            "before": len(before.elements),
            "after": len(after.elements),
        }
    before_types = Counter(element.element_type for element in before.elements)
    after_types = Counter(element.element_type for element in after.elements)
    if before_types != after_types:
        diff["element_types"] = {
            "before": dict(sorted(before_types.items())),
            "after": dict(sorted(after_types.items())),
        }
    return diff


def _sha256_bytes(value: bytes) -> str:
    if not value:
        return ""
    return hashlib.sha256(value).hexdigest()


def _mark_ids(mark_payload: dict[str, Any]) -> list[int]:
    marks = mark_payload.get("marks")
    if not isinstance(marks, list):
        return []
    ids: list[int] = []
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        try:
            ids.append(int(mark["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(ids)


def _observed_summary(receipt: str, diff: dict[str, Any]) -> str:
    chunks = []
    if receipt:
        chunks.append(f"receipt={str(receipt)[:500]}")
    if diff:
        chunks.append("diff=" + json.dumps(diff, sort_keys=True)[:500])
    return "; ".join(chunks) or "No visible state change was detected."


def _failure_reason(receipt: str) -> str | None:
    lower = str(receipt or "").lower()
    if "blocked" in lower:
        return "Action was blocked by policy or backend."
    if "selector-not-found" in lower or "not found" in lower:
        return "Target selector or required resource was not found."
    if "invalid" in lower:
        return "The UI rejected the submitted value as invalid."
    if "error" in lower or "exception" in lower or "failed" in lower:
        return "Backend receipt reported a failure."
    return None


def _new_blocker(
    before: AbstractUIState | None,
    after: AbstractUIState,
    receipt: str,
) -> str | None:
    if after.active_modal and (before is None or before.active_modal != after.active_modal):
        return f"New modal requires resolution: {after.active_modal}"
    reason = _failure_reason(receipt)
    if reason:
        return reason
    return None


def _repair_hint(failure_reason: str | None, new_blocker: str | None) -> str | None:
    text = f"{failure_reason or ''} {new_blocker or ''}".lower()
    if not text.strip():
        return None
    if "path" in text or "resource" in text:
        return "Validate or create the target resource before retrying."
    if "selector" in text:
        return "Re-snapshot the UI and choose a grounded target from current marks."
    if "modal" in text:
        return "Push a temporary modal-resolution subgoal."
    if "blocked" in text:
        return "Request approval or choose a safer alternative action."
    return "Replan using the latest observation and failure evidence."


def _receipt_indicates_success(receipt: str) -> bool:
    lower = str(receipt or "").lower()
    return any(token in lower for token in ("executed", "success", "ok", "saved"))


def _default_expected(action: UiAction) -> str:
    if action.action_type == "tool":
        return "The local tool returns structured output."
    if action.action_type == "explore":
        return "Bounded exploration produces state-change evidence."
    return f"The UI responds to {action.action_type} on {action.selector}."