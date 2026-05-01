"""Blocker-specific repair planning for the universal OS agent.

The runtime state identifies blockers; this module turns them into bounded,
locally executable repair actions or explicit non-executable repair decisions.
It keeps repair logic deterministic so the frontier model can advise without
owning recovery authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiAction

from .abstract_world_model import AbstractUIState
from .runtime_state import Blocker, OutcomeEvaluation


@dataclass(slots=True)
class RepairPlan:
    """A temporary plan for resolving the highest-priority blocker."""

    kind: str
    description: str
    mode: str
    rationale: str
    expected_observation: str
    action: UiAction | None = None
    can_execute: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "mode": self.mode,
            "rationale": self.rationale,
            "expected_observation": self.expected_observation,
            "can_execute": self.can_execute,
            "action": _action_payload(self.action),
            "evidence": dict(self.evidence),
        }


class BlockerRepairPlanner:
    """Create deterministic repair plans from active runtime blockers."""

    def propose(
        self,
        blockers: list[Blocker],
        state: AbstractUIState,
        objective: str,
        option_name: str = "",
        recent_reflections: list[OutcomeEvaluation] | None = None,
    ) -> RepairPlan | None:
        active = [blocker for blocker in blockers if blocker.active]
        if not active:
            return None
        blocker = active[-1]
        text = _joined_text(blocker, recent_reflections or [])
        if _requires_human_approval(text):
            return RepairPlan(
                kind="approval_required",
                description=blocker.description,
                mode="approval",
                rationale="The local safety gate blocked a high-risk action.",
                expected_observation="A human approval token is supplied or a safer path is chosen.",
                action=None,
                can_execute=False,
                evidence={"blocker": blocker.to_prompt_dict()},
            )
        if _selector_or_stale_target(text):
            return self._explore_repair(blocker, "selector_or_stale_target")
        if _resource_missing(text):
            return self._explore_repair(blocker, "missing_resource")
        if _invalid_input(text):
            return RepairPlan(
                kind="invalid_input",
                description=blocker.description,
                mode="ui",
                rationale="The UI rejected a value, so focus should remain on the current field for correction.",
                expected_observation="The invalid value is selected or ready to be replaced.",
                action=UiAction(
                    action_type="hotkey",
                    selector="app-window",
                    value="^a",
                    metadata={
                        "source": "blocker_repair",
                        "repair_kind": "invalid_input",
                        "expected_observation": (
                            "The active invalid value is selected for replacement."
                        ),
                    },
                ),
                evidence={"blocker": blocker.to_prompt_dict()},
            )
        if _modal_blocker(blocker, state, text):
            return self._modal_repair(blocker, state, objective, option_name)
        return self._explore_repair(blocker, "unknown_blocker")

    @staticmethod
    def _explore_repair(blocker: Blocker, repair_kind: str) -> RepairPlan:
        expected = "Fresh exploration produces grounded targets for replanning."
        return RepairPlan(
            kind=repair_kind,
            description=blocker.description,
            mode="explore",
            rationale="The current target cannot be trusted; re-snapshot and probe safely.",
            expected_observation=expected,
            action=UiAction(
                action_type="explore",
                selector="blocker_repair:resnapshot",
                metadata={
                    "source": "blocker_repair",
                    "repair_kind": repair_kind,
                    "expected_observation": expected,
                },
            ),
            evidence={"blocker": blocker.to_prompt_dict()},
        )

    @staticmethod
    def _modal_repair(
        blocker: Blocker,
        state: AbstractUIState,
        objective: str,
        option_name: str,
    ) -> RepairPlan | None:
        modal_name = state.active_modal or _modal_name_from_text(blocker.description)
        if _modal_is_goal_relevant(modal_name, objective, option_name):
            return None
        expected = "The unexpected modal closes or focus returns to the parent UI."
        return RepairPlan(
            kind="unexpected_modal",
            description=blocker.description,
            mode="ui",
            rationale="An unrelated modal is blocking the parent goal.",
            expected_observation=expected,
            action=UiAction(
                action_type="hotkey",
                selector="app-window",
                value="{ESC}",
                metadata={
                    "source": "blocker_repair",
                    "repair_kind": "unexpected_modal",
                    "expected_observation": expected,
                },
            ),
            evidence={"blocker": blocker.to_prompt_dict(), "modal": modal_name},
        )


def _action_payload(action: UiAction | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "action_type": action.action_type,
        "selector": action.selector,
        "value": action.value,
        "metadata": dict(action.metadata),
    }


def _joined_text(
    blocker: Blocker,
    reflections: list[OutcomeEvaluation],
) -> str:
    chunks = [blocker.kind, blocker.description, blocker.evidence, blocker.repair_hint]
    for item in reflections[-2:]:
        chunks.extend(
            [
                item.failure_reason or "",
                item.new_blocker or "",
                item.observed,
            ]
        )
    return " ".join(chunks).lower()


def _requires_human_approval(text: str) -> bool:
    return any(token in text for token in ("approval", "destructive", "policy"))


def _selector_or_stale_target(text: str) -> bool:
    return any(
        token in text
        for token in (
            "selector",
            "target selector",
            "not found",
            "stale",
            "missing tag",
            "unknown set-of-mark",
        )
    )


def _resource_missing(text: str) -> bool:
    return any(token in text for token in ("resource", "path", "file not found"))


def _invalid_input(text: str) -> bool:
    return any(token in text for token in ("invalid", "rejected", "validation"))


def _modal_blocker(
    blocker: Blocker,
    state: AbstractUIState,
    text: str,
) -> bool:
    return bool(state.active_modal) or blocker.kind == "modal" or "modal" in text


def _modal_name_from_text(text: str) -> str:
    match = re.search(r"modal active:\s*(?P<name>.+)$", text, flags=re.I)
    return match.group("name").strip() if match else ""


def _modal_is_goal_relevant(
    modal_name: str,
    objective: str,
    option_name: str,
) -> bool:
    haystack = f"{objective} {option_name}".lower()
    modal = modal_name.lower()
    if not modal:
        return False
    modal_tokens = {token for token in re.split(r"[^a-z0-9]+", modal) if len(token) > 2}
    if modal_tokens and any(token in haystack for token in modal_tokens):
        return True
    save_like = {"save", "export", "download", "upload", "file", "filename"}
    return "save" in modal and any(token in haystack for token in save_like)
