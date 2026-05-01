"""Explicit action-mode arbitration for the universal OS agent.

The agent should not default to UI clicks for every task. This module chooses
between local UI control, tool execution, research/navigation, filesystem work,
hybrid frontier reasoning, and bounded exploration before action selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .abstract_world_model import AbstractUIState
from .capability_profile import CapabilityProfile
from .runtime_state import AgentRuntimeState


AGENT_MODES = {"ui", "tool", "hybrid", "research", "filesystem", "explore"}


@dataclass(slots=True)
class ModeDecision:
    mode: str
    confidence: float
    rationale: str
    should_use_frontier: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale,
            "should_use_frontier": self.should_use_frontier,
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True)
class ModeContext:
    perceived_element_count: int = 0
    tool_action_available: bool = False
    similar_failures: list[dict[str, Any]] = field(default_factory=list)
    capability_profile: CapabilityProfile | None = None
    adapter_context: dict[str, Any] | None = None


class ModeArbiter:
    """Choose the safest and highest-leverage execution mode per option."""

    def choose(
        self,
        objective: str,
        option: Any,
        state: AbstractUIState,
        runtime_state: AgentRuntimeState,
        context: ModeContext | None = None,
    ) -> ModeDecision:
        context = context or ModeContext()
        option_name = str(getattr(option, "name", ""))
        lower = f"{objective} {option_name}".lower()
        blockers = [b for b in runtime_state.blocker_stack if b.active]
        candidates = [
            _blocker_decision(blockers),
            _profile_context_decision(context),
            _tool_decision(lower, option_name, context),
            _filesystem_decision(lower, option_name),
            _research_decision(lower, state),
            _uncertain_surface_decision(state, context),
        ]
        for decision in candidates:
            if decision is not None:
                return decision
        return _ui_decision(state)


def _blocker_decision(blockers: list[Any]) -> ModeDecision | None:
    if not blockers or not _blockers_need_exploration(blockers):
        return None
    return ModeDecision(
        mode="explore",
        confidence=0.9,
        rationale="Active blockers indicate stale or missing targets.",
        should_use_frontier=False,
        evidence={"blockers": [b.to_prompt_dict() for b in blockers[-3:]]},
    )


def _tool_decision(
    lower: str,
    option_name: str,
    context: ModeContext,
) -> ModeDecision | None:
    if not context.tool_action_available and not _tool_task(lower):
        return None
    return ModeDecision(
        mode="tool",
        confidence=0.88,
        rationale=("The objective is better served by deterministic code execution."),
        should_use_frontier=False,
        evidence={"option": option_name},
    )


def _filesystem_decision(lower: str, option_name: str) -> ModeDecision | None:
    if option_name in {"find_target", "stale_target_reground"}:
        return None
    if not _filesystem_task(lower):
        return None
    return ModeDecision(
        mode="filesystem",
        confidence=0.82,
        rationale=(
            "The option is a file operation and should use structured "
            "path-aware actions."
        ),
        should_use_frontier=False,
        evidence={"option": option_name},
    )


def _research_decision(
    lower: str,
    state: AbstractUIState,
) -> ModeDecision | None:
    if not _research_task(lower):
        return None
    if "find target" in lower:
        return None
    if state.app_context not in {"unknown", "browser", "chat_app"} and not any(
        cue in lower for cue in {"research", "look up", "compare", "investigate"}
    ):
        return None
    return ModeDecision(
        mode="research",
        confidence=0.78,
        rationale=(
            "The option needs information gathering before desktop manipulation."
        ),
        should_use_frontier=state.app_context in {"unknown", "browser"},
        evidence={"app_context": state.app_context},
    )


def _uncertain_surface_decision(
    state: AbstractUIState,
    context: ModeContext,
) -> ModeDecision | None:
    if state.app_context == "unknown" and context.perceived_element_count == 0:
        return ModeDecision(
            mode="explore",
            confidence=0.74,
            rationale="No reliable local affordances are visible.",
            should_use_frontier=False,
            evidence={"perceived_element_count": context.perceived_element_count},
        )
    if state.app_context != "unknown" and not context.similar_failures:
        return None
    return ModeDecision(
        mode="hybrid",
        confidence=0.68,
        rationale=(
            "Local perception is uncertain, so combine local marks with "
            "frontier semantics."
        ),
        should_use_frontier=True,
        evidence={
            "app_context": state.app_context,
            "similar_failures": len(context.similar_failures),
        },
    )


def _ui_decision(state: AbstractUIState) -> ModeDecision:
    return ModeDecision(
        mode="ui",
        confidence=0.72,
        rationale=("The visible UI has enough structure for local action selection."),
        should_use_frontier=False,
        evidence={
            "app_context": state.app_context,
            "interactive_count": sum(
                1 for item in state.elements if item.is_interactive
            ),
        },
    )


def _blockers_need_exploration(blockers: list[Any]) -> bool:
    text = " ".join(
        f"{b.kind} {b.description} {b.repair_hint}".lower() for b in blockers
    )
    return any(token in text for token in ("selector", "not found", "stale", "tag"))


def _tool_task(lower: str) -> bool:
    cues = {
        "analysis",
        "analyse",
        "analyze",
        "calculate",
        "compute",
        "quant",
        "statistics",
        "regression",
        "forecast",
        "script",
        "code",
        "pipeline",
    }
    return any(cue in lower for cue in cues)


def _filesystem_task(lower: str) -> bool:
    if any(
        phrase in lower
        for phrase in {
            "file path",
            "folder path",
            "directory",
            "filename",
            "file name",
            "open file",
            "save file",
            "select file",
            "choose file",
            "browse folder",
            "browse file",
        }
    ):
        return True
    verbs = {
        "copy",
        "move",
        "rename",
        "delete",
        "save",
        "open",
        "select",
        "choose",
        "browse",
    }
    nouns = {"folder", "path", "directory", "filename", "file name", "file"}
    return any(verb in lower for verb in verbs) and any(noun in lower for noun in nouns)


def _research_task(lower: str) -> bool:
    cues = {"research", "search", "find", "look up", "compare", "investigate"}
    return any(cue in lower for cue in cues)


def _profile_context_decision(context: ModeContext) -> ModeDecision | None:
    profile = context.capability_profile
    if profile is None:
        return None
    evidence = {
        "capability_profile": profile.to_prompt_dict(),
        "adapter_context": context.adapter_context or {},
    }
    if profile.recommended_mode == "explore":
        return ModeDecision(
            mode="explore",
            confidence=0.82,
            rationale=(
                "Capability profile indicates low-confidence or visual-heavy control."
            ),
            should_use_frontier=False,
            evidence=evidence,
        )
    if profile.recommended_mode == "tool" and profile.app_family == "terminal":
        return ModeDecision(
            mode="tool",
            confidence=0.76,
            rationale=(
                "Terminal-like surface should prefer structured tool execution."
            ),
            should_use_frontier=False,
            evidence=evidence,
        )
    if profile.recommended_mode == "hybrid":
        return ModeDecision(
            mode="hybrid",
            confidence=max(0.68, profile.confidence),
            rationale=(
                "Profile recommends combining local structure with semantic guidance."
            ),
            should_use_frontier=True,
            evidence=evidence,
        )
    return None


def available_modes() -> set[str]:
    return set(AGENT_MODES)
