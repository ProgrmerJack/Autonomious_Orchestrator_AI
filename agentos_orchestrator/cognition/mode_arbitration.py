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

_CONF_MIN = 0.35
_CONF_MAX = 0.95


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
    blocker_text = " ".join(
        f"{b.kind} {b.description} {b.repair_hint}".lower() for b in blockers
    )
    blocker_hits = _signal_hits(
        blocker_text,
        {"selector", "not found", "missing", "stale", "occluded", "tag"},
    )
    return ModeDecision(
        mode="explore",
        confidence=_bounded_confidence(0.72 + 0.05 * min(blocker_hits, 4)),
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
    cue_hits = _signal_hits(lower, _TOOL_TASK_CUES)
    confidence = _bounded_confidence(
        0.6 + 0.05 * min(cue_hits, 5) + (0.1 if context.tool_action_available else 0.0)
    )
    return ModeDecision(
        mode="tool",
        confidence=confidence,
        rationale=("The objective is better served by deterministic code execution."),
        should_use_frontier=False,
        evidence={"option": option_name, "tool_cue_hits": cue_hits},
    )


def _filesystem_decision(lower: str, option_name: str) -> ModeDecision | None:
    if option_name in {"find_target", "stale_target_reground"}:
        return None
    if not _filesystem_task(lower):
        return None
    fs_strength = _filesystem_signal_strength(lower)
    return ModeDecision(
        mode="filesystem",
        confidence=_bounded_confidence(0.62 + 0.26 * fs_strength),
        rationale=(
            "The option is a file operation and should use structured "
            "path-aware actions."
        ),
        should_use_frontier=False,
        evidence={"option": option_name, "filesystem_signal_strength": fs_strength},
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
    cue_hits = _signal_hits(lower, _RESEARCH_TASK_CUES)
    context_boost = (
        0.08 if state.app_context in {"unknown", "browser", "chat_app"} else 0.0
    )
    return ModeDecision(
        mode="research",
        confidence=_bounded_confidence(0.58 + 0.05 * min(cue_hits, 5) + context_boost),
        rationale=(
            "The option needs information gathering before desktop manipulation."
        ),
        should_use_frontier=state.app_context in {"unknown", "browser"},
        evidence={"app_context": state.app_context, "research_cue_hits": cue_hits},
    )


def _uncertain_surface_decision(
    state: AbstractUIState,
    context: ModeContext,
) -> ModeDecision | None:
    if state.app_context == "unknown" and context.perceived_element_count == 0:
        return ModeDecision(
            mode="explore",
            confidence=_bounded_confidence(
                0.68 + 0.03 * min(len(context.similar_failures), 4)
            ),
            rationale="No reliable local affordances are visible.",
            should_use_frontier=False,
            evidence={"perceived_element_count": context.perceived_element_count},
        )
    if state.app_context != "unknown" and not context.similar_failures:
        return None
    return ModeDecision(
        mode="hybrid",
        confidence=_bounded_confidence(
            0.6 + 0.04 * min(len(context.similar_failures), 5)
        ),
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
    interactive_count = sum(1 for item in state.elements if item.is_interactive)
    app_bonus = 0.08 if state.app_context not in {"unknown", "other"} else 0.0
    return ModeDecision(
        mode="ui",
        confidence=_bounded_confidence(
            0.5 + 0.04 * min(interactive_count, 6) + app_bonus
        ),
        rationale=("The visible UI has enough structure for local action selection."),
        should_use_frontier=False,
        evidence={
            "app_context": state.app_context,
            "interactive_count": interactive_count,
        },
    )


def _blockers_need_exploration(blockers: list[Any]) -> bool:
    text = " ".join(
        f"{b.kind} {b.description} {b.repair_hint}".lower() for b in blockers
    )
    return any(token in text for token in ("selector", "not found", "stale", "tag"))


def _tool_task(lower: str) -> bool:
    # Configurable cue sets — expressed as module-level constants so they can
    # be extended without editing decision logic.
    cues = _TOOL_TASK_CUES
    return any(cue in lower for cue in cues)


def _signal_hits(lower: str, cues: set[str]) -> int:
    return sum(1 for cue in cues if cue in lower)


def _filesystem_signal_strength(lower: str) -> float:
    phrase_hits = _signal_hits(lower, _FILESYSTEM_PHRASES)
    verb_hits = _signal_hits(lower, _FILESYSTEM_VERBS)
    noun_hits = _signal_hits(lower, _FILESYSTEM_NOUNS)
    # Phrase matches are strongest; verb+noun combination is secondary evidence.
    raw = min(
        1.0, 0.6 * min(phrase_hits, 2) / 2.0 + 0.4 * min(verb_hits * noun_hits, 4) / 4.0
    )
    return max(0.0, raw)


def _bounded_confidence(value: float) -> float:
    return max(_CONF_MIN, min(_CONF_MAX, value))


_TOOL_TASK_CUES = {
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


def _filesystem_task(lower: str) -> bool:
    if any(phrase in lower for phrase in _FILESYSTEM_PHRASES):
        return True
    return any(verb in lower for verb in _FILESYSTEM_VERBS) and any(
        noun in lower for noun in _FILESYSTEM_NOUNS
    )


_FILESYSTEM_PHRASES = {
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
_FILESYSTEM_VERBS = {
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
_FILESYSTEM_NOUNS = {"folder", "path", "directory", "filename", "file name", "file"}


def _research_task(lower: str) -> bool:
    return any(cue in lower for cue in _RESEARCH_TASK_CUES)


_RESEARCH_TASK_CUES = {
    "research",
    "search",
    "find",
    "look up",
    "compare",
    "investigate",
    "analyse",
    "analyze",
    "review",
}


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
            confidence=_bounded_confidence(profile.confidence + 0.1),
            rationale=(
                "Capability profile indicates low-confidence or visual-heavy control."
            ),
            should_use_frontier=False,
            evidence=evidence,
        )
    if profile.recommended_mode == "tool" and profile.app_family == "terminal":
        return ModeDecision(
            mode="tool",
            confidence=_bounded_confidence(max(profile.confidence, 0.68)),
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
