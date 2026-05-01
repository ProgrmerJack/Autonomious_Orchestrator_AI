"""Capability profiling for unknown desktop surfaces.

The universal agent should not treat every screen as the same kind of UI.
This module converts the current abstract state plus accessibility nodes into
a compact capability profile used by mode arbitration, app adapters, replay,
and training traces.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.app_family_registry import (
    app_family_names,
    family_for_context,
    profile_rules,
    spec_for_family,
)
from agentos_orchestrator.os_control.base import UiNode

from .abstract_world_model import AbstractUIState


CONTROL_CHANNELS = ("accessibility", "dom", "api", "ocr", "vision", "explore")
APP_FAMILIES = app_family_names()


@dataclass(slots=True)
class CapabilityProfile:
    app_family: str
    app_signature: str
    confidence: float
    control_channels: list[str]
    accessibility_quality: float
    dom_presence: float = 0.0
    ocr_quality: float = 0.0
    canvas_likelihood: float = 0.0
    modal_pressure: float = 0.0
    file_dialog_likelihood: float = 0.0
    clipboard_reliability: float = 0.5
    latency_sensitivity: float = 0.5
    recommended_mode: str = "ui"
    risks: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "app_family": self.app_family,
            "app_signature": self.app_signature,
            "confidence": round(float(self.confidence), 3),
            "control_channels": list(self.control_channels),
            "accessibility_quality": round(
                float(self.accessibility_quality),
                3,
            ),
            "dom_presence": round(float(self.dom_presence), 3),
            "ocr_quality": round(float(self.ocr_quality), 3),
            "canvas_likelihood": round(float(self.canvas_likelihood), 3),
            "modal_pressure": round(float(self.modal_pressure), 3),
            "file_dialog_likelihood": round(
                float(self.file_dialog_likelihood),
                3,
            ),
            "clipboard_reliability": round(
                float(self.clipboard_reliability),
                3,
            ),
            "latency_sensitivity": round(float(self.latency_sensitivity), 3),
            "recommended_mode": self.recommended_mode,
            "risks": list(self.risks),
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True)
class _ProfileSignals:
    family: str
    family_confidence: float
    accessibility: float
    dom: float
    canvas: float
    modal: float
    file_dialog: float
    ocr: float
    risks: list[str]
    channels: list[str]


class CapabilityProfiler:
    """Infer control capabilities for the current app/session."""

    def profile(
        self,
        state: AbstractUIState,
        nodes: list[UiNode] | None = None,
        screenshot_available: bool = False,
        last_latency_ms: float = 0.0,
    ) -> CapabilityProfile:
        nodes = nodes or []
        node_text = _node_text(nodes)
        signals = _collect_signals(
            state,
            nodes,
            node_text,
            screenshot_available,
        )
        return CapabilityProfile(
            app_family=signals.family,
            app_signature=_app_signature(state, nodes, signals.family),
            confidence=_clamp((signals.family_confidence + signals.accessibility) / 2),
            control_channels=signals.channels,
            accessibility_quality=signals.accessibility,
            dom_presence=signals.dom,
            ocr_quality=signals.ocr,
            canvas_likelihood=signals.canvas,
            modal_pressure=signals.modal,
            file_dialog_likelihood=signals.file_dialog,
            clipboard_reliability=_clipboard_reliability(
                signals.family,
                signals.accessibility,
            ),
            latency_sensitivity=_latency_sensitivity(
                last_latency_ms,
                signals.family,
            ),
            recommended_mode=_recommended_mode(
                signals.family,
                signals.accessibility,
                signals.canvas,
                signals.modal,
            ),
            risks=signals.risks,
            evidence=_profile_evidence(state, nodes),
        )


def _collect_signals(
    state: AbstractUIState,
    nodes: list[UiNode],
    node_text: str,
    screenshot_available: bool,
) -> _ProfileSignals:
    family, family_confidence = _classify_family(state, nodes, node_text)
    accessibility = _accessibility_quality(state, nodes)
    dom = _dom_presence(nodes, node_text, family)
    canvas = _canvas_likelihood(state, nodes, node_text)
    modal = _modal_pressure(state)
    file_dialog = _file_dialog_likelihood(state, node_text, family)
    ocr = _ocr_quality(screenshot_available, accessibility, canvas)
    return _ProfileSignals(
        family=family,
        family_confidence=family_confidence,
        accessibility=accessibility,
        dom=dom,
        canvas=canvas,
        modal=modal,
        file_dialog=file_dialog,
        ocr=ocr,
        risks=_risks(
            accessibility,
            canvas,
            modal,
            file_dialog,
            family_confidence,
        ),
        channels=_channels(accessibility, dom, ocr, canvas, family),
    )


def _profile_evidence(
    state: AbstractUIState,
    nodes: list[UiNode],
) -> dict[str, Any]:
    return {
        "app_context": state.app_context,
        "layout_mode": state.layout_mode,
        "node_count": len(nodes),
        "element_count": len(state.elements),
        "interactive_count": sum(1 for item in state.elements if item.is_interactive),
        "sample_names": _sample_names(nodes),
    }


def _modal_pressure(state: AbstractUIState) -> float:
    return 1.0 if state.active_modal or state.layout_mode == "modal_open" else 0.0


def _classify_family(
    state: AbstractUIState,
    nodes: list[UiNode],
    node_text: str,
) -> tuple[str, float]:
    context_family = family_for_context(state.app_context)
    if context_family is not None:
        return context_family, 0.82
    for family, score, cues in profile_rules():
        if any(cue in node_text for cue in cues):
            return family, score
    roles = {node.role.lower() for node in nodes}
    if "document" in roles and "edit" in roles:
        return "editor", 0.62
    return "unknown", 0.45


def _accessibility_quality(
    state: AbstractUIState,
    nodes: list[UiNode],
) -> float:
    named_nodes = sum(1 for node in nodes if node.name or node.role)
    enabled_nodes = sum(1 for node in nodes if node.enabled)
    node_score = named_nodes / max(len(nodes), 1)
    enabled_score = enabled_nodes / max(len(nodes), 1)
    interactive = sum(1 for item in state.elements if item.is_interactive)
    element_score = min(1.0, interactive / 5) if state.elements else 0.0
    if not nodes:
        return _clamp(element_score * 0.7)
    return _clamp(node_score * 0.45 + enabled_score * 0.25 + element_score * 0.3)


def _dom_presence(nodes: list[UiNode], node_text: str, family: str) -> float:
    if spec_for_family(family).family == "browser":
        return 0.65
    if spec_for_family(family).dom_like:
        return 0.55
    if any("dom" in str(node.metadata).lower() for node in nodes):
        return 0.75
    if "chrome_widgetwin" in node_text:
        return 0.5
    return 0.0


def _canvas_likelihood(
    state: AbstractUIState,
    nodes: list[UiNode],
    node_text: str,
) -> float:
    canvas_terms = ("canvas", "webgl", "image", "drawing", "preview")
    if any(term in node_text for term in canvas_terms):
        return 0.75
    visual = sum(
        1 for item in state.elements if item.element_type in {"image", "video"}
    )
    sparse_accessibility = len(nodes) <= 2 and len(state.elements) > 4
    if visual or sparse_accessibility:
        return 0.55
    return 0.1


def _file_dialog_likelihood(
    state: AbstractUIState,
    node_text: str,
    family: str,
) -> float:
    if family == "file_dialog":
        return 0.95
    if state.active_modal and any(term in node_text for term in ("save", "open")):
        return 0.75
    return 0.0


def _ocr_quality(
    screenshot_available: bool,
    accessibility: float,
    canvas: float,
) -> float:
    if not screenshot_available:
        return 0.0
    return _clamp(0.35 + canvas * 0.35 + (1.0 - accessibility) * 0.25)


def _channels(
    accessibility: float,
    dom: float,
    ocr: float,
    canvas: float,
    family: str,
) -> list[str]:
    channels = ["accessibility"] if accessibility >= 0.35 else []
    if dom >= 0.35:
        channels.append("dom")
    if spec_for_family(family).api_like:
        channels.append("api")
    if ocr >= 0.35:
        channels.append("ocr")
    if canvas >= 0.5:
        channels.append("vision")
    channels.append("explore")
    return list(dict.fromkeys(channels))


def _recommended_mode(
    family: str,
    accessibility: float,
    canvas: float,
    modal: float,
) -> str:
    if modal >= 0.7:
        return "ui"
    spec = spec_for_family(family)
    if spec.family == "unknown" and accessibility < 0.45:
        return "explore"
    if accessibility < 0.3 and canvas >= 0.5:
        return "explore"
    return spec.recommended_mode


def _risks(
    accessibility: float,
    canvas: float,
    modal: float,
    file_dialog: float,
    confidence: float,
) -> list[str]:
    risks = []
    if accessibility < 0.35:
        risks.append("low_accessibility_quality")
    if canvas >= 0.6:
        risks.append("canvas_or_image_heavy")
    if modal >= 0.7:
        risks.append("modal_active")
    if file_dialog >= 0.7:
        risks.append("file_dialog_requires_path_verification")
    if confidence < 0.55:
        risks.append("unknown_app_family")
    return risks


def _clipboard_reliability(family: str, accessibility: float) -> float:
    base = spec_for_family(family).clipboard_base
    return _clamp(base * 0.6 + accessibility * 0.4)


def _latency_sensitivity(last_latency_ms: float, family: str) -> float:
    base = spec_for_family(family).latency_base
    latency_penalty = min(0.25, max(0.0, last_latency_ms / 4000))
    return _clamp(base + latency_penalty)


def _node_text(nodes: list[UiNode]) -> str:
    chunks = []
    for node in nodes:
        chunks.extend([node.role, node.name, str(node.metadata)])
    return " ".join(chunks).lower()


def _sample_names(nodes: list[UiNode]) -> list[str]:
    names = [node.name for node in nodes if node.name]
    return names[:8]


def _app_signature(
    state: AbstractUIState,
    nodes: list[UiNode],
    family: str,
) -> str:
    normalized_names = [
        re.sub(r"\d+", "<n>", node.name.lower().strip()) for node in nodes if node.name
    ][:4]
    normalized_roles = sorted({node.role.lower() for node in nodes})[:4]
    interactive_types = sorted(
        {item.element_type.lower() for item in state.elements if item.is_interactive}
    )[:4]
    basis = "|".join(
        [
            family,
            state.layout_mode,
            state.focus_region,
            ",".join(normalized_roles),
            ",".join(normalized_names),
            ",".join(interactive_types),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"{family}:{digest}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def known_app_families() -> tuple[str, ...]:
    return APP_FAMILIES
