from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from typing import Any

from .base import UiNode


@dataclass(slots=True)
class SelectorCandidate:
    selector: str
    node_id: str
    role: str
    name: str
    score: float
    reasons: list[str] = dataclass_field(default_factory=list)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectorDebugReport:
    selector: str
    exact_matches: int
    candidates: list[SelectorCandidate]
    ready: bool
    guidance: str

    def asdict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "exact_matches": self.exact_matches,
            "candidates": [
                candidate.asdict() for candidate in self.candidates
            ],
            "ready": self.ready,
            "guidance": self.guidance,
        }


def debug_selector(
    selector: str,
    nodes: list[UiNode],
    limit: int = 8,
) -> SelectorDebugReport:
    """Explain selector matching against a structured UI snapshot."""

    cleaned = selector.strip()
    scored = [_score_node(cleaned, node) for node in nodes if cleaned]
    candidates = sorted(
        (candidate for candidate in scored if candidate.score > 0),
        key=lambda candidate: candidate.score,
        reverse=True,
    )[: max(1, limit)]
    exact_matches = sum(
        1 for candidate in candidates if candidate.score >= 100
    )
    ready = exact_matches == 1 or bool(
        candidates and candidates[0].score >= 80
    )
    if not cleaned:
        guidance = (
            "Provide a selector such as name=Save or "
            "automation_id=submit."
        )
    elif exact_matches == 1:
        guidance = (
            "One exact match found; the selector is ready for "
            "guarded action."
        )
    elif exact_matches > 1:
        guidance = (
            "Multiple exact matches found; add role= or "
            "automation_id= detail."
        )
    elif candidates:
        guidance = (
            "No exact match found; use the highest-ranked "
            "fallback candidate."
        )
    else:
        guidance = (
            "No candidate matched; take a fresh snapshot or "
            "broaden the selector."
        )
    return SelectorDebugReport(
        selector=cleaned,
        exact_matches=exact_matches,
        candidates=candidates,
        ready=ready,
        guidance=guidance,
    )


def _score_node(selector: str, node: UiNode) -> SelectorCandidate:
    selector_field, expected = _split_selector(selector)
    expected_norm = _normalize(expected)
    values = {
        "name": node.name,
        "role": node.role,
        "automation_id": str(node.metadata.get("automation_id") or ""),
        "class_name": str(node.metadata.get("class_name") or ""),
        "node_id": node.node_id,
    }
    reasons: list[str] = []
    score = 0.0
    if selector_field in values:
        score = _field_score(
            expected_norm,
            _normalize(values[selector_field]),
            selector_field,
        )
        if score >= 100:
            reasons.append(f"exact {selector_field} match")
        elif score > 0:
            reasons.append(f"partial {selector_field} match")
    else:
        for key, value in values.items():
            value_score = _field_score(expected_norm, _normalize(value), key)
            if value_score > score:
                score = value_score
                reasons = [f"fallback {key} match"]
    if node.enabled:
        score += 4
        reasons.append("enabled")
    if node.focused:
        score += 3
        reasons.append("focused")
    return SelectorCandidate(
        selector=_best_selector(node),
        node_id=node.node_id,
        role=node.role,
        name=node.name,
        score=round(score, 3),
        reasons=reasons,
        metadata=dict(node.metadata),
    )


def _split_selector(selector: str) -> tuple[str | None, str]:
    selector_field, separator, expected = selector.partition("=")
    allowed = {"name", "role", "automation_id", "class_name", "node_id"}
    if separator and selector_field.strip() in allowed:
        return selector_field.strip(), expected.strip()
    return None, selector.strip()


def _field_score(expected: str, actual: str, selector_field: str) -> float:
    if not expected or not actual:
        return 0.0
    weight = 1.0
    if selector_field == "automation_id":
        weight = 1.15
    elif selector_field == "name":
        weight = 1.05
    if expected == actual:
        return 100.0 * weight
    if expected in actual:
        return 78.0 * weight
    expected_terms = set(expected.split())
    actual_terms = set(actual.split())
    if not expected_terms:
        return 0.0
    overlap = len(expected_terms & actual_terms) / len(expected_terms)
    return 55.0 * overlap * weight


def _best_selector(node: UiNode) -> str:
    automation_id = str(node.metadata.get("automation_id") or "").strip()
    if automation_id:
        return f"automation_id={automation_id}"
    if node.name:
        return f"name={node.name}"
    return f"role={node.role}"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()
