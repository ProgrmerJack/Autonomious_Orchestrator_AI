"""Persistent affordance policy memory keyed by app signature.

The universal agent already learns within a single run, but unknown apps still
force it to rediscover the same reliable primitives on every new session.
This module stores successful action patterns by app signature so the next run
can reuse them before falling back to generic exploration.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentos_orchestrator.os_control.base import UiAction, UiNode


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "into",
    "from",
    "that",
    "this",
    "application",
    "app",
    "window",
    "task",
    "using",
}


@dataclass(slots=True)
class AffordancePolicyEntry:
    app_signature: str
    action_type: str
    selector: str
    control_channel: str
    objective_terms: list[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    value_hint: str = ""
    last_outcome: str = ""
    last_evidence: dict[str, Any] = field(default_factory=dict)

    def score(
        self,
        objective: str,
        nodes: list[UiNode] | None = None,
    ) -> float:
        objective_terms = set(_objective_terms(objective))
        stored_terms = set(self.objective_terms)
        overlap = len(objective_terms & stored_terms) / max(len(stored_terms), 1)
        reliability = (self.success_count + 1) / (
            self.success_count + self.failure_count + 2
        )
        familiarity = min(0.2, self.success_count * 0.04)
        node_bonus = (
            0.18 if _selector_matches_nodes(self.selector, nodes or []) else 0.0
        )
        failure_penalty = min(0.2, self.failure_count * 0.05)
        return max(
            0.0,
            min(
                1.0,
                reliability * 0.55
                + overlap * 0.2
                + familiarity
                + node_bonus
                - failure_penalty,
            ),
        )


class PersistentAffordancePolicyMemory:
    """Store and retrieve action affordances across runs."""

    def __init__(self, workspace_root: str | Path = ".") -> None:
        self.workspace_root = Path(workspace_root)
        self.path = self.workspace_root / ".agentos" / "affordance_policies.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, AffordancePolicyEntry] = {}
        self._load()

    def recommend_action(
        self,
        app_signature: str,
        objective: str,
        nodes: list[UiNode] | None = None,
        limit: int = 3,
    ) -> UiAction | None:
        if not app_signature:
            return None
        ranked = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.app_signature == app_signature and entry.success_count > 0
            ),
            key=lambda entry: entry.score(objective, nodes),
            reverse=True,
        )[:limit]
        if not ranked:
            return None
        best = ranked[0]
        best_score = best.score(objective, nodes)
        if best_score < 0.45:
            return None
        return UiAction(
            action_type=best.action_type,
            selector=best.selector,
            value=best.value_hint or None,
            metadata={
                "source": "policy_memory",
                "control_channel": best.control_channel,
                "policy_score": round(best_score, 3),
                "app_signature": app_signature,
                "expected_observation": (
                    "Reuse a previously successful affordance for this app signature."
                ),
            },
        )

    def preferred_channels(self, app_signature: str, limit: int = 3) -> list[str]:
        if not app_signature:
            return []
        channel_scores: dict[str, float] = {}
        for entry in self._entries.values():
            if entry.app_signature != app_signature or not entry.control_channel:
                continue
            score = (entry.success_count + 1) / (entry.failure_count + 1)
            channel_scores[entry.control_channel] = (
                channel_scores.get(
                    entry.control_channel,
                    0.0,
                )
                + score
            )
        return [
            channel
            for channel, _score in sorted(
                channel_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        ]

    def record(
        self,
        app_signature: str,
        objective: str,
        action: UiAction,
        success: bool,
        control_channel: str = "",
        observed: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if not app_signature or action.action_type == "explore":
            return
        entry = self._entries.setdefault(
            self._key(app_signature, action, control_channel),
            AffordancePolicyEntry(
                app_signature=app_signature,
                action_type=action.action_type,
                selector=action.selector,
                control_channel=control_channel
                or str(action.metadata.get("control_channel", "")),
            ),
        )
        if success:
            entry.success_count += 1
        else:
            entry.failure_count += 1
        entry.objective_terms = _merge_terms(
            entry.objective_terms, _objective_terms(objective)
        )
        if action.value and len(str(action.value)) <= 80:
            entry.value_hint = str(action.value)
        entry.last_outcome = str(observed or "")[:500]
        if evidence:
            entry.last_evidence = dict(evidence)
        self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._entries = {}
        for row in payload.get("entries", []):
            try:
                entry = AffordancePolicyEntry(**row)
            except TypeError:
                continue
            self._entries[
                self._key(
                    entry.app_signature,
                    UiAction(entry.action_type, entry.selector),
                    entry.control_channel,
                )
            ] = entry

    def _save(self) -> None:
        payload = {
            "version": 1,
            "entries": [asdict(entry) for entry in self._entries.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _key(app_signature: str, action: UiAction, control_channel: str) -> str:
        return "|".join(
            [
                app_signature,
                action.action_type,
                action.selector,
                control_channel or str(action.metadata.get("control_channel", "")),
            ]
        )


def _objective_terms(objective: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]{3,}", objective.lower())
    return [token for token in tokens if token not in _STOP_WORDS][:8]


def _merge_terms(existing: list[str], new_terms: list[str]) -> list[str]:
    merged = list(dict.fromkeys(existing + new_terms))
    return merged[:10]


def _selector_matches_nodes(selector: str, nodes: list[UiNode]) -> bool:
    selector_lower = selector.lower()
    for node in nodes:
        if node.node_id.lower() == selector_lower:
            return True
        if node.name and f"name={node.name}".lower() == selector_lower:
            return True
        if node.name and node.name.lower() in selector_lower:
            return True
    return False
