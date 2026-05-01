"""Active Inference exploration subroutine.

When the agent faces a UI with zero prior affordances, it halts the main
task and performs safe, exploratory actions to map state changes.
Based on the Free Energy Principle: the agent minimizes expected free energy
by choosing actions that reduce uncertainty about the environment.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentos_orchestrator.os_control.base import UiAction, UiNode


@dataclass(slots=True)
class ExplorationAction:
    """A single exploratory probe with expected information gain."""

    action: UiAction
    expected_info_gain: float
    safety_score: float  # 0.0-1.0, higher = safer
    rationale: str = ""


@dataclass(slots=True)
class ExplorationResult:
    """Outcome of an exploratory action."""

    action: UiAction
    pre_state_hash: str
    post_state_hash: str
    state_delta: list[str]
    info_gain: float
    safe: bool


class StateHasher(Protocol):
    """Protocol for producing a compact state fingerprint."""

    def hash(self, nodes: list[UiNode]) -> str:
        """Return a deterministic hash of the UI state."""


class ActiveInferenceExplorer:
    """Safe exploration engine for zero-prior-affordance UIs.

    Uses an information-gain heuristic to prioritize clicks on previously
    unvisited elements, while avoiding destructive actions.
    """

    # Actions considered safe for exploration
    SAFE_ACTION_TYPES = {"focus", "click", "hover", "type", "invoke"}
    DESTRUCTIVE_KEYWORDS = {
        "delete",
        "remove",
        "trash",
        "close",
        "quit",
        "exit",
        "format",
        "erase",
        "submit",
        "confirm",
        "purchase",
        "buy",
        "sell",
        "send",
        "transfer",
        "wire",
    }
    SEARCH_FIELD_KEYWORDS = {
        "search",
        "find",
        "query",
        "filter",
        "lookup",
        "address",
        "url",
    }
    CANVAS_OBJECTIVE_KEYWORDS = {
        "draw",
        "paint",
        "design",
        "sketch",
        "annotate",
        "write",
        "edit",
        "compose",
    }

    def __init__(self, max_probes: int = 6, random_seed: int | None = None) -> None:
        self.max_probes = max_probes
        self.rng = random.Random(random_seed)
        self._visited_selectors: set[str] = set()
        self._exploration_log: list[ExplorationResult] = []

    def explore(
        self,
        nodes: list[UiNode],
        objective: str,
        perform_fn: Any,
        snapshot_fn: Any,
    ) -> list[ExplorationResult]:
        """Run a bounded exploration loop and return the results.

        *perform_fn* receives a UiAction and returns a receipt string.
        *snapshot_fn* returns a list[UiNode].
        """
        results: list[ExplorationResult] = []
        for _ in range(self.max_probes):
            pre_nodes = snapshot_fn()
            pre_hash = self._state_hash(pre_nodes)
            action = self._choose_probe(pre_nodes, objective)
            if action is None:
                break
            try:
                receipt = perform_fn(action)
                post_nodes = snapshot_fn()
                post_hash = self._state_hash(post_nodes)
                delta = self._state_delta(pre_nodes, post_nodes)
                info_gain = self._compute_info_gain(pre_hash, post_hash, delta)
                result = ExplorationResult(
                    action=action,
                    pre_state_hash=pre_hash,
                    post_state_hash=post_hash,
                    state_delta=delta,
                    info_gain=info_gain,
                    safe=self._is_safe_receipt(receipt),
                )
                results.append(result)
                self._exploration_log.append(result)
                self._visited_selectors.add(self._visit_key(action))
            except Exception:
                self._visited_selectors.add(self._visit_key(action))
                continue
        return results

    def suggest_action(
        self,
        nodes: list[UiNode],
        objective: str,
        success_patterns: list[dict[str, Any]] | None = None,
    ) -> UiAction | None:
        """Return the best grounded primitive for the current UI.

        Unlike `explore`, this is a single-step proposal used by the main
        agent loop before it falls back to abstract planning.
        """
        candidates = self._rank_candidates(
            nodes,
            objective,
            success_patterns=success_patterns,
        )
        for candidate in candidates:
            if candidate.safety_score < 0.35:
                continue
            candidate.action.metadata.setdefault("source", "active_inference_grounding")
            candidate.action.metadata.setdefault(
                "expected_observation",
                "This grounded primitive should reveal a new affordance or advance the task.",
            )
            candidate.action.metadata.setdefault("rationale", candidate.rationale)
            return candidate.action
        return None

    def _choose_probe(
        self,
        nodes: list[UiNode],
        objective: str,
    ) -> UiAction | None:
        """Select the next exploratory action using expected info gain."""
        candidates = self._rank_candidates(nodes, objective)
        for candidate in candidates:
            if candidate.safety_score < 0.35:
                continue
            if self._visit_key(candidate.action) not in self._visited_selectors:
                return candidate.action
        return None

    def _rank_candidates(
        self,
        nodes: list[UiNode],
        objective: str,
        success_patterns: list[dict[str, Any]] | None = None,
    ) -> list[ExplorationAction]:
        """Rank UI nodes by expected exploration value."""
        actions: list[ExplorationAction] = []
        lower_obj = objective.lower()
        for node in nodes:
            if not node.enabled:
                continue
            selector = self._selector_for_node(node)
            for action_type, value, rationale_hint in self._candidate_actions_for_node(
                node,
                lower_obj,
            ):
                action = UiAction(
                    action_type=action_type,
                    selector=selector,
                    value=value,
                    metadata={
                        "exploration": True,
                        "role": node.role,
                        "node_id": node.node_id,
                    },
                )
                if self._visit_key(action) in self._visited_selectors:
                    continue
                safety = self._safety_score(node, action_type=action_type, value=value)
                if safety <= 0.0:
                    continue
                info_gain = self._expected_info_gain(
                    node,
                    lower_obj,
                    action_type=action_type,
                ) + self._success_pattern_bonus(
                    selector,
                    node,
                    action_type,
                    success_patterns,
                )
                actions.append(
                    ExplorationAction(
                        action=action,
                        expected_info_gain=min(1.0, info_gain),
                        safety_score=safety,
                        rationale=(
                            f"{action_type} {node.role} '{node.name}' via {rationale_hint} "
                            f"(safety={safety:.2f}, info_gain={min(1.0, info_gain):.2f})"
                        ),
                    )
                )
        # Sort by expected free energy reduction = info_gain * safety
        actions.sort(key=lambda a: a.expected_info_gain * a.safety_score, reverse=True)
        return actions

    @staticmethod
    def _safety_score(
        node: UiNode,
        action_type: str = "click",
        value: str | None = None,
    ) -> float:
        """Score how safe it is to interact with this node."""
        name_lower = node.name.lower()
        role = node.role.lower()
        score = 1.0
        for kw in ActiveInferenceExplorer.DESTRUCTIVE_KEYWORDS:
            if kw in name_lower:
                score -= 0.4
        if action_type == "focus":
            score += 0.05
        if action_type == "type":
            if any(kw in name_lower for kw in {"password", "passcode", "secret"}):
                return 0.0
            if any(
                kw in name_lower for kw in ActiveInferenceExplorer.SEARCH_FIELD_KEYWORDS
            ):
                score -= 0.05
            else:
                score -= 0.25
        if value and len(value) > 60:
            score -= 0.1
        if role in {"button", "menuitem", "hyperlink"}:
            score += 0.1
        if role in {"edit", "document", "canvas"} and action_type == "click":
            score -= 0.1  # May edit content unintentionally
        return max(0.0, min(1.0, score))

    @staticmethod
    def _expected_info_gain(
        node: UiNode,
        lower_obj: str,
        action_type: str = "click",
    ) -> float:
        """Estimate information gain from interacting with this node."""
        score = 0.5
        name = node.name.lower()
        role = node.role.lower()
        # Prefer interactive controls
        if role in {"button", "menu", "menuitem", "hyperlink", "tab"}:
            score += 0.3
        if action_type == "focus" and role in {"edit", "document", "canvas", "pane"}:
            score += 0.2
        if action_type == "type" and any(
            kw in name for kw in ActiveInferenceExplorer.SEARCH_FIELD_KEYWORDS
        ):
            score += 0.45
        # Prefer nodes whose name matches the objective
        for token in set(lower_obj.split()):
            if token and token in name:
                score += 0.2
        # Prefer previously unvisited roles
        if role in {"combobox", "splitbutton", "toolbar"}:
            score += 0.15
        return min(1.0, score)

    def _candidate_actions_for_node(
        self,
        node: UiNode,
        lower_obj: str,
    ) -> list[tuple[str, str | None, str]]:
        role = node.role.lower()
        name_lower = node.name.lower()
        if role in {"edit", "combobox"}:
            candidates: list[tuple[str, str | None, str]] = [
                ("focus", None, "focus editable control"),
            ]
            if self._is_search_like(name_lower, lower_obj):
                candidates.insert(
                    0,
                    (
                        "type",
                        self._probe_text(lower_obj),
                        "type into search-like field",
                    ),
                )
            return candidates
        if role in {"document", "canvas"}:
            candidates = [("focus", None, "focus workspace surface")]
            if any(kw in lower_obj for kw in self.CANVAS_OBJECTIVE_KEYWORDS):
                candidates.append(("click", None, "enter primary workspace"))
            return candidates
        if role in {
            "button",
            "menu",
            "menuitem",
            "hyperlink",
            "tab",
            "listitem",
            "treeitem",
            "toolbar",
        }:
            return [("click", None, "activate control")]
        return [("focus", None, "inspect control")]

    @staticmethod
    def _is_search_like(name_lower: str, lower_obj: str) -> bool:
        if any(
            kw in name_lower for kw in ActiveInferenceExplorer.SEARCH_FIELD_KEYWORDS
        ):
            return True
        return any(kw in lower_obj for kw in {"search", "find", "query", "lookup"})

    @staticmethod
    def _probe_text(lower_obj: str) -> str:
        stop_words = {
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
        }
        tokens = [
            token
            for token in lower_obj.split()
            if len(token) > 2 and token not in stop_words
        ]
        if not tokens:
            return "help"
        return " ".join(tokens[:4])[:48]

    @staticmethod
    def _success_pattern_bonus(
        selector: str,
        node: UiNode,
        action_type: str,
        success_patterns: list[dict[str, Any]] | None,
    ) -> float:
        if not success_patterns:
            return 0.0
        selector_lower = selector.lower()
        node_name = node.name.lower()
        bonus = 0.0
        for event in success_patterns:
            past_action = event.get("action")
            if not isinstance(past_action, UiAction):
                continue
            past_selector = str(past_action.selector or "").lower()
            if past_action.action_type == action_type:
                bonus += 0.03
            if node_name and node_name in past_selector:
                bonus += 0.05
            if selector_lower and selector_lower == past_selector:
                bonus += 0.07
        return min(0.2, bonus)

    @staticmethod
    def _state_hash(nodes: list[UiNode]) -> str:
        """Produce a compact deterministic hash of UI state."""
        parts = []
        for node in sorted(nodes, key=lambda n: n.node_id):
            parts.append(f"{node.node_id}:{node.role}:{node.name}:{int(node.enabled)}")
        return str(hash("|".join(parts)))

    @staticmethod
    def _state_delta(pre: list[UiNode], post: list[UiNode]) -> list[str]:
        """Describe what changed between two UI snapshots."""
        pre_map = {n.node_id: n for n in pre}
        post_map = {n.node_id: n for n in post}
        delta: list[str] = []
        for nid, post_node in post_map.items():
            if nid not in pre_map:
                delta.append(f"+{post_node.role}:{post_node.name}")
                continue
            pre_node = pre_map[nid]
            if pre_node.name != post_node.name:
                delta.append(f"~name:{nid}:{pre_node.name}->{post_node.name}")
            if pre_node.enabled != post_node.enabled:
                delta.append(f"~enabled:{nid}:{pre_node.enabled}->{post_node.enabled}")
            if pre_node.focused != post_node.focused:
                delta.append(f"~focused:{nid}")
        for nid in pre_map:
            if nid not in post_map:
                delta.append(f"-{pre_map[nid].role}:{pre_map[nid].name}")
        return delta

    @staticmethod
    def _compute_info_gain(pre_hash: str, post_hash: str, delta: list[str]) -> float:
        """Quantify information gain from a state transition."""
        if pre_hash == post_hash:
            return 0.0
        return min(1.0, 0.2 + 0.1 * len(delta))

    @staticmethod
    def _is_safe_receipt(receipt: str) -> bool:
        """Determine if a receipt indicates a safe action outcome."""
        try:
            payload = json.loads(receipt)
        except json.JSONDecodeError:
            return True
        status = str(payload.get("status", "")).lower()
        return status not in {"blocked", "error", "crashed", "access-denied"}

    @staticmethod
    def _selector_for_node(node: UiNode) -> str:
        if node.name:
            return f"name={node.name}"
        return node.node_id

    @staticmethod
    def _visit_key(action: UiAction) -> str:
        return f"{action.action_type}:{action.selector}"

    def get_affordance_map(self) -> dict[str, dict[str, Any]]:
        """Return a map of explored selectors to their observed effects."""
        mapping: dict[str, dict[str, Any]] = {}
        for result in self._exploration_log:
            mapping[self._visit_key(result.action)] = {
                "info_gain": result.info_gain,
                "safe": result.safe,
                "deltas": result.state_delta,
            }
        return mapping
