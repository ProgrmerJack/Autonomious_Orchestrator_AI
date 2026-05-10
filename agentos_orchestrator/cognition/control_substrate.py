"""Adaptive control substrate for universal desktop execution.

This module implements the first production slice of the universal OS-control
plan: canonical observations, explicit action proposals, pre-action decisions,
and an append-only control ledger. The classes are deterministic on purpose so
they can run before any frontier model or learned policy is trusted.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agentos_orchestrator.os_control.base import UiAction, UiNode

from .abstract_world_model import AbstractUIState
from .safety_gates import FormalSafetyVerifier, SafetyDecision, SafetyPolicy


CONTROL_CHANNELS = (
    "api",
    "code",
    "accessibility",
    "native",
    "vision",
    "clipboard",
    "manual",
)
ACTION_ROUTES = ("api_mcp", "code_tool", "structured_ui", "native_vision")
REVERSIBILITY = {"reversible", "hard_to_reverse", "irreversible"}


@dataclass(slots=True)
class ControlChannelScores:
    api: float = 0.0
    code: float = 0.0
    accessibility: float = 0.0
    native: float = 0.0
    vision: float = 0.0
    clipboard: float = 0.0
    manual: float = 0.0

    def asdict(self) -> dict[str, float]:
        return {
            key: round(float(value), 3)
            for key, value in asdict(self).items()
        }


@dataclass(slots=True)
class ObservationElement:
    node_id: str
    role: str
    name: str
    bounds: tuple[int, int, int, int] | None = None
    enabled: bool = True
    focused: bool = False
    channel_scores: ControlChannelScores = field(
        default_factory=ControlChannelScores,
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_scores"] = self.channel_scores.asdict()
        return payload


@dataclass(slots=True)
class ObservationFrame:
    frame_id: str
    created_at: float
    backend_name: str
    stable_fingerprint: str
    app_context: str = "unknown"
    layout_mode: str = "unknown"
    focused_element: str = ""
    active_modal: str = ""
    screenshot_hash: str = ""
    ocr_text: str = ""
    semantic_summary: dict[str, Any] = field(default_factory=dict)
    semantic_diff: dict[str, Any] = field(default_factory=dict)
    capability_profile: dict[str, Any] = field(default_factory=dict)
    discovered_surfaces: list[dict[str, Any]] = field(default_factory=list)
    elements: list[ObservationElement] = field(default_factory=list)
    previous_fingerprint: str = ""

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elements"] = [element.asdict() for element in self.elements]
        return payload


@dataclass(slots=True)
class ActionProposal:
    proposal_id: str
    action_type: str
    selector: str
    route: str
    fallback_routes: list[str]
    reversibility: str
    risk_score: float
    risk_level: str
    required_approval: bool
    expected_state_change: str
    verification_contract: dict[str, Any]
    observation_fingerprint: str
    rollback_notes: str = ""
    rationale: str = ""
    route_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_score"] = round(float(self.risk_score), 3)
        payload["route_scores"] = {
            key: round(float(value), 3)
            for key, value in self.route_scores.items()
        }
        return payload


@dataclass(slots=True)
class PreActionDecision:
    allowed: bool
    decision: str
    reason: str
    risk_score: float
    required_approval: bool = False
    approval_state: str = "not_required"
    diagnostician: str = "execute"
    violations: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_score"] = round(float(self.risk_score), 3)
        return payload


@dataclass(slots=True)
class GoalLock:
    objective: str
    objective_terms: list[str]
    allowed_domains: list[str]
    destructive_intent: bool
    external_navigation_intent: bool
    file_op_intent: bool
    required_surface: str = ""

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LedgerEvent:
    event_id: str
    entry_id: str
    stage: str
    created_at: float
    payload: dict[str, Any]

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationFrameBuilder:
    """Build canonical observation frames from local backend snapshots."""

    def build(
        self,
        *,
        nodes: list[UiNode],
        backend_name: str,
        abstract_state: AbstractUIState | None = None,
        previous: ObservationFrame | None = None,
        capability_profile: dict[str, Any] | None = None,
        discovered_surfaces: list[dict[str, Any]] | None = None,
        screenshot_hash: str = "",
        ocr_text: str = "",
    ) -> ObservationFrame:
        elements = [_element_from_node(node) for node in nodes]
        fingerprint = _stable_fingerprint(elements, abstract_state)
        previous_fingerprint = previous.stable_fingerprint if previous else ""
        if abstract_state is None:
            app_context = "unknown"
            layout_mode = "unknown"
            active_modal = ""
        else:
            app_context = abstract_state.app_context
            layout_mode = abstract_state.layout_mode
            active_modal = abstract_state.active_modal
        return ObservationFrame(
            frame_id=_short_hash(f"frame:{fingerprint}:{time.time_ns()}"),
            created_at=time.time(),
            backend_name=backend_name,
            stable_fingerprint=fingerprint,
            app_context=app_context,
            layout_mode=layout_mode,
            focused_element=_focused_element(elements),
            active_modal=active_modal,
            screenshot_hash=screenshot_hash,
            ocr_text=ocr_text,
            semantic_summary=_semantic_summary(elements, abstract_state),
            semantic_diff=_semantic_diff(previous, fingerprint, elements),
            capability_profile=dict(capability_profile or {}),
            discovered_surfaces=list(discovered_surfaces or []),
            elements=elements,
            previous_fingerprint=previous_fingerprint,
        )


class FourLaneActionRouter:
    """Route an action through the strongest safe control lane available."""

    def propose(
        self,
        *,
        action: UiAction,
        observation: ObservationFrame,
        objective: str = "",
    ) -> ActionProposal:
        route_scores = self._route_scores(action, observation, objective)
        route = max(route_scores.items(), key=lambda item: item[1])[0]
        risk_score = _risk_score(action, route, objective)
        reversibility = _reversibility(action)
        required_approval = _requires_approval(
            action,
            objective,
            reversibility,
        )
        contract = _verification_contract(action)
        proposal_seed = json.dumps(
            {
                "action_type": action.action_type,
                "selector": action.selector,
                "route": route,
                "fingerprint": observation.stable_fingerprint,
                "value": str(action.value or "")[:120],
            },
            sort_keys=True,
        )
        return ActionProposal(
            proposal_id=_short_hash(proposal_seed),
            action_type=action.action_type,
            selector=action.selector,
            route=route,
            fallback_routes=_fallback_routes(route, action),
            reversibility=reversibility,
            risk_score=risk_score,
            risk_level=_risk_level(risk_score),
            required_approval=required_approval,
            expected_state_change=str(
                contract.get("expected")
                or action.metadata.get("expected_observation")
                or "The action produces observable progress."
            ),
            verification_contract=contract,
            observation_fingerprint=observation.stable_fingerprint,
            rollback_notes=_rollback_notes(action, reversibility),
            rationale=_route_rationale(
                route,
                route_scores,
                action,
                observation,
            ),
            route_scores=route_scores,
            metadata={
                "backend_name": observation.backend_name,
                "frame_id": observation.frame_id,
            },
        )

    def _route_scores(
        self,
        action: UiAction,
        observation: ObservationFrame,
        objective: str,
    ) -> dict[str, float]:
        action_type = action.action_type.lower()
        selector = str(action.selector or "").lower()
        metadata = dict(action.metadata or {})
        lower_goal = f"{objective} {action_type} {selector}".lower()
        channel_max = _max_channel_scores(observation.elements)
        route_scores = {
            "api_mcp": 0.2 + 0.55 * channel_max.get("api", 0.0),
            "code_tool": 0.24 + 0.5 * channel_max.get("code", 0.0),
            "structured_ui": (
                0.28 + 0.55 * channel_max.get("accessibility", 0.0)
            ),
            "native_vision": 0.18
            + 0.35 * channel_max.get("native", 0.0)
            + 0.35 * channel_max.get("vision", 0.0),
        }
        if (
            action_type in _API_ACTIONS
            or metadata.get("control_channel") == "api"
        ):
            route_scores["api_mcp"] += 0.45
        if action_type in _CODE_ACTIONS or _code_task(lower_goal):
            route_scores["code_tool"] += 0.42
        if (
            _semantic_selector(selector)
            or action_type in _STRUCTURED_UI_ACTIONS
        ):
            route_scores["structured_ui"] += 0.3
        if action_type in _NATIVE_VISION_ACTIONS or _coordinate_action(action):
            route_scores["native_vision"] += 0.5
        canvas_likelihood = observation.capability_profile.get(
            "canvas_likelihood",
            0.0,
        )
        if canvas_likelihood >= 0.65:
            route_scores["native_vision"] += 0.12
        if observation.discovered_surfaces:
            route_scores["api_mcp"] += 0.12
        return {
            key: max(0.0, min(1.0, value))
            for key, value in route_scores.items()
        }


class PreActionVerifier:
    """Post-policy, pre-action verifier with deterministic diagnostics.

    Layers (innermost-first):
    1. FormalSafetyVerifier   — static pattern gates (blocklist keywords, paths)
    2. IntentVerifier         — intent-constraint compiler guards (payment, creds…)
    3. RiskGuardian           — policy-matrix risk score with approval threshold
    4. GoalLock conflict      — anti-drift check
    """

    def __init__(
        self,
        safety_verifier: FormalSafetyVerifier | None = None,
    ) -> None:
        high_stakes = frozenset(
            {"delete", "remove", "format", "trade", "submit", "purchase"},
        )
        self.safety_verifier = safety_verifier or FormalSafetyVerifier(
            SafetyPolicy(
                require_approval_for_destructive=True,
                high_stakes_actions=high_stakes,
            )
        )

        # Phase 4 — intent verifier (lazy import to avoid circular deps)
        try:
            from agentos_orchestrator.cognition.intent_verifier import (  # type: ignore[import]
                compile_intent,
                verify_action_intent,
            )
            self._compile_intent = compile_intent
            self._verify_action_intent = verify_action_intent
        except ImportError:
            self._compile_intent = None
            self._verify_action_intent = None

        # Phase 4 — risk guardian (lazy import)
        try:
            from agentos_orchestrator.cognition.risk_guardian import (  # type: ignore[import]
                assess_action_risk,
            )
            self._assess_action_risk = assess_action_risk
        except ImportError:
            self._assess_action_risk = None

    def verify(
        self,
        *,
        proposal: ActionProposal,
        action: UiAction,
        objective: str = "",
        observation: ObservationFrame | None = None,
        goal_lock: GoalLock | None = None,
        approval_token: str | None = None,
    ) -> PreActionDecision:
        # ── Layer 1: formal safety gates ────────────────────────────────── #
        safety = self.safety_verifier.verify_action(
            action,
            objective=objective,
            approval_token=approval_token,
        )
        if not safety.allowed:
            return _decision_from_safety(safety, proposal)

        # ── Layer 2: intent-constraint verifier (Phase 4) ────────────────── #
        if self._compile_intent is not None and self._verify_action_intent is not None:
            try:
                constraints = self._compile_intent(objective)
                intent_decision = self._verify_action_intent(
                    action, objective, constraints, proposal, goal_lock
                )
                diag = getattr(intent_decision, "diagnostician", "execute")
                if diag in ("abort",):
                    reason = getattr(intent_decision, "reason", "Intent constraint violated.")
                    return PreActionDecision(
                        allowed=False,
                        decision="abort",
                        reason=reason,
                        risk_score=max(proposal.risk_score, 0.9),
                        required_approval=True,
                        approval_state="missing",
                        diagnostician="abort",
                        evidence={
                            "proposal_id": proposal.proposal_id,
                            "intent_violated": getattr(
                                intent_decision, "violated_constraints", []
                            ),
                        },
                    )
                if diag in ("confirm",) and not approval_token:
                    reason = getattr(intent_decision, "reason", "Action requires approval.")
                    return PreActionDecision(
                        allowed=False,
                        decision="confirm",
                        reason=reason,
                        risk_score=max(proposal.risk_score, 0.75),
                        required_approval=True,
                        approval_state="missing",
                        diagnostician="confirm",
                        evidence={
                            "proposal_id": proposal.proposal_id,
                            "intent_violated": getattr(
                                intent_decision, "violated_constraints", []
                            ),
                        },
                    )
            except Exception:
                pass  # never block on verifier crash

        # ── Layer 3: risk guardian (Phase 4) ─────────────────────────────── #
        if self._assess_action_risk is not None:
            try:
                assessment = self._assess_action_risk(
                    action, proposal, observation, objective
                )
                if assessment.adjusted_risk >= 0.92:
                    return PreActionDecision(
                        allowed=False,
                        decision="abort",
                        reason=(
                            f"RiskGuardian: risk {assessment.adjusted_risk:.2f} "
                            "exceeds abort threshold."
                        ),
                        risk_score=assessment.adjusted_risk,
                        required_approval=True,
                        approval_state="missing",
                        diagnostician="abort",
                        evidence={
                            "proposal_id": proposal.proposal_id,
                            "risk_level": assessment.risk_level,
                        },
                    )
                if assessment.approval_required and not approval_token:
                    return PreActionDecision(
                        allowed=False,
                        decision="confirm",
                        reason=(
                            f"RiskGuardian: risk {assessment.adjusted_risk:.2f} "
                            "requires explicit approval."
                        ),
                        risk_score=assessment.adjusted_risk,
                        required_approval=True,
                        approval_state="missing",
                        diagnostician="confirm",
                        evidence={
                            "proposal_id": proposal.proposal_id,
                            "risk_level": assessment.risk_level,
                        },
                    )
            except Exception:
                pass  # never block on guardian crash

        # ── Approval gate (original) ─────────────────────────────────────── #
        if proposal.required_approval and not approval_token:
            return PreActionDecision(
                allowed=False,
                decision="confirm",
                reason="Action requires approval before execution.",
                risk_score=proposal.risk_score,
                required_approval=True,
                approval_state="missing",
                diagnostician="confirm",
                evidence={"proposal_id": proposal.proposal_id},
            )

        # ── Layer 4: goal-lock conflict check ────────────────────────────── #
        lock = goal_lock or build_goal_lock(
            objective=objective,
            observation=observation,
        )
        conflict = _goal_lock_conflict(
            action,
            objective,
            observation,
            lock,
        )
        if conflict:
            return PreActionDecision(
                allowed=False,
                decision="reflect",
                reason=conflict,
                risk_score=max(proposal.risk_score, 0.8),
                required_approval=False,
                approval_state="not_required",
                diagnostician="reflect",
                evidence={
                    "proposal_id": proposal.proposal_id,
                    "goal_lock": lock.asdict(),
                },
            )
        return PreActionDecision(
            allowed=True,
            decision="execute",
            reason="Pre-action verification passed.",
            risk_score=proposal.risk_score,
            required_approval=proposal.required_approval,
            approval_state="approved" if approval_token else "not_required",
            diagnostician="execute",
            evidence={"proposal_id": proposal.proposal_id},
        )


class AdaptiveControlLedger:
    """Append-only ledger for action provenance and failure promotion."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(
        cls,
        workspace_root: str | Path,
    ) -> "AdaptiveControlLedger":
        return cls(Path(workspace_root) / ".agentos" / "control_ledger.jsonl")

    def record_proposal(
        self,
        *,
        goal: str,
        observation: ObservationFrame,
        proposal: ActionProposal,
        decision: PreActionDecision,
        app_agent: str = "workflow",
        app_signature: str = "unknown",
    ) -> str:
        entry_id = _short_hash(
            f"entry:{goal}:{proposal.proposal_id}:{time.time_ns()}"
        )
        payload = {
            "entry_id": entry_id,
            "goal": goal,
            "observation_frame": observation.asdict(),
            "observation_fingerprint": observation.stable_fingerprint,
            "app_agent": app_agent,
            "app_signature": app_signature,
            "candidate_route": proposal.route,
            "fallback_routes": list(proposal.fallback_routes),
            "pre_action_verifier_result": decision.asdict(),
            "risk_guardian_score": proposal.risk_score,
            "policy_decision": decision.decision,
            "approval_state": decision.approval_state,
            "action_proposal": proposal.asdict(),
            "status": "proposed" if decision.allowed else "blocked",
        }
        self._append("proposal", entry_id, payload)
        return entry_id

    def record_completion(
        self,
        *,
        entry_id: str,
        receipt: Any,
        verification_result: dict[str, Any],
        repair_decision: dict[str, Any] | None = None,
        training_label: str = "success",
    ) -> None:
        matched = bool(verification_result.get("matched", True))
        payload = {
            "entry_id": entry_id,
            "execution_receipt": _json_safe(receipt),
            "post_action_verification_result": _json_safe(verification_result),
            "repair_decision": dict(repair_decision or {}),
            "training_label": (
                training_label if matched else "verification_failure"
            ),
            "regression_eligible": not matched,
            "status": "completed" if matched else "needs_repair",
        }
        self._append("completion", entry_id, payload)

    def record_failure_capsule(
        self,
        *,
        entry_id: str,
        reason: str,
        proposal: ActionProposal | None = None,
        observation: ObservationFrame | None = None,
    ) -> None:
        payload = {
            "entry_id": entry_id,
            "reason": reason,
            "action_proposal": proposal.asdict() if proposal else {},
            "observation_frame": observation.asdict() if observation else {},
            "training_label": "blocked_or_failed",
            "regression_eligible": True,
            "status": "failure_capsule",
        }
        self._append("failure", entry_id, payload)

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for line in lines[-max(1, limit):]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _append(
        self,
        stage: str,
        entry_id: str,
        payload: dict[str, Any],
    ) -> None:
        event = LedgerEvent(
            event_id=_short_hash(f"event:{stage}:{entry_id}:{time.time_ns()}"),
            entry_id=entry_id,
            stage=stage,
            created_at=time.time(),
            payload=_json_safe(payload),
        )
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.asdict(), sort_keys=True) + "\n")


def enrich_action_metadata(
    *,
    action: UiAction,
    observation: ObservationFrame,
    proposal: ActionProposal,
    decision: PreActionDecision,
    ledger_entry_id: str,
) -> UiAction:
    metadata = dict(action.metadata or {})
    control = {
        "observation_frame_id": observation.frame_id,
        "observation_fingerprint": observation.stable_fingerprint,
        "control_route": proposal.route,
        "fallback_routes": list(proposal.fallback_routes),
        "risk_score": round(float(proposal.risk_score), 3),
        "risk_level": proposal.risk_level,
        "required_approval": proposal.required_approval,
        "pre_action_decision": decision.asdict(),
        "proposal_id": proposal.proposal_id,
        "ledger_entry_id": ledger_entry_id,
    }
    for key in ("app_agent", "goal_lock", "speculation", "isolation"):
        if key in metadata:
            control[key] = _json_safe(metadata[key])
    metadata.setdefault("control_channel", proposal.route)
    metadata.setdefault("control_route", proposal.route)
    metadata.setdefault(
        "observation_fingerprint",
        observation.stable_fingerprint,
    )
    metadata.setdefault("pre_action_verification", decision.asdict())
    metadata["control"] = control
    return UiAction(
        action_type=action.action_type,
        selector=action.selector,
        value=action.value,
        metadata=metadata,
    )


def build_goal_lock(
    *,
    objective: str,
    observation: ObservationFrame | None = None,
) -> GoalLock:
    lower = str(objective or "").lower()
    domain_tokens = {
        "browser": ("browser", "web", "url", "search", "site", "page"),
        "file": ("file", "folder", "copy", "move", "rename", "delete"),
        "draw": ("draw", "paint", "canvas", "sketch", "illustration"),
        "sheet": ("sheet", "spreadsheet", "cell", "excel", "table"),
        "code": ("script", "code", "report", "analysis", "presentation"),
    }
    allowed_domains = [
        domain
        for domain, tokens in domain_tokens.items()
        if any(token in lower for token in tokens)
    ]
    required_surface = ""
    if observation is not None:
        required_surface = str(
            observation.capability_profile.get("app_family")
            or observation.app_context
            or ""
        )
    return GoalLock(
        objective=objective,
        objective_terms=re.findall(r"[a-z0-9_]{3,}", lower)[:12],
        allowed_domains=(
            allowed_domains + ["sheet"]
            if re.search(r"\b[a-z]{1,3}[0-9]{1,5}\b", lower)
            and "sheet" not in allowed_domains
            else allowed_domains
        ),
        destructive_intent=any(
            token in lower for token in ("delete", "remove", "format")
        ),
        external_navigation_intent=any(
            token in lower
            for token in (
                "browser",
                "web",
                "search",
                "url",
                "site",
                "find",
                "research",
                "lookup",
                "stock",
            )
        ),
        file_op_intent=any(
            token in lower
            for token in ("copy", "move", "rename", "delete", "folder", "file")
        ),
        required_surface=required_surface,
    )


def _element_from_node(node: UiNode) -> ObservationElement:
    metadata = dict(node.metadata or {})
    return ObservationElement(
        node_id=str(node.node_id or "unknown"),
        role=str(node.role or "Unknown"),
        name=str(node.name or ""),
        bounds=node.bounds,
        enabled=bool(node.enabled),
        focused=bool(node.focused),
        channel_scores=_channel_scores(node),
        metadata=_stable_metadata(metadata),
    )


def _channel_scores(node: UiNode) -> ControlChannelScores:
    metadata = dict(node.metadata or {})
    role = str(node.role or "").lower()
    name = str(node.name or "").lower()
    metadata_text = json.dumps(_json_safe(metadata), sort_keys=True).lower()
    has_bounds = node.bounds is not None
    is_editable = role in {"edit", "document", "textbox"} or "text" in name
    is_canvas = role in {"canvas", "chart", "image"} or "canvas" in name
    api_evidence = any(
        token in metadata_text
        for token in ("api", "endpoint", "openapi", "graphql", "devtools")
    )
    code_evidence = any(token in role + name for token in ("terminal", "file"))
    return ControlChannelScores(
        api=0.78 if api_evidence else 0.12,
        code=0.68 if code_evidence else 0.18,
        accessibility=0.82 if node.name or node.role else 0.25,
        native=0.7 if has_bounds else 0.3,
        vision=0.78 if has_bounds or is_canvas else 0.35,
        clipboard=0.72 if is_editable else 0.22,
        manual=0.2,
    )


def _stable_fingerprint(
    elements: list[ObservationElement],
    abstract_state: AbstractUIState | None,
) -> str:
    app_context = abstract_state.app_context if abstract_state else "unknown"
    layout_mode = abstract_state.layout_mode if abstract_state else "unknown"
    active_modal = abstract_state.active_modal if abstract_state else ""
    canonical = {
        "app_context": app_context,
        "layout_mode": layout_mode,
        "active_modal": active_modal,
        "elements": sorted(
            (_element_signature(element) for element in elements),
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8"),
    ).hexdigest()


def _element_signature(element: ObservationElement) -> dict[str, Any]:
    metadata = element.metadata
    return {
        "node_id": _normalize_text(element.node_id),
        "role": _normalize_text(element.role),
        "name": _normalize_text(element.name),
        "enabled": bool(element.enabled),
        "automation_id": _normalize_text(
            str(metadata.get("automation_id", "")),
        ),
        "class_name": _normalize_text(str(metadata.get("class_name", ""))),
        "panel_type": _normalize_text(str(metadata.get("panel_type", ""))),
    }


def _normalize_text(value: str) -> str:
    lowered = re.sub(r"\s+", " ", str(value or "").strip().lower())
    lowered = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "<time>", lowered)
    lowered = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", lowered)
    lowered = re.sub(r"\b\d+\b", "#", lowered)
    return lowered


def _stable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    stable_keys = {
        "automation_id",
        "class_name",
        "control_channel",
        "app_family",
        "panel_type",
        "backend",
        "native",
        "sandbox",
    }
    return {
        key: _json_safe(value)
        for key, value in metadata.items()
        if key in stable_keys
    }


def _semantic_summary(
    elements: list[ObservationElement],
    abstract_state: AbstractUIState | None,
) -> dict[str, Any]:
    roles: dict[str, int] = {}
    for element in elements:
        roles[element.role] = roles.get(element.role, 0) + 1
    app_context = abstract_state.app_context if abstract_state else "unknown"
    return {
        "node_count": len(elements),
        "focused": _focused_element(elements),
        "roles": roles,
        "app_context": app_context,
        "interactive_count": sum(1 for item in elements if item.enabled),
        "sample_names": [item.name for item in elements[:8] if item.name],
    }


def _semantic_diff(
    previous: ObservationFrame | None,
    fingerprint: str,
    elements: list[ObservationElement],
) -> dict[str, Any]:
    if previous is None:
        return {"changed": True, "reason": "initial_observation"}
    previous_ids = {element.node_id for element in previous.elements}
    current_ids = {element.node_id for element in elements}
    return {
        "changed": previous.stable_fingerprint != fingerprint,
        "added": sorted(current_ids - previous_ids)[:20],
        "removed": sorted(previous_ids - current_ids)[:20],
    }


def _focused_element(elements: Iterable[ObservationElement]) -> str:
    for element in elements:
        if element.focused:
            return element.node_id or element.name
    return ""


def _max_channel_scores(
    elements: list[ObservationElement],
) -> dict[str, float]:
    scores = {channel: 0.0 for channel in CONTROL_CHANNELS}
    for element in elements:
        element_scores = element.channel_scores.asdict()
        for channel in CONTROL_CHANNELS:
            scores[channel] = max(
                scores[channel],
                element_scores.get(channel, 0.0),
            )
    return scores


def _verification_contract(action: UiAction) -> dict[str, Any]:
    raw = action.metadata.get("verification_contract")
    if isinstance(raw, dict):
        return dict(raw)
    return {
        "kind": "state_changed",
        "expected": str(
            action.metadata.get("expected_observation")
            or f"The UI responds to {action.action_type}."
        ),
        "target": action.selector,
        "required": True,
    }


def _fallback_routes(route: str, action: UiAction) -> list[str]:
    if route == "api_mcp":
        return ["code_tool", "structured_ui", "native_vision"]
    if route == "code_tool":
        return ["structured_ui", "api_mcp", "native_vision"]
    if route == "structured_ui":
        return ["native_vision", "code_tool"]
    if action.action_type == "draw_path":
        return ["structured_ui", "manual"]
    return ["structured_ui", "manual"]


def _route_rationale(
    route: str,
    scores: dict[str, float],
    action: UiAction,
    observation: ObservationFrame,
) -> str:
    score = scores.get(route, 0.0)
    return (
        f"Selected {route} for {action.action_type} with score "
        f"{score:.2f} on {observation.backend_name}."
    )


def _risk_score(action: UiAction, route: str, objective: str) -> float:
    base = {
        "api_mcp": 0.2,
        "code_tool": 0.34,
        "structured_ui": 0.3,
        "native_vision": 0.55,
    }.get(route, 0.45)
    haystack = _action_haystack(action, objective)
    if any(token in haystack for token in _HIGH_RISK_TERMS):
        base += 0.35
    if _reversibility(action) == "irreversible":
        base += 0.25
    if action.action_type in {"type", "set_text", "set_value"}:
        base += 0.05
    return max(0.0, min(1.0, base))


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _reversibility(action: UiAction) -> str:
    action_type = action.action_type.lower()
    if action_type in {"delete_file", "format", "purchase", "trade", "submit"}:
        return "irreversible"
    if action_type in {"move_file", "rename_file", "install_package"}:
        return "hard_to_reverse"
    return "reversible"


def _requires_approval(
    action: UiAction,
    objective: str,
    reversibility: str,
) -> bool:
    haystack = _action_haystack(action, objective)
    if reversibility == "irreversible":
        return True
    return any(token in haystack for token in _APPROVAL_TERMS)


def _rollback_notes(action: UiAction, reversibility: str) -> str:
    if reversibility == "reversible":
        return (
            "Re-observe and undo through the same route if verification "
            "fails."
        )
    if action.action_type == "move_file":
        return "Record source and destination before execution for reversal."
    return "Requires confirmation or external recovery before execution."


def _decision_from_safety(
    safety: SafetyDecision,
    proposal: ActionProposal,
) -> PreActionDecision:
    return PreActionDecision(
        allowed=False,
        decision="confirm",
        reason=safety.reason,
        risk_score=max(proposal.risk_score, 0.75),
        required_approval=True,
        approval_state="missing",
        diagnostician="confirm",
        violations=[asdict(violation) for violation in safety.violations],
        evidence={
            "proposal_id": proposal.proposal_id,
            "solver": safety.solver,
        },
    )


def _goal_lock_conflict(
    action: UiAction,
    objective: str,
    observation: ObservationFrame | None = None,
    goal_lock: GoalLock | None = None,
) -> str:
    lower_goal = str(objective or "").lower()
    if not lower_goal:
        return ""
    lock = goal_lock or build_goal_lock(
        objective=objective,
        observation=observation,
    )
    haystack = _action_haystack(action, "")
    if "delete" in haystack and not lock.destructive_intent:
        return "Action drifts outside the declared workflow objective."
    if "send" in haystack and not any(
        term in lower_goal for term in ("send", "message")
    ):
        return "Action drifts outside the declared workflow objective."
    app_family = ""
    if observation is not None:
        app_family = str(
            observation.capability_profile.get("app_family")
            or observation.app_context
            or ""
        )
    if (
        action.action_type == "open_url"
        and not lock.external_navigation_intent
        and app_family != "browser"
    ):
        return (
            "Goal lock blocked browser navigation outside the declared "
            "objective."
        )
    if action.action_type == "draw_path" and not (
        "draw" in lock.allowed_domains
        or app_family == "design_canvas"
        or "canvas" in str(action.selector or "").lower()
    ):
        return "Goal lock blocked drawing outside a canvas-oriented objective."
    if action.action_type in {"cell_edit", "set_cell"} and not (
        "sheet" in lock.allowed_domains or app_family == "office_form"
    ):
        return "Goal lock blocked spreadsheet edits outside a sheet objective."
    if action.action_type in {
        "copy_file",
        "move_file",
        "rename_file",
        "delete_file",
    } and not lock.file_op_intent:
        return (
            "Goal lock blocked file mutation outside the declared file "
            "objective."
        )
    return ""


def _action_haystack(action: UiAction, objective: str) -> str:
    return " ".join(
        str(part).lower()
        for part in (
            objective,
            action.action_type,
            action.selector,
            action.value,
            json.dumps(_json_safe(action.metadata), sort_keys=True),
        )
        if part is not None
    )


def _semantic_selector(selector: str) -> bool:
    tokens = (
        "name=",
        "role=",
        "automation_id=",
        "class_name=",
        "node_id=",
    )
    return any(
        token in selector
        for token in tokens
    )


def _coordinate_action(action: UiAction) -> bool:
    metadata = dict(action.metadata or {})
    selector = str(action.selector or "")
    return bool(
        ("x" in metadata and "y" in metadata)
        or "bounds" in metadata
        or "bbox" in metadata
        or re.match(r"^\s*(point=)?-?\d+\s*,\s*-?\d+\s*$", selector)
    )


def _code_task(lower_goal: str) -> bool:
    return any(
        cue in lower_goal
        for cue in (
            "analyze",
            "analysis",
            "stock",
            "report",
            "presentation",
            "slides",
            "chart",
            "script",
            "convert",
            "generate",
            "file",
        )
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "asdict"):
            return value.asdict()
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


_API_ACTIONS = {"api_call", "mcp_call", "http_request", "browser_devtools"}
_CODE_ACTIONS = {
    "tool",
    "execute_command",
    "create_file",
    "write_file",
    "read_file",
    "copy_file",
    "move_file",
    "rename_file",
    "download_file",
    "upload_file",
}
_STRUCTURED_UI_ACTIONS = {
    "focus",
    "click",
    "invoke",
    "type",
    "set_text",
    "set_value",
    "hotkey",
    "launch_app",
    "open_url",
}
_NATIVE_VISION_ACTIONS = {"draw_path", "move_cursor", "scroll"}
_HIGH_RISK_TERMS = {
    "password",
    "credential",
    "secret",
    "token",
    "payment",
    "purchase",
    "trade",
    "order",
    "delete",
    "format",
    "permission",
    "admin",
    "install",
}
_APPROVAL_TERMS = {
    "payment",
    "purchase",
    "trade",
    "place order",
    "send message",
    "external message",
    "credential",
    "password",
    "permission",
    "install package",
    "system configuration",
}
