"""Risk guardian — action-to-target policy matrix and conformal risk scoring.

Phase 4 implementation. This module provides a deterministic risk guardian
that:

1. Maps (action_type × target_class × reversibility × data_sensitivity)
   to a baseline risk score and approval requirement.
2. Adjusts scores contextually from the observation frame and route.
3. Exposes a calibration hook so historical live-fire data can shift
   thresholds without rewriting the matrix.
4. Produces a structured RiskAssessment that feeds the PreActionVerifier
   and the AdaptiveControlLedger.

Design notes
────────────
* All scores are floats in [0.0, 1.0]. 0.0 = no risk, 1.0 = abort immediately.
* The APPROVAL_THRESHOLD (default 0.75) triggers ``confirm`` mode.
* The ABORT_THRESHOLD (default 0.92) triggers ``abort`` mode.
* Scores between thresholds trigger ``reflect`` or ``reroute`` depending on
  reversibility.
* The matrix is a frozen dataclass registry — adding new rows does not require
  changing verifier logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiAction

from .control_substrate import ActionProposal, ObservationFrame


# ─────────────────────────────────────────────────────────────────────────── #
# Thresholds                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

ABORT_THRESHOLD: float = 0.92
APPROVAL_THRESHOLD: float = 0.75
REFLECT_THRESHOLD: float = 0.50
REROUTE_THRESHOLD: float = 0.35


# ─────────────────────────────────────────────────────────────────────────── #
# Action-to-target policy matrix rows                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True, slots=True)
class PolicyMatrixRow:
    action_type: str          # matches UiAction.action_type (lower-cased prefix match)
    target_class: str         # "payment", "credential", "file", "system", "ui", "data", "*"
    reversibility: str        # "reversible" | "hard_to_reverse" | "irreversible"
    data_sensitivity: str     # "public" | "internal" | "confidential" | "secret"
    base_risk: float          # [0.0, 1.0]
    approval_required: bool
    diagnostician_floor: str  # minimum diagnostician level if risk threshold not met
    notes: str = ""


# The matrix encodes organisational policy. Each row is evaluated in order;
# the first matching row wins. A wildcard ("*") matches any value.
_POLICY_MATRIX: tuple[PolicyMatrixRow, ...] = (
    # ── Payment operations ──────────────────────────────────────────────── #
    PolicyMatrixRow("click", "payment", "irreversible", "confidential", 0.90, True,
                    "confirm", "Payment submit — always requires approval."),
    PolicyMatrixRow("type",  "payment", "irreversible", "confidential", 0.85, True,
                    "confirm", "Typing a payment value — approval required."),
    PolicyMatrixRow("*",     "payment", "irreversible", "confidential", 0.88, True,
                    "confirm", "Any payment-class action."),

    # ── Credential entry ─────────────────────────────────────────────────── #
    PolicyMatrixRow("type",  "credential", "hard_to_reverse", "secret", 0.82, True,
                    "confirm", "Typing a credential — approval required."),
    PolicyMatrixRow("paste", "credential", "hard_to_reverse", "secret", 0.82, True,
                    "confirm", "Pasting a credential — approval required."),
    PolicyMatrixRow("*",     "credential", "*",             "secret", 0.80, True,
                    "confirm", "Any credential-class action."),

    # ── External messaging ───────────────────────────────────────────────── #
    PolicyMatrixRow("click", "message_send", "irreversible", "internal", 0.88, True,
                    "confirm", "Sending an external message — approval required."),
    PolicyMatrixRow("submit","message_send", "irreversible", "internal", 0.88, True,
                    "confirm", "Submitting an external message."),
    PolicyMatrixRow("*",     "message_send", "irreversible", "*",       0.85, True,
                    "confirm", "Any external message dispatch."),

    # ── File / data deletion ─────────────────────────────────────────────── #
    PolicyMatrixRow("delete","file",         "irreversible", "*",       0.95, True,
                    "abort",   "File deletion — almost always irreversible."),
    PolicyMatrixRow("remove","file",         "irreversible", "*",       0.93, True,
                    "abort",   "File removal."),
    PolicyMatrixRow("*",     "file_delete",  "irreversible", "*",       0.94, True,
                    "abort",   "Any file-delete-class action."),

    # ── Trade / order placement ──────────────────────────────────────────── #
    PolicyMatrixRow("click", "trade",        "irreversible", "confidential", 0.96, True,
                    "abort",   "Trading order submit — extremely high risk."),
    PolicyMatrixRow("submit","trade",        "irreversible", "confidential", 0.96, True,
                    "abort",   "Trade submission."),
    PolicyMatrixRow("*",     "trade",        "*",             "*",       0.90, True,
                    "confirm", "Any trade-class action."),

    # ── Permission grants ────────────────────────────────────────────────── #
    PolicyMatrixRow("click", "permission",   "hard_to_reverse", "internal", 0.78, True,
                    "confirm", "Permission grant click."),
    PolicyMatrixRow("*",     "permission",   "*",             "*",       0.75, True,
                    "confirm", "Any permission-class action."),

    # ── Package installation ─────────────────────────────────────────────── #
    PolicyMatrixRow("*",     "package_install", "hard_to_reverse", "*",  0.72, True,
                    "confirm", "Package install — can introduce supply chain risk."),

    # ── System configuration ─────────────────────────────────────────────── #
    PolicyMatrixRow("*",     "system_config",  "hard_to_reverse", "*",  0.85, True,
                    "confirm", "System config change."),

    # ── Safe UI navigation (low risk) ────────────────────────────────────── #
    PolicyMatrixRow("scroll","ui",           "reversible",    "public", 0.02, False,
                    "execute", "Scroll — trivially reversible."),
    PolicyMatrixRow("hover", "ui",           "reversible",    "public", 0.01, False,
                    "execute", "Hover — no state change."),
    PolicyMatrixRow("focus", "ui",           "reversible",    "public", 0.03, False,
                    "execute", "Focus change."),
    PolicyMatrixRow("click", "ui",           "reversible",    "public", 0.12, False,
                    "execute", "Reversible UI click."),
    PolicyMatrixRow("hotkey","ui",           "reversible",    "public", 0.08, False,
                    "execute", "Reversible hotkey."),

    # ── Hard-to-reverse UI writes ─────────────────────────────────────────── #
    PolicyMatrixRow("type",  "ui",           "hard_to_reverse", "internal", 0.28, False,
                    "reflect", "Typing into a field — moderately risky."),
    PolicyMatrixRow("*",     "ui",           "hard_to_reverse", "internal", 0.30, False,
                    "reflect", "Hard-to-reverse UI action."),

    # ── Code execution ───────────────────────────────────────────────────── #
    PolicyMatrixRow("*",     "code_sandbox", "hard_to_reverse", "*",   0.40, False,
                    "reflect", "Sandboxed code — audit before use."),
    PolicyMatrixRow("*",     "code_host",    "hard_to_reverse", "*",   0.70, True,
                    "confirm", "Host code execution — approval recommended."),

    # ── Fallback ─────────────────────────────────────────────────────────── #
    PolicyMatrixRow("*",     "*",            "reversible",    "*",       0.10, False,
                    "execute", "Default reversible action."),
    PolicyMatrixRow("*",     "*",            "hard_to_reverse", "*",    0.35, False,
                    "reflect", "Default hard-to-reverse action."),
    PolicyMatrixRow("*",     "*",            "irreversible",  "*",       0.65, True,
                    "confirm", "Default irreversible action."),
    PolicyMatrixRow("*",     "*",            "*",             "*",       0.20, False,
                    "execute", "Catch-all default."),
)


# ─────────────────────────────────────────────────────────────────────────── #
# Data structures                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(slots=True)
class RiskAssessment:
    """Structured risk assessment for a single proposed action."""

    action_type: str
    target_class: str
    reversibility: str
    data_sensitivity: str
    base_risk: float
    adjusted_risk: float
    approval_required: bool
    diagnostician_recommendation: str  # execute | reflect | reroute | confirm | abort
    matched_policy_row: str            # notes field of the matched row
    context_adjustments: dict[str, float] = field(default_factory=dict)
    calibration_delta: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_level(self) -> str:
        if self.adjusted_risk >= ABORT_THRESHOLD:
            return "critical"
        if self.adjusted_risk >= APPROVAL_THRESHOLD:
            return "high"
        if self.adjusted_risk >= REFLECT_THRESHOLD:
            return "medium"
        if self.adjusted_risk >= REROUTE_THRESHOLD:
            return "low"
        return "negligible"

    def asdict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_class": self.target_class,
            "reversibility": self.reversibility,
            "data_sensitivity": self.data_sensitivity,
            "base_risk": round(self.base_risk, 3),
            "adjusted_risk": round(self.adjusted_risk, 3),
            "risk_level": self.risk_level,
            "approval_required": self.approval_required,
            "diagnostician_recommendation": self.diagnostician_recommendation,
            "matched_policy_row": self.matched_policy_row,
            "context_adjustments": {
                key: round(float(v), 3)
                for key, v in self.context_adjustments.items()
            },
            "calibration_delta": round(self.calibration_delta, 4),
            "evidence": dict(self.evidence),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Target class classifier                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

import re as _re  # noqa: E402 — intentional inline import

_PAYMENT_TERMS = _re.compile(
    r"\b(pay|payment|checkout|purchase|cart|billing|charge|invoice|stripe|"
    r"paypal|venmo|wire\.?transfer)\b",
    _re.I,
)
_CREDENTIAL_TERMS = _re.compile(
    r"\b(password|passwd|secret|api.?key|token|credential|auth|sign.?in|"
    r"oauth|bearer|session.?key)\b",
    _re.I,
)
_MESSAGE_TERMS = _re.compile(
    r"\b(send.?(?:email|message|sms|tweet|post)|reply|forward|broadcast)\b",
    _re.I,
)
_FILE_DELETE_TERMS = _re.compile(
    r"\b(delete|remove|trash|erase|format|wipe|shred|purge|rm(?:\s|$)|"
    r"-rf|truncate)\b",
    _re.I,
)
_TRADE_TERMS = _re.compile(
    r"\b(place.?order|buy.?stock|sell.?stock|limit.?order|market.?order|"
    r"stop.?loss|open.?position|trade.?(?:stock|crypto|option))\b",
    _re.I,
)
_PERMISSION_TERMS = _re.compile(
    r"\b(grant.?(?:access|permission|admin)|enable.?(?:admin|root)|"
    r"modify.?(?:acl|permission|rights))\b",
    _re.I,
)
_PACKAGE_TERMS = _re.compile(
    r"\b(pip.?install|npm.?install|apt.?install|brew.?install|"
    r"conda.?install|gem.?install|cargo.?install)\b",
    _re.I,
)
_SYSTEM_TERMS = _re.compile(
    r"\b(registry|reg.?(?:add|delete)|firewall|hosts.?file|bcdedit|"
    r"group.?policy|schtasks|netsh|sc.?(?:config|create))\b",
    _re.I,
)


def classify_target(action: UiAction) -> tuple[str, str]:
    """Return (target_class, data_sensitivity) from an action.

    target_class → one of the policy matrix rows' target_class values.
    data_sensitivity → "public" | "internal" | "confidential" | "secret"
    """
    combined = " ".join(
        filter(None, [action.action_type, str(action.selector or ""),
                      str(action.value or "")])
    ).lower()

    if _PAYMENT_TERMS.search(combined):
        return "payment", "confidential"
    if _CREDENTIAL_TERMS.search(combined):
        return "credential", "secret"
    if _FILE_DELETE_TERMS.search(combined):
        return "file_delete", "internal"
    if _TRADE_TERMS.search(combined):
        return "trade", "confidential"
    if _MESSAGE_TERMS.search(combined):
        return "message_send", "internal"
    if _PERMISSION_TERMS.search(combined):
        return "permission", "internal"
    if _PACKAGE_TERMS.search(combined):
        return "package_install", "internal"
    if _SYSTEM_TERMS.search(combined):
        return "system_config", "internal"

    # Code execution
    if action.action_type.lower() == "tool" or "sandbox" in combined:
        host = action.metadata.get("host_execution", False) if action.metadata else False
        return ("code_host" if host else "code_sandbox"), "internal"

    return "ui", "public"


def classify_reversibility(action: UiAction) -> str:
    """Classify an action's reversibility."""
    atype = action.action_type.lower()
    meta = action.metadata or {}
    explicit = meta.get("reversibility", "")
    if explicit in ("reversible", "hard_to_reverse", "irreversible"):
        return explicit
    if atype in ("scroll", "hover", "focus", "snapshot", "explore", "observe"):
        return "reversible"
    if atype in ("type", "hotkey", "click", "draw_path", "tool"):
        return "hard_to_reverse"
    if atype in ("delete", "remove", "submit", "trade", "purchase", "send"):
        return "irreversible"
    return "hard_to_reverse"


# ─────────────────────────────────────────────────────────────────────────── #
# Policy matrix lookup                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _row_matches(row: PolicyMatrixRow, action_type: str, target: str,
                 reversibility: str, sensitivity: str) -> bool:
    def _m(pattern: str, value: str) -> bool:
        return pattern == "*" or value.startswith(pattern)

    return (
        _m(row.action_type, action_type)
        and _m(row.target_class, target)
        and _m(row.reversibility, reversibility)
        and _m(row.data_sensitivity, sensitivity)
    )


def _lookup_policy(
    action_type: str,
    target_class: str,
    reversibility: str,
    data_sensitivity: str,
) -> PolicyMatrixRow:
    for row in _POLICY_MATRIX:
        if _row_matches(row, action_type, target_class, reversibility, data_sensitivity):
            return row
    # Should never happen — the catch-all row guarantees a match.
    return _POLICY_MATRIX[-1]


# ─────────────────────────────────────────────────────────────────────────── #
# Context adjustments                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _context_adjustments(
    action: UiAction,
    proposal: ActionProposal | None,
    observation: ObservationFrame | None,
) -> dict[str, float]:
    """Compute additive risk adjustments from execution context."""
    adjustments: dict[str, float] = {}

    # Previous failures on the same selector → risk creep
    if proposal is not None:
        if proposal.risk_score > 0.5:
            adjustments["high_base_risk_penalty"] = 0.05

    # Active modal → increased risk (unexpected state)
    if observation is not None and observation.active_modal:
        adjustments["active_modal_penalty"] = 0.08

    # Low-confidence observation → increased uncertainty risk
    if observation is not None:
        fingerprint = observation.stable_fingerprint
        if not fingerprint or fingerprint == "empty":
            adjustments["low_observation_confidence"] = 0.12

    # No discovered API surfaces → native route risk bump
    if observation is not None and not observation.discovered_surfaces:
        if proposal is not None and proposal.route == "native_vision":
            adjustments["native_without_api_surfaces"] = 0.06

    # Approval already present → reduce risk
    meta = action.metadata or {}
    if meta.get("approval_token") or meta.get("approved"):
        adjustments["approval_present"] = -0.15

    return adjustments


# ─────────────────────────────────────────────────────────────────────────── #
# Calibration hook                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class RiskCalibration:
    """Historical calibration layer for risk thresholds.

    Starts from zero-delta (deterministic policy matrix only) and can be
    updated with live-fire outcome data. The delta is bounded to [-0.15, 0.15]
    to prevent runaway calibration.
    """

    def __init__(self) -> None:
        self._deltas: dict[str, float] = {}  # keyed by target_class

    def record_outcome(
        self,
        target_class: str,
        predicted_risk: float,
        actual_incident: bool,
    ) -> None:
        """Update calibration delta from a live-fire outcome.

        If the agent predicted low risk (< 0.5) but an incident occurred,
        increase the delta for this target class. If predicted high risk
        but no incident, decrease it slightly.
        """
        current = self._deltas.get(target_class, 0.0)
        if actual_incident and predicted_risk < 0.5:
            current = min(current + 0.05, 0.15)
        elif not actual_incident and predicted_risk > 0.7:
            current = max(current - 0.02, -0.15)
        self._deltas[target_class] = current

    def delta_for(self, target_class: str) -> float:
        return self._deltas.get(target_class, 0.0)


_GLOBAL_CALIBRATION = RiskCalibration()


# ─────────────────────────────────────────────────────────────────────────── #
# Risk Guardian                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class RiskGuardian:
    """Score proposed actions using the policy matrix and context adjustments.

    This is the deterministic risk backbone described in CORA and the plan's
    Phase 4. It runs before frontier model calls so the model cannot override
    hard safety boundaries.
    """

    def __init__(self, calibration: RiskCalibration | None = None) -> None:
        self.calibration = calibration or _GLOBAL_CALIBRATION

    def assess(
        self,
        action: UiAction,
        *,
        proposal: ActionProposal | None = None,
        observation: ObservationFrame | None = None,
        objective: str = "",
    ) -> RiskAssessment:
        """Score the action and return a full risk assessment."""
        action_type = action.action_type.lower()
        target_class, data_sensitivity = classify_target(action)
        reversibility = classify_reversibility(action)

        row = _lookup_policy(action_type, target_class, reversibility, data_sensitivity)
        context = _context_adjustments(action, proposal, observation)
        calibration_delta = self.calibration.delta_for(target_class)

        raw_adjusted = row.base_risk + sum(context.values()) + calibration_delta
        adjusted_risk = max(0.0, min(1.0, raw_adjusted))

        # Derive diagnostician recommendation from adjusted score.
        diagnostician = _score_to_diagnostician(
            adjusted_risk,
            row.approval_required or (proposal.required_approval if proposal else False),
            row.diagnostician_floor,
        )

        return RiskAssessment(
            action_type=action_type,
            target_class=target_class,
            reversibility=reversibility,
            data_sensitivity=data_sensitivity,
            base_risk=row.base_risk,
            adjusted_risk=adjusted_risk,
            approval_required=(
                row.approval_required
                or adjusted_risk >= APPROVAL_THRESHOLD
            ),
            diagnostician_recommendation=diagnostician,
            matched_policy_row=row.notes,
            context_adjustments=context,
            calibration_delta=calibration_delta,
            evidence={
                "action_type": action_type,
                "selector": str(action.selector or "")[:120],
                "objective": objective[:200],
            },
        )

    def is_safe(
        self,
        action: UiAction,
        *,
        proposal: ActionProposal | None = None,
        observation: ObservationFrame | None = None,
        objective: str = "",
        abort_threshold: float = ABORT_THRESHOLD,
    ) -> bool:
        """Quick predicate: True when the action is below the abort threshold."""
        assessment = self.assess(
            action, proposal=proposal, observation=observation, objective=objective
        )
        return assessment.adjusted_risk < abort_threshold

    def record_outcome(
        self,
        action: UiAction,
        predicted_risk: float,
        actual_incident: bool,
    ) -> None:
        """Delegate outcome recording to the calibration layer."""
        _, target_class = classify_target(action)[0], classify_target(action)[0]
        self.calibration.record_outcome(target_class, predicted_risk, actual_incident)


def _score_to_diagnostician(
    score: float,
    approval_required: bool,
    floor: str,
) -> str:
    """Map a risk score (plus floor) to a diagnostician decision."""
    if score >= ABORT_THRESHOLD:
        return "abort"
    if score >= APPROVAL_THRESHOLD or approval_required:
        # Take the more severe of floor and "confirm".
        return _max_severity(floor, "confirm")
    if score >= REFLECT_THRESHOLD:
        return _max_severity(floor, "reflect")
    if score >= REROUTE_THRESHOLD:
        return _max_severity(floor, "reroute")
    return _max_severity(floor, "execute")


_SEVERITY = {
    "abort": 5,
    "confirm": 4,
    "reroute": 3,
    "reflect": 2,
    "repair": 1,
    "execute": 0,
}


def _max_severity(a: str, b: str) -> str:
    if _SEVERITY.get(a, 0) >= _SEVERITY.get(b, 0):
        return a
    return b


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level singleton                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

_GLOBAL_GUARDIAN = RiskGuardian()


def assess_action_risk(
    action: UiAction,
    *,
    proposal: ActionProposal | None = None,
    observation: ObservationFrame | None = None,
    objective: str = "",
) -> RiskAssessment:
    """Assess risk for a single action using the global guardian."""
    return _GLOBAL_GUARDIAN.assess(
        action, proposal=proposal, observation=observation, objective=objective
    )
