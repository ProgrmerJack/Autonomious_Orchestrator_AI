"""Intent constraint compiler and pre-action logic verifier.

Phase 4 implementation — translates a user objective into a set of typed
constraints, then checks proposed actions against those constraints *before*
execution to prevent intent/action mismatches.

Design notes
────────────
* Fully deterministic — no frontier model required. The intent layer is the
  safety backbone that runs even when models are offline.
* IntentConstraint objects are the "locked" form of the user goal. They are
  compiled once at task start and threaded through every verifier call.
* The diagnostician output follows the plan schema:
    confirm | reflect | repair | reroute | abort | execute
* Risk categories match the non-negotiable boundaries from the research plan:
    payment, credential, external_message, file_delete, trade, permission,
    private_data_transfer, package_install, system_config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.workflow.intent_parser import (
    parse_structured_intent,
)

from .control_substrate import ActionProposal, GoalLock


# ─────────────────────────────────────────────────────────────────────────── #
# Pattern registry                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

_RE = re.compile  # alias for brevity

_PAYMENT_RE = _RE(
    r"\b(pay|payment|checkout|purchase|buy(?!.?(?:time|in|out))|price|cart|"
    r"billing|charge|invoice|stripe|paypal|venmo|zelle|wire.?transfer)\b",
    re.I,
)
_CREDENTIAL_RE = _RE(
    r"\b(password|passwd|secret|token|api.?key|credential|auth|"
    r"sign.?in|log.?in|oauth|bearer|session.?key)\b",
    re.I,
)
_MESSAGING_RE = _RE(
    r"\b(send.?(?:email|message|sms|text|notification|alert|tweet|post)|"
    r"reply|forward|broadcast|publish.?(?:post|message))\b",
    re.I,
)
_DELETE_RE = _RE(
    r"\b(delete|remove|trash|erase|format|wipe|drop.?(?:table|database)|"
    r"truncate|rm(?:\s|$)|-rf|shred|destroy|purge)\b",
    re.I,
)
_TRADE_RE = _RE(
    r"\b((?:place|execute|submit).?order|buy.?stock|sell.?stock|"
    r"limit.?order|market.?order|stop.?loss|open.?position|close.?position|"
    r"trade.?(?:stock|option|future|crypto))\b",
    re.I,
)
_PERMISSION_RE = _RE(
    r"\b(grant.?(?:access|permission|admin)|enable.?(?:admin|root|sudo)|"
    r"install.?(?:extension|plugin|package|software)|"
    r"uninstall|modify.?(?:registry|hosts|sudoers))\b",
    re.I,
)
_PRIVATE_DATA_RE = _RE(
    r"\b(upload.?(?:private|personal|confidential|pii|phi|ssn|dob)|"
    r"transfer.?(?:file|data).?(?:to.?external|outside|third.?party)|"
    r"share.?(?:credential|password|token|private))\b",
    re.I,
)
_PACKAGE_INSTALL_RE = _RE(
    r"\b(pip.?install|npm.?install|apt.?install|brew.?install|"
    r"conda.?install|gem.?install|cargo.?install)\b",
    re.I,
)
_SYSTEM_CONFIG_RE = _RE(
    r"\b(modify.?(?:registry|firewall|hosts.?file|group.?policy)|"
    r"net.?sh|Set-MpPreference|bcdedit|reg.?(?:add|delete|modify)|"
    r"sc.?(?:config|create|delete)|schtasks)\b",
    re.I,
)

# Actions that are definitely NOT writing data externally
_SAFE_READ_TYPES = frozenset(
    {"snapshot", "explore", "scroll", "observe", "focus", "hover"},
)
# Actions that involve data transfer risk
_WRITE_ACTION_TYPES = frozenset(
    {"type", "set_value", "fill", "paste", "click_submit", "submit"},
)
# Actions that navigate to external destinations
_NAVIGATE_TYPES = (frozenset({"open_url", "navigate", "goto", "launch"}),)


# ─────────────────────────────────────────────────────────────────────────── #
# Data structures                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

DIAGNOSTICIAN_DECISIONS = frozenset(
    {"execute", "confirm", "reflect", "repair", "reroute", "abort"},
)

# Severity order: abort is most severe, execute is green light.
_SEVERITY = {
    "abort": 5,
    "confirm": 4,
    "reroute": 3,
    "reflect": 2,
    "repair": 1,
    "execute": 0,
}


@dataclass(slots=True)
class IntentConstraint:
    """A single compiled constraint derived from a user objective."""

    category: str
    description: str
    violated_by: list[str] = field(default_factory=list)
    # Diagnostician decision emitted when this constraint is violated.
    on_violation: str = "reflect"
    # Whether violation always blocks execution (True) or just raises caution.
    hard_block: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def check(self, action: UiAction, objective: str) -> bool:
        """Return True when the constraint is satisfied (no violation)."""
        raise NotImplementedError  # subclasses implement domain checks

    def asdict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "on_violation": self.on_violation,
            "hard_block": self.hard_block,
            "violated_by": list(self.violated_by),
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True)
class PatternConstraint(IntentConstraint):
    """Block actions whose selector/value match a sensitive pattern."""

    pattern: re.Pattern | None = field(default=None)
    match_value: bool = True
    match_selector: bool = False

    def check(self, action: UiAction, objective: str) -> bool:
        if self.pattern is None:
            return True
        if action.action_type.lower() in _SAFE_READ_TYPES:
            return True
        targets: list[str] = []
        if self.match_value and action.value is not None:
            targets.append(str(action.value))
        if self.match_selector:
            targets.append(str(action.selector or ""))
        combined = " ".join(targets)
        return not bool(self.pattern.search(combined))


@dataclass(slots=True)
class ObjectiveConstraint(IntentConstraint):
    """Verify an action is consistent with the stated objective."""

    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)

    def check(self, action: UiAction, objective: str) -> bool:
        if action.action_type.lower() in _SAFE_READ_TYPES:
            return True
        lower_action = f"{action.action_type} {action.selector} {action.value}".lower()
        for term in self.forbidden_terms:
            if term.lower() in lower_action:
                return False
        return True


@dataclass(slots=True)
class DiagnosticianDecision:
    """Structured output of the intent verifier."""

    allowed: bool
    diagnostician: str  # execute | confirm | reflect | repair | reroute | abort
    reason: str
    violated_constraints: list[dict[str, Any]] = field(default_factory=list)
    repair_hint: str = ""
    reroute_hint: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "diagnostician": self.diagnostician,
            "reason": self.reason,
            "violated_constraints": list(self.violated_constraints),
            "repair_hint": self.repair_hint,
            "reroute_hint": self.reroute_hint,
            "evidence": dict(self.evidence),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Constraint compiler                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


class IntentConstraintCompiler:
    """Compile a user objective string into a list of IntentConstraints.

    The constraints are purely structural — no frontier model required. The
    compiler infers risk categories from objective keywords and produces a
    constraint set that is checked against every proposed action.
    """

    def compile(self, objective: str) -> list[IntentConstraint]:
        """Compile the objective into constraints."""
        lower = objective.lower()
        parsed_intent = parse_structured_intent(objective)
        constraints: list[IntentConstraint] = []

        # Always active: visual prompt injection guard
        constraints.append(_VisualPromptInjectionGuard())

        # Category-specific constraints activated by objective keywords
        invoice_only_file_task = (
            parsed_intent.is_file_workflow()
            and "invoice" in lower
            and not any(
                token in lower
                for token in (
                    "pay",
                    "payment",
                    "checkout",
                    "purchase",
                    "billing",
                    "charge",
                    "stripe",
                    "paypal",
                    "venmo",
                    "zelle",
                    "wire",
                )
            )
        )
        payment_in_scope = (
            bool(_PAYMENT_RE.search(lower)) and not invoice_only_file_task
        )
        if invoice_only_file_task:
            pass
        elif not payment_in_scope:
            # Payment actions are only allowed if the objective explicitly
            # mentions payment. Otherwise block any payment-like actions.
            constraints.append(
                PatternConstraint(
                    category="payment_guard",
                    description=(
                        "Payment-related actions are blocked unless the"
                        " objective explicitly requests a payment operation."
                    ),
                    on_violation="abort",
                    hard_block=True,
                    pattern=_PAYMENT_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["payment", "purchase", "checkout"],
                )
            )
        else:
            # Payment IS in scope — still require confirmation.
            constraints.append(
                _ApprovalRequiredConstraint(
                    category="payment_confirmation",
                    description="Payment operations always require explicit confirmation.",
                    pattern=_PAYMENT_RE,
                    on_violation="confirm",
                    hard_block=True,
                )
            )

        if not _CREDENTIAL_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="credential_guard",
                    description=(
                        "Credential entry is blocked unless the objective"
                        " explicitly requires authentication."
                    ),
                    on_violation="abort",
                    hard_block=True,
                    pattern=_CREDENTIAL_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["password", "api_key", "token"],
                )
            )

        if not _MESSAGING_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="messaging_guard",
                    description=(
                        "Sending external messages is blocked unless the"
                        " objective explicitly requests it."
                    ),
                    on_violation="confirm",
                    hard_block=True,
                    pattern=_MESSAGING_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["send", "message", "email"],
                )
            )

        if not _DELETE_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="delete_guard",
                    description=(
                        "File or data deletion requires the objective to"
                        " explicitly include a delete/remove intent."
                    ),
                    on_violation="abort",
                    hard_block=True,
                    pattern=_DELETE_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["delete", "remove", "erase"],
                )
            )

        if not _TRADE_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="trade_guard",
                    description=(
                        "Trade or order placement requires an explicit trading"
                        " objective."
                    ),
                    on_violation="abort",
                    hard_block=True,
                    pattern=_TRADE_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["trade", "order", "position"],
                )
            )

        if not _PERMISSION_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="permission_guard",
                    description=(
                        "Permission grants or admin access require an explicit"
                        " objective."
                    ),
                    on_violation="confirm",
                    hard_block=True,
                    pattern=_PERMISSION_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["grant", "admin", "install"],
                )
            )

        if not _PACKAGE_INSTALL_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="package_install_guard",
                    description=(
                        "Package installation is blocked unless the objective"
                        " explicitly requests it."
                    ),
                    on_violation="confirm",
                    hard_block=True,
                    pattern=_PACKAGE_INSTALL_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["pip install", "npm install"],
                )
            )

        if not _SYSTEM_CONFIG_RE.search(lower):
            constraints.append(
                PatternConstraint(
                    category="system_config_guard",
                    description=(
                        "System configuration changes require an explicit"
                        " admin objective."
                    ),
                    on_violation="abort",
                    hard_block=True,
                    pattern=_SYSTEM_CONFIG_RE,
                    match_value=True,
                    match_selector=True,
                    violated_by=["registry", "firewall", "hosts"],
                )
            )

        if _PRIVATE_DATA_RE.search(lower):
            constraints.append(
                _ApprovalRequiredConstraint(
                    category="private_data_transfer_confirmation",
                    description=(
                        "Transferring private or personal data requires"
                        " explicit approval."
                    ),
                    pattern=_PRIVATE_DATA_RE,
                    on_violation="confirm",
                    hard_block=True,
                )
            )

        return constraints


# ─────────────────────────────────────────────────────────────────────────── #
# Specialised constraint implementations                                       #
# ─────────────────────────────────────────────────────────────────────────── #


@dataclass(slots=True)
class _VisualPromptInjectionGuard(IntentConstraint):
    """Detect injected instructions embedded in action values or selectors."""

    category: str = "visual_prompt_injection"
    description: str = (
        "Screen text claiming to override the agent's goal is treated as"
        " untrusted input and blocked."
    )
    on_violation: str = "abort"
    hard_block: bool = True

    # Common injection phrases (case-insensitive).
    _INJECTION_RE: re.Pattern = field(
        default=_RE(
            r"(ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions|"
            r"new\s+instructions?:\s|disregard\s+(?:your\s+)?(?:task|goal|instructions?)|"
            r"override\s+(?:your\s+)?(?:safety|instructions?|policy)|"
            r"your\s+new\s+(?:task|goal|objective|instruction)\s+is|"
            r"from\s+now\s+on\s+(?:you\s+(?:are|must|will|should))|"
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|uncensored))",
            re.I,
        ),
        init=False,
        repr=False,
    )

    def check(self, action: UiAction, objective: str) -> bool:
        haystack = " ".join(
            filter(
                None,
                [
                    str(action.value or ""),
                    str(action.selector or ""),
                    json_safe_metadata(action.metadata),
                ],
            )
        )
        return not bool(self._INJECTION_RE.search(haystack))

    def asdict(self) -> dict[str, Any]:
        return super().asdict()


@dataclass(slots=True)
class _ApprovalRequiredConstraint(IntentConstraint):
    """Require an approval token when the action matches a sensitive pattern."""

    category: str = "approval_required"
    description: str = "Action requires explicit approval."
    on_violation: str = "confirm"
    hard_block: bool = True
    pattern: re.Pattern | None = field(default=None)

    def check(self, action: UiAction, objective: str) -> bool:
        if self.pattern is None:
            return True
        if action.action_type.lower() in _SAFE_READ_TYPES:
            return True
        approval = action.metadata.get("approval_token") or action.metadata.get(
            "approved"
        )
        if approval:
            return True
        combined = f"{action.value or ''} {action.selector or ''}"
        return not bool(self.pattern.search(combined))

    def asdict(self) -> dict[str, Any]:
        return super().asdict()


# ─────────────────────────────────────────────────────────────────────────── #
# Logic verifier                                                               #
# ─────────────────────────────────────────────────────────────────────────── #


class IntentVerifier:
    """Check proposed actions against compiled intent constraints.

    Usage::

        compiler = IntentConstraintCompiler()
        verifier = IntentVerifier()

        constraints = compiler.compile("Open Chrome and search for quarterly results")
        decision = verifier.verify(action, objective, constraints)
        if not decision.allowed:
            # Handle via diagnostician decision
    """

    def verify(
        self,
        action: UiAction,
        objective: str,
        constraints: list[IntentConstraint],
        proposal: ActionProposal | None = None,
        goal_lock: GoalLock | None = None,
    ) -> DiagnosticianDecision:
        """Run all compiled constraints and return a diagnostician decision."""
        violated: list[IntentConstraint] = []
        for constraint in constraints:
            try:
                satisfied = constraint.check(action, objective)
            except Exception:  # noqa: BLE001 — never let a constraint crash the verifier
                satisfied = True
            if not satisfied:
                violated.append(constraint)

        if not violated:
            return DiagnosticianDecision(
                allowed=True,
                diagnostician="execute",
                reason="All intent constraints satisfied.",
                evidence={
                    "objective": objective,
                    "action_type": action.action_type,
                    "proposal_id": proposal.proposal_id if proposal else "",
                },
            )

        # Pick the most severe diagnostician decision across violations.
        worst = max(violated, key=lambda c: _SEVERITY.get(c.on_violation, 0))
        repair_hints = _build_repair_hints(worst, action)
        reroute_hint = _build_reroute_hint(worst, action)

        return DiagnosticianDecision(
            allowed=False,
            diagnostician=worst.on_violation,
            reason=f"[{worst.category}] {worst.description}",
            violated_constraints=[c.asdict() for c in violated],
            repair_hint=repair_hints,
            reroute_hint=reroute_hint,
            evidence={
                "objective": objective,
                "action_type": action.action_type,
                "selector": action.selector,
                "goal_lock": goal_lock.asdict() if goal_lock else {},
                "proposal_id": proposal.proposal_id if proposal else "",
            },
        )

    def verify_plan(
        self,
        actions: list[UiAction],
        objective: str,
        constraints: list[IntentConstraint],
        goal_lock: GoalLock | None = None,
    ) -> list[DiagnosticianDecision]:
        """Verify a full action sequence and return one decision per action."""
        return [
            self.verify(action, objective, constraints, goal_lock=goal_lock)
            for action in actions
        ]


# ─────────────────────────────────────────────────────────────────────────── #
# Convenience helpers                                                           #
# ─────────────────────────────────────────────────────────────────────────── #


def json_safe_metadata(metadata: dict | None) -> str:
    if not metadata:
        return ""
    try:
        import json

        return json.dumps(metadata, default=str)[:512]
    except Exception:  # noqa: BLE001
        return str(metadata)[:512]


def _build_repair_hints(
    constraint: IntentConstraint,
    action: UiAction,
) -> str:
    category = constraint.category
    if category in ("payment_guard", "payment_confirmation"):
        return (
            "Confirm with the user that a payment should be made and supply"
            " an approval token before retrying."
        )
    if category == "credential_guard":
        return (
            "Use a secure credential manager or ask the user to provide the"
            " credential through a safe channel rather than embedding it in"
            " the action value."
        )
    if category == "messaging_guard":
        return (
            "Draft the message for user review before sending. Present the"
            " recipient and content and request explicit approval."
        )
    if category in ("delete_guard",):
        return (
            "Move the item to a temporary location first (trash/archive) and"
            " confirm with the user before permanent deletion."
        )
    if category == "trade_guard":
        return (
            "Present the proposed trade (symbol, direction, size, price) to"
            " the user and require explicit confirmation."
        )
    if category in ("permission_guard", "package_install_guard"):
        return (
            "List the specific permissions or packages to be installed and"
            " request user approval before proceeding."
        )
    if category == "system_config_guard":
        return (
            "Show the exact registry/config change to the user and require"
            " approval. Consider a dry-run preview first."
        )
    if category == "visual_prompt_injection":
        return (
            "The action contains text that looks like an injected instruction."
            " Sanitize on-screen text before trusting it as an argument."
        )
    return (
        "Review the action against the user's objective and request approval if unsure."
    )


def _build_reroute_hint(
    constraint: IntentConstraint,
    action: UiAction,
) -> str:
    category = constraint.category
    if category in (
        "delete_guard",
        "system_config_guard",
        "trade_guard",
    ):
        return (
            "Consider using the code/tool lane to perform a dry-run preview"
            " and generate an approval artifact before committing."
        )
    if category in ("package_install_guard",):
        return (
            "Use the tool executor lane to install in a sandboxed environment"
            " and validate before applying to the host."
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level singleton helpers                                               #
# ─────────────────────────────────────────────────────────────────────────── #

_GLOBAL_COMPILER = IntentConstraintCompiler()
_GLOBAL_VERIFIER = IntentVerifier()


def compile_intent(objective: str) -> list[IntentConstraint]:
    """Compile an objective string into intent constraints (module-level API)."""
    return _GLOBAL_COMPILER.compile(objective)


def verify_action_intent(
    action: UiAction,
    objective: str,
    constraints: list[IntentConstraint] | None = None,
    *,
    proposal: ActionProposal | None = None,
    goal_lock: GoalLock | None = None,
) -> DiagnosticianDecision:
    """Verify a single action against compiled constraints.

    If ``constraints`` is None, they are compiled from ``objective``.
    """
    if constraints is None:
        constraints = compile_intent(objective)
    return _GLOBAL_VERIFIER.verify(
        action,
        objective,
        constraints,
        proposal=proposal,
        goal_lock=goal_lock,
    )
