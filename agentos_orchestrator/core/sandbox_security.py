from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import ActionRequest


_REAL_ISOLATION_BOUNDARIES = frozenset(
    {
        "cua",
        "firecracker",
        "kata",
        "microvm",
        "vm",
        "vm-backed-container",
    }
)


@dataclass(slots=True)
class SandboxSecurityAssessment:
    applies: bool
    hardened: bool
    boundary: str = ""
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def assess_sandbox_action(action: ActionRequest) -> SandboxSecurityAssessment:
    target = str(action.target or "").lower()
    applies = action.action_type == "sandbox.exec" or target.startswith("sandbox://")
    if not applies:
        return SandboxSecurityAssessment(applies=False, hardened=False)

    payload = dict(action.payload or {})
    security = payload.get("sandbox_security")
    if not isinstance(security, dict):
        security = {}

    boundary = (
        str(security.get("isolation_boundary") or security.get("boundary") or "")
        .strip()
        .lower()
    )
    allowlist = security.get("network_allowlist")
    egress_controlled = bool(
        security.get("egress_controlled")
        or security.get("egress_proxy")
        or (
            isinstance(allowlist, list) and any(str(item).strip() for item in allowlist)
        )
    )
    secret_stripping = bool(
        security.get("secret_stripping")
        or security.get("secrets_stripped")
        or security.get("secret_env_stripped")
    )
    hardened = (
        boundary in _REAL_ISOLATION_BOUNDARIES
        and egress_controlled
        and secret_stripping
    )

    reasons: list[str] = []
    if hardened:
        reasons.append(
            "sandbox request includes a real isolation boundary, egress controls, and secret stripping"
        )
    else:
        if boundary not in _REAL_ISOLATION_BOUNDARIES:
            reasons.append(
                "sandbox request does not prove a real VM-backed isolation boundary"
            )
        if not egress_controlled:
            reasons.append(
                "sandbox request does not prove egress controls or a network allowlist"
            )
        if not secret_stripping:
            reasons.append(
                "sandbox request does not prove secret stripping before execution"
            )

    return SandboxSecurityAssessment(
        applies=True,
        hardened=hardened,
        boundary=boundary,
        reasons=reasons,
        details=dict(security),
    )
