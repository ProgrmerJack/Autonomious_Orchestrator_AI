from __future__ import annotations

from dataclasses import dataclass, field

from .approvals import ApprovalStore, ApprovalTicket
from .policy import PermissionDecision, PermissionPolicy
from .trust import TrustDecision, TrustMonitor
from .types import ActionRequest


@dataclass(slots=True)
class AuthorizationDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    requires_approval: bool = False
    approval: ApprovalTicket | None = None
    trust: TrustDecision | None = None


class AuthorizationMiddleware:
    """Combines static policy and behavioral trust gates."""

    def __init__(
        self,
        policy: PermissionPolicy,
        approvals: ApprovalStore,
        trust: TrustMonitor,
    ) -> None:
        self.policy = policy
        self.approvals = approvals
        self.trust = trust

    def authorize(
        self,
        run_id: str,
        action: ActionRequest,
    ) -> AuthorizationDecision:
        if not action.approval_token:
            approved = self.approvals.find_approved_for(run_id, action)
            if approved is not None:
                action.approval_token = approved.token
        policy_decision = self.policy.evaluate(action)
        trust_decision = self.trust.assess(run_id, action)
        reasons = [*policy_decision.reasons, *trust_decision.reasons]

        if self._token_overrides(action, policy_decision, trust_decision):
            return AuthorizationDecision(
                allowed=True,
                reasons=[*reasons, "approved by human token"],
                trust=trust_decision,
            )

        if policy_decision.allowed and policy_decision.requires_approval:
            approval = None
            if action.approval_token:
                reasons.append("approval token is invalid for this action")
            else:
                approval = self.approvals.request(run_id, action, reasons)
            return AuthorizationDecision(
                allowed=False,
                reasons=reasons,
                requires_approval=True,
                approval=approval,
                trust=trust_decision,
            )

        if not policy_decision.allowed:
            approval = None
            if policy_decision.requires_approval:
                approval = self.approvals.request(run_id, action, reasons)
            return AuthorizationDecision(
                allowed=False,
                reasons=reasons,
                requires_approval=policy_decision.requires_approval,
                approval=approval,
                trust=trust_decision,
            )

        if trust_decision.requires_approval:
            approval = None
            if action.approval_token:
                reasons.append("approval token is invalid for this action")
            else:
                approval = self.approvals.request(run_id, action, reasons)
            return AuthorizationDecision(
                allowed=False,
                reasons=reasons,
                requires_approval=True,
                approval=approval,
                trust=trust_decision,
            )

        return AuthorizationDecision(
            allowed=True,
            reasons=reasons,
            trust=trust_decision,
        )

    def _token_overrides(
        self,
        action: ActionRequest,
        policy_decision: PermissionDecision,
        trust_decision: TrustDecision,
    ) -> bool:
        if not action.approval_token:
            return False
        if not self.approvals.is_approved_for(
            action.approval_token,
            action,
        ):
            return False
        return policy_decision.requires_approval or (
            policy_decision.allowed and trust_decision.requires_approval
        )
