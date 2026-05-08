from __future__ import annotations

from dataclasses import dataclass, field

from .approvals import ApprovalStore, ApprovalTicket
from .policy import PermissionDecision, PermissionPolicy
from .sandbox_security import assess_sandbox_action
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
        token_matches = False
        baseline_policy_decision: PermissionDecision | None = None
        if action.approval_token:
            token_matches = self.approvals.is_approved_for(
                action.approval_token,
                action,
            )
            baseline_policy_decision = self.policy.evaluate(
                self._without_approval_token(action)
            )
        policy_decision = self.policy.evaluate(action)
        trust_decision = self.trust.assess(run_id, action)
        reasons = [*policy_decision.reasons, *trust_decision.reasons]
        sandbox_assessment = assess_sandbox_action(action)
        if sandbox_assessment.applies:
            reasons.extend(sandbox_assessment.reasons)

        token_can_override = self._token_can_override(
            policy_decision,
            baseline_policy_decision,
            trust_decision,
        )

        if token_matches and token_can_override:
            return AuthorizationDecision(
                allowed=True,
                reasons=[*reasons, "approved by human token"],
                trust=trust_decision,
            )

        if action.approval_token and token_can_override and not token_matches:
            reasons.append("approval token is invalid for this action")

        effective_requires_approval = self._effective_requires_approval(
            policy_decision,
            baseline_policy_decision,
        )

        if policy_decision.allowed and policy_decision.requires_approval:
            approval = None
            if action.approval_token:
                if not token_matches:
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
            if effective_requires_approval and not action.approval_token:
                approval = self.approvals.request(run_id, action, reasons)
            return AuthorizationDecision(
                allowed=False,
                reasons=reasons,
                requires_approval=effective_requires_approval,
                approval=approval,
                trust=trust_decision,
            )

        if trust_decision.requires_approval:
            approval = None
            if action.approval_token:
                if not token_matches:
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

    @staticmethod
    def _without_approval_token(action: ActionRequest) -> ActionRequest:
        return ActionRequest(
            agent_id=action.agent_id,
            action_type=action.action_type,
            target=action.target,
            payload=dict(action.payload),
            approval_token=None,
        )

    @staticmethod
    def _effective_requires_approval(
        policy_decision: PermissionDecision,
        baseline_policy_decision: PermissionDecision | None,
    ) -> bool:
        return policy_decision.requires_approval or bool(
            baseline_policy_decision and baseline_policy_decision.requires_approval
        )

    def _token_can_override(
        self,
        policy_decision: PermissionDecision,
        baseline_policy_decision: PermissionDecision | None,
        trust_decision: TrustDecision,
    ) -> bool:
        effective_policy = baseline_policy_decision or policy_decision
        return effective_policy.requires_approval or (
            effective_policy.allowed and trust_decision.requires_approval
        )
