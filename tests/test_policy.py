from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.core.approvals import ApprovalStore
from agentos_orchestrator.core.authorization import AuthorizationMiddleware
from agentos_orchestrator.core.policy import PermissionPolicy
from agentos_orchestrator.core.trust import TrustMonitor
from agentos_orchestrator.core.types import ActionRequest


class PermissionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PermissionPolicy(
            {
                "default": "deny",
                "allow": {
                    "actions": ["mcp.call", "file.write", "network.fetch"],
                    "paths": ["runs/**"],
                    "network_hosts": ["example.com"],
                },
                "forbid": {
                    "actions": ["host.admin"],
                    "paths": ["C:/Users/*/.ssh/**"],
                    "keywords": ["password"],
                },
                "require_approval": {"actions": ["file.write"]},
            }
        )

    def test_allows_mcp_call(self) -> None:
        decision = self.policy.evaluate(
            ActionRequest("agent", "mcp.call", "mcp://research/search")
        )
        self.assertTrue(decision.allowed)

    def test_denies_forbidden_action(self) -> None:
        decision = self.policy.evaluate(
            ActionRequest("agent", "host.admin", "host://uac")
        )
        self.assertFalse(decision.allowed)

    def test_requires_approval_before_file_write(self) -> None:
        decision = self.policy.evaluate(
            ActionRequest("agent", "file.write", "runs/output.json")
        )
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.requires_approval)

    def test_allows_network_allowlist(self) -> None:
        decision = self.policy.evaluate(
            ActionRequest("agent", "network.fetch", "https://example.com/data")
        )
        self.assertTrue(decision.allowed)

    def test_approval_token_is_bound_to_exact_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.sqlite3"
            policy = PermissionPolicy(
                {
                    "default": "deny",
                    "allow": {"actions": ["os.act"], "paths": []},
                    "forbid": {"actions": [], "paths": []},
                    "require_approval": {"actions": ["os.act"]},
                }
            )
            approvals = ApprovalStore(db_path)
            middleware = AuthorizationMiddleware(
                policy,
                approvals,
                TrustMonitor(db_path),
            )
            first = ActionRequest("agent", "os.act", "windows-uia://name=A")
            requested = middleware.authorize("run_1", first)
            self.assertTrue(requested.requires_approval)
            assert requested.approval is not None
            approvals.approve(requested.approval.token)

            second = ActionRequest(
                "agent",
                "os.act",
                "windows-uia://name=B",
                approval_token=requested.approval.token,
            )
            rejected = middleware.authorize("run_1", second)

            self.assertFalse(rejected.allowed)
            self.assertIn(
                "approval token is invalid for this action",
                rejected.reasons,
            )

    def test_sandbox_exec_is_not_trust_escalated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.sqlite3"
            trust = TrustMonitor(db_path)

            decision = trust.assess(
                "run_sandbox_1",
                ActionRequest(
                    "agent",
                    "sandbox.exec",
                    "sandbox://virtual-desktop/browser-research",
                    payload={"action": "browse"},
                ),
            )

            self.assertFalse(decision.requires_approval)
            self.assertEqual(decision.score_delta, 0)
            self.assertIn("sandbox-confined execution action", decision.reasons)


if __name__ == "__main__":
    unittest.main()
