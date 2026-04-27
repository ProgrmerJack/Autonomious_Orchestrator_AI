"""Tests for GatewayCommandRouter — PC act approval, /run and /research
command routing and response formatting.

Split from test_config_gateway_core.py to keep each module narrowly scoped.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentos_orchestrator.cli import main
from agentos_orchestrator.gateway import ChannelMessage, GatewayCommandRouter

from tests.gateway_test_support import (
    FakeResearchEngine,
    base_network_hosts,
    new_orchestrator,
)

_ALLOW_POLICY = {
    "default": "deny",
    "allow": {
        "actions": [
            "file.write",
            "mcp.call",
            "mcp.list",
            "memory.commit",
            "network.fetch",
        ],
        "paths": ["runs/**", "memory://*", "mcp://*"],
        "network_hosts": base_network_hosts(),
    },
    "forbid": {"actions": [], "paths": []},
    "require_approval": {"actions": []},
}


class PcActApprovalTests(unittest.TestCase):
    def test_pc_act_requests_approval_before_backend_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {
                            "actions": ["os.act"],
                            "paths": [],
                            "network_hosts": base_network_hosts(),
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": ["os.act"]},
                    }
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--policy",
                        str(policy_path),
                        "--state",
                        str(root / "state.sqlite3"),
                        "--memory",
                        str(root / "memory.sqlite3"),
                        "pc-act",
                        "--action",
                        "invoke",
                        "--selector",
                        "name=Calculator",
                    ]
                )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertTrue(payload["requires_approval"])
            self.assertEqual(payload["approval"]["status"], "pending")


class GatewayRouterResearchTests(unittest.TestCase):
    def test_gateway_router_runs_research_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            orchestrator = new_orchestrator(root, _ALLOW_POLICY)
            orchestrator.worker.research_engine = FakeResearchEngine()
            router = GatewayCommandRouter(orchestrator)

            response = router.handle(
                ChannelMessage("telegram", "42", "/run test topic")
            )

            self.assertEqual(response.status, "completed")
            self.assertIn("report", response.payload)

    def test_gateway_router_supports_research_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            orchestrator = new_orchestrator(root, _ALLOW_POLICY)
            orchestrator.worker.research_engine = FakeResearchEngine()
            router = GatewayCommandRouter(orchestrator)

            response = router.handle(
                ChannelMessage(
                    "telegram",
                    "42",
                    "/research retrieval-augmented literature synthesis",
                )
            )

            self.assertEqual(response.status, "completed")
            report = response.payload["report"]
            self.assertIn("[multi-hour]", report["objective"])


if __name__ == "__main__":
    unittest.main()
