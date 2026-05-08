from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.gateway import ChannelMessage, GatewayCommandRouter
from agentos_orchestrator.os_control import VirtualDesktopSandboxBackend
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.workflow.service import (
    WorkflowVerificationError,
)

from tests.gateway_test_support import new_orchestrator


class GatewayDesktopWorkflowTests(unittest.TestCase):
    def _desktop_orchestrator(self, root: Path):
        orchestrator = new_orchestrator(
            root,
            {
                "default": "deny",
                "allow": {
                    "actions": ["os.act", "file.write"],
                    "paths": ["runs/**", "artifacts/**"],
                    "network_hosts": [],
                },
                "forbid": {"actions": [], "paths": []},
                "require_approval": {"actions": []},
            },
        )
        orchestrator.worker.pc_backend = VirtualDesktopSandboxBackend(
            root / "virtual_desktop_sandbox.json"
        )
        return orchestrator

    def test_gateway_router_executes_desktop_workflow_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(
                    ChannelMessage(
                        "telegram",
                        "42",
                        (
                            "/pc search for local pc automation and write a "
                            "report about local pc automation in word"
                        ),
                    )
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(response.status, "completed")
            self.assertIn("plan", response.payload)
            self.assertTrue(response.payload["artifacts"])
            self.assertTrue(
                any(
                    item["action_type"] == "open_url"
                    for item in response.payload["receipts"]
                )
            )
            self.assertTrue(
                any(
                    item["receipt"].get("launched") == "winword.exe"
                    for item in response.payload["receipts"]
                    if isinstance(item["receipt"], dict)
                )
            )
            self.assertTrue(
                any(
                    item.get("action_type") == "universal_agent_run"
                    for item in response.payload["receipts"]
                )
            )

    def test_run_command_auto_routes_actionable_objective_to_desktop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(
                    ChannelMessage(
                        "telegram",
                        "42",
                        "/run open foobar app and summarize current task progress",
                    )
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(response.status, "completed")
            self.assertIn("plan", response.payload)
            self.assertEqual(response.payload["plan"]["app_target"], "foobar.exe")
            self.assertTrue(
                any(
                    item.get("action_type") == "universal_agent_run"
                    for item in response.payload["receipts"]
                )
            )

    def test_gateway_router_requests_clarification_for_vague_desktop_task(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            response = router.handle(ChannelMessage("telegram", "42", "/pc do it"))

            self.assertEqual(response.status, "clarification_required")
            self.assertTrue(response.payload["plan"].get("requires_clarification"))

    def test_gateway_router_executes_multi_app_desktop_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(
                    ChannelMessage(
                        "telegram",
                        "42",
                        (
                            "/pc search for autonomous orchestration patterns "
                            "and write a report then create slides"
                        ),
                    )
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(response.status, "completed")
            self.assertEqual(response.payload["plan"]["mode"], "multi-app")
            self.assertTrue(
                any(
                    item["receipt"].get("launched") == "winword.exe"
                    for item in response.payload["receipts"]
                    if isinstance(item["receipt"], dict)
                )
            )
            self.assertTrue(
                any(
                    item["receipt"].get("launched") == "powerpnt.exe"
                    for item in response.payload["receipts"]
                    if isinstance(item["receipt"], dict)
                )
            )
            self.assertTrue(
                any(
                    item["action_type"] == "open_url"
                    for item in response.payload["receipts"]
                )
            )

    def test_gateway_router_returns_error_on_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            def fail_execute(_objective, _backend):
                raise WorkflowVerificationError(
                    action=UiAction(
                        action_type="type",
                        selector="name=Document Canvas",
                        value="broken",
                    ),
                    receipt='{"status": "executed"}',
                    verification={
                        "kind": "field_contains",
                        "matched": False,
                        "expected": "typed value visible",
                        "observed": "document canvas",
                        "required": True,
                        "reason": "typed value was not observed",
                        "evidence": {},
                    },
                    recovery={
                        "applied": False,
                        "attempted": False,
                        "reason": "",
                        "verification_failed": True,
                        "verification_reason": "typed value was not observed",
                    },
                )

            router.workflow_service.execute = fail_execute
            response = router.handle(
                ChannelMessage(
                    "telegram",
                    "42",
                    "/pc write a report about local pc automation",
                )
            )

            self.assertEqual(response.status, "error")
            self.assertEqual(response.payload["status"], "verification_failed")


if __name__ == "__main__":
    unittest.main()
