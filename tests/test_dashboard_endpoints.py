from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agentos_orchestrator.gateway import DashboardEventHub
from agentos_orchestrator.gateway.dashboard import create_dashboard_app

from tests.gateway_test_support import (
    FakeResearchEngine,
    base_network_hosts,
    new_orchestrator,
)


class DashboardEndpointsTests(unittest.TestCase):
    def _client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi dashboard extra is not installed")

        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        orchestrator = new_orchestrator(
            root,
            {
                "default": "deny",
                "allow": {
                    "actions": [
                        "file.write",
                        "mcp.call",
                        "mcp.list",
                        "memory.commit",
                        "network.fetch",
                        "os.act",
                        "os.snapshot",
                    ],
                    "paths": ["runs/**", "memory://*", "mcp://*"],
                    "network_hosts": base_network_hosts(),
                },
                "forbid": {"actions": [], "paths": []},
                "require_approval": {"actions": ["os.act"]},
            },
        )
        orchestrator.worker.research_engine = FakeResearchEngine()
        hub = DashboardEventHub()
        hub.attach(orchestrator.event_bus)
        app = create_dashboard_app(
            hub,
            orchestrator.approvals,
            orchestrator=orchestrator,
        )
        return temp_dir, TestClient(app)

    def test_dashboard_core_endpoints(self) -> None:
        temp_dir, client = self._client()
        with temp_dir:
            with client as api:
                status = api.get("/status").json()
                self.assertEqual(status["status"], "online")
                self.assertIn("pc_backends", status)

                setup = api.get("/setup/checks").json()
                self.assertIn("checks", setup)
                self.assertIn("benchmarks", setup)

                providers = api.get("/providers").json()
                self.assertTrue(
                    any(item["provider_id"] == "openai" for item in providers)
                )

                channels = api.get("/channels").json()
                self.assertTrue(
                    any(item["channel_id"] == "generic-webhook" for item in channels)
                )

                benchmarks = api.get("/benchmarks").json()
                self.assertIn("readiness_score", benchmarks)
                self.assertIn("golden_traces", benchmarks)

                daemon = api.get("/daemon/status").json()
                self.assertIn(
                    daemon["status"],
                    {"stopped", "stale", "running"},
                )

                commands = api.get("/commands").json()
                self.assertTrue(
                    any(item["command_id"] == "pc-research-smoke" for item in commands)
                )

    def test_dashboard_channel_and_pc_endpoints(self) -> None:
        temp_dir, client = self._client()
        with temp_dir:
            with client as api:
                workflow_plan = api.post(
                    "/pc/workflow/plan",
                    json={
                        "objective": (
                            "search for agent control and write a report "
                            "about agent control"
                        )
                    },
                ).json()
                self.assertEqual(workflow_plan["status"], "ok")
                self.assertEqual(workflow_plan["plan"]["mode"], "report")
                self.assertTrue(
                    any(
                        step["action_type"] == "open_url"
                        for step in workflow_plan["plan"]["steps"]
                    )
                )

                workflow_exec = api.post(
                    "/pc/workflow/execute",
                    json={
                        "objective": "write a report about agent control",
                        "backend": "virtual-desktop-sandbox",
                        "approval_token": None,
                    },
                ).json()
                self.assertIn(
                    workflow_exec["status"],
                    {"executed", "approval_required"},
                )

                replay = api.post("/benchmarks/replay", json={}).json()
                self.assertTrue(replay["passed"])

                policy = api.post(
                    "/policy/inspect",
                    json={
                        "action_type": "os.snapshot",
                        "target": "windows-uia://snapshot",
                    },
                ).json()
                self.assertTrue(policy["allowed"])

                command = api.post(
                    "/channels/command",
                    json={"text": "/quick-research dashboard channel topic"},
                ).json()
                self.assertEqual(command["status"], "completed")

                generic = api.post(
                    "/channels/generic",
                    json={"text": "/run generic channel topic"},
                ).json()
                self.assertEqual(generic["status"], "completed")

                slack = api.post(
                    "/channels/slack",
                    json={"event": {"text": "/run slack topic", "user": "U1"}},
                ).json()
                self.assertEqual(slack["status"], "completed")

                discord = api.post(
                    "/channels/discord",
                    json={
                        "content": "/run discord topic",
                        "author": {"id": "D1"},
                    },
                ).json()
                self.assertEqual(discord["status"], "completed")

                deliveries = api.get("/channels/deliveries").json()
                self.assertGreaterEqual(len(deliveries), 4)

                action = api.post(
                    "/pc/actions",
                    json={"action": "invoke", "selector": "name=Test"},
                ).json()
                self.assertEqual(action["status"], "approval_required")

                receipts = api.get("/pc/receipts").json()
                self.assertGreaterEqual(len(receipts), 1)

    def test_dashboard_background_job_endpoint(self) -> None:
        temp_dir, client = self._client()
        with temp_dir:
            with client as api:
                job = api.post(
                    "/runs",
                    json={
                        "objective": "dashboard topic",
                        "depth": "multi-hour",
                        "background": True,
                    },
                ).json()
                self.assertEqual(job["status"], "queued")
                self.assertTrue(job["objective"].startswith("[multi-hour]"))

                for _attempt in range(40):
                    job = api.get(f"/jobs/{job['job_id']}").json()
                    if job["status"] == "completed":
                        break
                    time.sleep(0.05)

                self.assertEqual(job["status"], "completed")
                self.assertTrue(job["run_id"].startswith("run_"))


if __name__ == "__main__":
    unittest.main()
