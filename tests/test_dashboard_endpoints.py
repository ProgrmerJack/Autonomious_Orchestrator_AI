from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agentos_orchestrator.gateway import DashboardEventHub
from agentos_orchestrator.gateway.dashboard import (
    DashboardRunManager,
    create_dashboard_app,
)

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

                eval_pack = api.get("/benchmarks/eval-pack").json()
                self.assertEqual(eval_pack["task_count"], 100)

                live_fire = api.post(
                    "/benchmarks/live-fire-eval",
                    json={
                        "backend": "virtual-desktop-sandbox",
                        "max_tasks": 2,
                        "windows_safe_pack": True,
                        "repeat": 2,
                        "promote_failures": False,
                    },
                ).json()
                self.assertEqual(live_fire["task_count"], 4)
                self.assertIn("replay_debug", live_fire)
                self.assertIn("training_summary", live_fire)
                self.assertIn("milestone", live_fire)

                review = api.get("/benchmarks/live-fire-review").json()
                self.assertIn("runs", review)
                self.assertIn("failed_tasks", review)
                self.assertIn("milestone", review)

                shadow = api.post(
                    "/benchmarks/live-fire-shadow-training",
                    json={"trajectory_paths": live_fire["trajectory_paths"]},
                ).json()
                self.assertTrue(shadow["advisory_only"])
                self.assertEqual(
                    shadow["head_order"],
                    ["outcome_critic", "option_policy", "affordance_ranker"],
                )
                self.assertTrue(shadow["ready_for_shadow_training"])

                debug = api.post("/debug/replay", json={}).json()
                self.assertIn("runs", debug)

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
                self.assertEqual(job["objective"], "dashboard topic")
                self.assertEqual(job["depth"], "multi-hour")
                self.assertTrue(job["run_objective"].startswith("[multi-hour] "))

                for _attempt in range(40):
                    job = api.get(f"/jobs/{job['job_id']}").json()
                    if job["status"] == "completed":
                        break
                    time.sleep(0.05)

                self.assertEqual(job["status"], "completed")
                self.assertTrue(job["run_id"].startswith("run_"))
                self.assertEqual(
                    job["report"]["objective"],
                    "[multi-hour] dashboard topic",
                )

    def test_dashboard_background_job_preserves_objective_depth_tag(self) -> None:
        temp_dir, client = self._client()
        with temp_dir:
            with client as api:
                job = api.post(
                    "/runs",
                    json={
                        "objective": "[multi-hour] depth-tagged dashboard topic",
                    },
                ).json()
                self.assertEqual(job["status"], "queued")
                self.assertEqual(job["depth"], "multi-hour")
                self.assertTrue(job["run_objective"].startswith("[multi-hour] "))

                for _attempt in range(40):
                    job = api.get(f"/jobs/{job['job_id']}").json()
                    if job["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.05)

                self.assertEqual(job["status"], "completed")

    def test_dashboard_background_job_marks_unexpected_exception_failed(self) -> None:
        class FailingOrchestrator:
            def run(self, objective: str, run_id: str | None = None):
                del objective, run_id
                raise TypeError("unexpected worker failure")

        manager = DashboardRunManager(
            FailingOrchestrator(),
            DashboardEventHub(),
            max_workers=1,
        )
        job = manager.start("dashboard topic", depth="multi-hour")

        for _attempt in range(40):
            job = manager.get_job(job["job_id"])
            if job is not None and job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "failed")
        self.assertIn("unexpected worker failure", job["error"])


if __name__ == "__main__":
    unittest.main()
