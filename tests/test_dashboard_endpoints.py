from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from agentos_orchestrator.gateway import DashboardEventHub
from agentos_orchestrator.gateway.dashboard import (
    DashboardRunManager,
    create_dashboard_app,
)
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.workflow.service import (
    WorkflowVerificationError,
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
        return temp_dir, app, TestClient(app)

    @staticmethod
    def _auth_headers(api, app) -> dict[str, str]:
        response = api.post(
            "/auth/session",
            json={"bootstrap_token": app.state.gateway_auth.bootstrap_token},
        )
        payload = response.json()
        return {
            "Authorization": f"Bearer {payload['session_token']}",
            "X-AgentOS-Csrf": payload["csrf_token"],
            "X-AgentOS-Unsafe": payload["unsafe_ack_value"],
        }

    def test_dashboard_requires_authenticated_session(self) -> None:
        temp_dir, _app, client = self._client()
        with temp_dir:
            with client as api:
                response = api.get("/status")
                self.assertEqual(response.status_code, 401)

    def test_dashboard_core_endpoints(self) -> None:
        temp_dir, app, client = self._client()
        with temp_dir:
            with client as api:
                headers = self._auth_headers(api, app)
                status = api.get("/status", headers=headers).json()
                self.assertEqual(status["status"], "online")
                self.assertIn("pc_backends", status)

                setup = api.get("/setup/checks", headers=headers).json()
                self.assertIn("checks", setup)
                self.assertIn("benchmarks", setup)

                providers = api.get("/providers", headers=headers).json()
                self.assertTrue(
                    any(item["provider_id"] == "openai" for item in providers)
                )

                channels = api.get("/channels", headers=headers).json()
                self.assertTrue(
                    any(item["channel_id"] == "generic-webhook" for item in channels)
                )

                benchmarks = api.get("/benchmarks", headers=headers).json()
                self.assertIn("readiness_score", benchmarks)
                self.assertIn("golden_traces", benchmarks)

                daemon = api.get("/daemon/status", headers=headers).json()
                self.assertIn(
                    daemon["status"],
                    {"stopped", "stale", "running"},
                )

                commands = api.get("/commands", headers=headers).json()
                self.assertTrue(
                    any(item["command_id"] == "pc-research-smoke" for item in commands)
                )

    def test_dashboard_channel_and_pc_endpoints(self) -> None:
        temp_dir, app, client = self._client()
        with temp_dir:
            with client as api:
                headers = self._auth_headers(api, app)
                workflow_plan = api.post(
                    "/pc/workflow/plan",
                    json={
                        "objective": (
                            "search for agent control and write a report "
                            "about agent control"
                        )
                    },
                    headers=headers,
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
                    headers=headers,
                ).json()
                self.assertIn(
                    workflow_exec["status"],
                    {"executed", "approval_required"},
                )

                replay = api.post("/benchmarks/replay", json={}, headers=headers).json()
                self.assertTrue(replay["passed"])

                eval_pack = api.get("/benchmarks/eval-pack", headers=headers).json()
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
                    headers=headers,
                ).json()
                self.assertEqual(live_fire["task_count"], 4)
                self.assertIn("replay_debug", live_fire)
                self.assertIn("training_summary", live_fire)
                self.assertIn("milestone", live_fire)

                review = api.get("/benchmarks/live-fire-review", headers=headers).json()
                self.assertIn("runs", review)
                self.assertIn("failed_tasks", review)
                self.assertIn("milestone", review)

                shadow = api.post(
                    "/benchmarks/live-fire-shadow-training",
                    json={"trajectory_paths": live_fire["trajectory_paths"]},
                    headers=headers,
                ).json()
                self.assertTrue(shadow["advisory_only"])
                self.assertEqual(
                    shadow["head_order"],
                    ["outcome_critic", "option_policy", "affordance_ranker"],
                )
                self.assertTrue(shadow["ready_for_shadow_training"])

                debug = api.post("/debug/replay", json={}, headers=headers).json()
                self.assertIn("runs", debug)

                policy = api.post(
                    "/policy/inspect",
                    json={
                        "action_type": "os.snapshot",
                        "target": "windows-uia://snapshot",
                    },
                    headers=headers,
                ).json()
                self.assertTrue(policy["allowed"])

                command = api.post(
                    "/channels/command",
                    json={"text": "/quick-research dashboard channel topic"},
                    headers=headers,
                ).json()
                self.assertEqual(command["status"], "completed")

                generic = api.post(
                    "/channels/generic",
                    json={"text": "/run generic channel topic"},
                    headers=headers,
                ).json()
                self.assertEqual(generic["status"], "completed")

                slack = api.post(
                    "/channels/slack",
                    json={"event": {"text": "/run slack topic", "user": "U1"}},
                    headers=headers,
                ).json()
                self.assertEqual(slack["status"], "completed")

                discord = api.post(
                    "/channels/discord",
                    json={
                        "content": "/run discord topic",
                        "author": {"id": "D1"},
                    },
                    headers=headers,
                ).json()
                self.assertEqual(discord["status"], "completed")

                deliveries = api.get("/channels/deliveries", headers=headers).json()
                self.assertGreaterEqual(len(deliveries), 4)

                action = api.post(
                    "/pc/actions",
                    json={"action": "invoke", "selector": "name=Test"},
                    headers=headers,
                ).json()
                self.assertEqual(action["status"], "approval_required")

                receipts = api.get("/pc/receipts", headers=headers).json()
                self.assertGreaterEqual(len(receipts), 1)

    def test_dashboard_workflow_execute_returns_verification_failure(self) -> None:
        temp_dir, app, client = self._client()
        with temp_dir:
            with client as api:
                headers = self._auth_headers(api, app)
                original_execute = app.state.workflow_service.execute

                def fail_execute(_objective, _backend):
                    raise WorkflowVerificationError(
                        action=UiAction(
                            action_type="type",
                            selector="name=Document Canvas",
                            value="broken",
                        ),
                        receipt=json.dumps({"status": "executed"}),
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

                app.state.workflow_service.execute = fail_execute
                try:
                    pending = api.post(
                        "/pc/workflow/execute",
                        json={
                            "objective": "write a report about agent control",
                            "backend": "virtual-desktop-sandbox",
                        },
                        headers=headers,
                    ).json()
                    self.assertEqual(pending["status"], "approval_required")
                    approval_token = pending["decision"]["approval"]["token"]
                    api.post(
                        f"/approvals/{approval_token}/approve",
                        headers=headers,
                    ).json()
                    payload = api.post(
                        "/pc/workflow/execute",
                        json={
                            "objective": "write a report about agent control",
                            "backend": "virtual-desktop-sandbox",
                            "approval_token": approval_token,
                        },
                        headers=headers,
                    ).json()
                finally:
                    app.state.workflow_service.execute = original_execute

                self.assertEqual(payload["status"], "verification_failed")
                self.assertIn("failure", payload)
                self.assertTrue(payload["failure"]["verification"]["required"])

    def test_dashboard_background_job_endpoint(self) -> None:
        temp_dir, app, client = self._client()
        with temp_dir:
            with client as api:
                headers = self._auth_headers(api, app)
                job = api.post(
                    "/runs",
                    json={
                        "objective": "dashboard topic",
                        "depth": "multi-hour",
                        "background": True,
                    },
                    headers=headers,
                ).json()
                self.assertEqual(job["status"], "queued")
                self.assertEqual(job["objective"], "dashboard topic")
                self.assertEqual(job["depth"], "multi-hour")
                self.assertTrue(job["run_objective"].startswith("[multi-hour] "))

                for _attempt in range(40):
                    job = api.get(
                        f"/jobs/{job['job_id']}",
                        headers=headers,
                    ).json()
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
        temp_dir, app, client = self._client()
        with temp_dir:
            with client as api:
                headers = self._auth_headers(api, app)
                job = api.post(
                    "/runs",
                    json={
                        "objective": "[multi-hour] depth-tagged dashboard topic",
                    },
                    headers=headers,
                ).json()
                self.assertEqual(job["status"], "queued")
                self.assertEqual(job["depth"], "multi-hour")
                self.assertTrue(job["run_objective"].startswith("[multi-hour] "))

                for _attempt in range(40):
                    job = api.get(
                        f"/jobs/{job['job_id']}",
                        headers=headers,
                    ).json()
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
