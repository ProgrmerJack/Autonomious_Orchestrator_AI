from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from agentos_orchestrator.product import (
    CommandRegistry,
    DaemonManager,
    WorkflowCommand,
    collect_product_status,
)
from agentos_orchestrator.sdk import AgentOSClient


class FakeClient(AgentOSClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((method, path, payload))
        return {"ok": True, "path": path, "payload": payload}


class ProductTests(unittest.TestCase):
    def test_product_status_reports_core_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {"actions": [], "paths": []},
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": ["os.act"]},
                    }
                ),
                encoding="utf-8",
            )

            status = collect_product_status(
                root,
                policy,
                root / ".agentos/state.sqlite3",
                root / ".agentos/memory.sqlite3",
                {"passed": True},
            )

            checks = {check.check_id: check for check in status.checks}
            self.assertEqual(checks["policy-safety"].status, "pass")
            self.assertIn("readiness_score", status.benchmarks)

    def test_provider_status_detects_environment_keys(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            status = collect_product_status(
                Path.cwd(),
                "examples/policies/deep_research.json",
                ".agentos/state.sqlite3",
                ".agentos/memory.sqlite3",
            )

        providers = {provider.provider_id: provider for provider in status.providers}
        self.assertTrue(providers["openai"].configured)

    def test_daemon_manager_reports_stopped_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DaemonManager(temp_dir)
            status = manager.status()

            self.assertEqual(status.status, "stopped")
            self.assertIsNone(status.launcher_pid)

    def test_sdk_shapes_run_and_command_requests(self) -> None:
        client = FakeClient()

        run = client.start_run("topic", depth="quick")
        command = client.command("/run topic")

        self.assertEqual(run["path"], "/runs")
        self.assertEqual(
            client.calls[0][2],
            {
                "objective": "topic",
                "depth": "quick",
                "background": True,
            },
        )
        self.assertEqual(command["path"], "/channels/command")

        self.assertEqual(client.daemon_status()["path"], "/daemon/status")
        self.assertEqual(
            client.golden_traces()["path"],
            "/benchmarks/golden-traces",
        )
        self.assertEqual(client.eval_pack()["path"], "/benchmarks/eval-pack")
        self.assertEqual(client.replay_debug()["path"], "/debug/replay")

    def test_sdk_shapes_benchmark_and_pc_requests(self) -> None:
        client = FakeClient()

        self.assertEqual(
            client.live_fire_eval(
                max_tasks=1,
                windows_safe_pack=True,
                repeat=2,
            )["path"],
            "/benchmarks/live-fire-eval",
        )
        payload = cast(dict[str, Any], client.calls[-1][2])
        self.assertTrue(payload["windows_safe_pack"])
        self.assertEqual(payload["repeat"], 2)
        self.assertEqual(
            client.live_fire_review()["path"],
            "/benchmarks/live-fire-review?limit=10",
        )
        self.assertEqual(
            client.promote_live_fire_failure("run", "task")["path"],
            "/benchmarks/live-fire-review/promote",
        )
        self.assertEqual(
            client.live_fire_shadow_training(["trace.jsonl"])["path"],
            "/benchmarks/live-fire-shadow-training",
        )
        self.assertEqual(
            client.pc_debug_selector("name=AgentOS")["path"],
            "/pc/debug-selector",
        )
        self.assertEqual(
            client.pc_workflow_plan("write a report")["path"],
            "/pc/workflow/plan",
        )
        self.assertEqual(
            client.pc_workflow_execute("write a report")["path"],
            "/pc/workflow/execute",
        )

    def test_command_registry_merges_defaults_and_user_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "commands.json"
            registry = CommandRegistry(path)
            saved = registry.save(
                WorkflowCommand(
                    command_id="audit",
                    label="Audit",
                    description="Custom audit",
                )
            )

            self.assertEqual(saved.objective_for("topic"), "topic")
            self.assertIsNotNone(registry.get("quick-research"))
            self.assertIsNotNone(registry.get("research"))
            loaded = registry.get("/audit")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.command_id, "audit")


if __name__ == "__main__":
    unittest.main()
