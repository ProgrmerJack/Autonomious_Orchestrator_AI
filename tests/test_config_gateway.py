from __future__ import annotations

import asyncio
import io
import json
import time
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentos_orchestrator.cli import main
from agentos_orchestrator.config import MarkdownAgentConfig
from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.core.types import Event
from agentos_orchestrator.gateway import (
    ChannelMessage,
    DashboardEventHub,
    DiscordWebhookAdapter,
    GatewayCommandRouter,
    HeartbeatScheduler,
    SlackWebhookAdapter,
    TelegramWebhookAdapter,
)
from agentos_orchestrator.gateway.dashboard import create_dashboard_app
from agentos_orchestrator.research import (
    DeepResearchEngine,
    ResearchBrief,
    ResearchSource,
)


class FakeResearchEngine(DeepResearchEngine):
    def run(self, objective: str, run_id: str) -> ResearchBrief:
        return ResearchBrief(
            objective=objective,
            query=objective,
            summary="Router test research completed.",
            sources=[
                ResearchSource(
                    provider="test",
                    title="Router Source",
                    url="https://example.com/router",
                    abstract="Router evidence.",
                )
            ],
            artifacts=[],
            confidence=0.88,
        )


class ConfigGatewayTests(unittest.TestCase):
    def test_markdown_config_and_cli_config_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "SOUL.md").write_text("# Soul\n", encoding="utf-8")
            (root / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
            (root / "HEARTBEAT.md").write_text(
                "enabled: true\ninterval_seconds: 120\n",
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["config", "--root", str(root)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["heartbeat_enabled"])
            self.assertEqual(payload["heartbeat"]["interval_seconds"], "120")

    def test_heartbeat_scheduler_respects_interval(self) -> None:
        config = MarkdownAgentConfig(
            soul="",
            agents="",
            heartbeat={
                "enabled": "true",
                "interval_seconds": "300",
                "max_background_turns": "2",
            },
        )
        scheduler = HeartbeatScheduler(config)

        first = scheduler.evaluate()
        second = scheduler.evaluate()

        self.assertTrue(first.due)
        self.assertEqual(first.max_background_turns, 2)
        self.assertFalse(second.due)

    def test_dashboard_event_hub_fans_out_messages(self) -> None:
        async def scenario() -> None:
            hub = DashboardEventHub()
            first_queue = hub.subscribe()
            second_queue = hub.subscribe()
            hub.publish_event(
                Event(
                    run_id="run_1",
                    type="run.started",
                    source="test",
                    payload={"ok": True},
                )
            )

            first = json.loads(await hub.next_message(first_queue))
            second = json.loads(await hub.next_message(second_queue))
            self.assertEqual(first["event"]["type"], "run.started")
            self.assertEqual(second["event"]["payload"]["ok"], True)
            hub.unsubscribe(first_queue)
            hub.unsubscribe(second_queue)

        asyncio.run(scenario())

    def test_telegram_adapter_parses_text_messages(self) -> None:
        adapter = TelegramWebhookAdapter()
        message = adapter.parse(
            {
                "update_id": 1,
                "message": {
                    "text": "/run research",
                    "chat": {"id": 42},
                },
            }
        )

        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.channel, "telegram")
        self.assertEqual(message.sender_id, "42")
        self.assertEqual(message.text, "/run research")

    def test_slack_and_discord_adapters_parse_text_messages(self) -> None:
        slack = SlackWebhookAdapter()
        discord = DiscordWebhookAdapter()

        slack_message = slack.parse(
            {"event": {"text": "/quick-research gui agents", "user": "U1"}}
        )
        discord_message = discord.parse(
            {"content": "/quick-research gui agents", "author": {"id": "D1"}}
        )

        self.assertIsNotNone(slack_message)
        self.assertIsNotNone(discord_message)
        assert slack_message is not None
        assert discord_message is not None
        self.assertEqual(slack_message.channel, "slack")
        self.assertEqual(discord_message.sender_id, "D1")

    def test_pc_act_requests_approval_before_backend_execution(self) -> None:
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
                            "network_hosts": [],
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

    def test_gateway_router_runs_research_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
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
                            "network_hosts": [
                                "api.openalex.org",
                                "api.semanticscholar.org",
                                "api.github.com",
                                "generativelanguage.googleapis.com",
                            ],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                encoding="utf-8",
            )
            orchestrator = ResearchOrchestrator.from_paths(
                policy_path=policy_path,
                state_path=root / "state.sqlite3",
                memory_path=root / "memory.sqlite3",
            )
            orchestrator.worker.research_engine = FakeResearchEngine()
            router = GatewayCommandRouter(orchestrator)

            response = router.handle(
                ChannelMessage("telegram", "42", "/run test topic")
            )

            self.assertEqual(response.status, "completed")
            self.assertIn("report", response.payload)

    def test_dashboard_operator_endpoints(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi dashboard extra is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
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
                            "network_hosts": [
                                "api.openalex.org",
                                "api.semanticscholar.org",
                                "api.github.com",
                                "generativelanguage.googleapis.com",
                            ],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": ["os.act"]},
                    }
                ),
                encoding="utf-8",
            )
            orchestrator = ResearchOrchestrator.from_paths(
                policy_path=policy_path,
                state_path=root / "state.sqlite3",
                memory_path=root / "memory.sqlite3",
            )
            orchestrator.worker.research_engine = FakeResearchEngine()
            hub = DashboardEventHub()
            hub.attach(orchestrator.event_bus)
            app = create_dashboard_app(
                hub,
                orchestrator.approvals,
                orchestrator=orchestrator,
            )

            with TestClient(app) as client:
                status = client.get("/status").json()
                self.assertEqual(status["status"], "online")
                self.assertIn("pc_backends", status)

                setup = client.get("/setup/checks").json()
                self.assertIn("checks", setup)
                self.assertIn("benchmarks", setup)

                providers = client.get("/providers").json()
                self.assertTrue(
                    any(item["provider_id"] == "openai" for item in providers)
                )

                channels = client.get("/channels").json()
                self.assertTrue(
                    any(
                        item["channel_id"] == "generic-webhook"
                        for item in channels
                    )
                )

                benchmarks = client.get("/benchmarks").json()
                self.assertIn("readiness_score", benchmarks)
                self.assertIn("golden_traces", benchmarks)

                daemon = client.get("/daemon/status").json()
                self.assertIn(
                    daemon["status"],
                    {"stopped", "stale", "running"},
                )

                commands = client.get("/commands").json()
                self.assertTrue(
                    any(
                        item["command_id"] == "pc-research-smoke"
                        for item in commands
                    )
                )

                traces = client.get("/benchmarks/golden-traces").json()
                self.assertIn("trace_count", traces)

                replay = client.post("/benchmarks/replay", json={}).json()
                self.assertTrue(replay["passed"])

                policy = client.post(
                    "/policy/inspect",
                    json={
                        "action_type": "os.snapshot",
                        "target": "windows-uia://snapshot",
                    },
                ).json()
                self.assertTrue(policy["allowed"])

                command = client.post(
                    "/channels/command",
                    json={"text": "/quick-research dashboard channel topic"},
                ).json()
                self.assertEqual(command["status"], "completed")

                generic = client.post(
                    "/channels/generic",
                    json={"text": "/run generic channel topic"},
                ).json()
                self.assertEqual(generic["status"], "completed")

                slack = client.post(
                    "/channels/slack",
                    json={"event": {"text": "/run slack topic", "user": "U1"}},
                ).json()
                self.assertEqual(slack["status"], "completed")

                discord = client.post(
                    "/channels/discord",
                    json={
                        "content": "/run discord topic",
                        "author": {"id": "D1"},
                    },
                ).json()
                self.assertEqual(discord["status"], "completed")

                deliveries = client.get("/channels/deliveries").json()
                self.assertGreaterEqual(len(deliveries), 4)

                action = client.post(
                    "/pc/actions",
                    json={"action": "invoke", "selector": "name=Test"},
                ).json()
                self.assertEqual(action["status"], "approval_required")

                receipts = client.get("/pc/receipts").json()
                self.assertGreaterEqual(len(receipts), 1)

                job = client.post(
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
                    job = client.get(f"/jobs/{job['job_id']}").json()
                    if job["status"] == "completed":
                        break
                    time.sleep(0.05)

                self.assertEqual(job["status"], "completed")
                self.assertTrue(job["run_id"].startswith("run_"))


if __name__ == "__main__":
    unittest.main()
