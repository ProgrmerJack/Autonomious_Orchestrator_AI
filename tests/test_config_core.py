"""Tests for MarkdownAgentConfig, CLI config command, HeartbeatScheduler,
DashboardEventHub, and channel webhook adapters (Telegram, Slack, Discord).

Split from test_config_gateway_core.py to keep each module under 100 lines
and each test method narrowly scoped.
"""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentos_orchestrator.cli import main
from agentos_orchestrator.config import MarkdownAgentConfig
from agentos_orchestrator.core.types import Event
from agentos_orchestrator.gateway import (
    DashboardEventHub,
    DiscordWebhookAdapter,
    HeartbeatScheduler,
    SlackWebhookAdapter,
    TelegramWebhookAdapter,
)


class MarkdownConfigTests(unittest.TestCase):
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


class DashboardEventHubTests(unittest.TestCase):
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


class ChannelAdapterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
