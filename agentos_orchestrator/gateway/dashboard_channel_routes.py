from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from agentos_orchestrator.gateway.channels import ChannelMessage
from agentos_orchestrator.gateway.dashboard_support import _record_channel_delivery
from agentos_orchestrator.product import WorkflowCommand


def register_dashboard_channel_routes(
    app: Any,
    fastapi: Any,
    command_registry: Any,
    command_router: Any,
    telegram: Any,
    generic_webhook: Any,
    slack: Any,
    discord: Any,
    channel_deliveries: deque[dict[str, Any]],
) -> None:
    @app.get("/commands")
    async def list_commands() -> list[dict]:
        return [command.asdict() for command in command_registry.list_commands()]

    @app.post("/commands")
    async def save_command(payload: dict) -> dict:
        command_id = str(payload.get("command_id") or "").strip().removeprefix("/")
        if not command_id:
            raise fastapi.HTTPException(
                status_code=400,
                detail="command_id is required",
            )
        command = WorkflowCommand(
            command_id=command_id,
            label=str(payload.get("label") or command_id),
            description=str(payload.get("description") or ""),
            enabled=bool(payload.get("enabled", True)),
        )
        return command_registry.save(command).asdict()

    @app.get("/channels/deliveries")
    async def channel_delivery_history() -> list[dict]:
        return list(channel_deliveries)

    @app.post("/channels/telegram")
    async def telegram_webhook(payload: dict) -> dict:
        message = telegram.parse(payload)
        if message is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="telegram payload did not contain text",
            )
        response = asdict(command_router.handle(message))
        _record_channel_delivery(channel_deliveries, message, response)
        return response

    @app.post("/channels/generic")
    async def generic_channel(payload: dict) -> dict:
        message = generic_webhook.parse(payload)
        if message is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="generic payload did not contain text",
            )
        response = asdict(command_router.handle(message))
        _record_channel_delivery(channel_deliveries, message, response)
        return response

    @app.post("/channels/slack")
    async def slack_webhook(payload: dict) -> dict:
        message = slack.parse(payload)
        if message is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="slack payload did not contain text",
            )
        response = asdict(command_router.handle(message))
        _record_channel_delivery(channel_deliveries, message, response)
        return response

    @app.post("/channels/discord")
    async def discord_webhook(payload: dict) -> dict:
        message = discord.parse(payload)
        if message is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="discord payload did not contain text",
            )
        response = asdict(command_router.handle(message))
        _record_channel_delivery(channel_deliveries, message, response)
        return response

    @app.post("/channels/command")
    async def command_channel(payload: dict) -> dict:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise fastapi.HTTPException(
                status_code=400,
                detail="text is required",
            )
        message = ChannelMessage(
            channel=str(payload.get("channel") or "dashboard"),
            sender_id=str(payload.get("sender_id") or "dashboard"),
            text=text,
            metadata={"source": "dashboard-command"},
        )
        response = asdict(command_router.handle(message))
        _record_channel_delivery(channel_deliveries, message, response)
        return response