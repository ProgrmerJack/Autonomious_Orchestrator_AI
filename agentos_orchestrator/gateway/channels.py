from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ChannelMessage:
    channel: str
    sender_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class ChannelAdapter(Protocol):
    name: str

    def parse(self, payload: dict) -> ChannelMessage | None:
        """Parse a webhook payload into an agent command."""


class TelegramWebhookAdapter:
    name = "telegram"

    def parse(self, payload: dict) -> ChannelMessage | None:
        message = payload.get("message") or payload.get("edited_message")
        if not isinstance(message, dict):
            return None
        text = message.get("text")
        chat = message.get("chat", {})
        if not text or not isinstance(chat, dict):
            return None
        return ChannelMessage(
            channel=self.name,
            sender_id=str(chat.get("id", "unknown")),
            text=str(text),
            metadata={"raw_update_id": payload.get("update_id")},
        )


class GenericWebhookAdapter:
    name = "generic-webhook"

    def parse(self, payload: dict) -> ChannelMessage | None:
        text = payload.get("text") or payload.get("message")
        if not text:
            return None
        metadata = payload.get("metadata")
        return ChannelMessage(
            channel=str(payload.get("channel") or self.name),
            sender_id=str(payload.get("sender_id") or "unknown"),
            text=str(text),
            metadata=metadata if isinstance(metadata, dict) else {},
        )


class SlackWebhookAdapter:
    name = "slack"

    def parse(self, payload: dict) -> ChannelMessage | None:
        raw_event = payload.get("event")
        event = raw_event if isinstance(raw_event, dict) else payload
        if not isinstance(event, dict):
            return None
        text = event.get("text") or payload.get("text")
        if not text:
            return None
        return ChannelMessage(
            channel=self.name,
            sender_id=str(
                event.get("user") or payload.get("user_id") or "unknown"
            ),
            text=str(text),
            metadata={
                "team_id": payload.get("team_id"),
                "channel_id": (
                    event.get("channel") or payload.get("channel_id")
                ),
                "event_id": payload.get("event_id"),
            },
        )


class DiscordWebhookAdapter:
    name = "discord"

    def parse(self, payload: dict) -> ChannelMessage | None:
        text = payload.get("content") or payload.get("text")
        if not text:
            return None
        raw_author = payload.get("author")
        author = raw_author if isinstance(raw_author, dict) else {}
        return ChannelMessage(
            channel=self.name,
            sender_id=str(
                author.get("id") or payload.get("user_id") or "unknown"
            ),
            text=str(text),
            metadata={
                "channel_id": payload.get("channel_id"),
                "guild_id": payload.get("guild_id"),
                "message_id": payload.get("id"),
            },
        )
