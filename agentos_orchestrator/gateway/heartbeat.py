from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agentos_orchestrator.config import MarkdownAgentConfig


@dataclass(slots=True)
class HeartbeatTurn:
    due: bool
    reason: str
    max_background_turns: int


class HeartbeatScheduler:
    """Determines when background autonomous turns may run."""

    def __init__(self, config: MarkdownAgentConfig) -> None:
        self.config = config
        self.last_turn_at: datetime | None = None

    def evaluate(self) -> HeartbeatTurn:
        max_turns = int(self.config.heartbeat.get("max_background_turns", "3"))
        if not self.config.heartbeat_enabled():
            return HeartbeatTurn(False, "heartbeat disabled", max_turns)
        now = datetime.now(tz=UTC)
        if self.last_turn_at is None:
            self.last_turn_at = now
            return HeartbeatTurn(True, "first heartbeat due", max_turns)
        interval = timedelta(seconds=self.config.heartbeat_interval())
        if now - self.last_turn_at >= interval:
            self.last_turn_at = now
            return HeartbeatTurn(True, "heartbeat interval elapsed", max_turns)
        return HeartbeatTurn(
            False,
            "heartbeat interval has not elapsed",
            max_turns,
        )
