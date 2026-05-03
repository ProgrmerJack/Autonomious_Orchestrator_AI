from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowCommand:
    command_id: str
    label: str
    description: str
    enabled: bool = True

    def objective_for(self, argument: str) -> str:
        return argument.strip()

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class CommandRegistry:
    """Reusable workflow command catalog for UI and channel surfaces."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None

    def list_commands(self) -> list[WorkflowCommand]:
        commands = {command.command_id: command for command in _default_commands()}
        for command in self._load_user_commands():
            commands[command.command_id] = command
        return sorted(commands.values(), key=lambda item: item.command_id)

    def get(self, command_id: str) -> WorkflowCommand | None:
        normalized = command_id.strip().removeprefix("/")
        for command in self.list_commands():
            if command.enabled and command.command_id == normalized:
                return command
        return None

    def save(self, command: WorkflowCommand) -> WorkflowCommand:
        if self.path is None:
            raise ValueError("command registry has no writable path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        commands = {item.command_id: item for item in self._load_user_commands()}
        commands[command.command_id] = command
        self.path.write_text(
            json.dumps(
                [item.asdict() for item in commands.values()],
                indent=2,
            ),
            encoding="utf-8",
        )
        return command

    def _load_user_commands(self) -> list[WorkflowCommand]:
        return [
            command
            for command in map(
                self._command_from_payload,
                self._read_user_commands_payload(),
            )
            if command is not None
        ]

    def _read_user_commands_payload(self) -> list[Any]:
        if self.path is None or not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return payload

    @staticmethod
    def _command_from_payload(item: Any) -> WorkflowCommand | None:
        if not isinstance(item, dict):
            return None
        try:
            command_id = str(item["command_id"]).strip().removeprefix("/")
        except (KeyError, TypeError, ValueError):
            return None
        return WorkflowCommand(
            command_id=command_id,
            label=str(item.get("label") or item["command_id"]),
            description=str(item.get("description") or ""),
            enabled=bool(item.get("enabled", True)),
        )


def _default_commands() -> list[WorkflowCommand]:
    return [
        WorkflowCommand(
            command_id="quick-research",
            label="Adaptive lookup",
            description="Run a prompt-scaled evidence-gathering workflow.",
        ),
        WorkflowCommand(
            command_id="research",
            label="Adaptive research",
            description=("Alias for prompt-scaled research from channel inputs."),
        ),
        WorkflowCommand(
            command_id="deep-research",
            label="Deep research",
            description=("Prompt-scaled long-horizon research with durable artifacts."),
        ),
        WorkflowCommand(
            command_id="pc-research-smoke",
            label="PC research smoke",
            description=(
                "Prompt-scaled research command that keeps PC-control evidence "
                "paths available when the request calls for them."
            ),
        ),
        WorkflowCommand(
            command_id="competitive-audit",
            label="Competitive audit",
            description=(
                "Prompt-scaled audit command; competitors and effort come from "
                "the request text."
            ),
        ),
    ]
