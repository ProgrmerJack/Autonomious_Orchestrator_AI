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
    template: str
    enabled: bool = True

    def render(self, argument: str) -> str:
        return self.template.format(argument=argument.strip())

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
        if self.path is None or not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        commands: list[WorkflowCommand] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                commands.append(
                    WorkflowCommand(
                        command_id=(str(item["command_id"]).strip().removeprefix("/")),
                        label=str(item.get("label") or item["command_id"]),
                        description=str(item.get("description") or ""),
                        template=str(item["template"]),
                        enabled=bool(item.get("enabled", True)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return commands


def _default_commands() -> list[WorkflowCommand]:
    return [
        WorkflowCommand(
            command_id="quick-research",
            label="Quick research",
            description="Fast evidence scan with structured sources.",
            template="[quick] {argument}",
        ),
        WorkflowCommand(
            command_id="research",
            label="Research",
            description=("Alias for multi-hour deep research from channel inputs."),
            template=(
                "[multi-hour] Execute paper-mode deep research with "
                "iterative retrieval and evidence-coverage gates for: "
                "{argument}"
            ),
        ),
        WorkflowCommand(
            command_id="deep-research",
            label="Multi-hour research",
            description=("Budgeted long-horizon research with durable artifacts."),
            template=(
                "[multi-hour] Execute paper-mode deep research with "
                "iterative retrieval, evidence-coverage gates, and active "
                "PC research actions (approval-gated) for: {argument}"
            ),
        ),
        WorkflowCommand(
            command_id="pc-research-smoke",
            label="PC research smoke",
            description=(
                "Research objective shaped to exercise PC-control evidence paths."
            ),
            template=(
                "[quick] Use local PC control, browser evidence, "
                "channel routing, and OpenCode/OpenClaw-style "
                "operator workflows to investigate: {argument}"
            ),
        ),
        WorkflowCommand(
            command_id="competitive-audit",
            label="Competitive audit",
            description=(
                "Compare AgentOS against OS-agent and coding-agent competitors."
            ),
            template=(
                "[standard] Compare AgentOS against OpenClaw, OpenCode, "
                "and OpenHands for: {argument}"
            ),
        ),
    ]
