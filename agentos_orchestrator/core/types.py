from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

JsonObject = dict[str, Any]
ActionType = Literal[
    "file.read",
    "file.write",
    "mcp.call",
    "mcp.list",
    "memory.commit",
    "network.fetch",
    "os.act",
    "os.snapshot",
    "sandbox.exec",
    "sandbox.spawn",
]


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True)
class Event:
    run_id: str
    type: str
    source: str
    payload: JsonObject
    event_id: str = field(default_factory=lambda: new_id("evt"))
    created_at: str = field(default_factory=utc_now)
    sequence: int | None = None


@dataclass(slots=True)
class ActionRequest:
    agent_id: str
    action_type: str
    target: str
    payload: JsonObject = field(default_factory=dict)
    approval_token: str | None = None


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    role: str
    objective: str
    declared_actions: list[ActionRequest]
    inputs: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class WorkerResult:
    task_id: str
    role: str
    summary: str
    artifacts: list[str] = field(default_factory=list)
    evidence: list[JsonObject] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class RunReport:
    run_id: str
    objective: str
    status: str
    worker_results: list[WorkerResult]
    synthesis: str
    checkpoint_path: str
