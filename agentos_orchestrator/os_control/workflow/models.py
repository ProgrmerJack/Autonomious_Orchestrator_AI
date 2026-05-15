from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DesktopWorkflowStep:
    action_type: str
    selector: str
    value: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowArtifact:
    path: str
    kind: str
    description: str


@dataclass(slots=True)
class DesktopWorkflowPlan:
    objective: str
    mode: str
    app_target: str | None
    summary: str
    steps: list[DesktopWorkflowStep]
    artifacts: list[WorkflowArtifact]
    intent: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sub_tasks: list[str] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
