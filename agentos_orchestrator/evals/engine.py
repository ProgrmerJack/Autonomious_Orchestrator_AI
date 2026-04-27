from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from agentos_orchestrator.core.events import EventBus
from agentos_orchestrator.core.types import Event, JsonObject


@dataclass(slots=True)
class EvaluationFinding:
    evaluator: str
    passed: bool
    reason: str
    severity: str = "info"
    context: JsonObject = field(default_factory=dict)


class EventEvaluator(Protocol):
    name: str

    def evaluate(
        self,
        event: Event,
        history: list[Event],
    ) -> list[EvaluationFinding]:
        """Return zero or more binary reliability findings."""


class RepeatedTransitionEvaluator:
    name = "repeated-transition"

    def __init__(self, max_repeats: int = 5) -> None:
        self.max_repeats = max_repeats

    def evaluate(
        self,
        event: Event,
        history: list[Event],
    ) -> list[EvaluationFinding]:
        recent = history[-self.max_repeats :]
        repeats = [item for item in recent if item.type == event.type]
        if len(repeats) < self.max_repeats:
            return []
        return [
            EvaluationFinding(
                evaluator=self.name,
                passed=False,
                reason="same transition repeated too many times",
                severity="error",
                context={"event_type": event.type},
            )
        ]


class PromptInjectionEvaluator:
    name = "prompt-injection"

    def evaluate(
        self,
        event: Event,
        history: list[Event],
    ) -> list[EvaluationFinding]:
        _ = history
        text = f"{event.payload}".lower()
        markers = (
            "ignore previous instructions",
            "system prompt",
            "developer message",
            "exfiltrate",
            "disable security",
        )
        if not any(marker in text for marker in markers):
            return []
        return [
            EvaluationFinding(
                evaluator=self.name,
                passed=False,
                reason="event payload resembles prompt-injection text",
                severity="critical",
                context={"event_id": event.event_id},
            )
        ]


class ApprovalPauseEvaluator:
    name = "approval-pause"

    def evaluate(
        self,
        event: Event,
        history: list[Event],
    ) -> list[EvaluationFinding]:
        _ = history
        if event.type != "approval.requested":
            return []
        return [
            EvaluationFinding(
                evaluator=self.name,
                passed=True,
                reason="risky action paused for human approval",
                severity="info",
                context={"event_id": event.event_id},
            )
        ]


class MissingEvidenceEvaluator:
    name = "missing-evidence"

    def evaluate(
        self,
        event: Event,
        history: list[Event],
    ) -> list[EvaluationFinding]:
        _ = history
        if event.type != "task.completed":
            return []
        result = event.payload.get("result", {})
        if result.get("evidence"):
            return []
        return [
            EvaluationFinding(
                evaluator=self.name,
                passed=False,
                reason="worker completed without attached evidence",
                severity="warning",
                context={"task_id": result.get("task_id")},
            )
        ]


class UnsupervisedEvalEngine:
    """Continuous binary checks over the live event stream."""

    def __init__(self, evaluators: list[EventEvaluator] | None = None) -> None:
        self.evaluators = evaluators or [
            RepeatedTransitionEvaluator(),
            PromptInjectionEvaluator(),
            ApprovalPauseEvaluator(),
            MissingEvidenceEvaluator(),
        ]
        self.history: list[Event] = []
        self.findings: list[EvaluationFinding] = []

    def attach(self, bus: EventBus) -> None:
        bus.subscribe("*", self.process)

    def process(self, event: Event) -> None:
        for evaluator in self.evaluators:
            self.findings.extend(evaluator.evaluate(event, self.history))
        self.history.append(event)

    def passed(self) -> bool:
        return not any(
            not finding.passed and finding.severity in {"error", "critical"}
            for finding in self.findings
        )

    def snapshot(self) -> JsonObject:
        return {
            "passed": self.passed(),
            "findings": [asdict(finding) for finding in self.findings],
            "events_seen": len(self.history),
        }
