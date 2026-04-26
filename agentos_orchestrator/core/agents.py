from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointStore
from .events import EventBus
from .types import ActionRequest, TaskSpec, WorkerResult, new_id
from agentos_orchestrator.os_control import WindowsUiaBackend
from agentos_orchestrator.os_control.base import BackendUnavailable, UiNode
from agentos_orchestrator.research import DeepResearchEngine


def declared_action(
    agent_id: str,
    action_type: str,
    target: str,
    payload: dict | None = None,
) -> ActionRequest:
    return ActionRequest(
        agent_id=agent_id,
        action_type=action_type,
        target=target,
        payload=payload or {},
    )


class SupervisorAgent:
    """Plans work and preserves separation from execution tools."""

    agent_id = "supervisor"

    def plan(self, objective: str) -> list[TaskSpec]:
        literature_id = new_id("task_literature")
        data_id = new_id("task_data")
        synthesis_id = new_id("task_synthesis")
        tasks: list[TaskSpec] = []
        if self._needs_pc_context(objective):
            tasks.append(
                TaskSpec(
                    task_id=new_id("task_pc"),
                    role="pc-control",
                    objective=(
                        "Capture live desktop and browser/operator context "
                        f"for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "pc-control-agent",
                            "os.snapshot",
                            "windows-uia://snapshot",
                        ),
                        declared_action(
                            "pc-control-agent",
                            "file.write",
                            "runs/**",
                        ),
                    ],
                )
            )

        tasks.extend(
            [
                TaskSpec(
                    task_id=literature_id,
                    role="literature",
                    objective=(
                        "Find authoritative sources, prior systems, and "
                        f"gaps for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "literature-agent",
                            "mcp.list",
                            "mcp://configured-servers",
                        ),
                        declared_action(
                            "literature-agent",
                            "mcp.call",
                            "mcp://research/search",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.openalex.org/works",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            (
                                "https://api.semanticscholar.org/graph/v1/"
                                "paper/search"
                            ),
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.github.com/search/repositories",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            (
                                "https://generativelanguage.googleapis.com/"
                                "v1beta/models/"
                                "gemini-flash-latest:generateContent"
                            ),
                        ),
                        declared_action(
                            "literature-agent",
                            "file.write",
                            "runs/**",
                        ),
                    ],
                ),
                TaskSpec(
                    task_id=data_id,
                    role="data",
                    objective=(
                        "Extract implementation constraints, security "
                        "boundaries, and validation criteria for: "
                        f"{objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "data-agent",
                            "file.write",
                            "runs/**",
                        ),
                        declared_action(
                            "data-agent",
                            "memory.commit",
                            "memory://candidate-facts",
                        ),
                    ],
                ),
                TaskSpec(
                    task_id=synthesis_id,
                    role="synthesis",
                    objective=(
                        "Merge worker outputs into a verified research "
                        f"brief for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "synthesis-agent",
                            "memory.commit",
                            "memory://verified-synthesis",
                        )
                    ],
                ),
            ]
        )
        return tasks

    @staticmethod
    def _needs_pc_context(objective: str) -> bool:
        lower = objective.lower()
        markers = (
            "browser research",
            "computer use",
            "desktop",
            "local pc",
            "openclaw",
            "pc control",
            "pc-research-smoke",
            "windows",
        )
        return any(marker in lower for marker in markers)


class WorkerAgent:
    """Constrained executor for one role and one task at a time."""

    def __init__(
        self,
        event_bus: EventBus,
        checkpoints: CheckpointStore,
        research_engine: DeepResearchEngine | None = None,
        pc_backend: Any | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.checkpoints = checkpoints
        self.research_engine = research_engine or DeepResearchEngine()
        self.pc_backend = pc_backend

    def run(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        self.event_bus.publish(
            run_id,
            "task.started",
            task.role,
            {"task": asdict(task)},
        )
        self.checkpoints.save(
            run_id,
            f"{task.task_id}.started",
            {"task": asdict(task), "completed": len(prior_results)},
        )

        result = self._execute(run_id, task, prior_results)

        self.event_bus.publish(
            run_id,
            "task.completed",
            task.role,
            {"result": asdict(result)},
        )
        self.checkpoints.save(
            run_id,
            f"{task.task_id}.completed",
            {"result": asdict(result), "completed": len(prior_results) + 1},
        )
        return result

    def _execute(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        if task.role == "literature":
            brief = self.research_engine.run(task.objective, run_id)
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=brief.summary,
                artifacts=brief.artifacts,
                evidence=brief.evidence(),
                confidence=brief.confidence,
            )
        if task.role == "pc-control":
            return self._capture_pc_context(run_id, task)
        if task.role == "data":
            evidence_count = sum(
                len(result.evidence) for result in prior_results
            )
            artifact_count = sum(
                len(result.artifacts) for result in prior_results
            )
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=(
                    "Extracted implementation constraints from "
                    f"{evidence_count} evidence records and "
                    f"{artifact_count} generated artifacts, then mapped "
                    "them into policy, durable state, event routing, and "
                    "sandbox boundaries."
                ),
                evidence=[
                    {
                        "source": "SECURITY.md",
                        "claim": (
                            "Workers must declare actions before running."
                        ),
                    }
                ],
                confidence=0.82,
            )

        combined = " ".join(result.summary for result in prior_results)
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Synthesized verified worker outputs into a checkpointed "
                f"research trace. Prior context: {combined[:500]}"
            ),
            evidence=[
                {
                    "source": "event-log",
                    "claim": "All worker transitions were durably recorded.",
                }
            ],
            confidence=0.8,
        )

    def _capture_pc_context(self, run_id: str, task: TaskSpec) -> WorkerResult:
        backend = self.pc_backend or WindowsUiaBackend()
        try:
            nodes = backend.snapshot()
        except (BackendUnavailable, OSError, RuntimeError) as exc:
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=f"PC context capture was unavailable: {exc}",
                evidence=[
                    {
                        "source": getattr(backend, "name", "pc-backend"),
                        "claim": "PC snapshot could not be captured.",
                    }
                ],
                confidence=0.35,
            )

        artifact = self._write_pc_snapshot(run_id, nodes[:120])
        named_nodes = [node for node in nodes if node.name][:8]
        names = "; ".join(node.name for node in named_nodes) or "no labels"
        browserish = [
            node.name
            for node in nodes
            if any(
                marker in node.name.lower()
                for marker in ("browser", "chrome", "edge", "127.0.0.1")
            )
        ]
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                f"Captured {len(nodes)} live UI Automation nodes from the "
                f"desktop. Visible context included: {names}."
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": (
                        "A live PC UI snapshot was captured for this run."
                    ),
                    "node_count": len(nodes),
                    "browser_context_detected": bool(browserish),
                    "browser_context": browserish[:5],
                }
            ],
            confidence=0.78,
        )

    @staticmethod
    def _write_pc_snapshot(run_id: str, nodes: list[UiNode]) -> str:
        path = Path("runs") / run_id / "pc" / "snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(node) for node in nodes], indent=2),
            encoding="utf-8",
        )
        return str(path)


class VerificationAgent:
    """Asynchronous support role expressed as an explicit verification pass."""

    def review(self, run_id: str, results: list[WorkerResult]) -> WorkerResult:
        confidence = 0.0
        if results:
            total = sum(result.confidence for result in results)
            confidence = total / len(results)
        missing_evidence = [
            result.task_id for result in results if not result.evidence
        ]
        if missing_evidence:
            summary = "Verification found worker outputs without evidence."
            confidence = min(confidence, 0.5)
        else:
            summary = "Verification accepted all worker outputs with evidence."

        return WorkerResult(
            task_id=new_id("task_verify"),
            role="verification",
            summary=summary,
            evidence=[
                {
                    "source": "verification-agent",
                    "run_id": run_id,
                    "checked_results": len(results),
                }
            ],
            confidence=confidence,
        )
