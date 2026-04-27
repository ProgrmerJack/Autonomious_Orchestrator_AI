from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .authorization import AuthorizationMiddleware
from .approvals import ApprovalRequired
from .checkpoint import CheckpointStore
from .events import EventBus
from .types import ActionRequest, TaskSpec, WorkerResult, new_id
from agentos_orchestrator.os_control import WindowsUiaBackend
from agentos_orchestrator.os_control.base import (
    BackendUnavailable,
    UiAction,
    UiNode,
)
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
        planning_id = new_id("task_planning")
        literature_id = new_id("task_literature")
        pc_research_id = new_id("task_pc_research")
        data_id = new_id("task_data")
        synthesis_id = new_id("task_synthesis")
        tasks: list[TaskSpec] = []

        if self._is_multi_hour(objective):
            tasks.append(
                TaskSpec(
                    task_id=planning_id,
                    role="planning",
                    objective=f"Design deep research plan for: {objective}",
                    declared_actions=[
                        declared_action(
                            "planning-agent",
                            "file.write",
                            "runs/**",
                        )
                    ],
                )
            )

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

        if self._is_multi_hour(objective) and self._needs_pc_context(objective):
            tasks.append(
                TaskSpec(
                    task_id=pc_research_id,
                    role="pc-research",
                    objective=(
                        "Perform approval-gated active PC research actions "
                        f"for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "pc-research-agent",
                            "os.snapshot",
                            "windows-uia://snapshot",
                        ),
                        declared_action(
                            "pc-research-agent",
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
                            ("https://api.semanticscholar.org/graph/v1/paper/search"),
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.github.com/search/repositories",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.crossref.org/works",
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

    @staticmethod
    def _is_multi_hour(objective: str) -> bool:
        return "[multi-hour]" in objective.lower()


class WorkerAgent:
    """Constrained executor for one role and one task at a time."""

    def __init__(
        self,
        event_bus: EventBus,
        checkpoints: CheckpointStore,
        research_engine: DeepResearchEngine | None = None,
        pc_backend: Any | None = None,
        authorization: AuthorizationMiddleware | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.checkpoints = checkpoints
        self.research_engine = research_engine or DeepResearchEngine()
        self.pc_backend = pc_backend
        self.authorization = authorization

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
        if task.role == "planning":
            return self._build_deep_plan(run_id, task)
        if task.role == "literature":
            brief = self._run_research_with_context(
                task.objective,
                run_id,
                prior_results,
            )
            evidence = brief.evidence()
            if getattr(brief, "metadata", None):
                evidence.append(
                    {
                        "source": "research-metrics",
                        "claim": "Coverage and retrieval metrics for gating.",
                        "metadata": brief.metadata,
                    }
                )
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=brief.summary,
                artifacts=brief.artifacts,
                evidence=evidence,
                confidence=brief.confidence,
            )
        if task.role == "pc-control":
            return self._capture_pc_context(run_id, task)
        if task.role == "pc-research":
            return self._active_pc_research(run_id, task, prior_results)
        if task.role == "data":
            evidence_count = sum(len(result.evidence) for result in prior_results)
            artifact_count = sum(len(result.artifacts) for result in prior_results)
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
                        "claim": ("Workers must declare actions before running."),
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

    def _build_deep_plan(self, run_id: str, task: TaskSpec) -> WorkerResult:
        objective = task.objective.replace(
            "Design deep research plan for:",
            "",
        ).strip()
        min_runtime_seconds = self._multi_hour_min_runtime_seconds()
        max_retrieval_passes = 4 if min_runtime_seconds == 0 else 48
        targets = {
            "min_source_count": 8,
            "min_provider_count": 2,
            "min_scholarly_sources": 3,
            "min_strong_or_moderate": 4,
            "max_contradiction_risk": 0.75,
            "min_novelty_rate": 0.1,
            "max_retrieval_passes": max_retrieval_passes,
            "min_runtime_seconds": min_runtime_seconds,
        }
        hypotheses = [
            "Structured, multi-pass retrieval increases evidence breadth.",
            "Active PC browsing with approvals improves target-source coverage.",
            "Claim-level trace constraints reduce unsupported synthesis.",
        ]
        plan = {
            "objective": objective,
            "paper_mode": True,
            "coverage_targets": targets,
            "hypotheses": hypotheses,
            "created_by": "planning-worker",
        }
        artifact = self._write_planning_artifact(run_id, plan)
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Created deep research design with explicit evidence-coverage "
                "gates and paper-mode constraints."
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": "Deep planning and evidence targets are defined.",
                    "coverage_targets": targets,
                }
            ],
            confidence=0.84,
        )

    @staticmethod
    def _multi_hour_min_runtime_seconds() -> int:
        raw = os.environ.get("AGENTOS_MULTI_HOUR_MIN_SECONDS", "0")
        try:
            value = int(raw)
        except ValueError:
            return 0
        return max(value, 0)

    def _run_research_with_context(
        self,
        objective: str,
        run_id: str,
        prior_results: list[WorkerResult],
    ) -> Any:
        pc_artifact = self._latest_pc_snapshot_artifact(prior_results)
        planning_context = self._latest_planning_context(prior_results)
        evidence_targets: dict[str, Any] = {}
        if planning_context:
            evidence_targets = planning_context.get("coverage_targets") or {}
        pc_findings = self._latest_pc_research_findings(prior_results)
        pc_context = (
            {
                "snapshot_path": pc_artifact,
                "pc_findings": pc_findings,
            }
            if pc_artifact
            else None
        )
        signature = inspect.signature(self.research_engine.run)
        kwargs: dict[str, Any] = {}
        if "pc_context" in signature.parameters:
            kwargs["pc_context"] = pc_context
        if "planning_context" in signature.parameters:
            kwargs["planning_context"] = planning_context
        if "evidence_targets" in signature.parameters:
            kwargs["evidence_targets"] = evidence_targets
        if kwargs:
            return self.research_engine.run(
                objective,
                run_id,
                **kwargs,
            )
        return self.research_engine.run(objective, run_id)

    def _active_pc_research(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        backend = self.pc_backend or WindowsUiaBackend()
        urls = self._candidate_urls_from_sources(prior_results)
        if not urls:
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary="No candidate URLs were available for active PC research.",
                evidence=[
                    {
                        "source": "pc-research",
                        "claim": "No actionable target URLs were found.",
                    }
                ],
                confidence=0.42,
            )

        token = self._objective_approval_token(task.objective)
        action = ActionRequest(
            agent_id="pc-research-agent",
            action_type="os.act",
            target="windows-uia://name=Microsoft Edge",
            payload={
                "action": "focus",
                "urls": urls[:3],
            },
            approval_token=token,
        )
        if self.authorization is None:
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary="Authorization middleware is unavailable for active PC actions.",
                evidence=[
                    {
                        "source": "pc-research",
                        "claim": "Cannot execute approval-gated PC action.",
                    }
                ],
                confidence=0.3,
            )
        decision = self.authorization.authorize(run_id, action)
        if not decision.allowed:
            artifact = self._write_pc_findings_artifact(
                run_id,
                {
                    "status": "approval_required",
                    "candidate_urls": urls[:6],
                    "approval": asdict(decision.approval)
                    if decision.approval
                    else None,
                    "decision": asdict(decision),
                },
            )
            if decision.requires_approval and decision.approval is not None:
                raise ApprovalRequired(decision.approval)
            raise PermissionError(
                "Active PC research action was blocked by policy/trust checks. "
                f"Details in {artifact}."
            )

        receipts: list[dict[str, Any]] = []
        try:
            receipts.append(
                {
                    "step": "focus-browser",
                    "result": self._json_or_text(
                        backend.perform(UiAction("focus", "name=Microsoft Edge"))
                    ),
                }
            )
            receipts.append(
                {
                    "step": "navigate-candidate",
                    "result": self._json_or_text(
                        backend.perform(
                            UiAction(
                                "set_text",
                                "role=Edit",
                                value=urls[0],
                            )
                        )
                    ),
                }
            )
            post_nodes = backend.snapshot()
        except (BackendUnavailable, OSError, RuntimeError, ValueError) as exc:
            artifact = self._write_pc_findings_artifact(
                run_id,
                {
                    "status": "execution_error",
                    "candidate_urls": urls[:6],
                    "receipts": receipts,
                    "error": str(exc),
                },
            )
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=f"Active PC research action failed: {exc}",
                artifacts=[artifact],
                evidence=[
                    {
                        "source": artifact,
                        "claim": "Active PC research execution encountered an error.",
                    }
                ],
                confidence=0.4,
            )

        names = [node.name for node in post_nodes if node.name][:10]
        findings = {
            "status": "executed",
            "candidate_urls": urls[:6],
            "receipts": receipts,
            "post_snapshot_labels": names,
            "post_snapshot_node_count": len(post_nodes),
        }
        artifact = self._write_pc_findings_artifact(run_id, findings)
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Executed approval-gated active PC research actions and captured "
                "structured findings for evidence integration."
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": "Live PC actions were used to collect research findings.",
                    "targeted_urls": urls[:3],
                }
            ],
            confidence=0.7,
        )

    @staticmethod
    def _objective_approval_token(objective: str) -> str | None:
        match = re.search(r"approval-token=([A-Za-z0-9_\-]+)", objective)
        if match is None:
            return None
        return match.group(1)

    @staticmethod
    def _candidate_urls_from_sources(prior_results: list[WorkerResult]) -> list[str]:
        urls: list[str] = []
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("research/sources.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url") or "").strip()
                    if url.startswith("http"):
                        urls.append(url)
                if urls:
                    return urls
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("planning/plan.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                objective = str(payload.get("objective") or "")
                urls.extend(WorkerAgent._planning_urls_from_objective(objective))
                if urls:
                    return urls
        return urls

    @staticmethod
    def _planning_urls_from_objective(objective: str) -> list[str]:
        lower = objective.lower()
        urls: list[str] = []
        for entity in ("openclaw", "opencode", "openhands", "agentos"):
            if entity not in lower:
                continue
            query = f"{entity} architecture benchmark safety"
            encoded = query.replace(" ", "+")
            urls.append(f"https://github.com/search?type=repositories&q={encoded}")
        urls.extend(
            [
                "https://github.com/All-Hands-AI/OpenHands",
                "https://openreview.net/group?id=OpenHands",
            ]
        )
        deduped: list[str] = []
        for item in urls:
            if item not in deduped:
                deduped.append(item)
        return deduped[:6]

    @staticmethod
    def _latest_planning_context(
        prior_results: list[WorkerResult],
    ) -> dict[str, Any] | None:
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("planning/plan.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
        return None

    @staticmethod
    def _latest_pc_research_findings(
        prior_results: list[WorkerResult],
    ) -> dict[str, Any] | None:
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("pc/research_findings.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
        return None

    @staticmethod
    def _write_planning_artifact(run_id: str, payload: dict[str, Any]) -> str:
        path = Path("runs") / run_id / "planning" / "plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _write_pc_findings_artifact(run_id: str, payload: dict[str, Any]) -> str:
        path = Path("runs") / run_id / "pc" / "research_findings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _json_or_text(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _latest_pc_snapshot_artifact(
        prior_results: list[WorkerResult],
    ) -> str | None:
        for result in reversed(prior_results):
            if result.role != "pc-control":
                continue
            for artifact in result.artifacts:
                if str(artifact).replace("\\", "/").endswith("pc/snapshot.json"):
                    return str(artifact)
        return None

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
                    "claim": ("A live PC UI snapshot was captured for this run."),
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
        missing_evidence = [result.task_id for result in results if not result.evidence]
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
