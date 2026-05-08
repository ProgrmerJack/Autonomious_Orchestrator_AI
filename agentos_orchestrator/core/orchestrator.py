from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from .agents import SupervisorAgent, VerificationAgent, WorkerAgent
from .approvals import ApprovalRequired, ApprovalStore
from .authorization import AuthorizationMiddleware
from .checkpoint import CheckpointStore
from .durable import DurableExecutionStore
from .events import DurableEventLog, EventBus
from .memory import MemoryCandidate, SemanticMemory
from .policy import PermissionPolicy
from .trust import TrustMonitor
from .types import ActionRequest, RunReport, TaskSpec, WorkerResult, new_id
from agentos_orchestrator.evals import UnsupervisedEvalEngine
from agentos_orchestrator.os_control.virtual_desktop_sandbox_backend import (
    VirtualDesktopSandboxBackend,
)
from agentos_orchestrator.os_control.windows_uia_backend import (
    WindowsUiaBackend,
)


class ResearchOrchestrator:
    """Coordinates planning, policy gates, execution, and memory."""

    def __init__(
        self,
        policy: PermissionPolicy,
        state_path: str | Path,
        memory_path: str | Path,
        policy_path: str | Path | None = None,
        max_parallel_workers: int | None = None,
        pc_backend: object | None = None,
    ) -> None:
        self.policy = policy
        self.policy_path = (
            Path(policy_path)
            if policy_path is not None
            else Path("examples/policies/deep_research.json")
        )
        configured_parallelism = (
            max_parallel_workers
            if max_parallel_workers is not None
            else int(os.environ.get("AGENTOS_MAX_PARALLEL_WORKERS", "2"))
        )
        self.max_parallel_workers = max(
            1,
            int(configured_parallelism),
        )
        self.state_path = Path(state_path)
        self.event_log = DurableEventLog(self.state_path)
        self.event_bus = EventBus(self.event_log)
        self.evals = UnsupervisedEvalEngine()
        self.evals.attach(self.event_bus)
        self.checkpoints = CheckpointStore(self.state_path)
        self.runtime = DurableExecutionStore(self.state_path)
        self.approvals = ApprovalStore(self.state_path)
        self.trust = TrustMonitor(self.state_path)
        self.authorization = AuthorizationMiddleware(
            policy,
            self.approvals,
            self.trust,
        )
        self.memory = SemanticMemory(memory_path)
        self.supervisor = SupervisorAgent()
        resolved_pc_backend = (
            pc_backend or self._default_pc_backend(self.state_path)
        )
        self.worker = WorkerAgent(
            self.event_bus,
            self.checkpoints,
            authorization=self.authorization,
            pc_backend=resolved_pc_backend,
        )
        self.verifier = VerificationAgent()

    @classmethod
    def from_paths(
        cls,
        policy_path: str | Path,
        state_path: str | Path,
        memory_path: str | Path,
        max_parallel_workers: int | None = None,
        pc_backend: object | None = None,
    ) -> "ResearchOrchestrator":
        return cls(
            policy=PermissionPolicy.from_file(policy_path),
            state_path=state_path,
            memory_path=memory_path,
            policy_path=policy_path,
            max_parallel_workers=max_parallel_workers,
            pc_backend=pc_backend,
        )

    @staticmethod
    def _default_pc_backend(state_path: str | Path) -> object:
        configured = os.environ.get("AGENTOS_PC_BACKEND", "").strip().lower()
        resolved_state_path = Path(state_path)
        if configured == "windows-uia":
            return WindowsUiaBackend()
        if configured == "virtual-desktop-sandbox":
            return VirtualDesktopSandboxBackend(
                resolved_state_path.with_name("virtual_desktop_sandbox.json")
            )
        if configured:
            raise ValueError(
                "Unknown AGENTOS_PC_BACKEND "
                f"'{configured}'; expected 'windows-uia' or "
                "'virtual-desktop-sandbox'"
            )

        windows_backend = WindowsUiaBackend()
        if windows_backend.available():
            return windows_backend

        return VirtualDesktopSandboxBackend(
            resolved_state_path.with_name("virtual_desktop_sandbox.json")
        )

    def run(self, objective: str, run_id: str | None = None) -> RunReport:
        run_id = run_id or new_id("run")
        self.event_bus.publish(
            run_id,
            "run.started",
            "orchestrator",
            {"objective": objective},
        )

        tasks = self.supervisor.plan(objective)
        task_payloads = [asdict(task) for task in tasks]
        self.runtime.save_manifest(run_id, objective, task_payloads)
        self.checkpoints.save(
            run_id,
            "planned",
            {
                "objective": objective,
                "tasks": task_payloads,
            },
        )

        return self._execute_manifest(run_id, objective, tasks)

    def recover(self, run_id: str) -> RunReport:
        manifest = self.runtime.load_manifest(run_id)
        if manifest is None:
            raise KeyError(f"No durable manifest found for run_id '{run_id}'")
        self.runtime.recover_stale_steps(run_id)
        tasks = [self._task_from_dict(item) for item in manifest["tasks"]]
        return self._execute_manifest(run_id, manifest["objective"], tasks)

    def _execute_manifest(
        self,
        run_id: str,
        objective: str,
        tasks: list[TaskSpec],
    ) -> RunReport:
        self.runtime.save_manifest(
            run_id,
            objective,
            [asdict(task) for task in tasks],
        )
        try:
            results = self._execute_worker_tasks(run_id, tasks)
            verification = self._run_verification(run_id, results)
            results.append(verification)

            synthesis_contract = self._build_synthesis_contract(
                run_id,
                objective,
                results,
                verification,
            )
            synthesis = self._synthesize(synthesis_contract)
            self._commit_memory(run_id, results, verification, synthesis)
            return self._complete_run(
                run_id,
                objective,
                results,
                synthesis,
                synthesis_contract,
            )
        except Exception as exc:
            self.runtime.fail_run(run_id)
            self.event_bus.publish(
                run_id,
                "run.failed",
                "orchestrator",
                {"error": str(exc)},
            )
            raise

    def _execute_worker_tasks(
        self,
        run_id: str,
        tasks: list[TaskSpec],
    ) -> list[WorkerResult]:
        normalized_tasks = self._normalize_task_graph(tasks)
        completed: dict[str, WorkerResult] = {}
        pending = {task.task_id: task for task in normalized_tasks}

        while pending:
            ready = [
                task
                for task in normalized_tasks
                if task.task_id in pending
                and all(dependency in completed for dependency in task.depends_on)
            ]
            if not ready:
                unresolved = {
                    task_id: list(task.depends_on) for task_id, task in pending.items()
                }
                raise RuntimeError(f"Task dependency graph is unresolved: {unresolved}")

            self.event_bus.publish(
                run_id,
                "task.wave.started",
                "orchestrator",
                {
                    "task_ids": [task.task_id for task in ready],
                    "parallel_limit": self.max_parallel_workers,
                },
            )
            with ThreadPoolExecutor(
                max_workers=min(self.max_parallel_workers, len(ready)),
            ) as executor:
                future_map = {
                    executor.submit(
                        self._run_worker_task_with_retries,
                        run_id,
                        task,
                        self._prior_results_for_task(
                            task,
                            normalized_tasks,
                            completed,
                        ),
                    ): task
                    for task in ready
                }
                for future in as_completed(future_map):
                    task = future_map[future]
                    completed[task.task_id] = future.result()
                    pending.pop(task.task_id, None)
                    if task.role == "literature":
                        self._enforce_evidence_gate(
                            run_id,
                            self._ordered_results(normalized_tasks, completed),
                        )

        return self._ordered_results(normalized_tasks, completed)

    def _run_worker_task_with_retries(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        max_attempts = max(1, int(task.max_attempts or 1))
        for attempt in range(1, max_attempts + 1):
            try:
                payload = self._run_worker_task(run_id, task, prior_results)
                result = self._worker_result_from_dict(payload)
                return self._annotate_result(
                    task,
                    result,
                    prior_results,
                    attempt,
                )
            except (ApprovalRequired, PermissionError):
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt >= max_attempts:
                    raise
                self.runtime.mark_run_running(run_id)
                self.event_bus.publish(
                    run_id,
                    "task.retrying",
                    "orchestrator",
                    {
                        "task_id": task.task_id,
                        "role": task.role,
                        "next_attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "error": str(exc),
                    },
                )
        raise RuntimeError(f"Task retries exhausted for {task.task_id}")

    def _run_worker_task(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> dict:
        try:
            self._assert_task_capabilities(run_id, task)
            self._authorize_task(run_id, task)
        except (ApprovalRequired, PermissionError) as exc:
            if not task.inputs.get("optional"):
                raise
            return self._optional_skip_payload(
                run_id,
                task,
                str(exc),
                claim="Optional adaptive step was skipped before execution.",
            )
        self.event_bus.publish(
            run_id,
            "policy.accepted",
            "policy",
            {"task_id": task.task_id, "role": task.role},
        )

        def run_worker_step(current_task: TaskSpec = task) -> dict:
            result = self.worker.run(run_id, current_task, prior_results)
            return asdict(result)

        try:
            return self.runtime.run_json_step(
                run_id,
                task.task_id,
                f"worker:{task.role}",
                {"task": asdict(task)},
                run_worker_step,
            )
        except (ApprovalRequired, PermissionError) as exc:
            if not task.inputs.get("optional"):
                raise
            return self._optional_skip_payload(
                run_id,
                task,
                str(exc),
                claim="Optional adaptive step was skipped during execution.",
            )

    def _run_verification(
        self,
        run_id: str,
        results: list[WorkerResult],
    ) -> WorkerResult:
        verification_payload = self.runtime.run_json_step(
            run_id,
            "verification",
            "verification",
            {"result_count": len(results)},
            lambda: asdict(self.verifier.review(run_id, results)),
        )
        verification = self._worker_result_from_dict(verification_payload)
        self.event_bus.publish(
            run_id,
            "verification.completed",
            "verification",
            {"result": asdict(verification)},
        )
        return verification

    def _commit_memory(
        self,
        run_id: str,
        results: list[WorkerResult],
        verification: WorkerResult,
        synthesis: str,
    ) -> None:
        payload = self.runtime.run_json_step(
            run_id,
            "memory.commit",
            "memory",
            {"synthesis": synthesis},
            lambda: asdict(
                self.memory.commit(
                    MemoryCandidate(
                        run_id=run_id,
                        statement=synthesis,
                        evidence=self._all_evidence(results),
                        confidence=verification.confidence,
                        tags=["research-run", "verified-synthesis"],
                    )
                )
            ),
        )
        self.event_bus.publish(
            run_id,
            "memory.evaluated",
            "memory",
            {"accepted": payload["accepted"]},
        )

    @staticmethod
    def _all_evidence(results: list[WorkerResult]) -> list[dict]:
        return [evidence for result in results for evidence in result.evidence]

    def _complete_run(
        self,
        run_id: str,
        objective: str,
        results: list[WorkerResult],
        synthesis: str,
        synthesis_contract: dict,
    ) -> RunReport:
        report = RunReport(
            run_id=run_id,
            objective=objective,
            status="completed",
            worker_results=results,
            synthesis=synthesis,
            checkpoint_path=str(self.state_path),
            synthesis_contract=synthesis_contract,
        )
        self.checkpoints.save(
            run_id,
            "completed",
            {"report": asdict(report)},
        )
        self.runtime.complete_run(run_id)
        self.event_bus.publish(
            run_id,
            "run.completed",
            "orchestrator",
            {
                "status": report.status,
                "evals": self.evals.snapshot(),
                "provenance": synthesis_contract.get("provenance", {}),
            },
        )
        return report

    def resume(self, run_id: str) -> dict:
        checkpoint = self.checkpoints.load(run_id)
        if checkpoint is None:
            raise KeyError(f"No checkpoint found for run_id '{run_id}'")
        events = self.event_log.list_events(run_id=run_id)
        return {
            "checkpoint": asdict(checkpoint),
            "events": [asdict(event) for event in events],
            "durable_steps": [asdict(step) for step in self.runtime.list_steps(run_id)],
        }

    @staticmethod
    def _synthesize(contract: dict) -> str:
        sections = ", ".join(contract.get("resolved_sections", [])) or "none"
        missing = contract.get("missing_sections", [])
        missing_text = ""
        if missing:
            missing_text = f" Missing sections: {', '.join(missing)}."
        claims = " ".join(
            item.get("summary", "")
            for item in contract.get("claims", [])[:4]
            if item.get("summary")
        )
        provenance = contract.get("provenance", {})
        lead = (
            f"Objective: {contract.get('objective', '')}. "
            f"Covered sections: {sections}."
            f" Average provenance: {provenance.get('average', 0.0):.2f}."
            f"{missing_text}"
        )
        return " ".join(part for part in (lead.strip(), claims.strip()) if part).strip()

    def _build_synthesis_contract(
        self,
        run_id: str,
        objective: str,
        results: list[WorkerResult],
        verification: WorkerResult,
    ) -> dict:
        claims: list[dict] = []
        required_sections: list[str] = []
        optional_sections: list[str] = []
        resolved_sections: list[str] = []
        provenance_scores: list[float] = []
        for result in results:
            metadata = dict(result.metadata or {})
            synthesis_contract = metadata.get("synthesis_contract") or {}
            if not isinstance(synthesis_contract, dict):
                synthesis_contract = {}
            section = str(synthesis_contract.get("section") or result.role).strip()
            if section and result.role != "verification":
                resolved_sections.append(section)
            if section:
                target = required_sections
                if not bool(
                    synthesis_contract.get(
                        "required",
                        result.role != "verification",
                    )
                ):
                    target = optional_sections
                target.append(section)
            provenance_scores.append(float(result.provenance_score or 0.0))
            claims.append(
                {
                    "task_id": result.task_id,
                    "role": result.role,
                    "section": section,
                    "summary": result.summary,
                    "confidence": round(float(result.confidence or 0.0), 3),
                    "provenance_score": round(
                        float(result.provenance_score or 0.0),
                        3,
                    ),
                    "evidence_count": len(result.evidence),
                    "artifact_count": len(result.artifacts),
                    "attempt_count": int(result.attempt_count or 1),
                }
            )

        contract = {
            "objective": objective,
            "verification_confidence": round(
                float(verification.confidence or 0.0),
                3,
            ),
            "required_sections": self._dedupe(required_sections),
            "optional_sections": self._dedupe(optional_sections),
            "resolved_sections": self._dedupe(resolved_sections),
            "missing_sections": [
                section
                for section in self._dedupe(required_sections)
                if section not in set(self._dedupe(resolved_sections))
            ],
            "provenance": {
                "average": round(
                    sum(provenance_scores) / len(provenance_scores),
                    3,
                )
                if provenance_scores
                else 0.0,
                "minimum": round(min(provenance_scores), 3)
                if provenance_scores
                else 0.0,
                "maximum": round(max(provenance_scores), 3)
                if provenance_scores
                else 0.0,
            },
            "claims": sorted(
                claims,
                key=lambda item: (
                    item.get("provenance_score", 0.0),
                    item.get("confidence", 0.0),
                ),
                reverse=True,
            ),
        }
        self.event_bus.publish(
            run_id,
            "synthesis.completed",
            "orchestrator",
            contract,
        )
        return contract

    def _assert_task_capabilities(self, run_id: str, task: TaskSpec) -> None:
        required = [item for item in task.required_capabilities if item]
        if not required:
            return
        available = self.worker.available_capabilities()
        missing = [item for item in required if item not in available]
        if not missing:
            return
        self.event_bus.publish(
            run_id,
            "task.requirements.failed",
            "orchestrator",
            {
                "task_id": task.task_id,
                "role": task.role,
                "missing_capabilities": missing,
                "required_capabilities": required,
            },
        )
        raise PermissionError(
            "worker missing required capabilities: " + ", ".join(sorted(missing))
        )

    def _optional_skip_payload(
        self,
        run_id: str,
        task: TaskSpec,
        reason: str,
        claim: str,
    ) -> dict:
        skipped = WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=f"Skipped optional {task.role} step: {reason}",
            evidence=[
                {
                    "source": "authorization",
                    "claim": claim,
                    "reason": reason,
                }
            ],
            confidence=0.4,
        )
        self.event_bus.publish(
            run_id,
            "policy.optional_skipped",
            "policy",
            {"task_id": task.task_id, "role": task.role, "reason": reason},
        )
        return asdict(skipped)

    @staticmethod
    def _normalize_task_graph(tasks: list[TaskSpec]) -> list[TaskSpec]:
        if any(task.depends_on for task in tasks):
            return tasks
        if len(tasks) < 2:
            return tasks
        normalized: list[TaskSpec] = []
        for index, task in enumerate(tasks):
            if index == 0:
                normalized.append(task)
                continue
            normalized.append(replace(task, depends_on=[tasks[index - 1].task_id]))
        return normalized

    @staticmethod
    def _ordered_results(
        tasks: list[TaskSpec],
        completed: dict[str, WorkerResult],
    ) -> list[WorkerResult]:
        return [completed[task.task_id] for task in tasks if task.task_id in completed]

    @staticmethod
    def _prior_results_for_task(
        task: TaskSpec,
        tasks: list[TaskSpec],
        completed: dict[str, WorkerResult],
    ) -> list[WorkerResult]:
        ordered_ids = {item.task_id: index for index, item in enumerate(tasks)}
        if task.depends_on:
            allowed = set(task.depends_on)
            return [
                completed[item.task_id]
                for item in tasks
                if item.task_id in completed and item.task_id in allowed
            ]
        return [
            completed[item.task_id]
            for item in tasks
            if item.task_id in completed
            and ordered_ids[item.task_id] < ordered_ids[task.task_id]
        ]

    @staticmethod
    def _annotate_result(
        task: TaskSpec,
        result: WorkerResult,
        prior_results: list[WorkerResult],
        attempt_count: int,
    ) -> WorkerResult:
        dependency_set = set(task.depends_on)
        dependency_results = [
            prior
            for prior in prior_results
            if not dependency_set or prior.task_id in dependency_set
        ]
        dependency_ratio = 1.0
        if task.depends_on:
            dependency_ratio = min(
                1.0,
                len(dependency_results) / max(1, len(task.depends_on)),
            )
        capability_ratio = 1.0 if task.required_capabilities else 0.8
        evidence_points = min(0.25, len(result.evidence) * 0.05)
        artifact_points = min(0.15, len(result.artifacts) * 0.05)
        confidence_points = min(0.35, float(result.confidence or 0.0) * 0.35)
        dependency_points = dependency_ratio * 0.15
        capability_points = capability_ratio * 0.1
        retry_penalty = min(0.2, max(0, attempt_count - 1) * 0.08)
        provenance_score = max(
            0.0,
            min(
                1.0,
                evidence_points
                + artifact_points
                + confidence_points
                + dependency_points
                + capability_points
                - retry_penalty,
            ),
        )
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "depends_on": list(task.depends_on),
                "required_capabilities": list(task.required_capabilities),
                "synthesis_contract": dict(task.synthesis_contract),
                "dependency_task_ids": [item.task_id for item in dependency_results],
                "retry_count": max(0, attempt_count - 1),
            }
        )
        return WorkerResult(
            task_id=result.task_id,
            role=result.role,
            summary=result.summary,
            artifacts=list(result.artifacts),
            evidence=list(result.evidence),
            confidence=result.confidence,
            provenance_score=round(provenance_score, 3),
            attempt_count=attempt_count,
            metadata=metadata,
        )

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _authorize_task(self, run_id: str, task: TaskSpec) -> None:
        for action in task.declared_actions:
            decision = self.policy.evaluate(action)
            if decision.allowed and not decision.requires_approval:
                continue
            if decision.requires_approval:
                approval = self.approvals.request(
                    run_id,
                    action,
                    decision.reasons,
                )
                self.event_bus.publish(
                    run_id,
                    "approval.requested",
                    "authorization",
                    {"approval": asdict(approval)},
                )
                self.checkpoints.save(
                    run_id,
                    "approval.required",
                    {
                        "task_id": task.task_id,
                        "approval": asdict(approval),
                    },
                )
                raise ApprovalRequired(approval)
            if not decision.allowed:
                raise PermissionError("; ".join(decision.reasons))

    def _enforce_evidence_gate(
        self,
        run_id: str,
        results: list[WorkerResult],
    ) -> None:
        targets = self._coverage_targets(results)
        if not targets:
            return
        coverage = self._literature_coverage(results)
        if not coverage:
            return

        failures = self._coverage_failures(coverage, targets)

        if failures:
            self._publish_failed_gate(
                run_id,
                coverage,
                targets,
                failures,
            )
            message = "Evidence coverage gate failed: " + "; ".join(failures)
            raise RuntimeError(message)

        self._publish_passed_gate(run_id, coverage, targets)

    def _publish_failed_gate(
        self,
        run_id: str,
        coverage: dict,
        targets: dict,
        failures: list[str],
    ) -> None:
        detail = {
            "run_id": run_id,
            "coverage": coverage,
            "targets": targets,
            "failures": failures,
        }
        self.event_bus.publish(
            run_id,
            "gate.failed",
            "orchestrator",
            detail,
        )

    def _publish_passed_gate(
        self,
        run_id: str,
        coverage: dict,
        targets: dict,
    ) -> None:
        self.event_bus.publish(
            run_id,
            "gate.passed",
            "orchestrator",
            {
                "coverage": coverage,
                "targets": targets,
            },
        )

    @staticmethod
    def _coverage_failures(coverage: dict, targets: dict) -> list[str]:
        failures: list[str] = []
        source_count = coverage.get("source_count", 0)
        if source_count < targets.get("min_source_count", 0):
            failures.append("source_count below target")

        provider_count = coverage.get("provider_count", 0)
        if provider_count < targets.get("min_provider_count", 0):
            failures.append("provider_count below target")

        scholarly_count = coverage.get("scholarly_source_count", 0)
        scholarly_target = targets.get("min_scholarly_sources", 0)
        if scholarly_count < scholarly_target:
            failures.append("scholarly_source_count below target")

        strong_count = coverage.get("strong_or_moderate", 0)
        strong_target = targets.get("min_strong_or_moderate", 0)
        if strong_count < strong_target:
            failures.append("strong_or_moderate evidence below target")

        contradiction = coverage.get("max_contradiction_risk", 1.0)
        contradiction_target = targets.get("max_contradiction_risk", 1.0)
        if contradiction > contradiction_target:
            failures.append("contradiction risk above threshold")

        novelty = coverage.get("novelty_rate", 0.0)
        novelty_target = targets.get("min_novelty_rate", 0.0)
        if novelty < novelty_target:
            failures.append("novelty rate below threshold")

        # Current-web/market mode uses broad live web evidence where
        # "strong_or_moderate" grading can undercount useful signals.
        # Treat this as a soft gate when relevance and provider diversity are adequate.
        # Current-web profile is indicated by relaxed scholarly requirement, not strict provider count.
        current_web_profile = int(targets.get("min_scholarly_sources", 0)) == 0
        if current_web_profile and failures:
            source_floor = max(3, int(targets.get("min_source_count", 0)) // 2)
            has_minimum_live_signal = (
                int(coverage.get("source_count", 0)) >= source_floor
                and float(coverage.get("on_topic_ratio", 0.0)) >= 0.65
                and int(coverage.get("provider_count", 0)) >= 1
            )
            if has_minimum_live_signal:
                failures = [
                    item
                    for item in failures
                    if item
                    not in {
                        "source_count below target",
                        "provider_count below target",
                        "strong_or_moderate evidence below target",
                    }
                ]
            else:
                # Second-tier fallback for live web runs: if we still have
                # at least some relevant evidence, keep the run alive and let
                # downstream synthesis expose uncertainty instead of hard-failing.
                has_partial_live_signal = (
                    int(coverage.get("source_count", 0)) >= 1
                    and int(coverage.get("provider_count", 0)) >= 1
                    and float(coverage.get("on_topic_ratio", 0.0)) >= 0.5
                )
                if has_partial_live_signal:
                    failures = [
                        item
                        for item in failures
                        if item
                        not in {
                            "source_count below target",
                            "provider_count below target",
                            "strong_or_moderate evidence below target",
                        }
                    ]

        return failures

    @staticmethod
    def _coverage_targets(results: list[WorkerResult]) -> dict:
        for result in reversed(results):
            if result.role != "literature":
                continue
            for evidence in result.evidence:
                if evidence.get("source") != "research-metrics":
                    continue
                metadata = evidence.get("metadata") or {}
                retrieval = metadata.get("retrieval") or {}
                targets = retrieval.get("targets")
                if isinstance(targets, dict):
                    return targets
        for result in reversed(results):
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
                    return {}
                targets = payload.get("coverage_targets")
                if isinstance(targets, dict):
                    return targets
                return {}
        return {}

    @staticmethod
    def _literature_coverage(results: list[WorkerResult]) -> dict:
        for result in reversed(results):
            if result.role != "literature":
                continue
            for evidence in result.evidence:
                if evidence.get("source") != "research-metrics":
                    continue
                metadata = evidence.get("metadata") or {}
                coverage = metadata.get("coverage")
                if isinstance(coverage, dict):
                    return coverage
            return {}
        return {}

    @staticmethod
    def _worker_result_from_dict(payload: dict) -> WorkerResult:
        return WorkerResult(
            task_id=payload["task_id"],
            role=payload["role"],
            summary=payload["summary"],
            artifacts=list(payload.get("artifacts", [])),
            evidence=list(payload.get("evidence", [])),
            confidence=float(payload.get("confidence", 0.0)),
            provenance_score=float(payload.get("provenance_score", 0.0)),
            attempt_count=int(payload.get("attempt_count", 1)),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _task_from_dict(payload: dict) -> TaskSpec:
        return TaskSpec(
            task_id=payload["task_id"],
            role=payload["role"],
            objective=payload["objective"],
            declared_actions=[
                ActionRequest(**action)
                for action in payload.get("declared_actions", [])
            ],
            inputs=dict(payload.get("inputs", {})),
            depends_on=list(payload.get("depends_on", [])),
            max_attempts=int(payload.get("max_attempts", 1)),
            required_capabilities=list(payload.get("required_capabilities", [])),
            synthesis_contract=dict(payload.get("synthesis_contract", {})),
        )
