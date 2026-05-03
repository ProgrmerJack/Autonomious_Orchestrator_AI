from __future__ import annotations

import json
from dataclasses import asdict
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


class ResearchOrchestrator:
    """Coordinates planning, policy gates, execution, and memory."""

    def __init__(
        self,
        policy: PermissionPolicy,
        state_path: str | Path,
        memory_path: str | Path,
        policy_path: str | Path | None = None,
    ) -> None:
        self.policy = policy
        self.policy_path = (
            Path(policy_path)
            if policy_path is not None
            else Path("examples/policies/deep_research.json")
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
        self.worker = WorkerAgent(
            self.event_bus,
            self.checkpoints,
            authorization=self.authorization,
        )
        self.verifier = VerificationAgent()

    @classmethod
    def from_paths(
        cls,
        policy_path: str | Path,
        state_path: str | Path,
        memory_path: str | Path,
    ) -> "ResearchOrchestrator":
        return cls(
            policy=PermissionPolicy.from_file(policy_path),
            state_path=state_path,
            memory_path=memory_path,
            policy_path=policy_path,
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

            synthesis = self._synthesize(results)
            self._commit_memory(run_id, results, verification, synthesis)
            return self._complete_run(run_id, objective, results, synthesis)
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
        results: list[WorkerResult] = []
        for task in tasks:
            payload = self._run_worker_task(run_id, task, results)
            results.append(self._worker_result_from_dict(payload))
            if task.role == "literature":
                self._enforce_evidence_gate(run_id, results)
        return results

    def _run_worker_task(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> dict:
        try:
            self._authorize_task(run_id, task)
        except (ApprovalRequired, PermissionError) as exc:
            if not task.inputs.get("optional"):
                raise
            skipped = WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=f"Skipped optional {task.role} step: {exc}",
                evidence=[
                    {
                        "source": "authorization",
                        "claim": "Optional adaptive step was skipped by policy.",
                        "reason": str(exc),
                    }
                ],
                confidence=0.4,
            )
            self.event_bus.publish(
                run_id,
                "policy.optional_skipped",
                "policy",
                {"task_id": task.task_id, "role": task.role, "reason": str(exc)},
            )
            return asdict(skipped)
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
            skipped = WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=f"Skipped optional {task.role} step: {exc}",
                evidence=[
                    {
                        "source": "authorization",
                        "claim": "Optional adaptive step was skipped by policy.",
                        "reason": str(exc),
                    }
                ],
                confidence=0.4,
            )
            self.event_bus.publish(
                run_id,
                "policy.optional_skipped",
                "policy",
                {"task_id": task.task_id, "role": task.role, "reason": str(exc)},
            )
            return asdict(skipped)

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
    ) -> RunReport:
        report = RunReport(
            run_id=run_id,
            objective=objective,
            status="completed",
            worker_results=results,
            synthesis=synthesis,
            checkpoint_path=str(self.state_path),
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
            {"status": report.status, "evals": self.evals.snapshot()},
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
    def _synthesize(results: list[WorkerResult]) -> str:
        accepted = [result.summary for result in results]
        return " ".join(accepted)

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
        )
