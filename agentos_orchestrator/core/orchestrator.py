from __future__ import annotations

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
        self.worker = WorkerAgent(self.event_bus, self.checkpoints)
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

    def run(self, objective: str) -> RunReport:
        run_id = new_id("run")
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

        return self._run_manifest(run_id, objective, tasks)

    def recover(self, run_id: str) -> RunReport:
        manifest = self.runtime.load_manifest(run_id)
        if manifest is None:
            raise KeyError(f"No durable manifest found for run_id '{run_id}'")
        self.runtime.recover_stale_steps(run_id)
        tasks = [self._task_from_dict(item) for item in manifest["tasks"]]
        return self._run_manifest(run_id, manifest["objective"], tasks)

    def _run_manifest(
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

        results: list[WorkerResult] = []
        for task in tasks:
            self._authorize_task(run_id, task)
            self.event_bus.publish(
                run_id,
                "policy.accepted",
                "policy",
                {"task_id": task.task_id, "role": task.role},
            )

            def run_worker_step(current_task: TaskSpec = task) -> dict:
                return asdict(self.worker.run(run_id, current_task, results))

            result_payload = self.runtime.run_json_step(
                run_id,
                task.task_id,
                f"worker:{task.role}",
                {"task": asdict(task)},
                run_worker_step,
            )
            results.append(self._worker_result_from_dict(result_payload))

        verification_payload = self.runtime.run_json_step(
            run_id,
            "verification",
            "verification",
            {"result_count": len(results)},
            lambda: asdict(self.verifier.review(run_id, results)),
        )
        verification = self._worker_result_from_dict(verification_payload)
        results.append(verification)
        self.event_bus.publish(
            run_id,
            "verification.completed",
            "verification",
            {"result": asdict(verification)},
        )

        synthesis = self._synthesize(results)
        memory_payload = self.runtime.run_json_step(
            run_id,
            "memory.commit",
            "memory",
            {"synthesis": synthesis},
            lambda: asdict(
                self.memory.commit(
                    MemoryCandidate(
                        run_id=run_id,
                        statement=synthesis,
                        evidence=[
                            evidence
                            for result in results
                            for evidence in result.evidence
                        ],
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
            {"accepted": memory_payload["accepted"]},
        )

        report = RunReport(
            run_id=run_id,
            objective=objective,
            status="completed",
            worker_results=results,
            synthesis=synthesis,
            checkpoint_path=str(self.state_path),
        )
        self.checkpoints.save(run_id, "completed", {"report": asdict(report)})
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
            "durable_steps": [
                asdict(step) for step in self.runtime.list_steps(run_id)
            ],
        }

    @staticmethod
    def _synthesize(results: list[WorkerResult]) -> str:
        accepted = [result.summary for result in results]
        return " ".join(accepted)

    def _authorize_task(self, run_id: str, task: TaskSpec) -> None:
        for action in task.declared_actions:
            decision = self.authorization.authorize(run_id, action)
            if decision.allowed:
                continue
            if decision.requires_approval and decision.approval is not None:
                self.event_bus.publish(
                    run_id,
                    "approval.requested",
                    "authorization",
                    {"approval": asdict(decision.approval)},
                )
                self.checkpoints.save(
                    run_id,
                    "approval.required",
                    {
                        "task_id": task.task_id,
                        "approval": asdict(decision.approval),
                    },
                )
                raise ApprovalRequired(decision.approval)
            raise PermissionError("; ".join(decision.reasons))

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
