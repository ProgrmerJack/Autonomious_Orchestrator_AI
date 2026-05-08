from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentos_orchestrator.cognition.live_fire_eval_recipes import (
    abstract_state,
    snapshot_nodes,
)
from agentos_orchestrator.cognition.verification_contracts import (
    VerificationContract,
    ensure_verification_contract,
    verify_action_contract,
)
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.os_control.workflow.artifact_writer import (
    WorkflowArtifactWriter,
)
from agentos_orchestrator.os_control.workflow.planner import (
    DesktopWorkflowPlanner,
)
from agentos_orchestrator.os_control.workflow.reasoner import (
    DesktopWorkflowReasoner,
)


class WorkflowVerificationError(RuntimeError):
    def __init__(
        self,
        *,
        action: UiAction,
        receipt: str,
        verification: dict[str, Any],
        recovery: dict[str, Any],
    ) -> None:
        self.action = action
        self.receipt = receipt
        self.verification = verification
        self.recovery = recovery
        reason = str(verification.get("reason") or "workflow verification failed")
        super().__init__(reason)

    def asdict(self) -> dict[str, Any]:
        return {
            "action_type": self.action.action_type,
            "selector": self.action.selector,
            "value": self.action.value,
            "verification": dict(self.verification),
            "recovery": dict(self.recovery),
            "receipt": DesktopWorkflowService._json_or_text(self.receipt),
        }


class DesktopWorkflowService:
    """Plan and execute desktop workflows via planner + writer modules."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.planner = DesktopWorkflowPlanner()
        self.writer = WorkflowArtifactWriter(workspace_root)
        self.reasoner = DesktopWorkflowReasoner()
        self.max_adaptive_steps = 4
        self._universal_agent: Any | None = None
        self._universal_backend_id: int | None = None

    def enable_universal_mode(
        self,
        backend: Any,
        use_mcts: bool = True,
        use_active_inference: bool = True,
        use_vla: bool = True,
        max_steps: int = 30,
    ) -> None:
        """Enable the production universal agent path.

        The legacy method name is preserved for compatibility, but the runtime
        now binds V2 only so the workflow service has a single cognitive path.
        """
        del use_mcts, use_active_inference
        self.enable_universal_mode_v2(
            backend,
            use_local_vla=use_vla,
            max_steps=max_steps,
        )

    def enable_universal_mode_v2(
        self,
        backend: Any,
        use_learned_model: bool = True,
        use_local_vla: bool = True,
        use_semantic_memory: bool = True,
        use_pixel_pomdp: bool = False,
        max_steps: int = 30,
        target_latency_ms: float = 100.0,
    ) -> None:
        """Enable Universal Desktop Agent v2 with real learned components.

        Uses:
        - Learned generative world model (MLP dynamics, online training)
        - Local fast VLA (zero API latency, <100ms loop)
        - Semantic dense memory (TF-IDF+SVD, captures meaning)
        - Pure pixel-based POMDP (no accessibility tree dependency)
        """
        from agentos_orchestrator.cognition.universal_agent_v2 import (
            UniversalDesktopAgentV2,
        )

        self._universal_agent = UniversalDesktopAgentV2(
            backend=backend,
            workspace_root=str(self.writer.workspace_root),
            use_learned_model=use_learned_model,
            use_local_vla=use_local_vla,
            use_semantic_memory=use_semantic_memory,
            use_pixel_pomdp=use_pixel_pomdp,
            max_steps=max_steps,
            target_latency_ms=target_latency_ms,
        )
        self._universal_backend_id = id(backend)

    def ensure_universal_mode(self, backend: Any, max_steps: int = 12) -> None:
        if self._universal_agent is not None and self._universal_backend_id == id(
            backend
        ):
            return
        self.enable_universal_mode_v2(backend, max_steps=max_steps)

    def plan(self, objective: str):
        return self.planner.plan(objective)

    def execute(self, objective: str, backend: Any) -> dict[str, Any]:
        plan = self.plan(objective)
        if plan.requires_clarification:
            return {
                "status": "clarification_required",
                "plan": plan.asdict(),
                "artifacts": [],
                "receipts": [],
            }
        written = self.writer.materialize(plan)
        receipts: list[dict[str, Any]] = []
        for step in plan.steps:
            (
                receipt,
                final_selector,
                recovery,
                verification,
            ) = self._perform_with_recovery(
                backend,
                step,
            )
            receipts.append(
                {
                    "action_type": step.action_type,
                    "selector": final_selector,
                    "description": step.description,
                    "receipt": self._json_or_text(receipt),
                    "recovery": recovery,
                    "verification": verification,
                }
            )
        self._execute_adaptive_steps(
            objective,
            plan,
            backend,
            receipts,
            step_budget=self._adaptive_step_budget(objective, plan),
        )
        # Universal agent cognitive loop: handles arbitrary tasks through state feedback.
        if self._universal_agent is not None:
            try:
                run = self._universal_agent.run_with_planned_bootstrap(objective, plan)
                receipt: dict[str, Any] = {
                    "run_id": run.run_id,
                    "success": run.success,
                    "steps": len(run.steps),
                    "exploration_probes": run.exploration_probes_used,
                    "mcts_simulations": run.mcts_simulations_run,
                    "final_state": run.final_state,
                }
                # V2 enriched fields
                if hasattr(run, "avg_latency_ms"):
                    receipt["avg_latency_ms"] = run.avg_latency_ms
                if hasattr(run, "model_used_ratio"):
                    receipt["model_used_ratio"] = run.model_used_ratio
                if hasattr(run, "adaptive_steps_used"):
                    receipt["adaptive_steps_used"] = run.adaptive_steps_used
                receipts.append(
                    {
                        "action_type": "universal_agent_run",
                        "selector": "cognitive-orchestrator",
                        "description": "Cognitive architecture universal agent execution",
                        "receipt": receipt,
                        "recovery": {"applied": False},
                        "adaptive": True,
                        "reasoner": "universal_cognitive_architecture",
                        "rationale": "System-2 deliberative reasoning with MCTS, POMDP, and active inference",
                    }
                )
            except Exception as exc:
                receipts.append(
                    {
                        "action_type": "universal_agent_run",
                        "selector": "cognitive-orchestrator",
                        "description": "Cognitive architecture execution failed",
                        "receipt": {"error": str(exc)},
                        "recovery": {"applied": False},
                        "adaptive": True,
                        "reasoner": "universal_cognitive_architecture",
                        "rationale": f"Universal agent error: {exc}",
                    }
                )
        return {
            "plan": plan.asdict(),
            "artifacts": written,
            "receipts": receipts,
        }

    def _execute_adaptive_steps(
        self,
        objective: str,
        plan: Any,
        backend: Any,
        receipts: list[dict[str, Any]],
        step_budget: int | None = None,
    ) -> None:
        budget = int(
            step_budget if step_budget is not None else self.max_adaptive_steps
        )
        budget = max(1, min(12, budget))
        for _ in range(budget):
            try:
                nodes = backend.snapshot()
            except (AttributeError, RuntimeError, ValueError, OSError):
                return
            decision = self.reasoner.next_decision(
                objective,
                plan,
                nodes,
                receipts,
            )
            if decision.done or decision.step is None:
                return
            (
                receipt,
                final_selector,
                recovery,
                verification,
            ) = self._perform_with_recovery(
                backend,
                decision.step,
            )
            receipts.append(
                {
                    "action_type": decision.step.action_type,
                    "selector": final_selector,
                    "description": decision.step.description,
                    "receipt": self._json_or_text(receipt),
                    "recovery": recovery,
                    "verification": verification,
                    "adaptive": True,
                    "reasoner": decision.engine,
                    "rationale": decision.rationale,
                }
            )

    def _adaptive_step_budget(self, objective: str, plan: Any) -> int:
        words = len(str(objective or "").split())
        sub_task_count = len(getattr(plan, "sub_tasks", []) or [])
        mode = str(getattr(plan, "mode", "") or "").lower()
        mode_boost = 2 if mode in {"script", "report", "spreadsheet", "drawing"} else 1
        complexity = words // 8 + sub_task_count + mode_boost
        return max(self.max_adaptive_steps, min(12, 2 + complexity))

    def _perform_with_recovery(
        self,
        backend: Any,
        step: Any,
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        before_nodes = snapshot_nodes(backend)
        before = abstract_state("unknown", before_nodes)
        recovery = self._empty_recovery()
        receipt, selector, action = self._perform_with_selector(
            backend,
            step,
            recovery,
        )
        after_nodes = snapshot_nodes(backend)
        after = abstract_state("unknown", after_nodes)
        verification = self._verify_step(action, before, after, receipt, recovery)
        return receipt, selector, recovery, verification

    def _perform_with_selector(
        self,
        backend: Any,
        step: Any,
        recovery: dict[str, Any],
    ) -> tuple[str, str, UiAction]:
        selector = step.selector
        if step.action_type != "launch_app":
            selector = self._preflight_selector(backend, step.selector, recovery)
        action = self._workflow_action(step, selector)
        receipt = backend.perform(action)
        selector, receipt, action = self._retry_with_fallback(
            backend,
            step,
            selector,
            receipt,
            recovery,
            action,
        )
        return receipt, selector, action

    @staticmethod
    def _empty_recovery() -> dict[str, Any]:
        return {
            "applied": False,
            "attempted": False,
            "reason": "",
        }

    def _preflight_selector(
        self,
        backend: Any,
        selector: str,
        recovery: dict[str, Any],
    ) -> str:
        suggested = self._suggest_selector(backend, selector)
        if suggested is not None and suggested != selector:
            recovery.update(
                {
                    "attempted": True,
                    "applied": True,
                    "reason": "preflight-fallback",
                    "from": selector,
                    "to": suggested,
                }
            )
            return suggested
        return selector

    @staticmethod
    def _perform_step(backend: Any, step: Any, selector: str) -> str:
        return backend.perform(DesktopWorkflowService._workflow_action(step, selector))

    @staticmethod
    def _workflow_action(step: Any, selector: str) -> UiAction:
        action = UiAction(
            action_type=step.action_type,
            selector=selector,
            value=step.value,
            metadata=dict(step.metadata or {}),
        )
        if "verification_contract" not in action.metadata:
            contract = DesktopWorkflowService._workflow_contract(
                step.action_type,
                selector,
            )
            if contract is not None:
                action.metadata["verification_contract"] = contract.asdict()
        ensure_verification_contract(action)
        return action

    @staticmethod
    def _workflow_contract(
        action_type: str,
        selector: str,
    ) -> VerificationContract | None:
        if action_type == "cell_edit":
            return VerificationContract(
                kind="receipt_success",
                expected="The backend receipt reports successful action execution.",
                target=selector,
                required=True,
            )
        if action_type == "draw_path":
            return VerificationContract(
                kind="receipt_success",
                expected="The backend receipt reports successful action execution.",
                target=selector,
                required=False,
            )
        if action_type not in {"type", "set_text", "set_value"}:
            return None
        lower_selector = str(selector or "").strip().lower()
        explicit_field_tokens = (
            "automation_id=1001",
            "class_name=edit",
            "role=edit",
            "file name",
            "address",
            "search",
            "text editor",
        )
        if any(token in lower_selector for token in explicit_field_tokens):
            return None
        return VerificationContract(
            kind="receipt_success",
            expected="The backend receipt reports successful text entry.",
            target=selector,
            required=False,
        )

    def _retry_with_fallback(
        self,
        backend: Any,
        step: Any,
        selector: str,
        receipt: str,
        recovery: dict[str, Any],
        action: UiAction,
    ) -> tuple[str, str, UiAction]:
        payload = self._json_or_text(receipt)
        if not self._needs_retry(payload):
            return selector, receipt, action
        retry_selector = self._suggest_selector(backend, step.selector)
        if retry_selector is None or retry_selector == selector:
            return selector, receipt, action
        recovery.update(
            {
                "attempted": True,
                "applied": True,
                "reason": "retry-fallback",
                "from": step.selector,
                "to": retry_selector,
            }
        )
        retry_action = self._workflow_action(step, retry_selector)
        retry_receipt = backend.perform(retry_action)
        return retry_selector, retry_receipt, retry_action

    @staticmethod
    def _verify_step(
        action: UiAction,
        before: Any,
        after: Any,
        receipt: str,
        recovery: dict[str, Any],
    ) -> dict[str, Any]:
        verification = verify_action_contract(action, before, after, receipt)
        if verification.required and not verification.matched:
            recovery["verification_failed"] = True
            recovery["verification_reason"] = verification.reason
            raise WorkflowVerificationError(
                action=action,
                receipt=receipt,
                verification=verification.asdict(),
                recovery=recovery,
            )
        return verification.asdict()

    @staticmethod
    def _needs_retry(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        status = str(payload.get("status") or "").lower()
        return status in {"selector-not-found", "not-found", "blocked"}

    @staticmethod
    def _suggest_selector(backend: Any, selector: str) -> str | None:
        try:
            nodes = backend.snapshot()
        except (AttributeError, RuntimeError, ValueError, OSError):
            return None
        report = debug_selector(selector, nodes, limit=5)
        if report.exact_matches == 1:
            return selector
        if not report.candidates:
            return None
        return report.candidates[0].selector

    @staticmethod
    def _json_or_text(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
