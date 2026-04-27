from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


class DesktopWorkflowService:
    """Plan and execute desktop workflows via planner + writer modules."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.planner = DesktopWorkflowPlanner()
        self.writer = WorkflowArtifactWriter(workspace_root)
        self.reasoner = DesktopWorkflowReasoner()
        self.max_adaptive_steps = 4
        self._universal_agent: Any | None = None

    def enable_universal_mode(
        self,
        backend: Any,
        use_mcts: bool = True,
        use_active_inference: bool = True,
        use_vla: bool = True,
        max_steps: int = 30,
    ) -> None:
        """Enable the cognitive-architecture universal agent.

        When enabled, execute() will first run the planned steps, then
        hand off to the UniversalDesktopAgent for arbitrary-task handling.
        """
        from agentos_orchestrator.cognition.universal_agent import (
            UniversalDesktopAgent,
        )

        self._universal_agent = UniversalDesktopAgent(
            backend=backend,
            workspace_root=str(self.writer.workspace_root),
            use_mcts=use_mcts,
            use_active_inference=use_active_inference,
            use_vla=use_vla,
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
            receipt, final_selector, recovery = self._perform_with_recovery(
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
                }
            )
        self._execute_adaptive_steps(
            objective,
            plan,
            backend,
            receipts,
        )
        # Universal agent cognitive loop: handles arbitrary tasks beyond templates
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
    ) -> None:
        for _ in range(self.max_adaptive_steps):
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
            receipt, final_selector, recovery = self._perform_with_recovery(
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
                    "adaptive": True,
                    "reasoner": decision.engine,
                    "rationale": decision.rationale,
                }
            )

    def _perform_with_recovery(
        self,
        backend: Any,
        step: Any,
    ) -> tuple[str, str, dict[str, Any]]:
        recovery = self._empty_recovery()
        receipt, selector = self._perform_with_selector(
            backend,
            step,
            recovery,
        )
        return receipt, selector, recovery

    def _perform_with_selector(
        self,
        backend: Any,
        step: Any,
        recovery: dict[str, Any],
    ) -> tuple[str, str]:
        selector = self._preflight_selector(backend, step.selector, recovery)
        receipt = self._perform_step(backend, step, selector)
        selector, receipt = self._retry_with_fallback(
            backend,
            step,
            selector,
            receipt,
            recovery,
        )
        return receipt, selector

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
        return backend.perform(
            UiAction(
                action_type=step.action_type,
                selector=selector,
                value=step.value,
                metadata=step.metadata,
            )
        )

    def _retry_with_fallback(
        self,
        backend: Any,
        step: Any,
        selector: str,
        receipt: str,
        recovery: dict[str, Any],
    ) -> tuple[str, str]:
        payload = self._json_or_text(receipt)
        if not self._needs_retry(payload):
            return selector, receipt
        retry_selector = self._suggest_selector(backend, step.selector)
        if retry_selector is None or retry_selector == selector:
            return selector, receipt
        recovery.update(
            {
                "attempted": True,
                "applied": True,
                "reason": "retry-fallback",
                "from": step.selector,
                "to": retry_selector,
            }
        )
        retry_receipt = self._perform_step(
            backend,
            step,
            retry_selector,
        )
        return retry_selector, retry_receipt

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
