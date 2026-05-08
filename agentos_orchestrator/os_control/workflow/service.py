from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any

from agentos_orchestrator.cognition.app_agent_runtime import (
    AppAgentRuntime,
    AppAgentSession,
)
from agentos_orchestrator.cognition.live_fire_eval_recipes import (
    abstract_state,
    snapshot_nodes,
)
from agentos_orchestrator.cognition.capability_profile import (
    CapabilityProfiler,
)
from agentos_orchestrator.cognition.control_substrate import (
    AdaptiveControlLedger,
    FourLaneActionRouter,
    GoalLock,
    ObservationFrame,
    ObservationFrameBuilder,
    PreActionVerifier,
    build_goal_lock,
    enrich_action_metadata,
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
from agentos_orchestrator.os_control.workflow.programmer import ProgrammerLane
from agentos_orchestrator.os_control.workflow.planner import (
    DesktopWorkflowPlanner,
)
from agentos_orchestrator.os_control.workflow.reasoner import (
    DesktopWorkflowReasoner,
)
from agentos_orchestrator.sandbox.providers import SandboxManager, SandboxSpec


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
        reason = str(
            verification.get("reason") or "workflow verification failed"
        )
        super().__init__(reason)

    def asdict(self) -> dict[str, Any]:
        return {
            "action_type": self.action.action_type,
            "selector": self.action.selector,
            "value": self.action.value,
            "verification": dict(self.verification),
            "recovery": dict(self.recovery),
            "receipt": _json_or_text(self.receipt),
        }


class DesktopWorkflowService:
    """Plan and execute desktop workflows via planner + writer modules."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.planner = DesktopWorkflowPlanner()
        self.writer = WorkflowArtifactWriter(workspace_root)
        self.programmer_lane = ProgrammerLane(workspace_root)
        self.reasoner = DesktopWorkflowReasoner()
        self.max_adaptive_steps = 4
        self._universal_agent: Any | None = None
        self._universal_backend_id: int | None = None
        self.capability_profiler = CapabilityProfiler()
        self.app_agent_runtime = AppAgentRuntime(self.writer.workspace_root)
        self.observation_builder = ObservationFrameBuilder()
        self.action_router = FourLaneActionRouter()
        self.pre_action_verifier = PreActionVerifier()
        self.control_ledger = AdaptiveControlLedger.from_workspace(
            self.writer.workspace_root,
        )
        self.sandbox_manager = SandboxManager()
        self._last_observation_frame: ObservationFrame | None = None
        self._last_capability_profile: Any | None = None
        self._last_app_agent_session: AppAgentSession | None = None

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
        same_backend = self._universal_backend_id == id(backend)
        if self._universal_agent is not None and same_backend:
            return
        self.enable_universal_mode_v2(backend, max_steps=max_steps)

    def plan(self, objective: str):
        return self.planner.plan(objective)

    def execute(self, objective: str, backend: Any) -> dict[str, Any]:
        plan = copy.deepcopy(self.plan(objective))
        if plan.requires_clarification:
            return {
                "status": "clarification_required",
                "plan": plan.asdict(),
                "artifacts": [],
                "receipts": [],
            }
        plan = self.programmer_lane.augment_plan(plan)
        written_by_path = {
            item["path"]: item
            for item in self.writer.materialize(
                plan,
                skip_paths=self.programmer_lane.reserved_paths(plan),
            )
        }
        receipts: list[dict[str, Any]] = []
        for index, step in enumerate(plan.steps):
            (
                receipt,
                final_selector,
                recovery,
                verification,
            ) = self._perform_with_recovery(
                backend,
                step,
                objective=objective,
                remaining_steps=plan.steps[index:],
            )
            receipt_payload = self._json_or_text(receipt)
            for artifact in self._generated_artifacts(receipt_payload):
                written_by_path[artifact["path"]] = artifact
            receipts.append(
                {
                    "action_type": step.action_type,
                    "selector": final_selector,
                    "description": step.description,
                    "receipt": receipt_payload,
                    "recovery": recovery,
                    "verification": verification,
                    "control": self._control_from_recovery(recovery),
                }
            )
        self._execute_adaptive_steps(
            objective,
            plan,
            backend,
            receipts,
            step_budget=self._adaptive_step_budget(objective, plan),
        )
        # Universal agent handles arbitrary tasks through state feedback.
        if self._universal_agent is not None:
            adaptation_readiness = self._universal_adaptation_readiness()
            try:
                run = self._universal_agent.run_with_planned_bootstrap(
                    objective,
                    plan,
                )
                universal_receipt: dict[str, Any] = {
                    "run_id": run.run_id,
                    "success": run.success,
                    "steps": len(run.steps),
                    "exploration_probes": run.exploration_probes_used,
                    "mcts_simulations": run.mcts_simulations_run,
                    "final_state": run.final_state,
                    "adaptation_readiness": adaptation_readiness,
                }
                # V2 enriched fields
                if hasattr(run, "avg_latency_ms"):
                    universal_receipt["avg_latency_ms"] = run.avg_latency_ms
                if hasattr(run, "model_used_ratio"):
                    universal_receipt["model_used_ratio"] = (
                        run.model_used_ratio
                    )
                if hasattr(run, "adaptive_steps_used"):
                    universal_receipt["adaptive_steps_used"] = (
                        run.adaptive_steps_used
                    )
                receipts.append(
                    {
                        "action_type": "universal_agent_run",
                        "selector": "cognitive-orchestrator",
                        "description": (
                            "Cognitive architecture universal agent execution"
                        ),
                        "receipt": universal_receipt,
                        "recovery": {"applied": False},
                        "adaptive": True,
                        "reasoner": "universal_cognitive_architecture",
                        "rationale": (
                            "System-2 deliberative reasoning with MCTS, "
                            "POMDP, and active inference"
                        ),
                    }
                )
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                receipts.append(
                    {
                        "action_type": "universal_agent_run",
                        "selector": "cognitive-orchestrator",
                        "description": (
                            "Cognitive architecture execution failed"
                        ),
                        "receipt": {
                            "error": str(exc),
                            "adaptation_readiness": adaptation_readiness,
                        },
                        "recovery": {"applied": False},
                        "adaptive": True,
                        "reasoner": "universal_cognitive_architecture",
                        "rationale": f"Universal agent error: {exc}",
                    }
                )
        return {
            "plan": plan.asdict(),
            "artifacts": list(written_by_path.values()),
            "receipts": receipts,
        }

    def _universal_adaptation_readiness(self) -> dict[str, Any]:
        readiness = getattr(
            self._universal_agent,
            "adaptation_readiness",
            None,
        )
        if readiness is not None and hasattr(readiness, "asdict"):
            return readiness.asdict()
        from agentos_orchestrator.cognition.adaptation_readiness import (
            collect_adaptation_readiness,
        )

        return collect_adaptation_readiness(
            self.writer.workspace_root,
        ).asdict()

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
                objective=objective,
                remaining_steps=[decision.step],
            )
            receipts.append(
                {
                    "action_type": decision.step.action_type,
                    "selector": final_selector,
                    "description": decision.step.description,
                    "receipt": self._json_or_text(receipt),
                    "recovery": recovery,
                    "verification": verification,
                    "control": self._control_from_recovery(recovery),
                    "adaptive": True,
                    "reasoner": decision.engine,
                    "rationale": decision.rationale,
                }
            )

    def _adaptive_step_budget(self, objective: str, plan: Any) -> int:
        words = len(str(objective or "").split())
        sub_task_count = len(getattr(plan, "sub_tasks", []) or [])
        mode = str(getattr(plan, "mode", "") or "").lower()
        boost_modes = {"script", "report", "spreadsheet", "drawing"}
        mode_boost = 2 if mode in boost_modes else 1
        complexity = words // 8 + sub_task_count + mode_boost
        return max(self.max_adaptive_steps, min(12, 2 + complexity))

    def _perform_with_recovery(
        self,
        backend: Any,
        step: Any,
        objective: str = "",
        remaining_steps: list[Any] | None = None,
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        before_nodes = snapshot_nodes(backend)
        before = abstract_state("unknown", before_nodes)
        recovery = self._empty_recovery()
        observation = self._observation_frame(
            backend,
            before_nodes,
            before,
            objective,
        )
        try:
            receipt, selector, action = self._perform_with_selector(
                backend,
                step,
                recovery,
                observation,
                objective,
                remaining_steps,
            )
        except WorkflowVerificationError as exc:
            self._record_control_completion(
                exc.action,
                exc.receipt,
                exc.verification,
                exc.recovery,
                training_label="verification_failure",
            )
            raise
        after_nodes = snapshot_nodes(backend)
        after = abstract_state("unknown", after_nodes)
        try:
            verification = self._verify_step(
                action,
                before,
                after,
                receipt,
                recovery,
            )
        except WorkflowVerificationError as exc:
            self._record_control_completion(
                action,
                receipt,
                exc.verification,
                exc.recovery,
                training_label="verification_failure",
            )
            raise
        self._record_control_completion(
            action,
            receipt,
            verification,
            recovery,
        )
        return receipt, selector, recovery, verification

    def _perform_with_selector(
        self,
        backend: Any,
        step: Any,
        recovery: dict[str, Any],
        observation: ObservationFrame,
        objective: str,
        remaining_steps: list[Any] | None,
    ) -> tuple[str, str, UiAction]:
        selector = step.selector
        if step.action_type not in {"launch_app", "tool"}:
            selector = self._preflight_selector(
                backend,
                step.selector,
                recovery,
            )
        action = self._workflow_action(step, selector)
        app_agent = self._last_app_agent_session
        if app_agent is not None:
            action = self.app_agent_runtime.enrich_action(
                action,
                app_agent,
                objective,
            )
        goal_lock = build_goal_lock(
            objective=objective,
            observation=observation,
        )
        speculation = self._speculative_preview(
            observation,
            objective,
            remaining_steps or [step],
            app_agent,
        )
        isolation = self._isolation_plan(action, observation, app_agent)
        action = self._annotate_action_runtime(
            action,
            objective,
            goal_lock,
            speculation,
            isolation,
        )
        action = self._prepare_control_action(
            action,
            observation,
            objective,
            recovery,
            goal_lock,
            app_agent,
        )
        receipt = self._perform_action(backend, action)
        if action.action_type == "tool":
            return receipt, selector, action
        selector, receipt, action = self._retry_with_fallback(
            backend,
            step,
            selector,
            receipt,
            recovery,
            action,
            observation,
            objective,
        )
        return receipt, selector, action

    def _prepare_control_action(
        self,
        action: UiAction,
        observation: ObservationFrame,
        objective: str,
        recovery: dict[str, Any],
        goal_lock: GoalLock | None = None,
        app_agent: AppAgentSession | None = None,
    ) -> UiAction:
        proposal = self.action_router.propose(
            action=action,
            observation=observation,
            objective=objective,
        )
        decision = self.pre_action_verifier.verify(
            proposal=proposal,
            action=action,
            objective=objective,
            observation=observation,
            goal_lock=goal_lock,
            approval_token=self._approval_token(action),
        )
        app_signature = str(
            (app_agent.skill_pack.app_signature if app_agent else "")
            or observation.capability_profile.get("app_signature")
            or observation.app_context
            or "unknown"
        )
        entry_id = self.control_ledger.record_proposal(
            goal=objective,
            observation=observation,
            proposal=proposal,
            decision=decision,
            app_agent=(
                app_agent.skill_pack.skill_pack_id
                if app_agent is not None
                else str(observation.app_context or "workflow")
            ),
            app_signature=app_signature,
        )
        enriched = enrich_action_metadata(
            action=action,
            observation=observation,
            proposal=proposal,
            decision=decision,
            ledger_entry_id=entry_id,
        )
        control = dict(enriched.metadata.get("control") or {})
        if app_agent is not None:
            control["app_agent"] = app_agent.asdict()
        if goal_lock is not None:
            control["goal_lock"] = goal_lock.asdict()
        if "speculation" in enriched.metadata:
            control["speculation"] = dict(enriched.metadata["speculation"])
        if "isolation" in enriched.metadata:
            control["isolation"] = dict(enriched.metadata["isolation"])
        enriched.metadata["control"] = control
        recovery["control"] = control
        recovery.setdefault("control_history", []).append(control)
        if not decision.allowed:
            blocked_receipt = json.dumps(
                {
                    "status": "blocked",
                    "reason": decision.reason,
                    "decision": decision.asdict(),
                },
                sort_keys=True,
            )
            verification = {
                "kind": "pre_action_verification",
                "matched": False,
                "expected": "The action is safe and aligned before execution.",
                "observed": decision.reason,
                "required": True,
                "reason": decision.reason,
                "evidence": decision.asdict(),
            }
            self.control_ledger.record_failure_capsule(
                entry_id=entry_id,
                reason=decision.reason,
                proposal=proposal,
                observation=observation,
            )
            raise WorkflowVerificationError(
                action=enriched,
                receipt=blocked_receipt,
                verification=verification,
                recovery=recovery,
            )
        return enriched

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
        action = DesktopWorkflowService._workflow_action(step, selector)
        return backend.perform(action)

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
                expected=(
                    "The backend receipt reports successful action execution."
                ),
                target=selector,
                required=True,
            )
        if action_type == "draw_path":
            return VerificationContract(
                kind="receipt_success",
                expected=(
                    "The backend receipt reports successful action execution."
                ),
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
        observation: ObservationFrame,
        objective: str,
    ) -> tuple[str, str, UiAction]:
        if action.action_type == "tool":
            return selector, receipt, action
        payload = self._json_or_text(receipt)
        if not self._needs_retry(payload):
            return selector, receipt, action
        retry_selector = self._suggest_selector(backend, step.selector)
        if retry_selector is None or retry_selector == selector:
            return selector, receipt, action
        self._record_control_completion(
            action,
            receipt,
            {
                "kind": "retry_preflight",
                "matched": False,
                "required": False,
                "reason": "selector retry requested",
            },
            recovery,
            training_label="retry",
        )
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
        retry_action = self._prepare_control_action(
            retry_action,
            observation,
            objective,
            recovery,
        )
        retry_receipt = backend.perform(retry_action)
        return retry_selector, retry_receipt, retry_action

    def _observation_frame(
        self,
        backend: Any,
        nodes: list[Any],
        state: Any,
        objective: str,
    ) -> ObservationFrame:
        profile = self.capability_profiler.profile(state, nodes)
        self._last_capability_profile = profile
        app_agent = self.app_agent_runtime.resolve(profile, objective, nodes)
        self._last_app_agent_session = app_agent
        backend_name = str(
            getattr(backend, "name", backend.__class__.__name__),
        )
        capability_profile = profile.to_prompt_dict()
        capability_profile["app_agent"] = app_agent.asdict()
        frame = self.observation_builder.build(
            nodes=nodes,
            backend_name=backend_name,
            abstract_state=state,
            previous=self._last_observation_frame,
            capability_profile=capability_profile,
        )
        if frame.app_context == "unknown":
            frame.app_context = app_agent.skill_pack.app_context
        self._last_observation_frame = frame
        return frame

    def _record_control_completion(
        self,
        action: UiAction,
        receipt: str,
        verification: dict[str, Any],
        recovery: dict[str, Any],
        training_label: str = "success",
    ) -> None:
        control = dict(action.metadata.get("control") or {})
        entry_id = str(control.get("ledger_entry_id") or "")
        receipt_payload = self._json_or_text(receipt)
        self.app_agent_runtime.record_outcome(
            action,
            verification_result=verification,
            receipt=receipt_payload,
        )
        if verification.get("kind") == "pre_action_verification" or not bool(
            verification.get("matched", True),
        ):
            benchmark = self._promote_workflow_failure_candidate(
                action,
                receipt_payload,
                verification,
                recovery,
            )
            if benchmark:
                updated = dict(recovery.get("control") or control)
                updated["benchmark_candidate"] = benchmark
                recovery["control"] = updated
        if not entry_id:
            return
        self.control_ledger.record_completion(
            entry_id=entry_id,
            receipt=receipt_payload,
            verification_result=verification,
            repair_decision=recovery,
            training_label=training_label,
        )

    @staticmethod
    def _control_from_recovery(recovery: dict[str, Any]) -> dict[str, Any]:
        return dict(recovery.get("control") or {})

    @staticmethod
    def _approval_token(action: UiAction) -> str | None:
        token = action.metadata.get("approval_token") or action.metadata.get(
            "approval",
        )
        return str(token) if token else None

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
        return _json_or_text(value)

    def _perform_action(self, backend: Any, action: UiAction) -> str:
        if action.action_type == "tool":
            return self.programmer_lane.execute_action(action)
        return backend.perform(action)

    @staticmethod
    def _annotate_action_runtime(
        action: UiAction,
        objective: str,
        goal_lock: GoalLock,
        speculation: dict[str, Any],
        isolation: dict[str, Any],
    ) -> UiAction:
        metadata = dict(action.metadata or {})
        metadata["workflow_objective"] = objective
        metadata["goal_lock"] = goal_lock.asdict()
        metadata["speculation"] = speculation
        metadata["isolation"] = isolation
        return UiAction(
            action_type=action.action_type,
            selector=action.selector,
            value=action.value,
            metadata=metadata,
        )

    def _speculative_preview(
        self,
        observation: ObservationFrame,
        objective: str,
        remaining_steps: list[Any],
        app_agent: AppAgentSession | None,
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for candidate_step in remaining_steps[:3]:
            candidate_action = self._workflow_action(
                candidate_step,
                candidate_step.selector,
            )
            if app_agent is not None:
                candidate_action = self.app_agent_runtime.enrich_action(
                    candidate_action,
                    app_agent,
                    objective,
                )
            proposal = self.action_router.propose(
                action=candidate_action,
                observation=observation,
                objective=objective,
            )
            candidates.append(
                {
                    "action_type": candidate_action.action_type,
                    "selector": candidate_action.selector,
                    "route": proposal.route,
                    "risk_score": round(float(proposal.risk_score), 3),
                    "description": getattr(candidate_step, "description", ""),
                }
            )
        commit_window = 1
        if candidates and all(
            float(item["risk_score"]) < 0.45 for item in candidates[:2]
        ):
            commit_window = min(2, len(candidates))
        preview_spec = SandboxSpec(
            provider="dry-run",
            image=(
                f"workflow-{app_agent.skill_pack.family}"
                if app_agent is not None
                else "workflow-desktop"
            ),
            metadata={
                "control_request": {
                    "kind": "speculative_preview",
                    "objective": objective,
                    "candidates": candidates,
                }
            },
        )
        preview = self.sandbox_manager.execute(
            preview_spec,
            ["speculative-preview", str(len(candidates))],
        )
        return {
            "candidate_count": len(candidates),
            "commit_window": commit_window,
            "candidates": candidates,
            "sandbox_preview": {
                "provider": preview.provider,
                "dry_run": preview.dry_run,
                "exit_code": preview.exit_code,
                "stdout": preview.stdout[:240],
            },
        }

    def _isolation_plan(
        self,
        action: UiAction,
        observation: ObservationFrame,
        app_agent: AppAgentSession | None,
    ) -> dict[str, Any]:
        provider = "dry-run"
        if app_agent is not None and (
            app_agent.skill_pack.visual_heavy
            or not app_agent.skill_pack.safe_windows
        ):
            provider = "agent-body"
        spec = SandboxSpec(
            provider=provider,
            image=(
                f"isolated-{app_agent.skill_pack.family}"
                if app_agent is not None
                else "isolated-desktop"
            ),
            mounts={str(self.writer.workspace_root): "/workspace"},
            metadata={
                "state_path": str(
                    self.writer.workspace_root
                    / ".agentos"
                    / "sandbox"
                    / f"{provider}.json"
                ),
                "control_request": {
                    "kind": "desktop_action_preview",
                    "action_type": action.action_type,
                    "selector": action.selector,
                    "backend": observation.backend_name,
                },
            },
        )
        preview = self.sandbox_manager.execute(
            spec,
            ["desktop-action", action.action_type, action.selector],
        )
        return {
            "provider": spec.provider,
            "image": spec.image,
            "dry_run": preview.dry_run,
            "exit_code": preview.exit_code,
            "stdout": preview.stdout[:240],
        }

    @staticmethod
    def _generated_artifacts(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        generated = payload.get("generated_outputs")
        if not isinstance(generated, list):
            return []
        return [
            dict(item)
            for item in generated
            if isinstance(item, dict) and item.get("path")
        ]

    def _promote_workflow_failure_candidate(
        self,
        action: UiAction,
        receipt: Any,
        verification: dict[str, Any],
        recovery: dict[str, Any],
    ) -> str:
        control = dict(action.metadata.get("control") or {})
        trace_id = str(
            control.get("ledger_entry_id")
            or f"workflow_{int(time.time())}"
        )
        failure_dir = (
            self.writer.workspace_root
            / "artifacts"
            / "live_fire_eval"
            / "workflow_failures"
        )
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"workflow_{trace_id}.json"
        payload = {
            "trace_id": f"workflow_{trace_id}",
            "source": "workflow_service",
            "objective": str(action.metadata.get("workflow_objective") or ""),
            "action": {
                "action_type": action.action_type,
                "selector": action.selector,
                "value": action.value,
                "metadata": dict(action.metadata),
            },
            "receipt": receipt,
            "verification": dict(verification),
            "recovery": dict(recovery),
            "control": control,
            "app_agent": dict(action.metadata.get("app_agent") or {}),
        }
        failure_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        golden_dir = (
            self.writer.workspace_root / "benchmarks" / "golden_traces"
        )
        golden_dir.mkdir(parents=True, exist_ok=True)
        golden_path = golden_dir / f"workflow_{trace_id}.json"
        if not golden_path.exists():
            golden_payload = dict(payload)
            golden_payload["status"] = "workflow_failure_candidate"
            golden_path.write_text(
                json.dumps(golden_payload, indent=2),
                encoding="utf-8",
            )
        return str(golden_path.relative_to(self.writer.workspace_root))
