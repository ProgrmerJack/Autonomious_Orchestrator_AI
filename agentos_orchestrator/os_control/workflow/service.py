from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
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
from agentos_orchestrator.cognition.control_surface_discovery import (
    GenericControlSurfaceDiscoverer,
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
from agentos_orchestrator.cognition.tool_executor import (
    QuantAnalysisRequest,
    ToolExecutor,
)
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.os_control.workflow.artifact_writer import (
    WorkflowArtifactWriter,
)
from agentos_orchestrator.os_control.workflow.intent_parser import (
    StructuredIntent,
    parse_structured_intent,
)
from agentos_orchestrator.os_control.workflow.models import DesktopWorkflowStep
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
        reason = str(verification.get("reason") or "workflow verification failed")
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
        self.tool_executor = ToolExecutor(
            self.writer.workspace_root / ".agentos",
        )
        self.programmer_lane = ProgrammerLane(
            workspace_root,
            tool_executor=self.tool_executor,
        )
        self.reasoner = DesktopWorkflowReasoner()
        self.max_adaptive_steps = 4
        self._universal_agent: Any | None = None
        self._universal_backend_id: int | None = None
        self.capability_profiler = CapabilityProfiler()
        self.app_agent_runtime = AppAgentRuntime(self.writer.workspace_root)
        self.control_surface_discoverer = GenericControlSurfaceDiscoverer(
            self.writer.workspace_root,
        )
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
        self._workflow_blackboard: dict[str, Any] = {}
        self._current_plan_intent: dict[str, Any] = {}

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
        plan = self._repair_bootstrap_plan(objective, plan, backend)
        self._current_plan_intent = dict(
            getattr(plan, "intent", {}) or parse_structured_intent(objective).asdict()
        )
        self._seed_workflow_blackboard(plan)
        if plan.requires_clarification:
            return {
                "status": "clarification_required",
                "plan": plan.asdict(),
                "artifacts": [],
                "receipts": [],
            }
        plan = self.programmer_lane.augment_plan(plan)
        if not getattr(plan, "intent", None):
            plan.intent = dict(self._current_plan_intent)
        written_by_path = {
            item["path"]: item
            for item in self.writer.materialize(
                plan,
                skip_paths=self._reserved_artifact_paths(plan),
            )
        }
        receipts: list[dict[str, Any]] = []
        for index, step in enumerate(plan.steps):
            materialized_step = self._materialize_handoff_step(step)
            (
                receipt,
                final_selector,
                recovery,
                verification,
            ) = self._perform_with_recovery(
                backend,
                materialized_step,
                objective=objective,
                remaining_steps=[
                    self._materialize_handoff_step(item) for item in plan.steps[index:]
                ],
            )
            receipt_payload = self._json_or_text(receipt)
            self._update_workflow_blackboard(
                materialized_step,
                final_selector,
                receipt_payload,
            )
            for artifact in self._generated_artifacts(receipt_payload):
                written_by_path[artifact["path"]] = artifact
            receipts.append(
                {
                    "action_type": materialized_step.action_type,
                    "selector": final_selector,
                    "description": materialized_step.description,
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
                    universal_receipt["model_used_ratio"] = run.model_used_ratio
                if hasattr(run, "adaptive_steps_used"):
                    universal_receipt["adaptive_steps_used"] = run.adaptive_steps_used
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
                        "description": ("Cognitive architecture execution failed"),
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

    def _repair_bootstrap_plan(
        self,
        objective: str,
        plan: Any,
        backend: Any,
    ) -> Any:
        intent_payload = dict(
            getattr(plan, "intent", {}) or parse_structured_intent(objective).asdict()
        )
        plan.intent = intent_payload
        if self._universal_agent is None:
            return plan
        repair_method = getattr(self._universal_agent, "repair_bootstrap_plan", None)
        if not callable(repair_method):
            return plan
        try:
            repaired = repair_method(objective, plan, backend=backend)
        except TypeError:
            repaired = repair_method(objective, plan)
        except (RuntimeError, ValueError, OSError):
            return plan
        if repaired is None:
            return plan
        if not getattr(repaired, "intent", None):
            repaired.intent = intent_payload
        return repaired

    def _seed_workflow_blackboard(self, plan: Any) -> None:
        intent = StructuredIntent.from_dict(getattr(plan, "intent", {}) or {})
        self._workflow_blackboard = {
            str(key): value
            for key, value in intent.entities.items()
            if str(value or "").strip()
        }
        if intent.source_surface:
            self._workflow_blackboard["source_surface"] = intent.source_surface
        if intent.destination_surface:
            self._workflow_blackboard["destination_surface"] = (
                intent.destination_surface
            )
        if intent.file_source_hint:
            self._workflow_blackboard["file_source"] = intent.file_source_hint
        if intent.file_destination_hint:
            self._workflow_blackboard["file_destination"] = intent.file_destination_hint
        self._workflow_blackboard["workflow_objective"] = str(plan.objective or "")

    def _materialize_handoff_step(self, step: Any) -> DesktopWorkflowStep:
        return DesktopWorkflowStep(
            action_type=step.action_type,
            selector=self._render_handoff_value(step.selector),
            value=self._render_handoff_value(step.value),
            description=step.description,
            metadata=self._render_handoff_value(dict(step.metadata or {})),
        )

    def _render_handoff_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(
                r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}",
                lambda match: str(
                    self._workflow_blackboard.get(match.group(1), match.group(0))
                ),
                value,
            )
        if isinstance(value, dict):
            return {
                str(key): self._render_handoff_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._render_handoff_value(item) for item in value]
        return value

    def _update_workflow_blackboard(
        self,
        step: Any,
        selector: str,
        receipt_payload: Any,
    ) -> None:
        del selector
        metadata = dict(step.metadata or {})
        handoff_write = metadata.get("handoff_write")
        if isinstance(handoff_write, dict):
            for key, value in handoff_write.items():
                rendered = self._render_handoff_value(value)
                if str(rendered or "").strip():
                    self._workflow_blackboard[str(key)] = rendered
        if step.action_type in {"open_url"} and step.value:
            self._workflow_blackboard["active_url"] = str(step.value)
        if step.action_type in {"set_clipboard", "clipboard_copy"} and step.value:
            self._workflow_blackboard["copied_text"] = str(step.value)
        if step.action_type in {"type", "set_text", "set_value"} and step.value:
            self._workflow_blackboard["last_written_text"] = str(step.value)
        if not isinstance(receipt_payload, dict):
            return
        clipboard = str(receipt_payload.get("clipboard") or "").strip()
        if clipboard:
            self._workflow_blackboard["copied_text"] = clipboard
        file_op = receipt_payload.get("file_op")
        if isinstance(file_op, dict):
            source = str(file_op.get("source") or "").strip()
            destination = str(
                file_op.get("destination") or file_op.get("new_name") or ""
            ).strip()
            if source:
                self._workflow_blackboard["file_source"] = source
            if destination:
                self._workflow_blackboard["file_destination"] = destination
        parsed_results = receipt_payload.get("parsed_results")
        if parsed_results:
            self._workflow_blackboard["extracted_fact"] = json.dumps(
                parsed_results,
                sort_keys=True,
            )[:500]

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
            materialized_step = self._materialize_handoff_step(decision.step)
            (
                receipt,
                final_selector,
                recovery,
                verification,
            ) = self._perform_with_recovery(
                backend,
                materialized_step,
                objective=objective,
                remaining_steps=[materialized_step],
            )
            receipt_payload = self._json_or_text(receipt)
            self._update_workflow_blackboard(
                materialized_step,
                final_selector,
                receipt_payload,
            )
            receipts.append(
                {
                    "action_type": materialized_step.action_type,
                    "selector": final_selector,
                    "description": materialized_step.description,
                    "receipt": receipt_payload,
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
        if step.action_type not in {
            "launch_app",
            "tool",
            "api_call",
            "mcp_call",
            "http_request",
            "browser_devtools",
        }:
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
        action = self._attach_workflow_handoff(action)
        goal_lock = build_goal_lock(
            objective=objective,
            observation=observation,
            intent_profile=self._current_plan_intent,
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

    def _attach_workflow_handoff(self, action: UiAction) -> UiAction:
        if not self._workflow_blackboard:
            return action
        metadata = dict(action.metadata or {})
        metadata["workflow_handoff"] = dict(self._workflow_blackboard)
        app_agent = dict(metadata.get("app_agent") or {})
        if app_agent:
            app_agent["handoff_entities"] = dict(self._workflow_blackboard)
            metadata["app_agent"] = app_agent
        return UiAction(
            action_type=action.action_type,
            selector=action.selector,
            value=action.value,
            metadata=metadata,
        )

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
        materialized = self._materialize_control_route(
            enriched,
            proposal=proposal,
            observation=observation,
            objective=objective,
            app_agent=app_agent,
        )
        if materialized is not enriched:
            control = dict(materialized.metadata.get("control") or control)
            recovery["control"] = control
            history = recovery.setdefault("control_history", [])
            if history:
                history[-1] = control
        return materialized

    @staticmethod
    def _empty_recovery() -> dict[str, Any]:
        return {
            "applied": False,
            "attempted": False,
            "reason": "",
        }

    def _materialize_control_route(
        self,
        action: UiAction,
        *,
        proposal: Any,
        observation: ObservationFrame,
        objective: str,
        app_agent: AppAgentSession | None,
    ) -> UiAction:
        route = (
            str(action.metadata.get("control_route") or proposal.route or "")
            .strip()
            .lower()
        )
        replacement: UiAction | None = None
        if route == "code_tool":
            replacement = self._code_tool_replacement(action)
        elif route == "api_mcp":
            replacement = self._api_tool_replacement(
                action,
                observation,
                objective,
                app_agent,
            )
        if replacement is None:
            return action
        return self._merge_materialized_action(action, replacement, route)

    def _code_tool_replacement(self, action: UiAction) -> UiAction | None:
        request = self._workspace_file_request(action)
        if request is None:
            return None
        if not self._workspace_file_request_is_actionable(request):
            return None
        verification_contract: dict[str, Any] = {
            "kind": "receipt_success",
            "expected": "The workspace file operation completes successfully.",
            "target": "tool_executor:workflow_workspace_file_op",
            "required": True,
        }
        target_path = str(request.get("target_path") or "").strip()
        if target_path and request.get("operation") != "delete":
            verification_contract = {
                "kind": "file_exists",
                "expected": f"The file exists at {target_path}.",
                "path": target_path,
                "required": True,
            }
        return UiAction(
            action_type="tool",
            selector="tool_executor:workflow_workspace_file_op",
            metadata={
                "tool_request": request,
                "control_channel": "code",
                "expected_observation": (
                    "The workspace file operation completes through the code tool lane."
                ),
                "verification_contract": verification_contract,
                "policy_anchor_action": {
                    "action_type": action.action_type,
                    "selector": action.selector,
                    "value": action.value,
                    "metadata": {
                        **dict(action.metadata or {}),
                        "control_channel": "code",
                    },
                },
            },
        )

    def _workspace_file_request_is_actionable(
        self,
        request: dict[str, Any],
    ) -> bool:
        operation = str(request.get("operation") or "").strip().lower()
        if operation not in {"copy", "move", "rename", "delete"}:
            return False
        source_raw = str(request.get("source") or "").strip()
        if not source_raw:
            return False
        try:
            source_path = self._resolve_workspace_path(source_raw)
        except ValueError:
            return False
        return source_path.exists()

    def _api_tool_replacement(
        self,
        action: UiAction,
        observation: ObservationFrame,
        objective: str,
        app_agent: AppAgentSession | None,
    ) -> UiAction | None:
        request = self._api_tool_request(
            action,
            observation,
            objective,
            app_agent,
        )
        if request is None:
            return None
        selector = str(request.get("selector") or "tool_executor:workflow_api")
        return UiAction(
            action_type="tool",
            selector=selector,
            metadata={
                "tool_request": request,
                "control_channel": "api",
                "expected_observation": str(request.get("expected_observation") or ""),
                "verification_contract": {
                    "kind": "receipt_success",
                    "expected": str(request.get("expected_observation") or ""),
                    "target": selector,
                    "required": True,
                },
                "policy_anchor_action": {
                    "action_type": "http_request",
                    "selector": str(
                        request.get("endpoint") or request.get("selector") or ""
                    ),
                    "metadata": {"control_channel": "api"},
                },
            },
        )

    @staticmethod
    def _merge_materialized_action(
        original: UiAction,
        replacement: UiAction,
        route: str,
    ) -> UiAction:
        metadata = dict(original.metadata or {})
        metadata.update(dict(replacement.metadata or {}))
        metadata.setdefault(
            "original_action",
            {
                "action_type": original.action_type,
                "selector": original.selector,
                "value": original.value,
            },
        )
        control = dict(metadata.get("control") or {})
        control["executed_via_route"] = route
        control["materialized_action_type"] = replacement.action_type
        control["materialized_selector"] = replacement.selector
        metadata["control"] = control
        return UiAction(
            action_type=replacement.action_type,
            selector=replacement.selector,
            value=replacement.value,
            metadata=metadata,
        )

    def _workspace_file_request(
        self,
        action: UiAction,
    ) -> dict[str, Any] | None:
        metadata = dict(action.metadata or {})
        operation = str(metadata.get("operation") or "").strip().lower()
        if not operation and action.action_type.endswith("_file"):
            operation = action.action_type[:-5]
        if operation not in {"copy", "move", "rename", "delete"}:
            return None
        request: dict[str, Any] = {
            "kind": "workspace_file_op",
            "operation": operation,
        }
        source, destination = self._parse_file_operation_payload(
            action,
            operation,
        )
        if operation in {"copy", "move"}:
            if not source or not destination:
                return None
            request["source"] = source
            request["destination"] = destination
            try:
                request["target_path"] = str(
                    self._resolve_workspace_path(destination),
                )
            except ValueError:
                pass
            return request
        if operation == "rename":
            if not source or not destination:
                return None
            request["source"] = source
            request["new_name"] = destination
            try:
                source_path = self._resolve_workspace_path(source)
                request["target_path"] = str(
                    (source_path.parent / destination).resolve(),
                )
            except ValueError:
                pass
            return request
        if not source:
            return None
        request["source"] = source
        return request

    @staticmethod
    def _parse_file_operation_payload(
        action: UiAction,
        operation: str,
    ) -> tuple[str, str]:
        metadata = dict(action.metadata or {})
        source = str(
            metadata.get("source")
            or metadata.get("path")
            or metadata.get("file_path")
            or ""
        ).strip()
        destination = str(
            metadata.get("destination") or metadata.get("new_name") or ""
        ).strip()
        if source and destination:
            return source, destination
        raw_value = str(action.value or "")
        if "->" not in raw_value:
            return source, destination
        left, right = raw_value.split("->", 1)
        left = left.strip()
        right = right.strip()
        if not source:
            source = left
        if not destination:
            destination = right
        if operation == "delete":
            return source or raw_value.strip(), ""
        return source, destination

    def _api_tool_request(
        self,
        action: UiAction,
        observation: ObservationFrame,
        objective: str,
        app_agent: AppAgentSession | None,
    ) -> dict[str, Any] | None:
        explicit = self._explicit_api_tool_request(action, objective)
        if explicit is not None:
            return explicit
        for surface in observation.discovered_surfaces:
            if not isinstance(surface, dict):
                continue
            surface_metadata = dict(surface.get("metadata") or {})
            auth_env_keys = [
                str(item)
                for item in surface_metadata.get("auth_env_keys", [])
                if str(item)
            ]
            workflow = surface.get("workflow") or []
            if isinstance(workflow, list) and workflow:
                selector = str(
                    surface.get("endpoint")
                    or surface_metadata.get("documentation_url")
                    or "tool_executor:workflow_api"
                )
                return {
                    "kind": "api_workflow",
                    "selector": f"tool_executor:workflow_api:{selector}",
                    "workflow": {
                        "steps": workflow,
                        "headers": {},
                        "auth_env_keys": auth_env_keys,
                    },
                    "auth_env_keys": auth_env_keys,
                    "objective": objective,
                    "expected_observation": (
                        "The synthesized API workflow returns structured HTTP results."
                    ),
                }
            endpoint = str(
                surface.get("endpoint") or surface_metadata.get("endpoint") or ""
            ).strip()
            if endpoint:
                return {
                    "kind": "http_probe",
                    "selector": (f"tool_executor:control_surface_probe:{endpoint}"),
                    "endpoint": endpoint,
                    "methods": ["OPTIONS", "GET"],
                    "objective": objective,
                    "expected_observation": (
                        "The API probe returns a structured HTTP result."
                    ),
                }
        if app_agent is None:
            return None
        policy_action = dict(app_agent.skill_pack.policy_action or {})
        action_type = str(policy_action.get("action_type") or "").strip().lower()
        selector = str(policy_action.get("selector") or "").strip()
        policy_metadata = dict(policy_action.get("metadata") or {})
        endpoint = selector or str(policy_metadata.get("endpoint") or "").strip()
        if action_type in {
            "api_call",
            "mcp_call",
            "http_request",
            "browser_devtools",
        } and endpoint.startswith(("http://", "https://")):
            return {
                "kind": "http_probe",
                "selector": f"tool_executor:control_surface_probe:{endpoint}",
                "endpoint": endpoint,
                "methods": ["OPTIONS", "GET"],
                "objective": objective,
                "expected_observation": (
                    "The remembered API affordance returns a structured HTTP result."
                ),
            }
        return None

    def _explicit_api_tool_request(
        self,
        action: UiAction,
        objective: str,
    ) -> dict[str, Any] | None:
        action_type = str(action.action_type or "").strip().lower()
        if action_type not in {
            "api_call",
            "mcp_call",
            "http_request",
            "browser_devtools",
        }:
            return None
        metadata = dict(action.metadata or {})
        auth_env_keys = [
            str(item) for item in metadata.get("auth_env_keys", []) if str(item)
        ]
        expected_observation = str(
            metadata.get("expected_observation")
            or "The direct API action returns a structured response."
        )
        workflow = metadata.get("workflow")
        if isinstance(workflow, list) and workflow:
            selector = str(
                metadata.get("endpoint")
                or action.selector
                or workflow[0].get("url")
                or "tool_executor:workflow_api"
            )
            headers = dict(metadata.get("headers") or {})
            return {
                "kind": "api_workflow",
                "selector": f"tool_executor:workflow_api:{selector}",
                "workflow": {
                    "steps": workflow,
                    "headers": headers,
                    "auth_env_keys": auth_env_keys,
                },
                "auth_env_keys": auth_env_keys,
                "objective": objective,
                "expected_observation": expected_observation,
            }

        endpoint = str(metadata.get("endpoint") or action.selector or "").strip()
        if not endpoint.startswith(("http://", "https://")):
            return None
        method = str(metadata.get("method") or "GET").strip().upper() or "GET"
        headers = dict(metadata.get("headers") or {})
        json_body = metadata.get("json_body")
        if json_body is None and "body" in metadata:
            json_body = metadata.get("body")
        if json_body is not None or method not in {"GET", "OPTIONS"} or headers:
            return {
                "kind": "api_workflow",
                "selector": f"tool_executor:workflow_api:{endpoint}",
                "workflow": {
                    "steps": [
                        {
                            "name": "explicit_action",
                            "method": method,
                            "url": endpoint,
                            "json_body": json_body,
                        }
                    ],
                    "headers": headers,
                    "auth_env_keys": auth_env_keys,
                },
                "auth_env_keys": auth_env_keys,
                "objective": objective,
                "expected_observation": expected_observation,
            }

        methods = [
            str(item).strip().upper()
            for item in metadata.get("methods", [])
            if str(item).strip()
        ]
        if not methods:
            methods = ["OPTIONS", "GET"] if method == "GET" else [method]
        return {
            "kind": "http_probe",
            "selector": f"tool_executor:control_surface_probe:{endpoint}",
            "endpoint": endpoint,
            "methods": methods,
            "objective": objective,
            "expected_observation": expected_observation,
        }

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        root = self.writer.workspace_root.resolve()
        candidate = Path(raw_path)
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Path '{raw_path}' escapes the workspace root.",
            )
        return resolved

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
        if action_type in {"set_clipboard", "clipboard_copy"}:
            return VerificationContract(
                kind="clipboard_contains",
                expected="The clipboard contains the transferred value.",
                target=selector,
                required=True,
            )
        if action_type == "cell_edit":
            return VerificationContract(
                kind="receipt_success",
                expected=("The backend receipt reports successful action execution."),
                target=selector,
                required=True,
            )
        if action_type == "draw_path":
            return VerificationContract(
                kind="receipt_success",
                expected=("The backend receipt reports successful action execution."),
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
        retry_receipt = self._perform_action(backend, retry_action)
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
        discovered_surfaces = self._discover_control_surfaces(
            profile,
            nodes,
            objective,
            app_agent,
        )
        frame = self.observation_builder.build(
            nodes=nodes,
            backend_name=backend_name,
            abstract_state=state,
            previous=self._last_observation_frame,
            capability_profile=capability_profile,
            discovered_surfaces=discovered_surfaces,
        )
        if frame.app_context == "unknown":
            frame.app_context = app_agent.skill_pack.app_context
        self._last_observation_frame = frame
        return frame

    def _discover_control_surfaces(
        self,
        profile: Any,
        nodes: list[Any],
        objective: str,
        app_agent: AppAgentSession,
    ) -> list[dict[str, Any]]:
        if not self._should_discover_control_surfaces(objective, app_agent):
            return []
        candidates = self.control_surface_discoverer.discover(
            profile,
            nodes,
            objective,
            preferred_channels=list(app_agent.skill_pack.preferred_channels),
            active_fingerprinting=self._should_probe_control_surfaces(
                objective,
                app_agent,
            ),
        )
        return [asdict(candidate) for candidate in candidates[:5]]

    @staticmethod
    def _should_discover_control_surfaces(
        objective: str,
        app_agent: AppAgentSession,
    ) -> bool:
        lower = str(objective or "").lower()
        if any(
            keyword in lower
            for keyword in (
                "api",
                "endpoint",
                "graphql",
                "json",
                "webhook",
                "developer",
                "service",
                "server",
            )
        ):
            return True
        if app_agent.skill_pack.api_like:
            return True
        policy_action = dict(app_agent.skill_pack.policy_action or {})
        return str(policy_action.get("action_type") or "").lower() in {
            "api_call",
            "mcp_call",
            "http_request",
            "browser_devtools",
        }

    @staticmethod
    def _should_probe_control_surfaces(
        objective: str,
        app_agent: AppAgentSession,
    ) -> bool:
        lower = str(objective or "").lower()
        if any(keyword in lower for keyword in ("localhost", "port", "graphql")):
            return True
        policy_action = dict(app_agent.skill_pack.policy_action or {})
        selector = str(policy_action.get("selector") or "")
        return selector.startswith(("http://", "https://"))

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
            return self._perform_tool_action(action)
        return backend.perform(action)

    def _perform_tool_action(self, action: UiAction) -> str:
        if action.selector == "tool_executor:workflow_programmer":
            return self.programmer_lane.execute_action(action)
        request = action.metadata.get("tool_request")
        if not isinstance(request, dict):
            return json.dumps(
                {
                    "status": "invalid-tool-request",
                    "success": False,
                    "error": "Missing workflow tool request metadata.",
                },
                sort_keys=True,
            )
        kind = str(request.get("kind") or "").strip().lower()
        if kind == "workspace_file_op":
            return self._execute_workspace_file_op(request)
        if kind == "workflow_research_brief":
            return self._execute_workflow_research_request(request)
        if kind in {"api_workflow", "http_probe"}:
            return self._execute_api_tool_request(request)
        return json.dumps(
            {
                "status": "invalid-tool-request",
                "success": False,
                "error": f"Unsupported workflow tool kind '{kind}'.",
            },
            sort_keys=True,
        )

    def _execute_workspace_file_op(self, request: dict[str, Any]) -> str:
        operation = str(request.get("operation") or "").strip().lower()
        try:
            source = self._resolve_workspace_path(
                str(request.get("source") or ""),
            )
            destination = None
            if operation in {"copy", "move"}:
                destination = self._resolve_workspace_path(
                    str(request.get("destination") or ""),
                )
            elif operation == "rename":
                new_name = str(request.get("new_name") or "").strip()
                if not new_name:
                    raise ValueError(
                        "Rename operation requires a new_name field.",
                    )
                destination = (source.parent / new_name).resolve()
                if not destination.is_relative_to(
                    self.writer.workspace_root.resolve(),
                ):
                    raise ValueError(
                        "Rename target escapes the workspace root.",
                    )

            if operation == "copy":
                if destination is None:
                    raise ValueError("Copy operation requires a destination.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
            elif operation == "move":
                if destination is None:
                    raise ValueError("Move operation requires a destination.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            elif operation == "rename":
                if destination is None:
                    raise ValueError(
                        "Rename operation requires a destination.",
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
            elif operation == "delete":
                if source.is_dir():
                    shutil.rmtree(source)
                else:
                    source.unlink()
            else:
                raise ValueError(f"Unsupported workspace file operation '{operation}'.")
        except (OSError, ValueError) as exc:
            return json.dumps(
                {
                    "status": "tool_error",
                    "success": False,
                    "operation": operation,
                    "error": str(exc),
                },
                sort_keys=True,
            )

        file_op: dict[str, Any] = {
            "operation": operation,
            "source": str(request.get("source") or ""),
            "status": "completed",
        }
        if operation in {"copy", "move"}:
            file_op["destination"] = str(request.get("destination") or "")
        elif operation == "rename":
            file_op["new_name"] = str(request.get("new_name") or "")
        return json.dumps(
            {
                "status": "file-op-executed",
                "success": True,
                "operation": operation,
                "file_op": file_op,
            },
            sort_keys=True,
        )

    def _execute_workflow_research_request(self, request: dict[str, Any]) -> str:
        from agentos_orchestrator.research.deep_research import DeepResearchEngine

        objective = str(request.get("objective") or "").strip()
        query = str(request.get("query") or objective).strip()
        output_path_raw = str(request.get("output_path") or "").strip()
        if not output_path_raw:
            return json.dumps(
                {
                    "status": "invalid-tool-request",
                    "success": False,
                    "error": "workflow_research_brief requires an output_path.",
                },
                sort_keys=True,
            )
        try:
            output_path = self._resolve_workspace_path(output_path_raw)
        except ValueError as exc:
            return json.dumps(
                {
                    "status": "tool_error",
                    "success": False,
                    "kind": "workflow_research_brief",
                    "error": str(exc),
                },
                sort_keys=True,
            )

        try:
            per_provider_limit = max(
                1,
                min(8, int(request.get("per_provider_limit") or 4)),
            )
        except (TypeError, ValueError):
            per_provider_limit = 4
        try:
            max_sources = max(1, min(20, int(request.get("max_sources") or 8)))
        except (TypeError, ValueError):
            max_sources = 8

        engine = DeepResearchEngine(
            self.writer.workspace_root,
            limit_per_provider=per_provider_limit,
            timeout_seconds=15,
        )
        providers = self._workflow_research_providers(engine, query)
        try:
            sources = engine._search_query_across_providers(
                query,
                providers,
                per_provider_limit,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return json.dumps(
                {
                    "status": "tool_error",
                    "success": False,
                    "kind": "workflow_research_brief",
                    "query": query,
                    "providers": sorted(providers),
                    "error": str(exc),
                },
                sort_keys=True,
            )

        curated_sources = self._dedupe_research_sources(sources)[:max_sources]
        if not curated_sources:
            return json.dumps(
                {
                    "status": "no_sources",
                    "success": False,
                    "kind": "workflow_research_brief",
                    "query": query,
                    "providers": sorted(providers),
                    "error": "No provider-backed sources were retrieved.",
                },
                sort_keys=True,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._workflow_research_brief_markdown(
            objective,
            query,
            curated_sources,
            sorted(providers),
        )
        output_path.write_text(content, encoding="utf-8")
        relative_path = self._workspace_relative_path(output_path)
        generated_outputs = [
            {
                "path": relative_path,
                "kind": "research-brief",
                "description": "Provider-backed workflow research brief.",
                "bytes": output_path.stat().st_size,
            }
        ]
        return json.dumps(
            {
                "status": "success",
                "success": True,
                "kind": "workflow_research_brief",
                "query": query,
                "providers": sorted(providers),
                "source_count": len(curated_sources),
                "generated_outputs": generated_outputs,
            },
            sort_keys=True,
        )

    def _execute_api_tool_request(self, request: dict[str, Any]) -> str:
        kind = str(request.get("kind") or "").strip().lower()
        objective = str(request.get("objective") or "")
        auth_env_keys = [
            str(item) for item in request.get("auth_env_keys", []) if str(item)
        ]
        if kind == "api_workflow":
            code = self.tool_executor.build_api_workflow_code(
                dict(request.get("workflow") or {}),
            )
        elif kind == "http_probe":
            code = self.tool_executor.build_http_probe_code(
                [str(request.get("endpoint") or "")],
                methods=[str(item) for item in request.get("methods", []) if str(item)],
            )
        else:
            return json.dumps(
                {
                    "status": "invalid-tool-request",
                    "success": False,
                    "error": f"Unsupported API tool kind '{kind}'.",
                },
                sort_keys=True,
            )
        result = self.tool_executor.run(
            QuantAnalysisRequest(
                objective=objective or f"Workflow {kind} execution",
                code=code,
                allow_network=True,
                timeout_seconds=20,
                expose_env_keys=auth_env_keys,
            ),
        )
        parsed_payload = self._api_tool_payload(kind, result.parsed_results)
        success = bool(result.success and parsed_payload)
        return json.dumps(
            {
                "status": "success" if success else "tool_error",
                "success": success,
                "kind": kind,
                "endpoint": str(request.get("endpoint") or ""),
                "tool_result": {
                    "success": result.success,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-1000:],
                    "error": result.error,
                    "parsed_results": result.parsed_results,
                    "artefacts": [str(path) for path in result.artefacts],
                    "elapsed_ms": result.elapsed_ms,
                },
                "payload": parsed_payload,
            },
            sort_keys=True,
        )

    @staticmethod
    def _api_tool_payload(kind: str, parsed_results: dict[str, Any]) -> Any:
        if kind == "api_workflow":
            payload = parsed_results.get("api_workflow")
            if isinstance(payload, list) and payload:
                if any(item.get("error") for item in payload if isinstance(item, dict)):
                    return None
                return payload
            return None
        payload = parsed_results.get("control_probe")
        if not isinstance(payload, dict) or not payload:
            return None
        for endpoint_result in payload.values():
            if not isinstance(endpoint_result, dict):
                continue
            if any(
                not isinstance(method_result, dict) or method_result.get("error")
                for method_result in endpoint_result.values()
            ):
                continue
            return payload
        return None

    def _reserved_artifact_paths(self, plan: Any) -> set[str]:
        reserved = set(self.programmer_lane.reserved_paths(plan))
        for artifact in getattr(plan, "artifacts", []) or []:
            if getattr(artifact, "kind", "") == "research-brief":
                reserved.add(str(getattr(artifact, "path", "")))
        return reserved

    @staticmethod
    def _workflow_research_providers(engine: Any, query: str) -> set[str]:
        if engine._looks_like_market_query(query):
            return {
                "sec-edgar",
                "earnings-data",
                "financial-portals",
                "stockanalysis",
                "fed-macro",
                "google-news-rss",
                "web-search",
                "bing-search",
                "seeking-alpha",
            }
        if engine._looks_like_academic_query(query):
            return {
                "openalex",
                "semantic-scholar",
                "crossref",
                "web-search",
                "bing-search",
                "google-news-rss",
            }
        providers = {"web-search", "bing-search", "google-news-rss"}
        if engine._looks_like_software_agent_query(query):
            providers.add("github-repositories")
        return providers

    @staticmethod
    def _dedupe_research_sources(sources: list[Any]) -> list[Any]:
        deduped: list[Any] = []
        seen: set[tuple[str, str]] = set()
        for source in sources:
            url = str(getattr(source, "url", "") or "").strip()
            title = str(getattr(source, "title", "") or "").strip().lower()
            key = (url, title)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    @classmethod
    def _workflow_research_brief_markdown(
        cls,
        objective: str,
        query: str,
        sources: list[Any],
        providers: list[str],
    ) -> str:
        lines = [
            "# Workflow Research Brief",
            "",
            "## Objective",
            objective or query,
            "",
            "## Query",
            query,
            "",
            "## Coverage",
            (
                f"Collected {len(sources)} provider-backed sources across "
                f"{', '.join(providers)} before any browser-first UI handoff."
            ),
            "",
            "## Sources",
        ]
        for index, source in enumerate(sources, start=1):
            title = str(
                getattr(source, "title", "") or getattr(source, "url", "source")
            )
            url = str(getattr(source, "url", "") or "")
            provider = str(getattr(source, "provider", "unknown") or "unknown")
            year = getattr(source, "year", None)
            claim = cls._workflow_source_claim(source)
            lines.extend(
                [
                    f"{index}. [{title}]({url})",
                    f"   Provider: {provider}" + (f" | Year: {year}" if year else ""),
                    f"   Evidence: {claim}",
                ]
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "Use this brief as the evidence-backed handoff input for any later authoring or app interaction.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _workflow_source_claim(source: Any) -> str:
        evidence = getattr(source, "evidence", None)
        if callable(evidence):
            claim = str((evidence() or {}).get("claim") or "").strip()
            if claim:
                return claim
        abstract = str(getattr(source, "abstract", "") or "").strip()
        if abstract:
            compact = re.sub(r"\s+", " ", abstract)
            return compact[:240]
        return str(getattr(source, "url", "") or "source snippet unavailable")

    def _workspace_relative_path(self, path: Path) -> str:
        return (
            path.resolve().relative_to(self.writer.workspace_root.resolve()).as_posix()
        )

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
            app_agent.skill_pack.visual_heavy or not app_agent.skill_pack.safe_windows
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
        trace_id = str(control.get("ledger_entry_id") or f"workflow_{int(time.time())}")
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
        golden_dir = self.writer.workspace_root / "benchmarks" / "golden_traces"
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
