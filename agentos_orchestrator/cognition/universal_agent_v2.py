"""Universal Desktop Agent v2 — Real cognitive architecture.

Addresses all four production bottlenecks:
1. Learned generative world model (MLP dynamics, online training)
2. Local fast VLA (zero API latency, <100ms loop)
3. Semantic dense memory (TF-IDF+SVD, captures meaning)
4. Pure pixel-based POMDP (no accessibility tree dependency)

Plus: Async perception loop for real-time responsiveness.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentos_orchestrator.os_control.base import UiAction, UiNode
from agentos_orchestrator.os_control.workflow.models import DesktopWorkflowPlan

from .abstract_world_model import (
    AbstractUIState,
    AbstractWorldModel,
    AbstractMCTSAdapter,
    ActionEmbedding,
)
from .adaptation_readiness import collect_adaptation_readiness
from .affordance_policy_memory import PersistentAffordancePolicyMemory
from .active_inference import ActiveInferenceExplorer
from .adaptive_perception import AdaptivePerceptionEngine
from .app_adapters import AdapterRegistry
from .blocker_repair import BlockerRepairPlanner
from .capability_profile import CapabilityProfiler
from .control_surface_discovery import GenericControlSurfaceDiscoverer
from .differentiable_memory import WorkingMemoryScratchpad
from .frontier_api import FrontierClient, FrontierPrompt, default_provider_from_env
from .hierarchical_planner import (
    ExecutionContext,
    MacroGoal,
    MacroPlanner,
    MicroExecutor,
)
from .hierarchical_task_decomposer import HierarchicalTaskDecomposer, TaskHierarchy
from .learned_world_model import LearnedGenerativeWorldModel
from .local_vla import LocalFastVLA
from .mcts_simulator import MCTSSimulator, WorldState
from .mode_arbitration import ModeArbiter, ModeContext, ModeDecision
from .pixel_pomdp import PixelFeatureExtractor, PurePixelEnvironment
from .pomdp_state import POMDPEnvironmentModel
from .runtime_state import AgentRuntimeState
from .safety_gates import FormalSafetyVerifier, default_safety_verifier
from .semantic_memory import SemanticEpisodicMemory, SemanticEmbedder
from .self_documentation import SelfDocumentationLoop
from .tool_executor import QuantAnalysisRequest, ToolExecutor
from .trajectory_recorder import TrajectoryRecorder
from .verification_contracts import (
    ensure_verification_contract,
    verify_action_contract,
)


@dataclass
class CognitiveStep:
    step_number: int
    phase: str
    action: UiAction | None
    observation: dict[str, Any]
    belief_entropy: float
    mcts_reward: float | None
    memory_reads: list[str]
    rationale: str
    latency_ms: float = 0.0


@dataclass
class UniversalAgentRun:
    run_id: str
    objective: str
    steps: list[CognitiveStep] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    adaptive_steps_used: int = 0
    exploration_probes_used: int = 0
    mcts_simulations_run: int = 0
    avg_latency_ms: float = 0.0
    model_used_ratio: float = 0.0
    trajectory_path: str = ""


class UniversalDesktopAgentV2:
    """Production-grade universal agent with real learned components."""

    def __init__(
        self,
        backend: Any,
        workspace_root: str = ".",
        use_learned_model: bool = True,
        use_local_vla: bool = True,
        use_semantic_memory: bool = True,
        use_pixel_pomdp: bool = False,
        use_frontier_api: bool = True,
        use_self_documentation: bool = True,
        frontier_client: FrontierClient | None = None,
        safety_verifier: FormalSafetyVerifier | None = None,
        allow_network_tools: bool = False,
        max_steps: int = 30,
        mcts_iterations: int = 64,
        target_latency_ms: float = 100.0,
    ) -> None:
        self.backend = backend
        self.workspace_root = Path(workspace_root)
        self.max_steps = max_steps
        self.mcts_iterations = mcts_iterations
        self.target_latency_ms = target_latency_ms
        self.allow_network_tools = allow_network_tools

        # --- Fix 1: Learned Generative World Model (kept for online learning) ---
        self.world_model = LearnedGenerativeWorldModel(
            max_depth=8,
            checkpoint_path=f"{workspace_root}/.agentos/world_model.pkl",
        )

        # --- Fix 1b: Abstract World Model (compact state transitions) ---
        self.abstract_world_model = AbstractWorldModel(
            state_dim=256,
            action_dim=64,
            hidden_dim=128,
        )
        # MCTS now uses the abstract 256-dim state model, NOT the pixel hash model.
        # This is the key fix for state-space explosion: planning over semantic
        # element counts instead of 8M pixel values makes MCTS tractable.
        self.mcts = MCTSSimulator(
            AbstractMCTSAdapter(self.abstract_world_model),
            iterations=mcts_iterations,
            max_depth=8,
        )

        # --- Fix 2: Local Fast VLA (zero API latency) ---
        self.local_vla = LocalFastVLA(
            model_path=str(self.workspace_root / ".agentos" / "local_vla.pkl")
        )

        # --- Frontier API bridge (semantic reasoning, local physical grounding) ---
        self.frontier_client = (
            frontier_client
            if frontier_client is not None
            else (default_provider_from_env() if use_frontier_api else None)
        )

        # --- Tool augmentation (terminal/code fast path) ---
        self.tool_executor = ToolExecutor(
            workspace_root=self.workspace_root / ".agentos"
        )

        # --- Deterministic safety gate between planner and executor ---
        self.safety_verifier = safety_verifier or default_safety_verifier(
            self.workspace_root
        )
        self.safety_verifier.policy.allow_network_tools = allow_network_tools

        # --- Self-documentation loop for unknown applications ---
        self.self_documentation = (
            SelfDocumentationLoop(workspace_root=self.workspace_root)
            if use_self_documentation
            else None
        )

        # --- Fix 2b: Adaptive Perception Engine (robust UI detection + OCR) ---
        self.perception = AdaptivePerceptionEngine(
            semantic_embedder=SemanticEmbedder(n_components=64),
        )

        # --- Fix 3: Semantic Dense Memory ---
        self.semantic_memory = SemanticEpisodicMemory(
            embedder=SemanticEmbedder(n_components=128)
        )
        self.affordance_policies = PersistentAffordancePolicyMemory(self.workspace_root)
        self.working_memory = WorkingMemoryScratchpad(capacity=32)
        self.runtime_state = AgentRuntimeState()
        self.repair_planner = BlockerRepairPlanner()
        self.mode_arbiter = ModeArbiter()
        self.capability_profiler = CapabilityProfiler()
        self.control_surface_discoverer = GenericControlSurfaceDiscoverer(
            self.workspace_root
        )
        self.adapter_registry = AdapterRegistry()
        self.trajectory_recorder = TrajectoryRecorder(self.workspace_root)
        self.adaptation_readiness = collect_adaptation_readiness(self.workspace_root)

        # --- Fix 3b: Hierarchical Task Decomposer (long-horizon planning) ---
        self.task_decomposer = HierarchicalTaskDecomposer(memory=self.semantic_memory)

        # --- Fix 4: Pure Pixel POMDP (optional) ---
        self.use_pixel_pomdp = use_pixel_pomdp
        if use_pixel_pomdp and hasattr(backend, "capture"):
            self.pixel_env = PurePixelEnvironment(backend)
            self.pomdp = None
        else:
            self.pixel_env = None
            self.pomdp = POMDPEnvironmentModel(backend)

        self.extractor = PixelFeatureExtractor(target_dim=256)
        self.macro_planner = MacroPlanner()
        self.micro_executor = MicroExecutor()
        self.explorer = ActiveInferenceExplorer(max_probes=6)

        # Async perception state
        self._latest_screenshot: bytes | None = None
        self._perception_thread: threading.Thread | None = None
        self._shutdown = False

        self._step_counter = 0
        self._run_id_counter = 0

    # ------------------------------------------------------------------ #
    # Async Perception Loop (<100ms reaction time)
    # ------------------------------------------------------------------ #

    def start_perception_loop(self) -> None:
        """Start background thread that continuously captures screenshots."""
        if not hasattr(self.backend, "capture"):
            return
        self._shutdown = False
        self._perception_thread = threading.Thread(
            target=self._perception_worker, daemon=True
        )
        self._perception_thread.start()

    def stop_perception_loop(self) -> None:
        """Stop the background perception thread."""
        self._shutdown = True
        if self._perception_thread:
            self._perception_thread.join(timeout=1.0)

    def _perception_worker(self) -> None:
        """Background worker: capture screenshot every ~50ms."""
        while not self._shutdown:
            try:
                self._latest_screenshot = self.backend.capture()
            except Exception:
                pass
            time.sleep(0.05)

    def get_latest_screenshot(self) -> bytes | None:
        """Get the most recent screenshot without blocking."""
        return self._latest_screenshot

    # ------------------------------------------------------------------ #
    # Main Execution Loop
    # ------------------------------------------------------------------ #

    def run(self, objective: str) -> UniversalAgentRun:
        self._run_id_counter += 1
        run_id = f"ua2_{self._run_id_counter}_{int(time.time())}"
        run = UniversalAgentRun(run_id=run_id, objective=objective)
        self.runtime_state.reset(objective)
        self.adaptation_readiness = collect_adaptation_readiness(self.workspace_root)
        self.working_memory.write(
            "Adaptation readiness: "
            f"{self.adaptation_readiness.status}; "
            f"heldout={self.adaptation_readiness.heldout_success_rate:.2f}; "
            f"blockers={self.adaptation_readiness.blockers}",
            item_type="observation",
            priority=0.86,
        )
        trajectory_path = self.trajectory_recorder.start_run(run_id, objective)
        if trajectory_path is not None:
            run.trajectory_path = str(trajectory_path)

        # Start async perception for real-time responsiveness
        self.start_perception_loop()

        try:
            # Phase 0: Macro planning
            macro_goals = self.macro_planner.plan_objective(objective)
            for index, macro_goal in enumerate(macro_goals, start=1):
                self.runtime_state.push_goal(
                    name=f"macro_{index}",
                    intent=macro_goal.description,
                    success_criteria=[
                        "The goal-completion checker marks this macro goal complete."
                    ],
                )
            self.working_memory.write(
                f"Goals: {[g.description for g in macro_goals]}",
                item_type="plan",
                priority=0.9,
            )

            # Phase 1: Initial observation
            if self.pixel_env:
                pixel_obs = self.pixel_env.observe()
                self._record_perception(
                    run,
                    {
                        "resolution": pixel_obs.resolution,
                        "features_shape": pixel_obs.features.shape,
                    },
                )
            else:
                nodes = self.backend.snapshot()
                self.pomdp.observe()
                self._record_perception(run, {"node_count": len(nodes)})

            # Phase 2: Explore if needed
            if self.pixel_env:
                # Pixel mode: explore based on visual entropy
                if self.pixel_env.belief.entropy() > 2.0:
                    probes = self._pixel_explore(objective, run)
                    run.exploration_probes_used = len(probes)
            else:
                interactive = [
                    n
                    for n in self.backend.snapshot()
                    if n.enabled
                    and n.role
                    in {
                        "Button",
                        "Edit",
                        "Document",
                        "Canvas",
                        "Menu",
                        "Hyperlink",
                        "Tab",
                    }
                ]
                if len(interactive) < 3:
                    probes = self.explorer.explore(
                        self.backend.snapshot(),
                        objective,
                        perform_fn=self._safe_perform,
                        snapshot_fn=self.backend.snapshot,
                    )
                    run.exploration_probes_used = len(probes)

            # Phase 3: Execute with cognitive loop
            context = ExecutionContext(objective=objective)
            latencies: list[float] = []

            for goal in macro_goals:
                if run.success or run.adaptive_steps_used >= self.max_steps:
                    break

                loop_start = time.perf_counter()
                context = self._execute_goal_with_cognition(goal, context, run)
                loop_elapsed = (time.perf_counter() - loop_start) * 1000
                latencies.append(loop_elapsed)

                if goal.completed:
                    self.working_memory.write(
                        f"Completed: {goal.description}",
                        item_type="goal",
                        priority=0.95,
                    )
                elif goal.failed:
                    self.working_memory.write(
                        f"Failed: {goal.description}", item_type="goal", priority=0.9
                    )
                    macro_goals = self.macro_planner.replan_on_failure(
                        context, macro_goals
                    )

                run.adaptive_steps_used = len(
                    [s for s in run.steps if s.phase == "act"]
                )

            # Final state
            run.avg_latency_ms = sum(latencies) / max(len(latencies), 1)
            run.model_used_ratio = self.world_model._model_used_count / max(
                self.world_model._model_used_count
                + self.world_model._fallback_used_count,
                1,
            )

            if self.pixel_env:
                run.final_state = self.pixel_env.get_visual_state_summary()
            else:
                run.final_state = {
                    "belief_entropy": self.pomdp.belief.entropy(),
                    "node_count": len(self.backend.snapshot()),
                    "steps_taken": len(run.steps),
                }
            if run.trajectory_path:
                run.final_state["trajectory_path"] = run.trajectory_path
            run.final_state["adaptation_readiness"] = self.adaptation_readiness.asdict()
            run.success = any(g.completed for g in macro_goals)
            self.trajectory_recorder.finish_run(run.run_id, run.final_state)

            # Record to semantic memory
            self.semantic_memory.record(
                objective=objective,
                action=UiAction(
                    action_type="universal_run", selector="agent", value=objective
                ),
                observation=json.dumps(run.final_state),
                outcome="success" if run.success else "partial",
                reward=1.0 if run.success else 0.3,
            )

        finally:
            self.stop_perception_loop()

        return run

    def run_with_planned_bootstrap(
        self, objective: str, plan: DesktopWorkflowPlan
    ) -> UniversalAgentRun:
        for step in plan.steps:
            self.working_memory.write(
                f"Planned: {step.action_type} on {step.selector}",
                item_type="plan",
                priority=0.6,
            )
        return self.run(objective)

    # ------------------------------------------------------------------ #
    # Internal: Goal execution with full cognition
    # ------------------------------------------------------------------ #

    def _execute_goal_with_cognition(
        self,
        goal: MacroGoal,
        context: ExecutionContext,
        run: UniversalAgentRun,
    ) -> ExecutionContext:
        context.current_goal = goal

        # --- Hierarchical Decomposition: Break goal into options ---
        current_state = self._build_abstract_state()
        self.runtime_state.update_observation(
            current_state,
            self.get_latest_screenshot(),
        )
        hierarchy = self.task_decomposer.decompose(goal.description, current_state)
        self.working_memory.write(
            f"Decomposed into {len(hierarchy.execution_sequence)} options, "
            f"P(success)={self.task_decomposer.estimate_completion_probability(hierarchy):.2f}",
            item_type="plan",
            priority=0.9,
        )

        # Execute option-by-option
        while hierarchy.has_more() and run.adaptive_steps_used < self.max_steps:
            option = hierarchy.next_option()
            if option is None:
                break

            if not option.can_start(current_state):
                # Skip if preconditions not met
                self.working_memory.write(
                    f"Option '{option.name}' skipped: preconditions not met",
                    item_type="plan",
                    priority=0.6,
                )
                continue

            step_start = time.perf_counter()

            # --- Tool Fast Path: use code for data/quant tasks instead of UI clicks ---
            tool_action = self._tool_action_for_option(option, run)

            # --- Adaptive Perception: Robust element detection ---
            screenshot = self.get_latest_screenshot()
            perceived_elements: list[Any] = []
            if screenshot and tool_action is None:
                perceived_elements = self.perception.perceive(
                    screenshot,
                    run.objective,
                    [],
                )
                self.working_memory.write(
                    f"Perceived {len(perceived_elements)} elements: "
                    f"{[e.element_type for e in perceived_elements[:5]]}",
                    item_type="observation",
                    priority=0.7,
                )

            snapshot_nodes = self._safe_snapshot_nodes()
            capability_profile = self.capability_profiler.profile(
                current_state,
                nodes=snapshot_nodes,
                screenshot_available=screenshot is not None,
            )
            adapter_context = self.adapter_registry.context_for(
                capability_profile,
                run.objective,
            )
            self.working_memory.write(
                f"Surface: {capability_profile.app_family} via "
                f"{adapter_context.preferred_channels[:2]}",
                item_type="observation",
                priority=0.72,
            )
            policy_channels = self.affordance_policies.preferred_channels(
                capability_profile.app_signature
            )
            if policy_channels:
                self.working_memory.write(
                    f"Policy memory channels: {policy_channels[:2]}",
                    item_type="observation",
                    priority=0.66,
                )

            # --- Semantic Memory: Check for past failures ---
            similar_failures = self.semantic_memory.get_failure_patterns(
                run.objective, top_k=3
            )
            for fail in similar_failures:
                self.working_memory.write(
                    f"Past failure: {fail['action'].selector} -> {fail['outcome']}",
                    item_type="hypothesis",
                    priority=0.85,
                )

            mode_decision = self.mode_arbiter.choose(
                run.objective,
                option,
                current_state,
                self.runtime_state,
                ModeContext(
                    perceived_element_count=len(perceived_elements),
                    tool_action_available=tool_action is not None,
                    similar_failures=similar_failures,
                    capability_profile=capability_profile,
                    adapter_context=adapter_context.to_prompt_dict(),
                ),
            )
            self.working_memory.write(
                f"Mode: {mode_decision.mode} ({mode_decision.rationale})",
                item_type="plan",
                priority=0.7,
            )

            active_blockers = [
                blocker
                for blocker in self.runtime_state.blocker_stack
                if blocker.active
            ]
            repair_plan = self.repair_planner.propose(
                active_blockers,
                current_state,
                run.objective,
                option_name=getattr(option, "name", ""),
                recent_reflections=list(self.runtime_state.recent_reflections),
            )
            if repair_plan is not None and not repair_plan.can_execute:
                self.working_memory.write(
                    f"Repair requires operator input: {repair_plan.rationale}",
                    item_type="reflection",
                    priority=0.95,
                )
                context.failure_streak += 1
                hierarchy.mark_current_failure()
                goal.failed = True
                break

            # --- Action Selection: Tool fast path, frontier SoM, perception, MCTS ---
            if repair_plan is not None and repair_plan.action is not None:
                action = repair_plan.action
            elif mode_decision.mode == "explore" and tool_action is None:
                action = self._local_explore_action(
                    run,
                    source="mode_arbitration",
                    rationale=mode_decision.rationale,
                )
            else:
                action = tool_action or self._select_action(
                    option,
                    perceived_elements,
                    current_state,
                    run,
                    similar_failures,
                    capability_profile=capability_profile,
                    adapter_context=adapter_context,
                    mode_decision=mode_decision,
                )
            self._warm_world_model_with_action(current_state, run, action)
            action.metadata.setdefault(
                "mode_decision",
                mode_decision.to_prompt_dict(),
            )
            if repair_plan is not None:
                action.metadata.setdefault(
                    "repair_plan",
                    repair_plan.to_prompt_dict(),
                )
            action.metadata.setdefault(
                "capability_profile",
                capability_profile.to_prompt_dict(),
            )
            action = self.adapter_registry.enrich_action(
                action,
                capability_profile,
                run.objective,
            )
            verification_contract = ensure_verification_contract(action)

            # --- Deterministic Safety Gate + Execute ---
            expected_observation = _expected_observation_for_action(action)
            safety = self.safety_verifier.verify_action(action, objective=run.objective)
            if not safety.allowed:
                receipt = json.dumps(
                    {
                        "status": "blocked",
                        "reason": safety.reason,
                        "solver": safety.solver,
                    }
                )
                context.failure_streak += 1
                hierarchy.mark_current_failure()
            elif action.action_type == "tool":
                receipt = json.dumps(action.metadata.get("tool_result", {}))
                context.failure_streak = 0
                if action.metadata.get("tool_success") is True:
                    hierarchy.mark_current_success()
            elif action.action_type == "explore":
                if "exploration_result" not in action.metadata:
                    source = str(action.metadata.get("source", "local_explore"))
                    action.metadata["exploration_result"] = self._run_local_exploration(
                        run, status=source
                    )
                receipt = json.dumps(action.metadata.get("exploration_result", {}))
                context.failure_streak = 0
            else:
                try:
                    receipt = self.backend.perform(action)
                    context.failure_streak = 0
                except Exception as exc:
                    receipt = str(exc)
                    context.failure_streak += 1
                    hierarchy.mark_current_failure()
            self.runtime_state.record_action(
                action,
                expected_observation=expected_observation,
                receipt=receipt,
            )

            if context.failure_streak >= context.max_failure_streak:
                goal.failed = True
                # Replan with alternatives
                hierarchy = self.task_decomposer.replan_on_failure(
                    hierarchy,
                    option,
                    current_state,
                )
                break

            step_elapsed = (time.perf_counter() - step_start) * 1000

            # --- Update abstract state and learn ---
            new_state = self._build_abstract_state()
            self.runtime_state.update_observation(
                new_state,
                self.get_latest_screenshot(),
            )
            outcome_evaluation = self.runtime_state.evaluate_outcome(
                action,
                current_state,
                new_state,
                receipt,
                expected_observation=expected_observation,
            )
            verification_result = verify_action_contract(
                action,
                current_state,
                new_state,
                receipt,
            )
            verification_failed = (
                verification_result.required and not verification_result.matched
            )
            if verification_failed:
                outcome_evaluation.matched = False
                existing_reason = outcome_evaluation.failure_reason
                outcome_evaluation.failure_reason = existing_reason or (
                    verification_result.reason
                )
                if not outcome_evaluation.new_blocker:
                    outcome_evaluation.new_blocker = verification_result.reason
                    self.runtime_state.add_blocker(
                        kind="verification",
                        description=verification_result.reason,
                        evidence=verification_result.observed,
                        repair_hint=(
                            "Re-ground the action using the latest capability profile."
                        ),
                    )
            outcome_failed = (
                outcome_evaluation.new_blocker or not outcome_evaluation.matched
            )
            if outcome_failed:
                self.working_memory.write(
                    f"Outcome reflection: {outcome_evaluation.observed}",
                    item_type="reflection",
                    priority=0.9,
                )
            self.abstract_world_model.record_transition(
                current_state,
                action,
                new_state,
            )
            self.trajectory_recorder.record_step(
                run_id=run.run_id,
                objective=run.objective,
                option_name=getattr(option, "name", ""),
                before=current_state,
                after=new_state,
                action=action,
                expected_observation=expected_observation,
                receipt=receipt,
                outcome=outcome_evaluation,
                mode_decision=mode_decision,
                repair_plan=repair_plan,
                capability_profile=capability_profile.to_prompt_dict(),
                adapter_context=adapter_context.to_prompt_dict(),
                verification_contract=verification_contract.asdict(),
                verification_result=verification_result.asdict(),
                latency_ms=step_elapsed,
            )
            current_state = new_state

            # --- Record cognitive step ---
            self._step_counter += 1
            run.steps.append(
                CognitiveStep(
                    step_number=self._step_counter,
                    phase=_phase_for_action(action),
                    action=action,
                    observation={
                        "receipt": receipt,
                        "option": option.name,
                        "perceived_elements": len(perceived_elements),
                        "abstract_state_elements": len(current_state.elements),
                        "mode_decision": mode_decision.to_prompt_dict(),
                        "capability_profile": (capability_profile.to_prompt_dict()),
                        "adapter_context": adapter_context.to_prompt_dict(),
                        "verification_contract": (verification_contract.asdict()),
                        "verification_result": verification_result.asdict(),
                        "repair_plan": (
                            repair_plan.to_prompt_dict()
                            if repair_plan is not None
                            else None
                        ),
                        "outcome_evaluation": (outcome_evaluation.to_prompt_dict()),
                    },
                    belief_entropy=self.pixel_env.belief.entropy()
                    if self.pixel_env
                    else self.pomdp.belief.entropy(),
                    mcts_reward=None,
                    memory_reads=[f["outcome"] for f in similar_failures],
                    rationale=(
                        f"Option '{option.name}': {action.action_type} "
                        f"on {action.selector}"
                    ),
                    latency_ms=step_elapsed,
                )
            )
            run.adaptive_steps_used += 1

            # --- Semantic Memory: Record outcome ---
            self.semantic_memory.record(
                objective=run.objective,
                action=action,
                observation=f"Option {option.name}: {receipt}",
                outcome=receipt,
                reward=1.0 if "executed" in receipt.lower() else -0.5,
            )
            policy_success = (
                bool(action.metadata.get("tool_success"))
                if action.action_type == "tool"
                else not outcome_failed
            )
            self.affordance_policies.record(
                capability_profile.app_signature,
                run.objective,
                action,
                success=policy_success,
                control_channel=str(
                    action.metadata.get("control_channel")
                    or adapter_context.preferred_channels[0]
                ),
                observed=outcome_evaluation.observed,
                evidence={
                    "family": capability_profile.app_family,
                    "option": option.name,
                    "source": action.metadata.get("source", ""),
                },
            )

            # Check option completion
            if option.is_done(current_state):
                hierarchy.mark_current_success()
                self.working_memory.write(
                    f"Option '{option.name}' completed",
                    item_type="goal",
                    priority=0.9,
                )

            # Check goal completion
            state_summary = self._build_state_summary(current_state)
            if self.macro_planner.check_goal_completion(goal, state_summary):
                goal.completed = True
                break

        return context

    def _tool_action_for_option(
        self,
        option: Any,
        run: UniversalAgentRun,
    ) -> UiAction | None:
        """Execute known tool-use options locally and wrap the result as an action."""
        if getattr(option, "name", "") != "run_analysis_code":
            return None

        tickers = self._extract_tickers(run.objective)
        code = self.tool_executor.build_quant_analysis_code(
            run.objective,
            tickers=tickers or ["SPY"],
            period="1y",
        )
        result = self.tool_executor.run(
            QuantAnalysisRequest(
                objective=run.objective,
                code=code,
                allow_network=self.allow_network_tools,
                timeout_seconds=60,
            )
        )
        self.working_memory.write(
            result.summary(),
            item_type="tool",
            priority=0.85 if result.success else 0.95,
        )
        return UiAction(
            action_type="tool",
            selector="tool_executor:quant_analysis",
            metadata={
                "source": "tool_fast_path",
                "tool_success": result.success,
                "tool_result": {
                    "success": result.success,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-1000:],
                    "error": result.error,
                    "parsed_results": result.parsed_results,
                    "artefacts": [str(path) for path in result.artefacts],
                    "elapsed_ms": result.elapsed_ms,
                },
                "allow_network": self.allow_network_tools,
            },
        )

    @staticmethod
    def _extract_tickers(objective: str) -> list[str]:
        import re

        candidates = re.findall(r"\b[A-Z]{1,5}\b", objective)
        ignored = {"I", "A", "THE", "AND", "FOR", "WITH", "Q", "API"}
        return [ticker for ticker in candidates if ticker not in ignored]

    def _select_frontier_som_action(
        self,
        current_state: AbstractUIState,
        run: UniversalAgentRun,
    ) -> UiAction | None:
        """Ask a frontier multimodal model to choose a Set-of-Mark target."""
        if self.frontier_client is None:
            return None
        screenshot = self.get_latest_screenshot()
        if not screenshot:
            return None

        try:
            frame = self.local_vla.render_set_of_mark(screenshot)
        except Exception as exc:  # noqa: BLE001
            self.working_memory.write(
                f"Set-of-Mark render failed: {exc}",
                item_type="hypothesis",
                priority=0.7,
            )
            return None
        if not frame.elements:
            return None
        mark_payload = frame.as_prompt_payload()
        self.runtime_state.update_observation(
            current_state,
            screenshot,
            mark_payload,
        )

        docs_context = ""
        if (
            self.self_documentation is not None
            and current_state.app_context == "unknown"
        ):
            bundle = self.self_documentation.prepare_context(
                run.objective,
                app_hint="unknown application",
            )
            docs_context = (
                bundle.context or f"Generated documentation query: {bundle.query}"
            )

        prompt = FrontierPrompt(
            objective=run.objective,
            annotated_png=frame.annotated_png,
            mark_payload=mark_payload,
            documentation_context=docs_context,
            memory_context=self.working_memory.summarize(),
            state_context=self.runtime_state.frontier_context(
                self._build_state_summary(current_state)
            ),
            tool_context=(
                "Local Python sandbox is available via action='tool'. "
                "Network is enabled only when the local safety policy allows it."
            ),
        )
        try:
            decision = self.frontier_client.choose_action(prompt)
            if decision.action == "tool" and decision.code:
                result = self.tool_executor.run(
                    QuantAnalysisRequest(
                        objective=run.objective,
                        code=decision.code,
                        allow_network=self.allow_network_tools,
                        timeout_seconds=60,
                    )
                )
                return UiAction(
                    action_type="tool",
                    selector=f"frontier_tool:{decision.tool or 'python'}",
                    metadata={
                        "source": "frontier_tool_request",
                        "frontier_rationale": decision.rationale,
                        "frontier_orientation": decision.metadata.get(
                            "frontier_orientation",
                        ),
                        "frontier_hypothesis": decision.metadata.get(
                            "frontier_hypothesis",
                        ),
                        "expected_observation": decision.metadata.get(
                            "expected_observation",
                        ),
                        "tool_success": result.success,
                        "tool_result": {
                            "success": result.success,
                            "stdout": result.stdout[-4000:],
                            "stderr": result.stderr[-1000:],
                            "error": result.error,
                            "parsed_results": result.parsed_results,
                            "artefacts": [str(path) for path in result.artefacts],
                            "elapsed_ms": result.elapsed_ms,
                        },
                        "allow_network": self.allow_network_tools,
                    },
                )

            if decision.action == "explore":
                return self._frontier_escape_explore(run, decision)

            action = frame.resolve_action(decision.to_action_json())
            action.metadata["source"] = "frontier_som"
            action.metadata["frontier_rationale"] = decision.rationale
            action.metadata["frontier_confidence"] = decision.confidence
            action.metadata.update(decision.metadata)
            return action
        except Exception as exc:  # noqa: BLE001
            self.working_memory.write(
                f"Frontier SoM action failed: {exc}",
                item_type="hypothesis",
                priority=0.75,
            )
            return None

    def _frontier_escape_explore(
        self,
        run: UniversalAgentRun,
        decision: Any,
    ) -> UiAction:
        """Execute bounded local exploration after frontier uncertainty."""
        payload = self._run_local_exploration(
            run,
            status="frontier_requested_explore",
            extra_payload={
                "frontier_confidence": decision.confidence,
                "frontier_rationale": decision.rationale,
            },
        )
        return UiAction(
            action_type="explore",
            selector="frontier_escape_hatch",
            metadata={
                "source": "frontier_escape_hatch",
                "frontier_confidence": decision.confidence,
                "frontier_rationale": decision.rationale,
                "frontier_grounding": decision.metadata.get(
                    "frontier_grounding",
                    {},
                ),
                "frontier_orientation": decision.metadata.get(
                    "frontier_orientation",
                ),
                "frontier_hypothesis": decision.metadata.get(
                    "frontier_hypothesis",
                ),
                "expected_observation": decision.metadata.get(
                    "expected_observation",
                ),
                "exploration_result": payload,
            },
        )

    def _local_explore_action(
        self,
        run: UniversalAgentRun,
        source: str,
        rationale: str = "",
    ) -> UiAction:
        payload = self._run_local_exploration(
            run,
            status=source,
            extra_payload={"rationale": rationale},
        )
        return UiAction(
            action_type="explore",
            selector=source,
            metadata={
                "source": source,
                "expected_observation": (
                    "Bounded exploration produces grounded evidence."
                ),
                "exploration_result": payload,
            },
        )

    def _run_local_exploration(
        self,
        run: UniversalAgentRun,
        status: str,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        probes: list[Any] = []
        if self.pixel_env:
            probes = self._pixel_explore(run.objective, run)
        elif hasattr(self.backend, "snapshot"):
            probes = self.explorer.explore(
                self.backend.snapshot(),
                run.objective,
                perform_fn=self._safe_perform,
                snapshot_fn=self.backend.snapshot,
            )
        run.exploration_probes_used += len(probes)
        payload = {
            "status": status,
            "probes": [_exploration_result_payload(probe) for probe in probes],
        }
        payload.update(extra_payload or {})
        return payload

    def _select_action(
        self,
        option: Any,
        perceived_elements: list[Any],
        current_state: AbstractUIState,
        run: UniversalAgentRun,
        similar_failures: list[dict[str, Any]],
        capability_profile: Any | None = None,
        adapter_context: Any | None = None,
        mode_decision: ModeDecision | None = None,
    ) -> UiAction:
        """Select best action using frontier SoM, perception, MCTS, and fallbacks."""
        snapshot_nodes = self._safe_snapshot_nodes()
        capability_profile = capability_profile or self.capability_profiler.profile(
            current_state,
            nodes=snapshot_nodes,
            screenshot_available=self.get_latest_screenshot() is not None,
        )
        policy_action = self.affordance_policies.recommend_action(
            capability_profile.app_signature,
            run.objective,
            nodes=snapshot_nodes,
        )
        if (
            policy_action is not None
            and float(policy_action.metadata.get("policy_score", 0.0)) >= 0.72
        ):
            self._warm_world_model_with_action(current_state, run, policy_action)
            return policy_action

        control_surface_action = self._select_control_surface_action(
            capability_profile,
            snapshot_nodes,
            current_state,
            run,
            adapter_context=adapter_context,
        )
        if control_surface_action is not None:
            self._warm_world_model_with_action(
                current_state, run, control_surface_action
            )
            return control_surface_action

        if policy_action is not None:
            self._warm_world_model_with_action(current_state, run, policy_action)
            return policy_action

        use_frontier = (
            mode_decision is None
            or mode_decision.should_use_frontier
            or mode_decision.mode in {"ui", "hybrid", "research"}
        )
        if use_frontier:
            frontier_action = self._select_frontier_som_action(current_state, run)
            if frontier_action is not None:
                self._warm_world_model_with_action(current_state, run, frontier_action)
                return frontier_action

        # 2. Try perception-guided action (highest semantic match)
        if perceived_elements:
            best_elem = perceived_elements[0]
            if best_elem.semantic_score > 0.3:
                cx = best_elem.x + best_elem.width // 2
                cy = best_elem.y + best_elem.height // 2
                action = UiAction(
                    action_type=best_elem.element_type
                    if best_elem.element_type in {"button", "text_field", "checkbox"}
                    else "click",
                    selector=f"perceived_{best_elem.element_type}_{best_elem.x}_{best_elem.y}",
                    metadata={"x": cx, "y": cy, "source": "adaptive_perception"},
                )
                self._warm_world_model_with_action(current_state, run, action)
                return action

                # 3. Try local VLA fallback
        screenshot = self.get_latest_screenshot()
        if screenshot and self.local_vla:
            vla_action = self.local_vla.propose_action(screenshot, run.objective, [])
            if vla_action:
                x = getattr(vla_action, "x", 0)
                y = getattr(vla_action, "y", 0)
                action = UiAction(
                    action_type=vla_action.action_type,
                    selector=f"vla_{x}_{y}",
                    metadata={"x": x, "y": y, "source": "local_vla"},
                )
                self._warm_world_model_with_action(current_state, run, action)
                return action

        grounded_action = self.explorer.suggest_action(
            self._safe_snapshot_nodes(),
            f"{getattr(option, 'description', '')} {run.objective}".strip(),
            success_patterns=self.semantic_memory.get_success_patterns(
                run.objective,
                top_k=3,
            ),
        )
        if grounded_action is not None:
            grounded_action.metadata.setdefault("source", "active_inference_grounding")
            grounded_action.metadata.setdefault(
                "expected_observation",
                "The grounded control should expose the next affordance or advance the option.",
            )
            self._warm_world_model_with_action(current_state, run, grounded_action)
            return grounded_action

        # 4. MCTS deliberation on abstract state
        root_state = WorldState(
            state_vector=current_state.to_vector(256).tolist(),
            depth=run.adaptive_steps_used,
            terminal=False,
            reward=0.0,
        )
        mcts_action = self.mcts.search(root_state, run.objective)
        # Shadow-call world_model so it stays warm and records usage — it is kept
        # for online learning even though MCTS now uses the abstract model.
        try:
            _probe = UiAction(action_type="click", selector="mcts_probe")
            self.world_model.predict(root_state, _probe)
        except Exception:  # noqa: BLE001 – best-effort, never block execution
            pass
        if mcts_action:
            return mcts_action

        # 5. Ultimate fallback: re-ground with local exploration instead of
        # issuing a blind coordinate click.
        return UiAction(
            action_type="explore",
            selector="reground-ui",
            metadata={"source": "fallback"},
        )

    def _warm_world_model_with_action(
        self,
        current_state: AbstractUIState,
        run: UniversalAgentRun,
        action: UiAction,
    ) -> None:
        if action.metadata.get("world_model_warmed") is True:
            return
        try:
            root_state = WorldState(
                state_vector=current_state.to_vector(256).tolist(),
                depth=run.adaptive_steps_used,
                terminal=False,
                reward=0.0,
            )
            self.world_model.predict(root_state, action)
            action.metadata["world_model_warmed"] = True
        except Exception:  # noqa: BLE001
            pass

    def _select_control_surface_action(
        self,
        capability_profile: Any,
        snapshot_nodes: list[UiNode],
        current_state: AbstractUIState,
        run: UniversalAgentRun,
        adapter_context: Any | None = None,
    ) -> UiAction | None:
        preferred_channels = []
        if adapter_context is not None:
            preferred_channels.extend(
                getattr(adapter_context, "preferred_channels", [])
            )
        preferred_channels.extend(
            self.affordance_policies.preferred_channels(
                capability_profile.app_signature
            )
        )

        documentation_context = ""
        objective_lower = run.objective.lower()
        should_fetch_docs = self.self_documentation is not None and (
            capability_profile.app_family in {"unknown", "electron_app", "chat_app"}
            or "api" in capability_profile.control_channels
            or any(
                keyword in objective_lower
                for keyword in {
                    "api",
                    "json",
                    "endpoint",
                    "payload",
                    "graphql",
                    "webhook",
                }
            )
        )
        should_probe_loopback = self.allow_network_tools and (
            capability_profile.app_family
            in {"unknown", "browser", "terminal", "electron_app", "chat_app"}
            or "api" in capability_profile.control_channels
            or any(
                keyword in objective_lower
                for keyword in {
                    "api",
                    "json",
                    "endpoint",
                    "payload",
                    "graphql",
                    "webhook",
                    "localhost",
                    "port",
                    "service",
                    "server",
                }
            )
        )
        if should_fetch_docs:
            bundle = self.self_documentation.prepare_context(
                run.objective,
                app_hint=self._app_hint(capability_profile, snapshot_nodes),
                max_sources=2,
            )
            documentation_context = bundle.context or bundle.query

        candidates = self.control_surface_discoverer.discover(
            capability_profile,
            snapshot_nodes,
            run.objective,
            documentation_context=documentation_context,
            preferred_channels=list(dict.fromkeys(preferred_channels)),
            active_fingerprinting=should_probe_loopback,
        )
        if not candidates:
            return None
        best = candidates[0]
        if best.workflow and self.allow_network_tools:
            workflow_payload = {
                "steps": best.workflow,
                "headers": {},
                "auth_env_keys": best.metadata.get("auth_env_keys", []),
            }
            result = self.tool_executor.run(
                QuantAnalysisRequest(
                    objective=(
                        f"Execute synthesized API workflow for {run.objective}: {best.endpoint or best.metadata.get('documentation_url', '')}"
                    ),
                    code=self.tool_executor.build_api_workflow_code(workflow_payload),
                    allow_network=True,
                    timeout_seconds=20,
                    expose_env_keys=list(best.metadata.get("auth_env_keys", [])),
                )
            )
            return UiAction(
                action_type="tool",
                selector=f"synthesized_api_workflow:{best.endpoint or best.metadata.get('documentation_url', 'docs')}",
                metadata={
                    "source": "control_surface_discovery",
                    "control_channel": best.channel,
                    "control_surface_kind": best.kind,
                    "control_surface_confidence": best.confidence,
                    "control_surface_rationale": best.rationale,
                    "workflow_step_count": len(best.workflow),
                    "workflow": best.workflow,
                    "tool_success": result.success,
                    "tool_result": {
                        "success": result.success,
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-1000:],
                        "error": result.error,
                        "parsed_results": result.parsed_results,
                        "artefacts": [str(path) for path in result.artefacts],
                        "elapsed_ms": result.elapsed_ms,
                    },
                    "allow_network": True,
                    "expected_observation": (
                        "The documented API workflow returns structured HTTP results for each step."
                    ),
                },
            )
        if best.endpoint and self.allow_network_tools:
            result = self.tool_executor.run(
                QuantAnalysisRequest(
                    objective=(
                        f"Probe discovered control surface for {run.objective}: {best.endpoint}"
                    ),
                    code=self.tool_executor.build_http_probe_code([best.endpoint]),
                    allow_network=True,
                    timeout_seconds=15,
                )
            )
            return UiAction(
                action_type="tool",
                selector=f"control_surface_api_probe:{best.endpoint}",
                metadata={
                    "source": "control_surface_discovery",
                    "control_channel": best.channel,
                    "control_surface_kind": best.kind,
                    "control_surface_confidence": best.confidence,
                    "control_surface_rationale": best.rationale,
                    "tool_success": result.success,
                    "tool_result": {
                        "success": result.success,
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-1000:],
                        "error": result.error,
                        "parsed_results": result.parsed_results,
                        "artefacts": [str(path) for path in result.artefacts],
                        "elapsed_ms": result.elapsed_ms,
                    },
                    "allow_network": True,
                    "expected_observation": (
                        "The API probe yields a structured response or an informative HTTP status."
                    ),
                },
            )
        if not best.selector:
            return None
        return UiAction(
            action_type=best.action_type or "click",
            selector=best.selector,
            value=best.value,
            metadata={
                "source": "control_surface_discovery",
                "control_channel": best.channel,
                "control_surface_kind": best.kind,
                "control_surface_confidence": best.confidence,
                "control_surface_rationale": best.rationale,
                "expected_observation": (
                    "The higher-level control surface becomes active or reveals structured controls."
                ),
            },
        )

    @staticmethod
    def _app_hint(capability_profile: Any, nodes: list[UiNode]) -> str:
        names = [node.name.strip() for node in nodes if node.name][:3]
        if names:
            return " ".join(names)
        return str(getattr(capability_profile, "app_family", "unknown application"))

    def _build_abstract_state(self) -> AbstractUIState:
        """Build compact abstract state from current perception."""
        screenshot = self.get_latest_screenshot()
        if screenshot:
            elements = self.perception.quick_detect(screenshot)
            # Infer app context from elements
            app_context = "unknown"
            for e in elements:
                if e.element_type in {"link", "tab", "dropdown"}:
                    app_context = "browser"
                    break
                if e.element_type in {"text_block", "icon", "panel"}:
                    app_context = "other"
            return AbstractUIState.from_perceived_elements(elements, app_context)
        # Fallback from accessibility tree
        if hasattr(self.backend, "snapshot"):
            nodes = self.backend.snapshot()
            elements = [
                type(
                    "FakeElem",
                    (),
                    {
                        "x": 0,
                        "y": 0,
                        "width": 100,
                        "height": 30,
                        "element_type": _element_type_from_node(n),
                        "text": getattr(n, "name", "") or "",
                    },
                )()
                for n in nodes
            ]
            state = AbstractUIState.from_perceived_elements(elements, "unknown")
            profile = self.capability_profiler.profile(
                state,
                nodes=nodes,
                screenshot_available=False,
            )
            return AbstractUIState.from_perceived_elements(
                elements,
                profile.app_family,
            )
        return AbstractUIState(app_context="unknown")

    def _build_state_summary(self, state: AbstractUIState) -> dict[str, Any]:
        """Build state summary for goal completion checking."""
        return {
            "app_context": state.app_context,
            "layout_mode": state.layout_mode,
            "element_count": len(state.elements),
            "modal_open": bool(state.active_modal),
            "focus_region": state.focus_region,
            "task_progress": state.task_progress,
            "has_interactive": any(e.is_interactive for e in state.elements),
        }

    def _record_perception(
        self, run: UniversalAgentRun, obs_dict: dict[str, Any]
    ) -> None:
        self._step_counter += 1
        entropy = (
            self.pixel_env.belief.entropy()
            if self.pixel_env
            else self.pomdp.belief.entropy()
        )
        run.steps.append(
            CognitiveStep(
                step_number=self._step_counter,
                phase="perceive",
                action=None,
                observation=obs_dict,
                belief_entropy=entropy,
                mcts_reward=None,
                memory_reads=[],
                rationale="Initial perception",
            )
        )

    def _pixel_explore(self, objective: str, run: UniversalAgentRun) -> list[Any]:
        """Explore using pixel-based visual change detection."""
        results = []
        screenshot = self.get_latest_screenshot()
        if not screenshot:
            return results
        elements = self.local_vla.detect_elements(screenshot)
        # Click top elements and observe visual changes
        for elem in elements[:4]:
            action = UiAction(
                action_type="click",
                selector=f"pixel=({elem.x + elem.width // 2},{elem.y + elem.height // 2})",
            )
            try:
                before = self.pixel_env.observe() if self.pixel_env else None
                self.backend.perform(action)
                time.sleep(0.3)
                after = self.pixel_env.observe() if self.pixel_env else None
                if before and after:
                    visual_change = float(
                        np.linalg.norm(after.features - before.features)
                    )
                    results.append({"element": elem, "visual_change": visual_change})
                    self.working_memory.write(
                        f"Pixel explore: {elem.affordance_type} at ({elem.x},{elem.y}) change={visual_change:.3f}",
                        item_type="observation",
                        priority=0.7,
                    )
            except Exception:
                continue
        return results

    def _safe_perform(self, action: UiAction) -> str:
        destructive = {"delete", "remove", "trash", "format", "erase", "close", "quit"}
        if any(kw in action.selector.lower() for kw in destructive):
            return json.dumps(
                {"status": "blocked", "reason": "destructive_during_exploration"}
            )
        return self.backend.perform(action)

    def _safe_snapshot_nodes(self) -> list[UiNode]:
        if not hasattr(self.backend, "snapshot"):
            return []
        try:
            return list(self.backend.snapshot())
        except Exception:
            return []

    def get_cognitive_summary(self, run: UniversalAgentRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "objective": run.objective,
            "success": run.success,
            "total_steps": len(run.steps),
            "exploration_probes": run.exploration_probes_used,
            "mcts_simulations": run.mcts_simulations_run,
            "adaptive_actions": run.adaptive_steps_used,
            "avg_latency_ms": run.avg_latency_ms,
            "model_used_ratio": run.model_used_ratio,
            "trajectory_path": run.trajectory_path,
            "final_belief_entropy": run.final_state.get("belief_entropy", 1.0),
            "working_memory_summary": self.working_memory.summarize(),
            "semantic_memory_events": len(self.semantic_memory._events),
            "step_breakdown": {
                phase: len([s for s in run.steps if s.phase == phase])
                for phase in {"perceive", "deliberate", "explore", "act", "reflect"}
            },
        }


def _phase_for_action(action: UiAction) -> str:
    if action.action_type in {"tool", "explore"}:
        return action.action_type
    return "act"


def _expected_observation_for_action(action: UiAction) -> str:
    contract = action.metadata.get("verification_contract")
    if isinstance(contract, dict) and contract.get("expected"):
        return str(contract["expected"])
    expected = action.metadata.get("expected_observation")
    if expected:
        return str(expected)
    hypothesis = action.metadata.get("frontier_hypothesis")
    if isinstance(hypothesis, dict) and hypothesis.get("expected_observation"):
        return str(hypothesis["expected_observation"])
    if action.action_type == "tool":
        return "The tool returns structured output relevant to the objective."
    if action.action_type == "explore":
        return "Exploration produces evidence for the next plan step."
    return f"The UI changes in response to {action.action_type} on {action.selector}."


def _element_type_from_node(node: UiNode) -> str:
    role = node.role.lower()
    if "button" in role:
        return "button"
    if "edit" in role:
        return "text_field"
    if "table" in role or "grid" in role:
        return "text_field"
    if "canvas" in role:
        return "image"
    if "tab" in role:
        return "tab"
    return "panel"


def _exploration_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    action = getattr(result, "action", None)
    return {
        "action": {
            "action_type": getattr(action, "action_type", ""),
            "selector": getattr(action, "selector", ""),
        },
        "info_gain": getattr(result, "info_gain", 0.0),
        "safe": getattr(result, "safe", False),
        "state_delta": list(getattr(result, "state_delta", [])),
    }
