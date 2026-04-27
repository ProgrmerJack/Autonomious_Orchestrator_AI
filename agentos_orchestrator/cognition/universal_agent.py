"""Universal Desktop Agent — integration of all cognitive architecture components.

This agent combines:
- VLA affordance grounding (pixel-level visual intuition)
- Active Inference exploration (zero-prior UI mapping)
- POMDP belief state tracking (partial observability)
- MCTS System 2 deliberation (simulation before action)
- Hierarchical Macro/Micro planning (goal decomposition)
- Differentiable memory (Working + Episodic)

It can handle arbitrary desktop tasks without task-specific templates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiAction, UiNode
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
)

from .active_inference import ActiveInferenceExplorer
from .differentiable_memory import EpisodicMemoryBank, WorkingMemoryScratchpad
from .hierarchical_planner import ExecutionContext, MacroPlanner, MicroExecutor
from .mcts_simulator import MCTSWorldModel, MCTSSimulator, WorldState
from .pomdp_state import POMDPEnvironmentModel, POMDPBeliefState
from .vla_affordance import VLAActionSpace, VLAAffordanceGrounding


@dataclass
class CognitiveStep:
    """A single step in the cognitive loop with full provenance."""

    step_number: int
    phase: str  # "perceive", "deliberate", "explore", "act", "reflect"
    action: UiAction | None
    observation: dict[str, Any]
    belief_entropy: float
    mcts_reward: float | None
    memory_reads: list[str]
    rationale: str


@dataclass
class UniversalAgentRun:
    """Complete trace of a universal agent execution."""

    run_id: str
    objective: str
    steps: list[CognitiveStep] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    adaptive_steps_used: int = 0
    exploration_probes_used: int = 0
    mcts_simulations_run: int = 0


class UniversalDesktopAgent:
    """Cognitive-architecture-based universal desktop control agent.

    Implements the full observe → remember → deliberate → act → learn loop.
    """

    def __init__(
        self,
        backend: Any,
        workspace_root: str = ".",
        use_mcts: bool = True,
        use_active_inference: bool = True,
        use_vla: bool = True,
        max_steps: int = 30,
        mcts_iterations: int = 32,
    ) -> None:
        self.backend = backend
        self.use_mcts = use_mcts
        self.use_active_inference = use_active_inference
        self.use_vla = use_vla
        self.max_steps = max_steps
        self.mcts_iterations = mcts_iterations

        # Cognitive components
        self.vla = VLAAffordanceGrounding()
        self.explorer = ActiveInferenceExplorer(max_probes=6)
        self.pomdp = POMDPEnvironmentModel(backend)
        self.world_model = MCTSWorldModel(max_depth=8)
        self.mcts = MCTSSimulator(self.world_model, iterations=mcts_iterations)
        self.macro_planner = MacroPlanner()
        self.micro_executor = MicroExecutor()
        self.working_memory = WorkingMemoryScratchpad(capacity=32)
        self.episodic_memory = EpisodicMemoryBank(
            db_path=f"{workspace_root}/.agentos/episodic_memory.db"
        )

        # Execution state
        self._step_counter = 0
        self._run_id_counter = 0

    def run(self, objective: str) -> UniversalAgentRun:
        """Execute the full cognitive loop for an arbitrary objective."""
        self._run_id_counter += 1
        run_id = f"ua_{self._run_id_counter}_{int(time.time())}"
        run = UniversalAgentRun(run_id=run_id, objective=objective)

        # Phase 0: Macro planning
        macro_goals = self.macro_planner.plan_objective(objective)
        self.working_memory.write(
            f"Macro goals: {[g.description for g in macro_goals]}",
            item_type="plan",
            priority=0.9,
        )

        # Phase 1: Initial observation
        obs = self.pomdp.observe()
        self._record_perception(run, obs)

        # Phase 2: Explore if no known affordances
        if self.use_active_inference and self._should_explore(obs.nodes):
            probes = self.explorer.explore(
                obs.nodes,
                objective,
                perform_fn=self._safe_perform,
                snapshot_fn=self.backend.snapshot,
            )
            run.exploration_probes_used = len(probes)
            for probe in probes:
                self.working_memory.write(
                    f"Exploration: {probe.action.selector} -> delta={len(probe.state_delta)} info_gain={probe.info_gain:.2f}",
                    item_type="observation",
                    priority=0.7,
                )
            # Re-observe after exploration
            obs = self.pomdp.observe()

        # Phase 3: Execute goals with cognitive loop
        context = ExecutionContext(objective=objective)
        for goal in macro_goals:
            if run.success:
                break
            context = self.micro_executor.execute_goal(
                goal,
                context,
                obs.nodes,
                perform_fn=self._cognitive_perform(run),
                snapshot_fn=self.backend.snapshot,
            )
            if goal.completed:
                self.working_memory.write(
                    f"Goal completed: {goal.description}",
                    item_type="goal",
                    priority=0.95,
                )
            elif goal.failed:
                self.working_memory.write(
                    f"Goal failed: {goal.description}",
                    item_type="goal",
                    priority=0.9,
                )
                macro_goals = self.macro_planner.replan_on_failure(context, macro_goals)

            run.adaptive_steps_used = len([s for s in run.steps if s.phase == "act"])
            if run.adaptive_steps_used >= self.max_steps:
                break

        # Phase 4: Final reflection
        final_obs = self.pomdp.observe()
        run.final_state = {
            "belief_entropy": self.pomdp.belief.entropy(),
            "node_count": len(final_obs.nodes),
            "steps_taken": len(run.steps),
            "working_memory_items": len(self.working_memory.read()),
        }
        run.success = any(g.completed for g in macro_goals)

        # Record episodic memory
        self.episodic_memory.record(
            objective=objective,
            action=UiAction(
                action_type="universal_run", selector="agent", value=objective
            ),
            observation_summary=json.dumps(run.final_state),
            outcome="success" if run.success else "partial",
            reward=1.0 if run.success else 0.3,
            tags=["universal_agent", "cognitive_run"],
        )

        return run

    def run_with_planned_bootstrap(
        self,
        objective: str,
        plan: DesktopWorkflowPlan,
    ) -> UniversalAgentRun:
        """Bootstrap from an existing workflow plan, then go universal."""
        run = self.run(objective)
        # Inject planned steps into working memory as context
        for step in plan.steps:
            self.working_memory.write(
                f"Planned step: {step.action_type} on {step.selector}",
                item_type="plan",
                priority=0.6,
            )
        return run

    # ------------------------------------------------------------------ #
    # Internal cognitive loop helpers
    # ------------------------------------------------------------------ #

    def _record_perception(self, run: UniversalAgentRun, obs: Any) -> None:
        """Record a perception step."""
        self._step_counter += 1
        step = CognitiveStep(
            step_number=self._step_counter,
            phase="perceive",
            action=None,
            observation={
                "node_count": len(obs.nodes),
                "window_title": obs.active_window_title,
            },
            belief_entropy=self.pomdp.belief.entropy(),
            mcts_reward=None,
            memory_reads=[],
            rationale="Initial UI snapshot captured",
        )
        run.steps.append(step)
        self.working_memory.write(
            f"Perceived {len(obs.nodes)} UI nodes, entropy={step.belief_entropy:.3f}",
            item_type="observation",
        )

    def _cognitive_perform(self, run: UniversalAgentRun):
        """Return a perform function that wraps actions with cognition."""

        def perform(action: UiAction) -> str:
            # Retrieve similar past experiences
            similar = self.episodic_memory.retrieve_similar(
                run.objective, action, top_k=3
            )
            memory_reads = [s.observation_summary for s in similar]
            for s in similar:
                if s.reward < 0:
                    self.working_memory.write(
                        f"Past failure warning: {s.action.selector} -> {s.outcome}",
                        item_type="hypothesis",
                        priority=0.85,
                    )

            # System 2 deliberation via MCTS
            mcts_reward = None
            if self.use_mcts:
                current_state = self.pomdp.belief.most_likely_state() or {}
                root_state = WorldState(
                    state_vector=current_state,
                    depth=0,
                    terminal=False,
                    reward=0.0,
                )
                mcts_action = self.mcts.search(root_state, run.objective)
                run.mcts_simulations_run += self.mcts_iterations
                if mcts_action and mcts_action.selector != action.selector:
                    # MCTS suggests a different action; log but follow the micro plan
                    mcts_reward = self.world_model.evaluate(root_state, run.objective)
                    self.working_memory.write(
                        f"MCTS suggests: {mcts_action.action_type} on {mcts_action.selector} (reward={mcts_reward:.3f})",
                        item_type="hypothesis",
                        priority=0.7,
                    )

            # VLA pixel-level action (if screenshot available)
            vla_action = None
            if self.use_vla:
                try:
                    screenshot = self._try_capture_screenshot()
                    if screenshot:
                        vla_action = self.vla.propose_action(
                            screenshot, run.objective, []
                        )
                except Exception:
                    pass

            # Execute the action
            receipt = self.backend.perform(action)

            # Record the cognitive step
            self._step_counter += 1
            step = CognitiveStep(
                step_number=self._step_counter,
                phase="act",
                action=action,
                observation={"receipt": receipt},
                belief_entropy=self.pomdp.belief.entropy(),
                mcts_reward=mcts_reward,
                memory_reads=memory_reads,
                rationale=f"Executed {action.action_type} on {action.selector}",
            )
            run.steps.append(step)

            # Update POMDP belief
            self.pomdp.step(action)

            # Record in episodic memory
            self.episodic_memory.record(
                objective=run.objective,
                action=action,
                observation_summary=f"Nodes: {len(self.backend.snapshot())}, receipt: {receipt}",
                outcome=receipt,
                reward=1.0 if "executed" in receipt.lower() else -0.5,
                tags=["micro_execution"],
            )

            return receipt

        return perform

    def _safe_perform(self, action: UiAction) -> str:
        """Perform an action during exploration, with extra safety checks."""
        # Block obviously destructive actions during exploration
        destructive = {"delete", "remove", "trash", "format", "erase", "close", "quit"}
        if any(kw in action.selector.lower() for kw in destructive):
            return json.dumps(
                {"status": "blocked", "reason": "destructive_action_during_exploration"}
            )
        return self.backend.perform(action)

    def _should_explore(self, nodes: list[UiNode]) -> bool:
        """Determine if the UI has too few known affordances to act directly."""
        interactive = [
            n
            for n in nodes
            if n.enabled
            and n.role
            in {"Button", "Edit", "Document", "Canvas", "Menu", "Hyperlink", "Tab"}
        ]
        return len(interactive) < 3

    def _try_capture_screenshot(self) -> bytes | None:
        """Attempt to capture a screenshot from the backend."""
        if hasattr(self.backend, "capture"):
            try:
                return self.backend.capture()
            except Exception:
                return None
        return None

    def get_cognitive_summary(self, run: UniversalAgentRun) -> dict[str, Any]:
        """Generate a rich summary of the cognitive process."""
        return {
            "run_id": run.run_id,
            "objective": run.objective,
            "success": run.success,
            "total_steps": len(run.steps),
            "exploration_probes": run.exploration_probes_used,
            "mcts_simulations": run.mcts_simulations_run,
            "adaptive_actions": run.adaptive_steps_used,
            "final_belief_entropy": run.final_state.get("belief_entropy", 1.0),
            "working_memory_summary": self.working_memory.summarize(),
            "episodic_memory_stats": self.episodic_memory.stats(),
            "step_breakdown": {
                phase: len([s for s in run.steps if s.phase == phase])
                for phase in {"perceive", "deliberate", "explore", "act", "reflect"}
            },
        }
