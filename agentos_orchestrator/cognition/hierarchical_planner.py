"""Hierarchical task decomposition: Macro-Planner + Micro-Executor.

Macro-Planner: Sets long-term goals and maintains overarching state.
Micro-Executor: Handles immediate, step-by-step UI actions.
The Macro-planner only intervenes if the Micro-executor reports persistent
failure, preventing infinite loops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiAction, UiNode


@dataclass(slots=True)
class MacroGoal:
    """A high-level objective with success criteria."""

    goal_id: str
    description: str
    success_criteria: list[str]
    max_micro_attempts: int = 10
    sub_goals: list["MacroGoal"] = field(default_factory=list)
    completed: bool = False
    failed: bool = False


@dataclass(slots=True)
class MicroStep:
    """A concrete UI action to execute."""

    step_id: str
    action: UiAction
    expected_outcome: str
    fallback_steps: list[UiAction] = field(default_factory=list)
    completed: bool = False
    failed: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ExecutionContext:
    """Shared context between macro and micro layers."""

    objective: str
    current_goal: MacroGoal | None = None
    current_step: MicroStep | None = None
    step_history: list[tuple[MicroStep, str]] = field(default_factory=list)
    failure_streak: int = 0
    max_failure_streak: int = 3


class MacroPlanner:
    """Long-term goal setter and task decomposer.

    Takes a natural-language objective and decomposes it into a tree of
    MacroGoals. Monitors Micro-executor progress and replans on persistent
    failure.
    """

    def __init__(self) -> None:
        self._goal_counter = 0

    def plan_objective(self, objective: str) -> list[MacroGoal]:
        """Decompose an objective into top-level macro goals."""
        self._goal_counter = 0
        lower = objective.lower()
        goals: list[MacroGoal] = []

        # Detect app-launch goals
        if any(kw in lower for kw in {"open", "launch", "start"}):
            goals.append(
                self._make_goal(
                    f"Launch the target application for: {objective}",
                    ["Target app is open and visible"],
                )
            )

        # Detect content-creation goals
        if any(kw in lower for kw in {"write", "create", "draw", "make", "edit"}):
            goals.append(
                self._make_goal(
                    f"Create or modify content for: {objective}",
                    [
                        "Content has been entered into the workspace",
                        "Changes are saved or staged",
                    ],
                )
            )

        # Detect research/knowledge goals
        if any(
            kw in lower
            for kw in {"find", "search", "analyze", "compare", "research", "look up"}
        ):
            goals.append(
                self._make_goal(
                    f"Gather information for: {objective}",
                    [
                        "Relevant information has been retrieved",
                        "Information is accessible in the workspace",
                    ],
                )
            )

        # Detect file-operation goals
        if any(
            kw in lower
            for kw in {"save", "export", "download", "move", "copy", "delete"}
        ):
            goals.append(
                self._make_goal(
                    f"Perform file operation for: {objective}",
                    [
                        "File operation completed successfully",
                        "File exists at expected path or is removed",
                    ],
                )
            )

        # Fallback: if no patterns matched, create a single exploratory goal
        if not goals:
            goals.append(
                self._make_goal(
                    f"Explore and accomplish: {objective}",
                    ["Objective appears satisfied based on UI state"],
                )
            )

        return goals

    def replan_on_failure(
        self,
        context: ExecutionContext,
        current_goals: list[MacroGoal],
    ) -> list[MacroGoal]:
        """When micro-executor fails persistently, replan at macro level."""
        if context.failure_streak < context.max_failure_streak:
            return current_goals

        # Escalation: break current goal into smaller sub-goals
        for goal in current_goals:
            if goal.completed or goal.failed:
                continue
            if not goal.sub_goals:
                goal.sub_goals = self._decompose_further(goal)
            return current_goals
        return current_goals

    def check_goal_completion(
        self, goal: MacroGoal, state_summary: dict[str, Any]
    ) -> bool:
        """Evaluate whether success criteria are met."""
        for criterion in goal.success_criteria:
            lower_crit = criterion.lower()
            state_str = json.dumps(state_summary).lower()
            # Simple keyword matching against state summary
            if any(
                token in state_str for token in lower_crit.split() if len(token) > 3
            ):
                continue
            # If criterion is about saved changes
            if (
                "saved" in lower_crit
                and state_summary.get("has_unsaved_changes") is False
            ):
                continue
            if "open" in lower_crit and state_summary.get("app_open"):
                continue
            return False
        return True

    def _make_goal(self, description: str, criteria: list[str]) -> MacroGoal:
        self._goal_counter += 1
        return MacroGoal(
            goal_id=f"goal_{self._goal_counter}",
            description=description,
            success_criteria=criteria,
        )

    @staticmethod
    def _decompose_further(goal: MacroGoal) -> list[MacroGoal]:
        """Break a stuck goal into smaller pieces."""
        sub_goals: list[MacroGoal] = []
        desc = goal.description.lower()
        if "launch" in desc or "open" in desc:
            sub_goals.append(
                MacroGoal(
                    goal_id=f"{goal.goal_id}_sub1",
                    description="Identify the correct application executable",
                    success_criteria=["Executable path or name is known"],
                    max_micro_attempts=3,
                )
            )
            sub_goals.append(
                MacroGoal(
                    goal_id=f"{goal.goal_id}_sub2",
                    description="Launch the application via shell or UI",
                    success_criteria=["Application window is visible"],
                    max_micro_attempts=5,
                )
            )
        elif "create" in desc or "write" in desc:
            sub_goals.append(
                MacroGoal(
                    goal_id=f"{goal.goal_id}_sub1",
                    description="Focus the correct input/workspace area",
                    success_criteria=["Cursor is in an editable region"],
                    max_micro_attempts=5,
                )
            )
            sub_goals.append(
                MacroGoal(
                    goal_id=f"{goal.goal_id}_sub2",
                    description="Enter the required content",
                    success_criteria=["Content matches the intent"],
                    max_micro_attempts=8,
                )
            )
            sub_goals.append(
                MacroGoal(
                    goal_id=f"{goal.goal_id}_sub3",
                    description="Save or confirm the changes",
                    success_criteria=["File is saved or changes are committed"],
                    max_micro_attempts=5,
                )
            )
        else:
            sub_goals.append(
                MacroGoal(
                    goal_id=f"{goal.goal_id}_sub1",
                    description="Explore the UI to locate relevant controls",
                    success_criteria=[
                        "At least one relevant control has been identified"
                    ],
                    max_micro_attempts=6,
                )
            )
        return sub_goals


class MicroExecutor:
    """Step-by-step UI action executor with retry and fallback logic.

    Reports persistent failures back to the MacroPlanner for replanning.
    """

    def __init__(self, max_steps_per_goal: int = 20) -> None:
        self.max_steps_per_goal = max_steps_per_goal
        self._step_counter = 0

    def execute_goal(
        self,
        goal: MacroGoal,
        context: ExecutionContext,
        nodes: list[UiNode],
        perform_fn: Any,
        snapshot_fn: Any,
    ) -> ExecutionContext:
        """Execute micro-steps until goal completes, fails, or step limit reached."""
        context.current_goal = goal
        steps = self._plan_micro_steps(goal, nodes)
        for step in steps:
            if goal.completed or goal.failed:
                break
            context.current_step = step
            for attempt in range(step.max_retries):
                try:
                    receipt = perform_fn(step.action)
                    context.step_history.append((step, receipt))
                    step.completed = True
                    context.failure_streak = 0
                    break
                except Exception as exc:
                    step.retry_count += 1
                    receipt = str(exc)
                    context.step_history.append((step, receipt))
                    if attempt < len(step.fallback_steps):
                        step.action = step.fallback_steps[attempt]
            if not step.completed:
                context.failure_streak += 1
                step.failed = True
                if context.failure_streak >= context.max_failure_streak:
                    goal.failed = True
                    break
            # Refresh UI state for next step
            nodes = snapshot_fn()
        # Evaluate goal completion
        state_summary = self._summarize_state(nodes, context)
        if MacroPlanner().check_goal_completion(goal, state_summary):
            goal.completed = True
        return context

    def _plan_micro_steps(
        self, goal: MacroGoal, nodes: list[UiNode]
    ) -> list[MicroStep]:
        """Generate concrete UI steps for a macro goal."""
        steps: list[MicroStep] = []
        self._step_counter += 1
        desc = goal.description.lower()

        # Find best interactive nodes
        clickable = [
            n
            for n in nodes
            if n.enabled and n.role in {"Button", "Menu", "Hyperlink", "Tab"}
        ]
        editable = [
            n
            for n in nodes
            if n.enabled and n.role in {"Edit", "Document", "Canvas", "Text"}
        ]

        if "launch" in desc or "open" in desc:
            # Try to find a launch button or menu item
            for node in clickable:
                if any(
                    kw in node.name.lower() for kw in {"start", "open", "launch", "run"}
                ):
                    steps.append(
                        self._make_step(
                            UiAction(action_type="click", selector=f"name={node.name}"),
                            f"Click {node.name} to launch",
                        )
                    )
                    break
            if not steps:
                steps.append(
                    self._make_step(
                        UiAction(
                            action_type="hotkey", selector="app-window", value="^r"
                        ),
                        "Open run dialog",
                        [
                            UiAction(
                                action_type="type",
                                selector="name=Open",
                                value="app.exe",
                            )
                        ],
                    )
                )

        elif "create" in desc or "write" in desc or "enter" in desc:
            # Focus best editable surface
            if editable:
                best = editable[0]
                steps.append(
                    self._make_step(
                        UiAction(action_type="focus", selector=f"name={best.name}"),
                        f"Focus {best.name}",
                    )
                )
                steps.append(
                    self._make_step(
                        UiAction(
                            action_type="type",
                            selector=f"name={best.name}",
                            value="Content placeholder",
                        ),
                        "Enter content",
                    )
                )
            else:
                steps.append(
                    self._make_step(
                        UiAction(action_type="click", selector="name=Workspace"),
                        "Click workspace area",
                    )
                )

        elif "save" in desc:
            steps.append(
                self._make_step(
                    UiAction(action_type="hotkey", selector="app-window", value="^s"),
                    "Save with Ctrl+S",
                )
            )

        elif "explore" in desc or "find" in desc:
            for node in clickable[:3]:
                steps.append(
                    self._make_step(
                        UiAction(action_type="click", selector=f"name={node.name}"),
                        f"Explore {node.name}",
                    )
                )

        return steps[: self.max_steps_per_goal]

    def _make_step(
        self,
        action: UiAction,
        expected: str,
        fallbacks: list[UiAction] | None = None,
    ) -> MicroStep:
        self._step_counter += 1
        return MicroStep(
            step_id=f"step_{self._step_counter}",
            action=action,
            expected_outcome=expected,
            fallback_steps=fallbacks or [],
        )

    @staticmethod
    def _summarize_state(
        nodes: list[UiNode], context: ExecutionContext
    ) -> dict[str, Any]:
        """Create a summary dict for goal completion checking."""
        return {
            "node_count": len(nodes),
            "has_editable": any(
                n.role in {"Edit", "Document", "Canvas"} for n in nodes
            ),
            "has_focused": any(n.focused for n in nodes),
            "steps_taken": len(context.step_history),
            "failures": context.failure_streak,
        }
