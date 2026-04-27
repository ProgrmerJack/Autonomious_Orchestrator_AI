"""System 2 deliberative reasoning via Monte Carlo Tree Search (MCTS).

Before emitting an action, the agent simulates multiple future trajectories
using a World Model. It explores the decision tree: if I do A_t, the system
transitions to S_{t+1}. If S_{t+1} results in an error, backtrack.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentos_orchestrator.os_control.base import UiAction


@dataclass
class WorldState:
    """A hypothetical state used during MCTS simulation."""

    state_vector: dict[str, Any]
    depth: int = 0
    terminal: bool = False
    reward: float = 0.0


class WorldModel(Protocol):
    """Protocol for simulating state transitions without real execution."""

    def predict(self, state: WorldState, action: UiAction) -> WorldState:
        """Predict next state given current state and action."""

    def evaluate(self, state: WorldState, objective: str) -> float:
        """Return a reward for being in this state w.r.t. the objective."""

    def is_terminal(self, state: WorldState, objective: str) -> bool:
        """Check if this state satisfies the objective or is a dead end."""

    def available_actions(self, state: WorldState) -> list[UiAction]:
        """Return actions available from this state."""


@dataclass
class MCTSNode:
    """A node in the Monte Carlo search tree."""

    state: WorldState
    parent: "MCTSNode | None" = None
    action: UiAction | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[UiAction] | None = None

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def best_child(self, c: float = 1.414) -> "MCTSNode":
        """Select child with highest UCT score."""
        choices = [
            (
                child.total_reward / child.visits
                + c * math.sqrt(math.log(self.visits) / child.visits),
                child,
            )
            for child in self.children
            if child.visits > 0
        ]
        if not choices:
            return self.children[0] if self.children else self
        return max(choices, key=lambda x: x[0])[1]

    def uct_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_reward / self.visits
        explore = (
            c * math.sqrt(math.log(self.parent.visits) / self.visits)
            if self.parent and self.parent.visits > 0
            else 0
        )
        return exploit + explore


class MCTSWorldModel:
    """Simple learned world model for desktop environments.

    Uses a lightweight rule-based transition model plus history-based
    heuristics. In a full deployment this would be a neural network
    (e.g., a transformer dynamics model or a JEPA world model).
    """

    def __init__(self, max_depth: int = 8) -> None:
        self.max_depth = max_depth
        self._history: list[tuple[dict[str, Any], UiAction, dict[str, Any]]] = []

    def predict(self, state: WorldState, action: UiAction) -> WorldState:
        """Predict next state using simple transition rules."""
        next_vector = dict(state.state_vector)
        next_vector["depth"] = state.depth + 1
        next_vector["last_action"] = action.action_type
        next_vector["last_selector"] = action.selector

        # Rule: clicking may change focused element
        if action.action_type == "click":
            next_vector["focused_element"] = action.selector
            next_vector["click_count"] = next_vector.get("click_count", 0) + 1
        # Rule: typing sets has_unsaved
        if action.action_type == "type" and action.value:
            next_vector["has_unsaved_changes"] = True
            next_vector["last_typed_length"] = len(action.value)
        # Rule: save hotkey clears unsaved
        if (
            action.action_type == "hotkey"
            and action.value
            and "s" in action.value.lower()
        ):
            next_vector["has_unsaved_changes"] = False
        # Rule: opening a URL changes app context
        if action.action_type == "open_url":
            next_vector["app_context"] = "browser"
            next_vector["url_opened"] = True

        # Check for stagnation penalty
        recent_actions = next_vector.get("recent_actions", [])
        recent_actions.append(action.action_type)
        next_vector["recent_actions"] = recent_actions[-10:]

        terminal = state.depth >= self.max_depth
        return WorldState(
            state_vector=next_vector,
            depth=state.depth + 1,
            terminal=terminal,
            reward=0.0,
        )

    def evaluate(self, state: WorldState, objective: str) -> float:
        """Heuristic reward based on progress toward objective."""
        vec = state.state_vector
        score = 0.0
        lower_obj = objective.lower()

        # Reward for reduced entropy / uncertainty
        score += 0.1 * (1.0 - vec.get("belief_entropy", 1.0))

        # Reward for making progress (new actions)
        recent = vec.get("recent_actions", [])
        if len(set(recent)) > 1:
            score += 0.2

        # Reward for matching objective keywords
        for token in lower_obj.split():
            if token in str(vec.get("focused_element", "")).lower():
                score += 0.3
            if token in str(vec.get("app_context", "")).lower():
                score += 0.2

        # Penalty for stagnation (repeating same action)
        if len(recent) >= 3 and len(set(recent[-3:])) == 1:
            score -= 0.4

        # Penalty for deep states without progress
        if state.depth > self.max_depth // 2 and score < 0.3:
            score -= 0.2

        return score

    def is_terminal(self, state: WorldState, objective: str) -> bool:
        """Check if objective appears satisfied or we hit max depth."""
        if state.depth >= self.max_depth:
            return True
        lower_obj = objective.lower()
        vec = state.state_vector
        # Heuristic: if we typed/saved and objective mentions creating content
        if "write" in lower_obj or "create" in lower_obj or "draw" in lower_obj:
            if (
                vec.get("has_unsaved_changes") is False
                and vec.get("last_typed_length", 0) > 10
            ):
                return True
        if "open" in lower_obj and vec.get("url_opened"):
            return True
        return False

    def available_actions(self, state: WorldState) -> list[UiAction]:
        """Generate candidate actions from current state."""
        actions: list[UiAction] = []
        vec = state.state_vector
        # Generic click on focused element
        focused = vec.get("focused_element", "app-window")
        actions.append(UiAction(action_type="click", selector=focused))
        # Type if there's a focused input
        if "edit" in focused.lower() or "document" in focused.lower():
            actions.append(
                UiAction(action_type="type", selector=focused, value="placeholder text")
            )
        # Save hotkey
        actions.append(
            UiAction(action_type="hotkey", selector="app-window", value="^s")
        )
        # Focus a menu
        actions.append(UiAction(action_type="focus", selector="name=Menu"))
        # Scroll
        actions.append(
            UiAction(action_type="scroll", selector="app-window", value="-3")
        )
        return actions

    def record_transition(
        self,
        before: dict[str, Any],
        action: UiAction,
        after: dict[str, Any],
    ) -> None:
        """Record a real transition to improve the model over time."""
        self._history.append((before, action, after))


class MCTSSimulator:
    """Monte Carlo Tree Search for deliberative action selection.

    Given a current belief state and objective, runs MCTS to find the
    action with highest expected cumulative reward.
    """

    def __init__(
        self,
        world_model: WorldModel,
        iterations: int = 64,
        max_depth: int = 8,
        exploration_constant: float = 1.414,
    ) -> None:
        self.world_model = world_model
        self.iterations = iterations
        self.max_depth = max_depth
        self.c = exploration_constant
        self.rng = random.Random(42)

    def search(self, root_state: WorldState, objective: str) -> UiAction | None:
        """Run MCTS and return the best action from the root."""
        root = MCTSNode(state=root_state)
        root.untried_actions = self.world_model.available_actions(root_state)

        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node, objective)
            self._backpropagate(node, reward)

        if not root.children:
            return None
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCT."""
        while not node.state.terminal and node.is_fully_expanded():
            node = node.best_child(self.c)
        if not node.state.terminal and node.untried_actions:
            return self._expand(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        assert node.untried_actions is not None
        action = node.untried_actions.pop(self.rng.randrange(len(node.untried_actions)))
        next_state = self.world_model.predict(node.state, action)
        child = MCTSNode(
            state=next_state,
            parent=node,
            action=action,
            untried_actions=self.world_model.available_actions(next_state)
            if not next_state.terminal
            else [],
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode, objective: str) -> float:
        """Rollout from node to terminal state."""
        state = node.state
        cumulative_reward = 0.0
        discount = 1.0
        for _ in range(self.max_depth - state.depth):
            if self.world_model.is_terminal(state, objective):
                break
            actions = self.world_model.available_actions(state)
            if not actions:
                break
            action = self.rng.choice(actions)
            state = self.world_model.predict(state, action)
            cumulative_reward += discount * self.world_model.evaluate(state, objective)
            discount *= 0.95
        return cumulative_reward

    @staticmethod
    def _backpropagate(node: MCTSNode | None, reward: float) -> None:
        """Propagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
