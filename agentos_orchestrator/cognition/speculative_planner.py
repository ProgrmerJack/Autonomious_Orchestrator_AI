"""Speculative planning via Monte Carlo Tree Search with safe-commit semantics.

Phase 5 implementation.

The SpeculativePlanner wraps MCTSWorldModel to generate candidate action
sequences in a dry-run / sandbox mode before committing any action to the
host desktop.  Only the first *verified low-risk prefix* is committed; the
rest is discarded and regenerated after re-observation.

Design guarantees
─────────────────
1. No irreversible action is committed without an explicit approval token.
2. Every speculative action is scored against the RiskGuardian before
   commitment; actions above ABORT_THRESHOLD are hard-rejected.
3. Stale plans (produced against an old observation) are invalidated on
   re-observation; the planner rebuilds from scratch.
4. A per-session budget cap (MAX_SPECULATIVE_STEPS) prevents the planner
   from exhausting compute on degenerate search trees.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agentos_orchestrator.cognition.mcts_simulator import (
    MCTSNode,
    MCTSWorldModel,
    WorldState,
)
from agentos_orchestrator.os_control.base import UiAction

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────── #
# Configuration constants                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

MAX_SPECULATIVE_STEPS: int = 24  # hard cap on plan length
MCTS_ITERATIONS: int = 80  # simulations per search
MIN_PREFIX_SCORE: float = 0.35  # minimum cumulative reward to commit
STALE_OBSERVATION_HASH_MISMATCH = "stale_hash"
IRREVERSIBLE_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "file_delete",
        "shell_exec",
        "send_message",
        "trade_order",
        "payment",
        "bulk_delete",
    }
)


# ─────────────────────────────────────────────────────────────────────────── #
# Data structures                                                               #
# ─────────────────────────────────────────────────────────────────────────── #


@dataclass(slots=True)
class SpeculativeAction:
    """A single action inside a speculative plan, annotated with risk."""

    action: UiAction
    speculative_reward: float  # estimated reward for this step
    risk_score: float  # 0–1; from RiskGuardian or heuristic
    requires_approval: bool
    is_irreversible: bool


@dataclass(slots=True)
class SpeculativePlan:
    """An ordered sequence of speculative actions + provenance metadata."""

    plan_id: str
    objective: str
    observation_hash: str  # hash of ObservationFrame at plan-build time
    actions: list[SpeculativeAction] = field(default_factory=list)
    committed_count: int = 0  # how many have been committed so far
    is_stale: bool = False
    build_timestamp: float = field(default_factory=time.time)

    # ── Derived helpers ─────────────────────────────────────────────────── #

    @property
    def remaining(self) -> list[SpeculativeAction]:
        return self.actions[self.committed_count :]

    @property
    def next_action(self) -> SpeculativeAction | None:
        remaining = self.remaining
        return remaining[0] if remaining else None

    @property
    def is_exhausted(self) -> bool:
        return self.committed_count >= len(self.actions)

    @property
    def cumulative_reward(self) -> float:
        return sum(a.speculative_reward for a in self.actions)

    def mark_committed(self) -> None:
        self.committed_count += 1

    def invalidate(self, reason: str = "") -> None:
        self.is_stale = True
        log.info("SpeculativePlan %s invalidated: %s", self.plan_id, reason)


@dataclass(slots=True)
class CommitDecision:
    """The planner's verdict on whether to commit the next speculative action."""

    should_commit: bool
    action: UiAction | None
    reason: str
    requires_approval: bool
    risk_score: float
    plan_id: str
    step_index: int


# ─────────────────────────────────────────────────────────────────────────── #
# Risk scoring — thin shim so the planner is self-contained                    #
# ─────────────────────────────────────────────────────────────────────────── #


def _quick_risk_score(action: UiAction) -> tuple[float, bool]:
    """Lightweight heuristic risk score when RiskGuardian is unavailable.

    Returns (score_0_to_1, requires_approval).
    """
    at = (action.action_type or "").lower()
    sel = (action.selector or "").lower()
    val = (action.value or "").lower()

    if at in IRREVERSIBLE_ACTION_TYPES:
        return 0.92, True
    if at == "shell_exec":
        return 0.85, True
    if at in ("file_delete", "bulk_delete"):
        return 0.95, True
    if "password" in sel or "credential" in val:
        return 0.82, True
    if at in ("click", "focus", "scroll", "hover"):
        return 0.05, False
    if at in ("type", "hotkey"):
        return 0.12, False
    if at == "screenshot":
        return 0.01, False
    return 0.25, False


try:
    from agentos_orchestrator.cognition.risk_guardian import (  # type: ignore
        assess_action_risk,
        RiskAssessment,
    )

    def _risk_score(
        action: UiAction,
        proposal: Any | None,
        observation: Any | None,
        objective: str,
    ) -> tuple[float, bool]:
        try:
            assessment: RiskAssessment = assess_action_risk(
                action, proposal, observation, objective
            )
            return assessment.adjusted_risk, assessment.approval_required
        except Exception:
            return _quick_risk_score(action)

except ImportError:

    def _risk_score(
        action: UiAction,
        proposal: Any | None,
        observation: Any | None,
        objective: str,
    ) -> tuple[float, bool]:
        return _quick_risk_score(action)


# ─────────────────────────────────────────────────────────────────────────── #
# Observation hashing                                                           #
# ─────────────────────────────────────────────────────────────────────────── #


def _hash_observation(observation: Any) -> str:
    """Produce a short hash of the observation for staleness detection."""
    if observation is None:
        return "none"
    try:
        raw = json.dumps(
            {
                "elements": len(getattr(observation, "elements", [])),
                "frame_id": getattr(observation, "frame_id", ""),
                "timestamp": str(getattr(observation, "captured_at", ""))[:16],
            },
            sort_keys=True,
        ).encode()
    except Exception:
        raw = str(observation).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────── #
# MCTS search → speculative plan                                                #
# ─────────────────────────────────────────────────────────────────────────── #


class _MCTSSearch:
    """Thin driver around MCTSWorldModel for SpeculativePlanner."""

    def __init__(self, world_model: MCTSWorldModel, iterations: int = MCTS_ITERATIONS):
        self._wm = world_model
        self._iterations = iterations

    def search(
        self,
        root_state: WorldState,
        objective: str,
        max_depth: int,
    ) -> list[UiAction]:
        """Run MCTS and return the best action sequence."""
        import math

        root = MCTSNode(state=root_state)
        root.untried_actions = self._wm.available_actions(root_state)

        for _ in range(self._iterations):
            node = self._select(root)
            if (
                not self._wm.is_terminal(node.state, objective)
                and not node.is_fully_expanded()
            ):
                node = self._expand(node, objective)
            reward = self._simulate(node.state, objective, max_depth)
            self._backpropagate(node, reward)

        # Extract best path from root
        return self._extract_path(root, max_depth)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node: MCTSNode, objective: str) -> MCTSNode:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop(0)
        next_state = self._wm.predict(node.state, action)
        child = MCTSNode(state=next_state, parent=node, action=action)
        child.untried_actions = self._wm.available_actions(next_state)
        node.children.append(child)
        return child

    def _simulate(self, state: WorldState, objective: str, max_depth: int) -> float:
        current = state
        total_reward = 0.0
        depth = 0
        while not self._wm.is_terminal(current, objective) and depth < max_depth:
            actions = self._wm.available_actions(current)
            if not actions:
                break
            action = actions[depth % len(actions)]
            current = self._wm.predict(current, action)
            total_reward += self._wm.evaluate(current, objective)
            depth += 1
        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _extract_path(self, root: MCTSNode, max_depth: int) -> list[UiAction]:
        path: list[UiAction] = []
        node = root
        while node.children and len(path) < max_depth:
            best = max(
                node.children,
                key=lambda c: c.total_reward / max(c.visits, 1),
            )
            if best.action:
                path.append(best.action)
            node = best
        return path


# ─────────────────────────────────────────────────────────────────────────── #
# SpeculativePlanner                                                            #
# ─────────────────────────────────────────────────────────────────────────── #


class SpeculativePlanner:
    """Generates, validates, and incrementally commits speculative action plans.

    Usage
    ─────
    planner = SpeculativePlanner()
    plan    = planner.build_plan(objective, observation, initial_state_vector)
    decision = planner.next_commit(plan, observation, proposal=None)
    if decision.should_commit:
        execute(decision.action)
        plan.mark_committed()
    """

    def __init__(
        self,
        world_model: MCTSWorldModel | None = None,
        mcts_iterations: int = MCTS_ITERATIONS,
        max_plan_length: int = MAX_SPECULATIVE_STEPS,
        min_prefix_score: float = MIN_PREFIX_SCORE,
        abort_risk_threshold: float = 0.92,
    ) -> None:
        self._wm = world_model or MCTSWorldModel(max_depth=max_plan_length)
        self._search = _MCTSSearch(self._wm, iterations=mcts_iterations)
        self._max_plan_length = max_plan_length
        self._min_prefix_score = min_prefix_score
        self._abort_threshold = abort_risk_threshold
        self._plan_counter = 0

    # ── Public API ──────────────────────────────────────────────────────── #

    def build_plan(
        self,
        objective: str,
        observation: Any | None,
        initial_state_vector: dict[str, Any] | None = None,
        proposal: Any | None = None,
    ) -> SpeculativePlan:
        """Run MCTS search and return a scored speculative plan.

        Every candidate action is risk-scored; actions above the abort
        threshold are silently dropped from the plan.
        """
        self._plan_counter += 1
        plan_id = f"specplan-{self._plan_counter:04d}"
        obs_hash = _hash_observation(observation)

        state_vec: dict[str, Any] = dict(initial_state_vector or {})
        if observation is not None:
            state_vec.setdefault("belief_entropy", 0.5)
            state_vec.setdefault("focused_element", "app-window")

        root_state = WorldState(state_vector=state_vec, depth=0)

        log.info("SpeculativePlanner: building plan %s (obs=%s)", plan_id, obs_hash)

        try:
            raw_actions = self._search.search(
                root_state, objective, self._max_plan_length
            )
        except Exception as exc:
            log.warning("MCTS search failed: %s — using empty plan", exc)
            raw_actions = []

        # Score and filter
        spec_actions: list[SpeculativeAction] = []
        simulated_state = root_state
        for action in raw_actions:
            risk, needs_approval = _risk_score(action, proposal, observation, objective)
            is_irrev = action.action_type in IRREVERSIBLE_ACTION_TYPES

            if risk >= self._abort_threshold:
                log.warning(
                    "Speculative action %s risk=%.2f exceeds abort threshold; dropping",
                    action.action_type,
                    risk,
                )
                continue

            reward = self._wm.evaluate(simulated_state, objective)
            spec_actions.append(
                SpeculativeAction(
                    action=action,
                    speculative_reward=reward,
                    risk_score=risk,
                    requires_approval=needs_approval or is_irrev,
                    is_irreversible=is_irrev,
                )
            )
            simulated_state = self._wm.predict(simulated_state, action)

        plan = SpeculativePlan(
            plan_id=plan_id,
            objective=objective,
            observation_hash=obs_hash,
            actions=spec_actions,
        )
        log.info(
            "Plan %s: %d actions, cumulative_reward=%.2f",
            plan_id,
            len(spec_actions),
            plan.cumulative_reward,
        )
        return plan

    def next_commit(
        self,
        plan: SpeculativePlan,
        current_observation: Any | None,
        proposal: Any | None = None,
        approval_token: str | None = None,
    ) -> CommitDecision:
        """Evaluate whether the next action in the plan is safe to commit.

        The decision is NO_COMMIT when:
        - The plan is stale (observation changed).
        - The plan is exhausted.
        - The next action requires approval but no token is present.
        - The re-evaluated risk exceeds the abort threshold.
        """
        plan_id = plan.plan_id
        step = plan.committed_count

        # ── Staleness check ─────────────────────────────────────────────── #
        current_hash = _hash_observation(current_observation)
        if current_hash != plan.observation_hash and current_observation is not None:
            plan.invalidate(STALE_OBSERVATION_HASH_MISMATCH)
            return CommitDecision(
                should_commit=False,
                action=None,
                reason="Plan stale — observation changed; rebuild required.",
                requires_approval=False,
                risk_score=0.0,
                plan_id=plan_id,
                step_index=step,
            )

        if plan.is_stale:
            return CommitDecision(
                should_commit=False,
                action=None,
                reason="Plan is marked stale.",
                requires_approval=False,
                risk_score=0.0,
                plan_id=plan_id,
                step_index=step,
            )

        if plan.is_exhausted:
            return CommitDecision(
                should_commit=False,
                action=None,
                reason="Plan exhausted.",
                requires_approval=False,
                risk_score=0.0,
                plan_id=plan_id,
                step_index=step,
            )

        spec = plan.next_action
        assert spec is not None  # guaranteed by is_exhausted check above

        # ── Re-evaluate risk at commit time ──────────────────────────────── #
        live_risk, live_approval = _risk_score(
            spec.action, proposal, current_observation, plan.objective
        )
        effective_risk = max(spec.risk_score, live_risk)
        effective_approval = spec.requires_approval or live_approval

        if effective_risk >= self._abort_threshold:
            plan.invalidate(
                f"action {spec.action.action_type} live_risk={effective_risk:.2f}"
            )
            return CommitDecision(
                should_commit=False,
                action=spec.action,
                reason=f"Live risk {effective_risk:.2f} ≥ abort threshold {self._abort_threshold}.",
                requires_approval=True,
                risk_score=effective_risk,
                plan_id=plan_id,
                step_index=step,
            )

        # ── Approval gate for irreversible/high-risk actions ─────────────── #
        if effective_approval and approval_token is None:
            return CommitDecision(
                should_commit=False,
                action=spec.action,
                reason="Action requires approval token before commitment.",
                requires_approval=True,
                risk_score=effective_risk,
                plan_id=plan_id,
                step_index=step,
            )

        # ── Score gate: discard low-reward plans early ───────────────────── #
        if spec.speculative_reward < self._min_prefix_score and step == 0:
            return CommitDecision(
                should_commit=False,
                action=spec.action,
                reason=f"First step reward {spec.speculative_reward:.2f} below threshold; rebuild.",
                requires_approval=False,
                risk_score=effective_risk,
                plan_id=plan_id,
                step_index=step,
            )

        return CommitDecision(
            should_commit=True,
            action=spec.action,
            reason="Action is safe and plan is current.",
            requires_approval=False,
            risk_score=effective_risk,
            plan_id=plan_id,
            step_index=step,
        )

    def refresh_plan(
        self,
        old_plan: SpeculativePlan,
        new_observation: Any | None,
        proposal: Any | None = None,
    ) -> SpeculativePlan:
        """Build a fresh plan from the current observation, discarding old."""
        old_plan.invalidate("explicit refresh")
        return self.build_plan(
            objective=old_plan.objective,
            observation=new_observation,
            proposal=proposal,
        )

    def record_outcome(
        self,
        action: UiAction,
        success: bool,
        actual_risk: float | None = None,
    ) -> None:
        """Feed execution outcome back to the world model for calibration.

        Currently records to the MCTS history; in a full deployment this
        would update a neural dynamics model.
        """
        # The MCTSWorldModel stores history for future heuristic improvement
        if hasattr(self._wm, "_history"):
            self._wm._history.append(({}, action, {"success": success}))  # type: ignore[attr-defined]
        if actual_risk is not None:
            log.debug(
                "SpeculativePlanner outcome: %s success=%s actual_risk=%.2f",
                action.action_type,
                success,
                actual_risk,
            )


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level singleton                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

_DEFAULT_PLANNER: SpeculativePlanner | None = None


def get_speculative_planner(
    *,
    world_model: MCTSWorldModel | None = None,
    mcts_iterations: int = MCTS_ITERATIONS,
    max_plan_length: int = MAX_SPECULATIVE_STEPS,
) -> SpeculativePlanner:
    """Return the shared SpeculativePlanner instance (lazily constructed)."""
    global _DEFAULT_PLANNER
    if _DEFAULT_PLANNER is None:
        _DEFAULT_PLANNER = SpeculativePlanner(
            world_model=world_model,
            mcts_iterations=mcts_iterations,
            max_plan_length=max_plan_length,
        )
    return _DEFAULT_PLANNER
