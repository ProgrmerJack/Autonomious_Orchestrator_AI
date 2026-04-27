"""POMDP state representation for desktop environments.

Treats the OS/application as a Partially Observable Markov Decision Process.
The agent maintains a belief state over possible world configurations because
it cannot observe the full internal state of every application.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.os_control.base import UiNode


@dataclass(slots=True)
class UIStateObservation:
    """A single observation of the UI at a point in time."""

    timestamp: float
    nodes: list[UiNode]
    screenshot_hash: str = ""
    active_window_title: str = ""
    cursor_position: tuple[int, int] | None = None
    keyboard_focus: str = ""


@dataclass(slots=True)
class ActionOutcome:
    """Result of taking an action in the environment."""

    action_type: str
    selector: str
    receipt: dict[str, Any] | str
    duration_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class BeliefParticle:
    """A single hypothesis about the current world state."""

    state_vector: dict[str, Any]
    weight: float = 1.0
    last_observation_hash: str = ""


@dataclass
class POMDPBeliefState:
    """Particle-filter belief state over possible desktop configurations.

    Instead of assuming we know the exact state, we maintain a distribution
    of hypotheses. Each particle represents one possible configuration of
    hidden state variables (e.g., whether a dialog is open, which tab is
    active, whether a file has been saved).
    """

    particles: list[BeliefParticle] = field(default_factory=list)
    history: list[tuple[UIStateObservation, ActionOutcome]] = field(
        default_factory=list
    )
    current_observation: UIStateObservation | None = None

    def update(
        self,
        observation: UIStateObservation,
        action: ActionOutcome,
    ) -> None:
        """Bayesian update: reweight particles based on observation likelihood."""
        self.history.append((observation, action))
        self.current_observation = observation
        obs_hash = self._observation_hash(observation)
        for particle in self.particles:
            likelihood = self._observation_likelihood(particle, observation)
            particle.weight *= likelihood
            particle.last_observation_hash = obs_hash
        self._normalize_weights()
        self._resample_if_degenerate()

    def predict(self, action_type: str, selector: str) -> list[BeliefParticle]:
        """Propagate particles forward through a transition model."""
        new_particles: list[BeliefParticle] = []
        for particle in self.particles:
            next_state = self._transition(particle.state_vector, action_type, selector)
            new_particles.append(
                BeliefParticle(
                    state_vector=next_state,
                    weight=particle.weight,
                    last_observation_hash=particle.last_observation_hash,
                )
            )
        return new_particles

    def most_likely_state(self) -> dict[str, Any] | None:
        """Return the state vector of the highest-weight particle."""
        if not self.particles:
            return None
        best = max(self.particles, key=lambda p: p.weight)
        return best.state_vector

    def entropy(self) -> float:
        """Measure of uncertainty in the belief state."""
        if not self.particles:
            return 1.0
        weights = [p.weight for p in self.particles]
        import math

        entropy = 0.0
        for w in weights:
            if w > 0:
                entropy -= w * math.log2(w)
        return entropy

    def initialize_from_observation(
        self, observation: UIStateObservation, n_particles: int = 10
    ) -> None:
        """Bootstrap belief state from first observation."""
        self.current_observation = observation
        self.particles = []
        base_state = self._observation_to_state(observation)
        for i in range(n_particles):
            # Slightly perturbed copies to represent initial uncertainty
            perturbed = dict(base_state)
            perturbed["_particle_id"] = i
            perturbed["_uncertainty"] = 0.1 * (i + 1)
            self.particles.append(
                BeliefParticle(state_vector=perturbed, weight=1.0 / n_particles)
            )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _observation_hash(observation: UIStateObservation) -> str:
        node_ids = [n.node_id for n in observation.nodes]
        return str(hash(f"{observation.active_window_title}:{','.join(node_ids)}"))

    @staticmethod
    def _observation_to_state(observation: UIStateObservation) -> dict[str, Any]:
        """Convert an observation into a structured state vector."""
        return {
            "node_count": len(observation.nodes),
            "window_title": observation.active_window_title,
            "has_keyboard_focus": bool(observation.keyboard_focus),
            "roles": [n.role for n in observation.nodes],
            "names": [n.name for n in observation.nodes],
            "enabled_count": sum(1 for n in observation.nodes if n.enabled),
            "focused_count": sum(1 for n in observation.nodes if n.focused),
        }

    @staticmethod
    def _observation_likelihood(
        particle: BeliefParticle, observation: UIStateObservation
    ) -> float:
        """Compute P(observation | state_hypothesis)."""
        state = particle.state_vector
        score = 1.0
        # Penalize mismatches in node count
        expected_count = state.get("node_count", 0)
        actual_count = len(observation.nodes)
        if expected_count > 0:
            ratio = min(expected_count, actual_count) / max(
                expected_count, actual_count
            )
            score *= 0.5 + 0.5 * ratio
        # Penalize window title mismatch
        if (
            state.get("window_title")
            and state["window_title"] != observation.active_window_title
        ):
            score *= 0.7
        return max(0.1, score)

    @staticmethod
    def _transition(
        state: dict[str, Any],
        action_type: str,
        selector: str,
    ) -> dict[str, Any]:
        """Apply a simple transition model."""
        next_state = dict(state)
        if action_type == "click":
            next_state["last_clicked"] = selector
            next_state["click_count"] = next_state.get("click_count", 0) + 1
        elif action_type == "type":
            next_state["last_typed"] = selector
            next_state["has_unsaved_changes"] = True
        elif action_type == "hotkey" and "save" in selector.lower():
            next_state["has_unsaved_changes"] = False
        return next_state

    def _normalize_weights(self) -> None:
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total

    def _resample_if_degenerate(self, threshold: float = 0.5) -> None:
        """Resample particles if effective sample size drops below threshold."""
        if not self.particles:
            return
        ess = 1.0 / sum(p.weight**2 for p in self.particles)
        if ess < threshold * len(self.particles):
            new_particles: list[BeliefParticle] = []
            for _ in self.particles:
                # Simple multinomial resample
                import random

                r = random.random()
                cumsum = 0.0
                for p in self.particles:
                    cumsum += p.weight
                    if r <= cumsum:
                        new_particles.append(
                            BeliefParticle(
                                state_vector=dict(p.state_vector),
                                weight=1.0 / len(self.particles),
                            )
                        )
                        break
            self.particles = new_particles


class POMDPEnvironmentModel:
    """Wrapper that bridges the real desktop backend with POMDP belief updates.

    Provides a clean interface: observe() -> UIStateObservation, act() -> ActionOutcome.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.belief = POMDPBeliefState()
        self._step_count = 0

    def observe(self) -> UIStateObservation:
        """Capture current observation from the backend."""
        import time

        nodes = self.backend.snapshot()
        obs = UIStateObservation(
            timestamp=time.time(),
            nodes=nodes,
            active_window_title="",
            cursor_position=None,
            keyboard_focus="",
        )
        if self._step_count == 0:
            self.belief.initialize_from_observation(obs)
        self._step_count += 1
        return obs

    def act(self, action: Any) -> ActionOutcome:
        """Execute an action and record the outcome."""
        import time

        start = time.time()
        try:
            receipt = self.backend.perform(action)
            duration = (time.time() - start) * 1000
            return ActionOutcome(
                action_type=action.action_type,
                selector=action.selector,
                receipt=receipt,
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.time() - start) * 1000
            return ActionOutcome(
                action_type=action.action_type,
                selector=action.selector,
                receipt=str(exc),
                duration_ms=duration,
                error=str(exc),
            )

    def step(self, action: Any) -> tuple[UIStateObservation, ActionOutcome]:
        """Full POMDP step: act, then observe."""
        outcome = self.act(action)
        # Brief pause for UI to settle
        import time

        time.sleep(0.3)
        observation = self.observe()
        self.belief.update(observation, outcome)
        return observation, outcome
