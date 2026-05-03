"""Abstract World Model — Compact UI State Transition Prediction.

Addresses "State-Space Explosion" by predicting structured UI state changes
instead of raw pixels. The state space is:
- Element inventory (what interactive elements exist)
- App context (which app is active)
- Task progress (which subtask is complete)
- Screen layout (abstract regions: header, sidebar, main, modal)

This makes MCTS tractable: instead of predicting 4K pixels, we predict
which elements appear/disappear and how layout changes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentos_orchestrator.os_control.base import UiAction

from .learned_world_model import MLPDynamics, MLPConfig


@dataclass
class UIElementState:
    """Abstract representation of a UI element."""

    element_type: str  # button, text_field, menu, panel, etc.
    region: str  # header, sidebar, main, modal, floating
    relative_x: float  # 0-1, relative to region
    relative_y: float  # 0-1, relative to region
    is_interactive: bool
    semantic_label: str = ""  # "submit", "cancel", "search", etc.


@dataclass
class AbstractUIState:
    """Compact abstract representation of a screen state."""

    app_context: str = "unknown"  # "browser", "file_explorer", "text_editor", etc.
    layout_mode: str = "full"  # "full", "split", "modal_open", "overlay"
    elements: list[UIElementState] = field(default_factory=list)
    active_modal: str = ""  # Which modal/dialog is open, if any
    focus_region: str = "main"  # Where user attention is
    task_progress: dict[str, float] = field(default_factory=dict)
    # Semantic vector of the overall screen (128-dim)
    screen_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(128, dtype=np.float32),
    )

    def to_vector(self, target_dim: int = 256) -> np.ndarray:
        """Flatten to a fixed-dimension vector for the dynamics model."""
        # App context one-hot (8 dims)
        apps = [
            "browser",
            "file_explorer",
            "text_editor",
            "spreadsheet",
            "media",
            "settings",
            "terminal",
            "other",
        ]
        app_vec = np.zeros(len(apps), dtype=np.float32)
        if self.app_context in apps:
            app_vec[apps.index(self.app_context)] = 1.0

        # Layout mode one-hot (4 dims)
        layouts = ["full", "split", "modal_open", "overlay"]
        layout_vec = np.zeros(len(layouts), dtype=np.float32)
        if self.layout_mode in layouts:
            layout_vec[layouts.index(self.layout_mode)] = 1.0

        # Element counts by type (16 dims)
        elem_types = [
            "button",
            "text_field",
            "checkbox",
            "menu",
            "panel",
            "icon",
            "tab",
            "scrollbar",
            "dropdown",
            "link",
            "image",
            "video",
            "table",
            "chart",
            "notification",
            "other",
        ]
        type_counts = np.zeros(len(elem_types), dtype=np.float32)
        for e in self.elements:
            if e.element_type in elem_types:
                type_counts[elem_types.index(e.element_type)] += 1.0
        # Normalize
        total = max(len(self.elements), 1)
        type_counts = type_counts / total

        # Region distribution (5 dims)
        regions = ["header", "sidebar", "main", "modal", "floating"]
        region_counts = np.zeros(len(regions), dtype=np.float32)
        for e in self.elements:
            if e.region in regions:
                region_counts[regions.index(e.region)] += 1.0
        region_counts = region_counts / total

        # Focus and modal presence (9 dims)
        focus_vec = np.zeros(len(regions), dtype=np.float32)
        if self.focus_region in regions:
            focus_vec[regions.index(self.focus_region)] = 1.0
        modal_vec = np.array(
            [
                1.0 if self.active_modal else 0.0,
                len(self.active_modal) / 50.0,  # Normalized length
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )

        # Task progress (8 dims)
        progress_vec = np.zeros(8, dtype=np.float32)
        for i, (task, pct) in enumerate(list(self.task_progress.items())[:8]):
            progress_vec[i] = pct

        # Screen embedding (128 dims)
        emb = self.screen_embedding[:128]
        if len(emb) < 128:
            emb = np.pad(emb, (0, 128 - len(emb)))

        vec = np.concatenate(
            [
                app_vec,
                layout_vec,
                type_counts,
                region_counts,
                focus_vec,
                modal_vec,
                progress_vec,
                emb,
            ]
        ).astype(np.float32)

        # Pad or truncate to target_dim
        if len(vec) < target_dim:
            vec = np.pad(vec, (0, target_dim - len(vec)))
        elif len(vec) > target_dim:
            vec = vec[:target_dim]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    @classmethod
    def from_perceived_elements(
        cls,
        elements: list[Any],
        app_context: str = "unknown",
    ) -> "AbstractUIState":
        """Build abstract state from perceived elements."""
        ui_elements: list[UIElementState] = []
        for e in elements:
            # Map screen position to region
            cx = getattr(e, "x", 0) + getattr(e, "width", 0) / 2
            cy = getattr(e, "y", 0) + getattr(e, "height", 0) / 2
            region = _screen_pos_to_region(cx, cy, 1920, 1080)
            ui_elements.append(
                UIElementState(
                    element_type=getattr(e, "element_type", "unknown"),
                    region=region,
                    relative_x=cx / 1920.0,
                    relative_y=cy / 1080.0,
                    is_interactive=getattr(e, "element_type", "")
                    in {
                        "button",
                        "text_field",
                        "checkbox",
                        "menu",
                        "dropdown",
                        "link",
                        "icon",
                        "tab",
                    },
                    semantic_label=getattr(e, "text", ""),
                )
            )
        return cls(
            app_context=app_context,
            layout_mode="full",
            elements=ui_elements,
        )


def _screen_pos_to_region(x: float, y: float, w: float, h: float) -> str:
    """Map screen coordinates to abstract region."""
    rx, ry = x / w, y / h
    if ry < 0.12:
        return "header"
    if rx < 0.18:
        return "sidebar"
    if ry > 0.85:
        return "floating"
    return "main"


@dataclass
class ActionEmbedding:
    """Compact embedding of a UI action."""

    action_type: str
    target_type: str  # What kind of element is targeted
    target_region: str  # Where on screen
    semantic_intent: str  # "submit", "navigate", "input", etc.

    def to_vector(self, target_dim: int = 64) -> np.ndarray:
        """One-hot + semantic embedding of action."""
        action_types = ["click", "type", "scroll", "drag", "hotkey", "wait"]
        a_vec = np.zeros(len(action_types), dtype=np.float32)
        if self.action_type in action_types:
            a_vec[action_types.index(self.action_type)] = 1.0

        target_types = [
            "button",
            "text_field",
            "menu",
            "panel",
            "icon",
            "link",
            "other",
        ]
        t_vec = np.zeros(len(target_types), dtype=np.float32)
        if self.target_type in target_types:
            t_vec[target_types.index(self.target_type)] = 1.0

        regions = ["header", "sidebar", "main", "modal", "floating"]
        r_vec = np.zeros(len(regions), dtype=np.float32)
        if self.target_region in regions:
            r_vec[regions.index(self.target_region)] = 1.0

        # Semantic intent hash-based (32 dims)
        intent_vec = np.zeros(32, dtype=np.float32)
        h = hash(self.semantic_intent) % 32
        intent_vec[h] = 1.0

        vec = np.concatenate([a_vec, t_vec, r_vec, intent_vec])
        if len(vec) < target_dim:
            vec = np.pad(vec, (0, target_dim - len(vec)))
        elif len(vec) > target_dim:
            vec = vec[:target_dim]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)


class AbstractWorldModel:
    """Predicts abstract UI state transitions, not raw pixels.

    This makes planning tractable: the state space is ~256 dims
    instead of 8M pixels (4K screen). The model learns:
    - "Clicking a submit button in main region → modal_open with notification"
    - "Typing in search field → elements update, focus stays main"
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 64,
        hidden_dim: int = 128,
        min_training_samples: int = 8,
        max_training_samples: int = 5000,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_training_samples = min_training_samples
        self.dynamics = MLPDynamics(
            MLPConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                learning_rate=0.005,
                max_training_samples=max_training_samples,
            )
        )
        self._fallback_count = 0
        self._model_count = 0

    def predict_next_state(
        self,
        current: AbstractUIState,
        action: UiAction,
        action_emb: ActionEmbedding | None = None,
    ) -> AbstractUIState:
        """Predict next abstract state given current state and action."""
        s_vec = current.to_vector(self.state_dim)

        if action_emb is None:
            action_emb = self._embed_action(action, current)
        a_vec = action_emb.to_vector(self.action_dim)

        if len(self.dynamics._buffer) >= self.min_training_samples:
            delta = self.dynamics.forward(s_vec, a_vec)
            delta = delta[: self.state_dim]  # Ensure matching shape
            self._model_count += 1
        else:
            # Fallback: use heuristics
            delta = self._heuristic_delta(current, action)
            delta = delta[: self.state_dim]  # Ensure matching shape
            self._fallback_count += 1

        next_vec = s_vec + delta
        return self._vector_to_state(next_vec, current)

    def record_transition(
        self,
        before: AbstractUIState,
        action: UiAction,
        after: AbstractUIState,
        action_emb: ActionEmbedding | None = None,
    ) -> None:
        """Record real transition for online learning."""
        s_before = before.to_vector(self.state_dim)
        s_after = after.to_vector(self.state_dim)
        if action_emb is None:
            action_emb = self._embed_action(action, before)
        a_vec = action_emb.to_vector(self.action_dim)
        self.dynamics.record_transition(s_before, a_vec, s_after)
        # Train immediately (online learning)
        if len(self.dynamics._buffer) >= 4:
            self.dynamics.train_step(s_before, a_vec, s_after)

    def train_epoch(self) -> float:
        """Train on all buffered transitions. Returns avg loss."""
        if len(self.dynamics._buffer) < 4:
            return 0.0
        losses: list[float] = []
        for s, a, ns in self.dynamics._buffer:
            loss = self.dynamics.train_step(s, a, ns)
            losses.append(loss)
        return sum(losses) / max(len(losses), 1)

    def save(self, path: Path) -> None:
        """Save model checkpoint."""
        self.dynamics.save(path)

    def load(self, path: Path) -> bool:
        """Load model checkpoint."""
        return self.dynamics.load(path)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _embed_action(action: UiAction, state: AbstractUIState) -> ActionEmbedding:
        """Create action embedding from raw action + state context."""
        action_type = getattr(action, "action_type", "click")
        selector = getattr(action, "selector", "")
        # Infer target type from selector
        target_type = "other"
        if "btn" in selector or "button" in selector:
            target_type = "button"
        elif "edit" in selector or "input" in selector or "text" in selector:
            target_type = "text_field"
        elif "menu" in selector:
            target_type = "menu"
        elif "icon" in selector:
            target_type = "icon"
        elif "link" in selector or "a[" in selector:
            target_type = "link"

        # Infer region from action coordinates if available
        target_region = "main"
        x = getattr(action, "x", None)
        y = getattr(action, "y", None)
        if x is not None and y is not None:
            target_region = _screen_pos_to_region(x, y, 1920, 1080)
        elif state.focus_region:
            target_region = state.focus_region

        # Infer semantic intent
        intent = "interact"
        if action_type == "click":
            intent = "submit" if "submit" in selector else "navigate"
        elif action_type == "type":
            intent = "input"
        elif action_type == "scroll":
            intent = "navigate"
        elif action_type == "hotkey":
            intent = "shortcut"

        return ActionEmbedding(
            action_type=action_type,
            target_type=target_type,
            target_region=target_region,
            semantic_intent=intent,
        )

    @staticmethod
    def _heuristic_delta(state: AbstractUIState, action: UiAction) -> np.ndarray:
        """Rule-based state transition when model is untrained."""
        delta = np.zeros(256, dtype=np.float32)
        action_type = getattr(action, "action_type", "click")
        selector = getattr(action, "selector", "").lower()

        # Clicking submit → expect modal/notification
        if action_type == "click" and any(
            kw in selector for kw in {"submit", "ok", "save", "apply"}
        ):
            # Boost modal_open indicator (at index ~29)
            delta[29] = 1.0
            # Add notification element
            delta[45] = 1.0  # notification count

        # Clicking cancel/close → dismiss modal
        if action_type == "click" and any(
            kw in selector for kw in {"cancel", "close", "dismiss", "x"}
        ):
            delta[29] = -1.0  # modal closes
            delta[30] = -0.5  # modal length drops

        # Typing in search → expect results to appear
        if action_type == "type" and any(
            kw in selector for kw in {"search", "find", "query"}
        ):
            delta[42:46] = 0.3  # Increase panel/table elements

        # Scrolling → focus may shift
        if action_type == "scroll":
            delta[20:25] = 0.1  # Slight region changes

        return delta

    @staticmethod
    def _vector_to_state(
        vec: np.ndarray, reference: AbstractUIState
    ) -> AbstractUIState:
        """Convert state vector back to AbstractUIState (approximate)."""
        # This is approximate; keep the reference structure but update
        # task progress from the vector
        new_state = AbstractUIState(
            app_context=reference.app_context,
            layout_mode=reference.layout_mode,
            elements=list(reference.elements),
            active_modal=reference.active_modal,
            focus_region=reference.focus_region,
            task_progress=dict(reference.task_progress),
            screen_embedding=vec[128:256].astype(np.float32)
            if len(vec) >= 256
            else vec[-128:].astype(np.float32),
        )
        # Update task progress from vector slice
        progress_slice = vec[48:56] if len(vec) >= 56 else vec[-8:]
        tasks = list(new_state.task_progress.keys())
        for i, val in enumerate(progress_slice[: len(tasks)]):
            if i < len(tasks):
                new_state.task_progress[tasks[i]] = float(np.clip(val, 0.0, 1.0))
        return new_state

    @property
    def is_trained(self) -> bool:
        return len(self.dynamics._buffer) >= self.min_training_samples

    @property
    def model_usage_ratio(self) -> float:
        total = self._model_count + self._fallback_count
        return self._model_count / max(total, 1)


# ======================================================================== #
# AbstractMCTSAdapter — bridges AbstractWorldModel ↔ MCTSSimulator          #
#                                                                            #
# Addresses "State-Space Explosion": MCTS now plans over compact 256-dim    #
# abstract state vectors instead of raw pixel hashes.  Simulating "what     #
# happens if I click this button" costs one MLP forward pass (~0.1ms)       #
# instead of generating a full video frame.                                  #
# ======================================================================== #


class AbstractMCTSAdapter:
    """Wraps AbstractWorldModel as a WorldModel compatible with MCTSSimulator.

    The WorldModel protocol requires: predict, evaluate, is_terminal,
    available_actions — all operating on WorldState objects.

    AbstractMCTSAdapter converts between the flat WorldState.state_vector
    (list[float]) and the structured AbstractUIState, delegating dynamics
    to AbstractWorldModel which works in 256-dim semantic space rather than
    pixel space.  This is what makes MCTS tractable on real desktop UIs.
    """

    # Fixed set of candidate actions the planner proposes during MCTS
    _CANDIDATE_ACTIONS: list[dict] = [
        {"action_type": "click", "selector": "main_button", "region": "main"},
        {"action_type": "click", "selector": "header_nav", "region": "header"},
        {"action_type": "click", "selector": "sidebar_item", "region": "sidebar"},
        {"action_type": "click", "selector": "modal_confirm", "region": "modal"},
        {"action_type": "click", "selector": "modal_cancel", "region": "modal"},
        {"action_type": "type", "selector": "search_input", "region": "header"},
        {"action_type": "type", "selector": "form_field", "region": "main"},
        {"action_type": "scroll", "selector": "main_content", "region": "main"},
        {"action_type": "hotkey", "selector": "ctrl_s", "region": "main"},
        {"action_type": "hotkey", "selector": "escape", "region": "main"},
    ]
    _APP_CONTEXTS = [
        "browser",
        "file_explorer",
        "text_editor",
        "spreadsheet",
        "media",
        "settings",
        "terminal",
        "other",
    ]
    _ELEMENT_TYPES = [
        "button",
        "text_field",
        "checkbox",
        "menu",
        "panel",
        "icon",
        "tab",
        "scrollbar",
        "dropdown",
        "link",
        "image",
        "video",
        "table",
        "chart",
        "notification",
        "other",
    ]
    _REGIONS = ["header", "sidebar", "main", "modal", "floating"]

    def __init__(self, model: "AbstractWorldModel") -> None:
        self._model = model

    # ------------------------------------------------------------------ #
    # WorldModel protocol                                                  #
    # ------------------------------------------------------------------ #

    def predict(self, state: "WorldState", action: UiAction) -> "WorldState":
        """Predict next WorldState using abstract dynamics model."""
        from .mcts_simulator import WorldState as WS

        current = self._world_state_to_abstract(state)
        next_abstract = self._model.predict_next_state(current, action)
        next_vec = next_abstract.to_vector(self._model.state_dim)
        return WS(
            state_vector=next_vec.tolist(),
            depth=state.depth + 1,
            terminal=False,
            reward=0.0,
        )

    def evaluate(self, state: "WorldState", objective: str) -> float:
        """Reward = how much the abstract state resembles goal completion.

        Heuristics:
        - task_progress slots in the vector are near index 48-56
        - If a modal_open indicator (index 29) fires after a submit → reward
        - If we have notification elements (index ~44) → small reward
        """
        from .mcts_simulator import WorldState as WS

        vec = self._to_float_array(state.state_vector)
        if len(vec) < 56:
            return 0.0

        # task_progress slice: average as a proxy for goal advancement
        progress = float(np.mean(vec[48:56]))

        # Modal-open after submit = intermediate goal achieved
        modal_bonus = float(np.clip(vec[29], 0, 1)) * 0.2 if len(vec) > 29 else 0.0

        # Notification bonus (result appeared)
        notification_bonus = (
            float(np.clip(vec[44], 0, 1)) * 0.1 if len(vec) > 44 else 0.0
        )

        return float(
            np.clip(progress * 0.7 + modal_bonus + notification_bonus, 0.0, 1.0)
        )

    def is_terminal(self, state: "WorldState", objective: str) -> bool:
        """Terminal if task progress avg > 0.8 or depth exceeds safe limit."""
        vec = self._to_float_array(state.state_vector)
        if len(vec) >= 56:
            progress = float(np.mean(vec[48:56]))
            if progress > 0.8:
                return True
        return state.depth >= 12

    def available_actions(self, state: "WorldState") -> list[UiAction]:
        """Return candidate actions inferred from the current abstract UI."""
        vec = self._to_float_array(state.state_vector)
        modal_active = self._modal_active(vec)

        inferred = self._inferred_actions(vec, modal_active)
        if inferred:
            return inferred

        actions: list[UiAction] = []
        for cand in self._CANDIDATE_ACTIONS:
            region = cand["region"]
            if modal_active and region not in {"modal", "main"}:
                continue  # Blocked behind modal
            if not modal_active and region == "modal":
                continue  # No modal to interact with
            actions.append(
                UiAction(
                    action_type=cand["action_type"],
                    selector=cand["selector"],
                    metadata={"region": region, "source": "abstract_mcts"},
                )
            )
        return actions

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _world_state_to_abstract(self, state: "WorldState") -> AbstractUIState:
        """Convert flat WorldState vector → structured AbstractUIState."""
        vec = self._to_float_array(state.state_vector)
        abstract = AbstractUIState()
        if len(vec) > 0:
            abstract.screen_embedding = np.array(
                vec[128:256] if len(vec) >= 256 else vec[:128], dtype=np.float32
            )
        return abstract

    @staticmethod
    def _to_float_array(state_vector: Any) -> np.ndarray:
        """Safely convert state_vector (list or dict) to ndarray."""
        if isinstance(state_vector, np.ndarray):
            return state_vector.astype(np.float32)
        if isinstance(state_vector, list):
            return np.array(state_vector, dtype=np.float32)
        if isinstance(state_vector, dict):
            return np.array(list(state_vector.values()), dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    def _inferred_actions(self, vec: np.ndarray, modal_active: bool) -> list[UiAction]:
        app_context = self._dominant_label(
            vec[: len(self._APP_CONTEXTS)], self._APP_CONTEXTS
        )
        region_slice = vec[28:33] if len(vec) >= 33 else np.zeros(5, dtype=np.float32)
        focus_slice = vec[33:38] if len(vec) >= 38 else np.zeros(5, dtype=np.float32)
        dominant_region = self._dominant_label(region_slice, self._REGIONS) or "main"
        focus_region = (
            self._dominant_label(focus_slice, self._REGIONS) or dominant_region
        )
        region = "modal" if modal_active else focus_region
        type_slice = vec[12:28] if len(vec) >= 28 else np.zeros(16, dtype=np.float32)
        type_weights = {
            label: float(type_slice[index])
            for index, label in enumerate(self._ELEMENT_TYPES)
        }

        actions: list[UiAction] = []
        if modal_active:
            if type_weights.get("text_field", 0.0) > 0.01:
                self._append_candidate(actions, "type", "inferred_modal_field", "modal")
            if type_weights.get("button", 0.0) > 0.01:
                self._append_candidate(
                    actions, "click", "inferred_modal_confirm", "modal"
                )
            self._append_candidate(actions, "hotkey", "escape", "modal", value="escape")
            return actions

        if type_weights.get("text_field", 0.0) > 0.01:
            selector = (
                "inferred_search_input"
                if app_context in {"browser", "file_explorer"}
                or dominant_region == "header"
                else "inferred_form_field"
            )
            self._append_candidate(actions, "type", selector, region)

        if type_weights.get("button", 0.0) > 0.01:
            self._append_candidate(actions, "click", "inferred_primary_button", region)

        if (
            sum(
                type_weights.get(label, 0.0)
                for label in {"menu", "dropdown", "link", "tab"}
            )
            > 0.01
        ):
            nav_region = "header" if region_slice[0] >= region_slice[1] else "sidebar"
            self._append_candidate(actions, "click", "inferred_navigation", nav_region)

        if (
            sum(type_weights.get(label, 0.0) for label in {"panel", "table", "chart"})
            > 0.01
        ):
            self._append_candidate(actions, "scroll", "inferred_content", "main")
            self._append_candidate(actions, "click", "inferred_result_panel", "main")

        if app_context in {"text_editor", "spreadsheet", "media", "other"}:
            self._append_candidate(actions, "focus", "inferred_workspace", region)

        return actions[:6]

    @staticmethod
    def _append_candidate(
        actions: list[UiAction],
        action_type: str,
        selector: str,
        region: str,
        value: str | None = None,
    ) -> None:
        if any(
            action.action_type == action_type and action.selector == selector
            for action in actions
        ):
            return
        actions.append(
            UiAction(
                action_type=action_type,
                selector=selector,
                value=value,
                metadata={"region": region, "source": "abstract_mcts_dynamic"},
            )
        )

    @staticmethod
    def _dominant_label(vec: np.ndarray, labels: list[str]) -> str:
        if len(vec) == 0 or not np.any(vec):
            return ""
        return labels[int(np.argmax(vec))]

    @staticmethod
    def _modal_active(vec: np.ndarray) -> bool:
        layout_modal = len(vec) > 10 and vec[10] > 0.2
        modal_presence = len(vec) > 38 and vec[38] > 0.2
        legacy_modal = len(vec) > 29 and vec[29] > 0.5
        return bool(layout_modal or modal_presence or legacy_modal)
