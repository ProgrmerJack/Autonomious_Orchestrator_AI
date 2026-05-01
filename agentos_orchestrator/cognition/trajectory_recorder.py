"""Training-ready trajectory recording for universal OS-agent runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agentos_orchestrator.os_control.base import UiAction

from .abstract_world_model import AbstractUIState
from .runtime_state import (
    diff_ui_states,
    summarize_ui_state,
)


class TrajectoryRecorder:
    """Persist action/state/outcome traces as JSONL training artifacts."""

    def __init__(
        self,
        workspace_root: str | Path,
        enabled: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.enabled = enabled
        self.root = self.workspace_root / ".agentos" / "trajectories"
        self.current_path: Path | None = None

    def start_run(self, run_id: str, objective: str) -> Path | None:
        if not self.enabled:
            self.current_path = None
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / f"{_safe_name(run_id)}.jsonl"
        header = {
            "schema_version": 1,
            "event": "run_started",
            "run_id": run_id,
            "objective": objective,
            "timestamp": time.time(),
        }
        self._append(header)
        return self.current_path

    def finish_run(self, run_id: str, final_state: dict[str, Any]) -> None:
        self._append(
            {
                "schema_version": 1,
                "event": "run_finished",
                "run_id": run_id,
                "timestamp": time.time(),
                "final_state": final_state,
            }
        )

    def record_step(self, **step: Any) -> None:
        before = _required_state(step, "before")
        after = _required_state(step, "after")
        action = _required_action(step)
        outcome = step["outcome"]
        mode_decision = step.get("mode_decision")
        repair_plan = step.get("repair_plan")
        payload = {
            "schema_version": 1,
            "event": "step",
            "run_id": step.get("run_id", ""),
            "objective": step.get("objective", ""),
            "timestamp": time.time(),
            "option": step.get("option_name", ""),
            "mode_decision": (
                mode_decision.to_prompt_dict() if mode_decision else None
            ),
            "capability_profile": step.get("capability_profile"),
            "adapter_context": step.get("adapter_context"),
            "repair_plan": (repair_plan.to_prompt_dict() if repair_plan else None),
            "before": summarize_ui_state(before),
            "after": summarize_ui_state(after),
            "diff": diff_ui_states(before, after),
            "action": _action_payload(action),
            "expected_observation": step.get("expected_observation", ""),
            "receipt": str(step.get("receipt", ""))[:4000],
            "outcome_evaluation": outcome.to_prompt_dict(),
            "verification_contract": step.get("verification_contract"),
            "verification_result": step.get("verification_result"),
            "latency_ms": step.get("latency_ms", 0.0),
        }
        self._append(payload)

    def _append(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.current_path is None:
            return
        with self.current_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _action_payload(action: UiAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type,
        "selector": action.selector,
        "value": action.value,
        "metadata": dict(action.metadata),
    }


def _required_state(step: dict[str, Any], key: str) -> AbstractUIState:
    value = step[key]
    if not isinstance(value, AbstractUIState):
        raise TypeError(f"{key} must be an AbstractUIState")
    return value


def _required_action(step: dict[str, Any]) -> UiAction:
    action = step["action"]
    if not isinstance(action, UiAction):
        raise TypeError("action must be a UiAction")
    return action


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def trajectory_root_name() -> str:
    return "trajectories"
