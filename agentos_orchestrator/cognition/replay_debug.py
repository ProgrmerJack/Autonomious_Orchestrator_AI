"""Replay/debug payloads for recorded universal-agent trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_replay_debug(
    workspace_root: str | Path,
    run_id: str = "",
    limit: int = 1,
) -> dict[str, Any]:
    paths = _trajectory_paths(Path(workspace_root), run_id, limit)
    runs = [_load_path(path) for path in paths]
    return {
        "run_count": len(runs),
        "runs": runs,
        "schema_version": 1,
    }


def _trajectory_paths(root: Path, run_id: str, limit: int) -> list[Path]:
    trace_root = root / ".agentos" / "trajectories"
    if not trace_root.exists():
        return []
    paths = sorted(
        trace_root.glob("*.jsonl"),
        key=lambda item: item.stat().st_mtime,
    )
    if run_id:
        paths = [path for path in paths if run_id in path.stem]
    return list(reversed(paths))[: max(1, limit)]


def _load_path(path: Path) -> dict[str, Any]:
    events = _events(path)
    started = next(
        (event for event in events if event.get("event") == "run_started"),
        {},
    )
    finished = next(
        (event for event in events if event.get("event") == "run_finished"),
        {},
    )
    steps = [_debug_step(event) for event in events if event.get("event") == "step"]
    return {
        "path": str(path),
        "run_id": started.get("run_id") or finished.get("run_id") or path.stem,
        "objective": started.get("objective", ""),
        "step_count": len(steps),
        "success": bool(finished.get("final_state", {}).get("success", False)),
        "steps": steps,
        "final_state": finished.get("final_state", {}),
    }


def _events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _debug_step(event: dict[str, Any]) -> dict[str, Any]:
    action = _dict(event.get("action"))
    metadata = _dict(action.get("metadata"))
    return {
        "option": event.get("option", ""),
        "state": {
            "before": event.get("before", {}),
            "after": event.get("after", {}),
            "diff": event.get("diff", {}),
        },
        "chosen_mode": event.get("mode_decision"),
        "capability_profile": event.get("capability_profile")
        or metadata.get("capability_profile"),
        "adapter_context": event.get("adapter_context")
        or metadata.get("adapter_context"),
        "action": {
            "action_type": action.get("action_type"),
            "selector": action.get("selector"),
            "value": action.get("value"),
            "control_channel": metadata.get("control_channel"),
        },
        "expected_observation": event.get("expected_observation", ""),
        "verification_contract": event.get("verification_contract")
        or metadata.get("verification_contract"),
        "verification_result": event.get("verification_result"),
        "receipt": event.get("receipt", ""),
        "outcome": event.get("outcome_evaluation"),
        "blocker": _blocker(event),
        "repair": event.get("repair_plan"),
        "latency_ms": event.get("latency_ms", 0.0),
    }


def _blocker(event: dict[str, Any]) -> dict[str, Any] | None:
    outcome = _dict(event.get("outcome_evaluation"))
    if not outcome.get("new_blocker") and not outcome.get("failure_reason"):
        return None
    return {
        "new_blocker": outcome.get("new_blocker"),
        "failure_reason": outcome.get("failure_reason"),
        "suggested_repair": outcome.get("suggested_repair"),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
