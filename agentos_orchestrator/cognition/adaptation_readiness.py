from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AdaptationReadiness:
    status: str
    connected: bool
    latest_training_path: str = ""
    latest_live_fire_path: str = ""
    grounding_examples: int = 0
    world_model_transitions: int = 0
    heldout_success_rate: float = 0.0
    heldout_task_count: int = 0
    underfilled: bool = False
    missing_total: int = 0
    blockers: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def collect_adaptation_readiness(workspace_root: str | Path) -> AdaptationReadiness:
    root = Path(workspace_root)
    training_path = _latest_training_state(root)
    live_fire_path = _latest_live_fire_summary(root)
    training = _read_json(training_path)
    live_fire = _read_json(live_fire_path)

    scale = _latest_scale_report(training)
    underfill = _latest_underfill_report(training)
    heldout = dict(live_fire.get("heldout_metrics") or {})
    grounding = _safe_int(scale.get("grounding_examples"))
    transitions = _safe_int(scale.get("world_model_transitions"))
    success_rate = _safe_float(heldout.get("success_rate"))
    heldout_count = _safe_int(heldout.get("task_count"))
    blockers: list[str] = []
    if not training_path:
        blockers.append("no_adaptation_training_artifact")
    if not live_fire_path:
        blockers.append("no_heldout_live_fire_artifact")
    if underfill.get("underfilled"):
        blockers.append("dataset_underfill")
    if not bool(scale.get("meets_minimum_scale")):
        blockers.append("scale_target_not_met")
    if heldout_count == 0 or success_rate < 0.8:
        blockers.append("heldout_success_target_not_met")

    guidance = _guidance(blockers)
    connected = bool(training_path and live_fire_path)
    status = "ready" if connected and not blockers else "needs_training_or_eval"
    if connected and blockers == ["scale_target_not_met"]:
        status = "bootstrap_ready_scale_incomplete"
    return AdaptationReadiness(
        status=status,
        connected=connected,
        latest_training_path=str(training_path) if training_path else "",
        latest_live_fire_path=str(live_fire_path) if live_fire_path else "",
        grounding_examples=grounding,
        world_model_transitions=transitions,
        heldout_success_rate=success_rate,
        heldout_task_count=heldout_count,
        underfilled=bool(underfill.get("underfilled")),
        missing_total=_safe_int(underfill.get("missing_total")),
        blockers=blockers,
        guidance=guidance,
    )


def _latest_training_state(root: Path) -> Path | None:
    candidates = list(
        (root / ".agentos").glob("adaptation_longrun*/**/long_run_result.json")
    )
    candidates.extend(
        (root / ".agentos").glob("adaptation_longrun*/**/long_run_state.json")
    )
    candidates.extend((root / ".agentos").glob("adaptation*/**/training_result.json"))
    return _newest(candidates)


def _latest_live_fire_summary(root: Path) -> Path | None:
    candidates = list((root / ".agentos" / "live_fire_eval").glob("*.json"))
    candidates.extend((root / "artifacts" / "live_fire_eval").glob("*.json"))
    return _newest(candidates)


def _newest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_scale_report(training: dict[str, Any]) -> dict[str, Any]:
    scale = training.get("scale_report")
    if isinstance(scale, dict):
        return scale
    shard_results = list(training.get("shard_results") or [])
    if shard_results:
        scale = dict(shard_results[-1]).get("scale_report")
        if isinstance(scale, dict):
            return scale
    return {}


def _latest_underfill_report(training: dict[str, Any]) -> dict[str, Any]:
    underfill = training.get("underfill")
    if isinstance(underfill, dict):
        return underfill
    shard_results = list(training.get("shard_results") or [])
    if shard_results:
        underfill = dict(shard_results[-1]).get("underfill")
        if isinstance(underfill, dict):
            return underfill
    return {}


def _guidance(blockers: list[str]) -> list[str]:
    messages = {
        "no_adaptation_training_artifact": "Run pc-train-adaptation-longrun so learned grounding and world-model state are persisted.",
        "no_heldout_live_fire_artifact": "Run held-out live-fire eval against the virtual sandbox before claiming generality.",
        "dataset_underfill": "Resume long-run training with fresh GUI-Actor or OSWorld cache candidates to recover missing budget.",
        "scale_target_not_met": "Continue shard training toward the 100K+ minimum scale target.",
        "heldout_success_target_not_met": "Promote failures, shadow-train, then rerun held-out live-fire eval.",
    }
    return [messages[item] for item in blockers if item in messages]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
