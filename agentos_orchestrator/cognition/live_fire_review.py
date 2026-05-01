from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .live_fire_eval import (
    LiveFireTaskResult,
    _promote_failure,
)
from .replay_debug import load_replay_debug
from .trajectory_training import SHADOW_HEAD_ORDER, TrajectoryTrainingBuilder

TRIAGE_CLASSES = (
    "selector_grounding",
    "modal_handling",
    "path_validation",
    "backend_limitation",
    "policy_block",
    "unknown",
)


@dataclass(frozen=True)
class LiveFireFailureReview:
    run_id: str
    task_id: str
    surface: str
    intent: str
    classification: str
    durable: bool
    promotable: bool
    failure_reason: str
    replay_payload: dict[str, Any]
    existing_golden_trace: str = ""

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveFireRunReview:
    run_id: str
    summary_path: str
    backend: str
    success: bool
    passed: int
    failed: int
    task_count: int
    created_at: float
    promoted_traces: list[str] = field(default_factory=list)
    failures: list[LiveFireFailureReview] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = [item.asdict() for item in self.failures]
        return payload


def load_live_fire_reviews(
    workspace_root: str | Path,
    limit: int = 10,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve(strict=False)
    runs = [_review_path(root, path) for path in _summary_paths(root, limit)]
    flat_failures = [failure for run in runs for failure in run.failures]
    return {
        "schema_version": 1,
        "runs": [run.asdict() for run in runs],
        "failed_tasks": [failure.asdict() for failure in flat_failures],
        "milestone": _milestone(runs),
        "triage_classes": list(TRIAGE_CLASSES),
    }


def promote_live_fire_failure(
    workspace_root: str | Path,
    run_id: str,
    task_id: str,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve(strict=False)
    review = _find_failure(root, run_id, task_id)
    if not review:
        return {
            "status": "not_found",
            "run_id": run_id,
            "task_id": task_id,
        }
    if not review.promotable:
        return {
            "status": "not_promotable",
            "run_id": run_id,
            "task_id": task_id,
            "classification": review.classification,
            "durable": review.durable,
        }
    path = _promote_failure(run_id, _task_from_review(review), root)
    _record_promoted_trace(root, run_id, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["failure_review"] = review.asdict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "status": "promoted",
        "run_id": run_id,
        "task_id": task_id,
        "path": str(path),
        "classification": review.classification,
    }


def write_shadow_training_heads(
    workspace_root: str | Path,
    trajectory_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve(strict=False)
    paths = trajectory_paths or _default_shadow_trajectories(root)
    summary = TrajectoryTrainingBuilder(root).write_shadow_head_datasets(
        output_dir=output_dir,
        paths=list(paths),
        head_order=SHADOW_HEAD_ORDER,
    )
    summary["source_paths"] = [str(path) for path in paths]
    return summary


def classify_failure(task_result: dict[str, Any]) -> str:
    text = _failure_text(task_result)
    for classification, tokens in _TRIAGE_RULES:
        if any(token in text for token in tokens):
            return classification
    return "unknown"


_TRIAGE_RULES = (
    ("policy_block", ("policy-blocked", "unsafe", "blocked")),
    ("modal_handling", ("modal", "dialog")),
    ("path_validation", ("path", "file_exists", "outside")),
    ("selector_grounding", ("selector", "target", "typed value")),
    ("backend_limitation", ("backend", "exception", "not available")),
)


def _failure_text(task_result: dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in (
            task_result.get("failure_reason"),
            _dict(task_result.get("verification_result")).get("message"),
            _dict(task_result.get("verification")).get("kind"),
            task_result.get("intent"),
            task_result.get("surface"),
            json.dumps(task_result.get("receipts") or []),
        )
    ).lower()


def _summary_paths(root: Path, limit: int) -> list[Path]:
    summary_dir = root / ".agentos" / "live_fire_eval"
    if not summary_dir.exists():
        return []
    paths = sorted(
        summary_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[: max(1, limit)]


def _review_path(root: Path, path: Path) -> LiveFireRunReview:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or path.stem)
    replay = load_replay_debug(root, run_id=run_id, limit=25)
    promoted = [str(item) for item in payload.get("promoted_traces") or []]
    failures = [
        _failure_review(root, run_id, item, replay, promoted)
        for item in payload.get("task_results", [])
        if not bool(item.get("success"))
    ]
    return LiveFireRunReview(
        run_id=run_id,
        summary_path=str(path),
        backend=_backend_from_summary(payload, run_id),
        success=bool(payload.get("success")),
        passed=_int_field(payload, "passed"),
        failed=_int_field(payload, "failed", len(failures)),
        task_count=_int_field(payload, "task_count"),
        created_at=float(path.stat().st_mtime),
        promoted_traces=promoted,
        failures=failures,
    )


def _failure_review(
    root: Path,
    run_id: str,
    task_result: dict[str, Any],
    replay: dict[str, Any],
    promoted: list[str],
) -> LiveFireFailureReview:
    classification = classify_failure(task_result)
    failure_reason = str(task_result.get("failure_reason") or "")
    task_id = str(task_result.get("task_id") or "")
    existing = _existing_trace(root, run_id, task_id, promoted)
    replay_payload = _task_replay_payload(replay, task_id)
    durable = classification != "unknown" and bool(failure_reason)
    return LiveFireFailureReview(
        run_id=run_id,
        task_id=task_id,
        surface=str(task_result.get("surface") or ""),
        intent=str(task_result.get("intent") or ""),
        classification=classification,
        durable=durable,
        promotable=durable and not existing,
        failure_reason=failure_reason,
        replay_payload=replay_payload,
        existing_golden_trace=existing,
    )


def _task_replay_payload(
    replay: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    for run in replay.get("runs", []):
        for step in run.get("steps", []):
            if step.get("task_id") == task_id:
                return {"run": run.get("run_id"), "step": step}
    return replay


def _existing_trace(
    root: Path,
    run_id: str,
    task_id: str,
    promoted: list[str],
) -> str:
    for promoted_path in promoted:
        if task_id and task_id in Path(promoted_path).stem:
            return promoted_path
    golden_dir = root / "benchmarks" / "golden_traces"
    if not golden_dir.exists():
        return ""
    patterns = (
        f"live_fire_{task_id}_*.json",
        f"live_fire_{run_id}_{task_id}_*.json",
    )
    for pattern in patterns:
        for trace_path in golden_dir.glob(pattern):
            return str(trace_path)
    return ""


def _find_failure(
    root: Path,
    run_id: str,
    task_id: str,
) -> LiveFireFailureReview | None:
    reviews = load_live_fire_reviews(root, limit=50)
    for failure in reviews["failed_tasks"]:
        if failure["run_id"] == run_id and failure["task_id"] == task_id:
            return LiveFireFailureReview(**failure)
    return None


def _task_from_review(review: LiveFireFailureReview) -> LiveFireTaskResult:
    return LiveFireTaskResult(
        task_id=review.task_id,
        surface=review.surface,
        intent=review.intent,
        success=False,
        trajectory_path="",
        action_count=0,
        failure_reason=review.failure_reason,
    )


def _milestone(runs: list[LiveFireRunReview]) -> dict[str, Any]:
    real_windows_tasks = _real_windows_task_count(runs)
    promoted_count = len(_promoted_traces(runs))
    unsafe_blocks = _unsafe_block_count(runs)
    return {
        "real_windows_task_target": 50,
        "durable_failure_target": 10,
        "real_windows_tasks": real_windows_tasks,
        "durable_promoted_failures": promoted_count,
        "unsafe_action_blocks": unsafe_blocks,
        "ready_to_widen_scope": (
            real_windows_tasks >= 50 and promoted_count >= 10 and unsafe_blocks == 0
        ),
    }


def _default_shadow_trajectories(root: Path) -> list[str]:
    trajectory_dir = root / ".agentos" / "trajectories"
    if not trajectory_dir.exists():
        return []
    run_ids = ("full_100_virtual_live_fire", "windows_uia_editor_smoke")
    paths: list[str] = []
    for run_id in run_ids:
        paths.extend(
            str(path) for path in sorted(trajectory_dir.glob(f"{run_id}*.jsonl"))
        )
    return paths


def _int_field(
    payload: dict[str, Any],
    key: str,
    default: int = 0,
) -> int:
    return int(payload.get(key) or default)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _real_windows_task_count(runs: list[LiveFireRunReview]) -> int:
    return sum(run.task_count for run in runs if _is_windows_uia(run))


def _backend_from_summary(payload: dict[str, Any], run_id: str) -> str:
    backend = str(payload.get("backend") or "")
    if backend:
        return backend
    if "windows_uia" in run_id or "windows-uia" in run_id:
        return "windows-uia"
    paths = " ".join(str(path) for path in payload.get("trajectory_paths") or [])
    return "windows-uia" if "windows_uia" in paths else ""


def _is_windows_uia(run: LiveFireRunReview) -> bool:
    return run.backend in {"windows-uia", "WindowsUiaBackend"}


def _promoted_traces(runs: list[LiveFireRunReview]) -> set[str]:
    return {
        path
        for run in runs
        for path in (
            list(run.promoted_traces)
            + [failure.existing_golden_trace for failure in run.failures]
        )
        if path
    }


def _record_promoted_trace(root: Path, run_id: str, path: Path) -> None:
    summary_path = root / ".agentos" / "live_fire_eval" / f"{run_id}.json"
    if not summary_path.exists():
        return
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    promoted = [str(item) for item in payload.get("promoted_traces") or []]
    trace_path = str(path)
    if trace_path not in promoted:
        promoted.append(trace_path)
    payload["promoted_traces"] = promoted
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _unsafe_block_count(runs: list[LiveFireRunReview]) -> int:
    return sum(
        1
        for run in runs
        for failure in run.failures
        if failure.classification == "policy_block"
    )
