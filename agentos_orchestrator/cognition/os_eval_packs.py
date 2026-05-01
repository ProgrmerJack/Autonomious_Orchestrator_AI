"""Synthetic/live-fire eval pack definitions for app-family coverage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agentos_orchestrator.app_family_registry import (
    eval_surface_families,
    live_fire_families,
)

SURFACE_FAMILIES = eval_surface_families()
TASK_INTENTS = (
    "open_app",
    "find_target",
    "fill_form",
    "save_file",
    "attach_file",
    "recover_modal",
    "switch_window",
    "export_artifact",
    "copy_paste",
    "wait_delayed_dialog",
    "recover_focus",
    "verify_outcome",
    "scroll_and_select",
    "use_shortcut",
    "invalid_input_repair",
    "stale_target_reground",
    "clipboard_roundtrip",
    "path_validation",
    "tool_vs_ui_choice",
    "approval_boundary",
)


@dataclass(slots=True)
class EvalTask:
    task_id: str
    surface: str
    intent: str
    objective: str
    expected_verifications: list[str]
    failure_modes: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    live_fire: bool = False

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalPack:
    name: str
    version: int
    tasks: list[EvalTask]

    def asdict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task_count": len(self.tasks),
            "surfaces": sorted({task.surface for task in self.tasks}),
            "tasks": [task.asdict() for task in self.tasks],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        by_surface = {surface: 0 for surface in SURFACE_FAMILIES}
        for task in self.tasks:
            by_surface[task.surface] = by_surface.get(task.surface, 0) + 1
        return {
            "task_count": len(self.tasks),
            "surface_counts": by_surface,
            "metric_names": TRACKED_METRICS,
            "ready_for_live_fire": all(
                by_surface.get(surface, 0) >= 20 for surface in SURFACE_FAMILIES
            ),
        }


TRACKED_METRICS = [
    "success_rate",
    "intervention_count",
    "retry_count",
    "recovery_rate",
    "latency_ms",
    "unsafe_action_blocks",
]


def build_universal_app_eval_pack() -> EvalPack:
    tasks: list[EvalTask] = []
    for surface in SURFACE_FAMILIES:
        for index, intent in enumerate(TASK_INTENTS, start=1):
            tasks.append(_task(surface, intent, index))
    return EvalPack(name="universal_app_surface_v1", version=1, tasks=tasks)


def eval_pack_payload(max_tasks: int | None = 100) -> dict[str, Any]:
    pack = build_universal_app_eval_pack()
    if max_tasks is not None and len(pack.tasks) > max_tasks:
        pack = EvalPack(
            name=pack.name,
            version=pack.version,
            tasks=_round_robin_task_subset(pack.tasks, max_tasks),
        )
    payload = pack.asdict()
    payload["full_task_count"] = len(build_universal_app_eval_pack().tasks)
    return payload


def _round_robin_task_subset(tasks: list[EvalTask], max_tasks: int) -> list[EvalTask]:
    if max_tasks <= 0:
        return []
    by_surface: dict[str, list[EvalTask]] = {
        surface: [] for surface in SURFACE_FAMILIES
    }
    for task in tasks:
        by_surface.setdefault(task.surface, []).append(task)
    selected: list[EvalTask] = []
    round_index = 0
    while len(selected) < max_tasks:
        added = False
        for surface in SURFACE_FAMILIES:
            surface_tasks = by_surface.get(surface, [])
            if round_index < len(surface_tasks):
                selected.append(surface_tasks[round_index])
                added = True
                if len(selected) >= max_tasks:
                    break
        if not added:
            break
        round_index += 1
    return selected


def _task(surface: str, intent: str, index: int) -> EvalTask:
    objective = _objective(surface, intent)
    return EvalTask(
        task_id=f"{surface}_{index:02d}_{intent}",
        surface=surface,
        intent=intent,
        objective=objective,
        expected_verifications=_verifications(intent),
        failure_modes=_failure_modes(intent),
        metrics=list(TRACKED_METRICS),
        live_fire=surface in live_fire_families(),
    )


def _objective(surface: str, intent: str) -> str:
    readable = intent.replace("_", " ")
    surface_name = surface.replace("_", " ")
    return f"On a {surface_name} surface, {readable} and verify the result."


def _verifications(intent: str) -> list[str]:
    mapping = {
        "fill_form": ["field_contains"],
        "save_file": ["file_exists", "modal_closed"],
        "attach_file": ["file_exists", "state_changed"],
        "recover_modal": ["modal_closed"],
        "export_artifact": ["export_hash_changed", "file_exists"],
        "copy_paste": ["clipboard_contains", "field_contains"],
        "wait_delayed_dialog": ["modal_closed", "state_changed"],
        "recover_focus": ["window_title_changed", "state_changed"],
        "invalid_input_repair": ["field_contains", "state_changed"],
        "path_validation": ["file_exists"],
        "tool_vs_ui_choice": ["receipt_success"],
    }
    return mapping.get(intent, ["state_changed"])


def _failure_modes(intent: str) -> list[str]:
    mapping = {
        "recover_modal": ["surprise_modal"],
        "wait_delayed_dialog": ["delayed_dialog"],
        "recover_focus": ["focus_theft"],
        "invalid_input_repair": ["invalid_input"],
        "stale_target_reground": ["stale_screenshot"],
        "path_validation": ["invalid_path"],
        "tool_vs_ui_choice": ["tool_vs_ui_routing"],
        "approval_boundary": ["policy_block"],
    }
    return mapping.get(intent, [])
