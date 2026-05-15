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
    workflow_surfaces: list[str] = field(default_factory=list)

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
            "cross_app_task_count": sum(
                1 for task in self.tasks if len(task.workflow_surfaces or []) > 1
            ),
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

PRODUCT_DEFAULT_PACK = "combined"


def build_universal_app_eval_pack() -> EvalPack:
    tasks: list[EvalTask] = []
    for surface in SURFACE_FAMILIES:
        for index, intent in enumerate(TASK_INTENTS, start=1):
            tasks.append(_task(surface, intent, index))
    return EvalPack(name="universal_app_surface_v1", version=1, tasks=tasks)


def build_real_user_handoff_eval_pack() -> EvalPack:
    tasks = [
        EvalTask(
            task_id="browser_editor_clipboard_address",
            surface="browser",
            intent="browser_editor_handoff",
            objective=(
                "Search Chrome for the nearest UPS store and copy the address into "
                "Notepad."
            ),
            expected_verifications=["clipboard_contains", "field_contains"],
            failure_modes=["browser_to_editor_handoff", "copy_text_vs_copy_file"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["browser", "editor"],
        ),
        EvalTask(
            task_id="file_explorer_invoice_rename",
            surface="file_explorer",
            intent="local_file_handoff",
            objective=(
                "Find the latest invoice PDF in Downloads and rename it to "
                "april-invoice.pdf."
            ),
            expected_verifications=["file_exists", "state_changed"],
            failure_modes=["local_file_vs_browser_route", "rename_target_miss"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["file_explorer"],
        ),
        EvalTask(
            task_id="chat_search_draft_reply",
            surface="chat_app",
            intent="chat_reply_handoff",
            objective=(
                "Open Slack, search for messages about Q4 planning, and draft a "
                "short reply summary."
            ),
            expected_verifications=["field_contains", "state_changed"],
            failure_modes=["chat_search_miss", "unsafe_send"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["chat_app"],
        ),
        EvalTask(
            task_id="pdf_editor_note_transfer",
            surface="pdf_viewer",
            intent="pdf_editor_handoff",
            objective=(
                "Open the PDF, extract the renewal clause note, and paste it into "
                "Notepad."
            ),
            expected_verifications=["clipboard_contains", "field_contains"],
            failure_modes=["pdf_extraction_miss", "clipboard_handoff_loss"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["pdf_viewer", "editor"],
        ),
    ]
    return EvalPack(name="real_user_handoff_v1", version=1, tasks=tasks)


def build_everyday_family_eval_pack() -> EvalPack:
    tasks = [
        EvalTask(
            task_id="email_attachment_send",
            surface="email",
            intent="email_send_attachment",
            objective=(
                "Open Outlook, attach the latest invoice PDF from Downloads to an "
                "email for Alex, and send it."
            ),
            expected_verifications=["send_outcome"],
            failure_modes=["attachment_selection_miss", "unsafe_send"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["file_explorer", "email"],
        ),
        EvalTask(
            task_id="calendar_invite_from_email",
            surface="calendar",
            intent="calendar_invite_from_email",
            objective=(
                "Find the Zoom invite in Outlook email, add it to the calendar, "
                "and create the invite."
            ),
            expected_verifications=["invite_outcome"],
            failure_modes=["email_lookup_miss", "calendar_invite_missing"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["email", "calendar"],
        ),
        EvalTask(
            task_id="settings_toggle_night_light",
            surface="settings",
            intent="settings_toggle_night_light",
            objective=(
                "Open Windows Settings, search for Night Light, and turn it on."
            ),
            expected_verifications=["toggle_state"],
            failure_modes=["settings_search_miss", "toggle_proof_missing"],
            metrics=list(TRACKED_METRICS),
            live_fire=True,
            workflow_surfaces=["settings"],
        ),
    ]
    return EvalPack(name="everyday_family_realism_v1", version=1, tasks=tasks)


def build_combined_live_fire_eval_pack() -> EvalPack:
    tasks = [
        *build_universal_app_eval_pack().tasks,
        *build_real_user_handoff_eval_pack().tasks,
        *build_everyday_family_eval_pack().tasks,
    ]
    return EvalPack(name="combined_live_fire_v1", version=1, tasks=tasks)


def normalize_eval_pack_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized in {"", "default", "product"}:
        return PRODUCT_DEFAULT_PACK
    if normalized in {"universal", "handoff", "everyday", "combined"}:
        return normalized
    return PRODUCT_DEFAULT_PACK


def resolve_eval_pack(name: str) -> EvalPack:
    builders = {
        "universal": build_universal_app_eval_pack,
        "handoff": build_real_user_handoff_eval_pack,
        "everyday": build_everyday_family_eval_pack,
        "combined": build_combined_live_fire_eval_pack,
    }
    return builders[normalize_eval_pack_name(name)]()


def available_eval_packs() -> dict[str, dict[str, Any]]:
    packs = {
        "universal": build_universal_app_eval_pack(),
        "handoff": build_real_user_handoff_eval_pack(),
        "everyday": build_everyday_family_eval_pack(),
        "combined": build_combined_live_fire_eval_pack(),
    }
    return {
        name: {
            "name": pack.name,
            "task_count": len(pack.tasks),
            "surfaces": sorted({task.surface for task in pack.tasks}),
            "cross_app_task_count": sum(
                1 for task in pack.tasks if len(task.workflow_surfaces or []) > 1
            ),
        }
        for name, pack in packs.items()
    }


def eval_pack_payload(
    pack: str = PRODUCT_DEFAULT_PACK,
    max_tasks: int | None = 100,
) -> dict[str, Any]:
    selected_pack = normalize_eval_pack_name(pack)
    full_pack = resolve_eval_pack(selected_pack)
    pack_payload = full_pack
    if max_tasks is not None and len(full_pack.tasks) > max_tasks:
        pack_payload = EvalPack(
            name=full_pack.name,
            version=full_pack.version,
            tasks=_round_robin_task_subset(full_pack.tasks, max_tasks),
        )
    payload = pack_payload.asdict()
    payload["pack"] = selected_pack
    payload["product_default_pack"] = PRODUCT_DEFAULT_PACK
    payload["available_packs"] = available_eval_packs()
    payload["full_task_count"] = len(full_pack.tasks)
    return payload


def _round_robin_task_subset(tasks: list[EvalTask], max_tasks: int) -> list[EvalTask]:
    if max_tasks <= 0:
        return []
    by_surface: dict[str, list[EvalTask]] = {}
    for task in tasks:
        by_surface.setdefault(task.surface, []).append(task)
    surface_order = list(dict.fromkeys(task.surface for task in tasks))
    selected: list[EvalTask] = []
    round_index = 0
    while len(selected) < max_tasks:
        added = False
        for surface in surface_order:
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
        workflow_surfaces=[surface],
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
        "email_send_attachment": ["send_outcome"],
        "calendar_invite_from_email": ["invite_outcome"],
        "settings_toggle_night_light": ["toggle_state"],
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
