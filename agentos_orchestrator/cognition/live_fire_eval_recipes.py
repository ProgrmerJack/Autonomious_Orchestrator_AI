"""Action recipes and state adapters for live-fire eval tasks."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from agentos_orchestrator.app_family_registry import (
    app_context_for_family,
    launch_target_for_family,
    primary_selector_for_family,
)
from agentos_orchestrator.os_control.base import UiAction, UiNode

from .abstract_world_model import AbstractUIState, UIElementState
from .os_eval_packs import EvalTask
from .verification_contracts import VerificationContract


def actions_for_task(
    task: EvalTask,
    artifact_root: Path,
    run_id: str,
) -> list[UiAction]:
    workflow = _workflow_actions_for_task(task, artifact_root, run_id)
    if workflow is not None:
        return workflow
    setup = _surface_setup(task.surface)
    main = _intent_action(task, artifact_root, run_id)
    return [setup] if main is None else [setup, main]


def abstract_state(surface: str, nodes: list[UiNode]) -> AbstractUIState:
    return AbstractUIState(
        app_context=_app_context(surface),
        layout_mode=_layout_mode(nodes),
        elements=[_element_from_node(node) for node in nodes],
        active_modal=_active_modal(nodes),
        focus_region=_focus_region(nodes),
    )


def snapshot_nodes(backend: Any) -> list[UiNode]:
    try:
        return list(backend.snapshot())
    except (OSError, RuntimeError, ValueError, AttributeError):
        return []


def receipt_payload(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return {"raw": str(value)[:500]}
    return payload if isinstance(payload, dict) else {"raw": str(payload)}


def json_receipt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _surface_setup(surface: str) -> UiAction:
    launch_target = _launch_target(surface)
    return UiAction(
        "launch_app",
        launch_target,
        launch_target,
        metadata={
            "verification_contract": VerificationContract(
                kind="process_launched",
                expected=f"{surface} surface launches.",
                required=True,
            ).asdict()
        },
    )


def _intent_action(
    task: EvalTask,
    artifact_root: Path,
    run_id: str,
) -> UiAction | None:
    if task.intent == "open_app":
        return None
    if _file_intent(task.intent):
        return _file_action(task, artifact_root, run_id)
    builders: dict[str, Callable[[EvalTask], UiAction]] = {
        "tool_vs_ui_choice": _tool_action,
        "approval_boundary": _approval_action,
        "stale_target_reground": _explore_action,
        "recover_modal": _modal_action,
        "wait_delayed_dialog": _modal_action,
        "switch_window": _shortcut_action,
        "recover_focus": _shortcut_action,
        "use_shortcut": _shortcut_action,
        "fill_form": _field_action,
        "copy_paste": _field_action,
        "clipboard_roundtrip": _field_action,
        "invalid_input_repair": _valid_field_action,
    }
    return builders.get(task.intent, _default_click_action)(task)


def _workflow_actions_for_task(
    task: EvalTask,
    artifact_root: Path,
    run_id: str,
) -> list[UiAction] | None:
    if task.intent == "browser_editor_handoff":
        query = "nearest UPS store"
        value = "Address result for: nearest UPS store"
        return [
            _surface_setup("browser"),
            UiAction(
                "open_url",
                "browser-address-bar",
                "https://www.bing.com/search?q=nearest+UPS+store",
                metadata={
                    "verification_contract": VerificationContract(
                        kind="state_changed",
                        expected="The browser search result loads for the requested address.",
                        target="browser-address-bar",
                        value=query,
                    ).asdict()
                },
            ),
            UiAction(
                "set_clipboard",
                "workflow-clipboard",
                value,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="clipboard_contains",
                        expected="Clipboard contains the copied address.",
                        value=value,
                        required=True,
                    ).asdict()
                },
            ),
            _surface_setup("editor"),
            UiAction(
                "set_text",
                "document-canvas",
                value,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected="The destination editor contains the copied address.",
                        target="document-canvas",
                        value=value,
                    ).asdict()
                },
            ),
        ]
    if task.intent == "local_file_handoff":
        staging_dir = artifact_root / "live_fire_inputs" / run_id
        source_path = str((staging_dir / "invoice.pdf").resolve())
        target_path = str((staging_dir / "april-invoice.pdf").resolve())
        return [
            UiAction(
                "launch_app",
                launch_target_for_family("file_explorer"),
                value=launch_target_for_family("file_explorer"),
                metadata={
                    "create_artifact_path": source_path,
                    "artifact_content": "Invoice PDF fixture",
                },
            ),
            UiAction(
                "rename_file",
                "explorer-file-list",
                f"{source_path} -> {target_path}",
                metadata={
                    "operation": "rename",
                    "source": source_path,
                    "new_name": target_path,
                    "path": target_path,
                    "create_artifact_path": target_path,
                    "artifact_content": "Invoice PDF fixture",
                    "verification_contract": VerificationContract(
                        kind="file_exists",
                        expected="The renamed invoice file exists.",
                        target="explorer-file-list",
                        path=target_path,
                    ).asdict(),
                },
            ),
        ]
    if task.intent == "chat_reply_handoff":
        query = "Q4 planning"
        reply = "Draft summary for: Q4 planning"
        return [
            _surface_setup("chat_app"),
            UiAction(
                "type",
                "conversation-list",
                f"Search messages: {query}",
                metadata={
                    "verification_contract": VerificationContract(
                        kind="state_changed",
                        expected="The chat search query is applied.",
                        target="conversation-list",
                        value=query,
                    ).asdict()
                },
            ),
            UiAction(
                "type",
                "chat-composer",
                reply,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected="The draft reply is present in the chat composer.",
                        target="chat-composer",
                        value=reply,
                    ).asdict()
                },
            ),
        ]
    if task.intent == "pdf_editor_handoff":
        note = "PDF note: renewal clause"
        return [
            _surface_setup("pdf_viewer"),
            UiAction(
                "type",
                "pdf-search-box",
                "renewal clause",
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected="The PDF search box contains the requested query.",
                        target="pdf-search-box",
                        value="renewal clause",
                    ).asdict()
                },
            ),
            UiAction(
                "set_clipboard",
                "workflow-clipboard",
                note,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="clipboard_contains",
                        expected="Clipboard contains the extracted PDF note.",
                        value=note,
                        required=True,
                    ).asdict()
                },
            ),
            _surface_setup("editor"),
            UiAction(
                "set_text",
                "document-canvas",
                note,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected="The destination editor contains the extracted PDF note.",
                        target="document-canvas",
                        value=note,
                    ).asdict()
                },
            ),
        ]
    if task.intent == "email_send_attachment":
        staging_dir = artifact_root / "live_fire_inputs" / run_id
        source_path = str((staging_dir / "invoice.pdf").resolve())
        attachment_name = Path(source_path).name
        recipient = "Alex"
        return [
            UiAction(
                "launch_app",
                launch_target_for_family("file_explorer"),
                value=launch_target_for_family("file_explorer"),
                metadata={
                    "create_artifact_path": source_path,
                    "artifact_content": "Invoice PDF fixture",
                },
            ),
            UiAction(
                "launch_app",
                launch_target_for_family("email"),
                value=launch_target_for_family("email"),
            ),
            UiAction(
                "set_text",
                "email-to-field",
                recipient,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected=(
                            "The email recipient field contains the requested "
                            "recipient."
                        ),
                        target="email-to-field",
                        value=recipient,
                    ).asdict()
                },
            ),
            UiAction(
                "set_text",
                "email-attachment-field",
                source_path,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected=(
                            "The email attachment field contains the selected file."
                        ),
                        target="email-attachment-field",
                        value=source_path,
                    ).asdict()
                },
            ),
            UiAction(
                "click",
                "email-send-button",
                f"Send email to {recipient}",
                metadata={
                    "recipient": recipient,
                    "attachment": attachment_name,
                    "verification_contract": VerificationContract(
                        kind="send_outcome",
                        expected=(
                            "The final email state proves the message was sent "
                            "with the requested attachment."
                        ),
                        target="email-status-text",
                        value=f"Sent email to {recipient}",
                        metadata={
                            "recipient": recipient,
                            "attachment": attachment_name,
                            "expected_state": "sent",
                        },
                    ).asdict(),
                },
            ),
        ]
    if task.intent == "calendar_invite_from_email":
        query = "Zoom invite"
        return [
            UiAction(
                "launch_app",
                launch_target_for_family("email"),
                value=launch_target_for_family("email"),
            ),
            UiAction(
                "set_text",
                "email-search-box",
                query,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected=(
                            "The email search box contains the requested invite query."
                        ),
                        target="email-search-box",
                        value=query,
                    ).asdict()
                },
            ),
            UiAction(
                "launch_app",
                launch_target_for_family("calendar"),
                value=launch_target_for_family("calendar"),
            ),
            UiAction(
                "set_text",
                "calendar-event-editor",
                query,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected=(
                            "The calendar event editor contains the invite title."
                        ),
                        target="calendar-event-editor",
                        value=query,
                    ).asdict()
                },
            ),
            UiAction(
                "click",
                "calendar-invite-button",
                f"Create invite for {query}",
                metadata={
                    "event_title": query,
                    "verification_contract": VerificationContract(
                        kind="invite_outcome",
                        expected=(
                            "The final calendar state proves the invite was "
                            "created for the requested event."
                        ),
                        target="calendar-details",
                        value=f"Invite created for {query}",
                        metadata={
                            "event_title": query,
                            "expected_state": "invited",
                        },
                    ).asdict(),
                },
            ),
        ]
    if task.intent == "settings_toggle_night_light":
        setting_name = "Night Light"
        return [
            UiAction(
                "launch_app",
                launch_target_for_family("settings"),
                value=launch_target_for_family("settings"),
            ),
            UiAction(
                "set_text",
                "settings-search-box",
                setting_name,
                metadata={
                    "verification_contract": VerificationContract(
                        kind="field_contains",
                        expected=(
                            "The Settings search box contains the requested "
                            "setting name."
                        ),
                        target="settings-search-box",
                        value=setting_name,
                    ).asdict()
                },
            ),
            UiAction(
                "click",
                "settings-toggle",
                f"{setting_name}:on",
                metadata={
                    "setting_name": setting_name,
                    "setting_state": "on",
                    "verification_contract": VerificationContract(
                        kind="toggle_state",
                        expected=(
                            "The final Settings state proves the requested toggle "
                            "is on."
                        ),
                        target="settings-status-text",
                        value=f"{setting_name} is on",
                        metadata={
                            "setting_name": setting_name,
                            "expected_state": "on",
                        },
                    ).asdict(),
                },
            ),
        ]
    return None


def _tool_action(task: EvalTask) -> UiAction:
    return _contracted_action("tool", "tool_executor:live_fire_eval", task)


def _approval_action(task: EvalTask) -> UiAction:
    del task
    return UiAction("delete_file", "protected-live-fire-target")


def _explore_action(task: EvalTask) -> UiAction:
    return _contracted_action("explore", "live-fire-reground", task)


def _modal_action(task: EvalTask) -> UiAction:
    del task
    return UiAction(
        "hotkey",
        "app-window",
        "{ESC}",
        metadata={
            "verification_contract": VerificationContract(
                kind="modal_closed",
                expected="Modal state is closed or remains non-modal.",
                required=True,
            ).asdict()
        },
    )


def _shortcut_action(task: EvalTask) -> UiAction:
    return _receipt_action("hotkey", "app-window", "%{TAB}", task)


def _valid_field_action(task: EvalTask) -> UiAction:
    return _field_action(task, value="valid-agentos-value")


def _default_click_action(task: EvalTask) -> UiAction:
    return _contracted_action("click", _surface_selector(task.surface), task)


def _contracted_action(
    action_type: str,
    selector: str,
    task: EvalTask,
    value: str | None = None,
) -> UiAction:
    kind = (
        task.expected_verifications[0]
        if task.expected_verifications
        else "state_changed"
    )
    return UiAction(
        action_type,
        selector,
        value,
        metadata={
            "verification_contract": VerificationContract(
                kind=kind,
                expected=f"{task.intent} verifies with {kind}.",
                target=selector,
                value=value or "",
                required=task.intent != "stale_target_reground",
            ).asdict()
        },
    )


def _field_action(
    task: EvalTask,
    value: str = "AgentOS live-fire value",
) -> UiAction:
    selector = _surface_selector(task.surface)
    return UiAction(
        "set_text",
        selector,
        value,
        metadata={
            "verification_contract": VerificationContract(
                kind="field_contains",
                expected="The target field contains the live-fire value.",
                target=selector,
                value=value,
            ).asdict()
        },
    )


def _file_action(task: EvalTask, artifact_root: Path, run_id: str) -> UiAction:
    path = artifact_root / run_id / f"{task.task_id}.txt"
    selector = _surface_selector(task.surface)
    return UiAction(
        "set_text",
        selector,
        str(path),
        metadata={
            "create_artifact_path": str(path),
            "artifact_content": f"{task.task_id}\n{task.objective}\n",
            "path": str(path),
            "verification_contract": VerificationContract(
                kind="file_exists",
                expected=f"Live-fire artifact exists at {path}.",
                target=selector,
                path=str(path),
            ).asdict(),
        },
    )


def _receipt_action(
    action_type: str,
    selector: str,
    value: str,
    task: EvalTask,
) -> UiAction:
    return UiAction(
        action_type,
        selector,
        value,
        metadata={
            "verification_contract": VerificationContract(
                kind="receipt_success",
                expected=f"{task.intent} backend receipt reports progress.",
                target=selector,
            ).asdict()
        },
    )


def _surface_selector(surface: str) -> str:
    return primary_selector_for_family(surface)


def _launch_target(surface: str) -> str:
    return launch_target_for_family(surface)


def _file_intent(intent: str) -> bool:
    return intent in {
        "save_file",
        "attach_file",
        "export_artifact",
        "path_validation",
    }


def _element_from_node(node: UiNode) -> UIElementState:
    element_type = _element_type(node)
    return UIElementState(
        element_type=element_type,
        region=_region(node),
        relative_x=0.5,
        relative_y=0.5,
        is_interactive=node.enabled and element_type != "panel",
        semantic_label=_semantic_label(node),
    )


def _app_context(surface: str) -> str:
    return app_context_for_family(surface)


def _layout_mode(nodes: list[UiNode]) -> str:
    return "modal_open" if _active_modal(nodes) else "full"


def _active_modal(nodes: list[UiNode]) -> str:
    for node in nodes:
        if "save as" in node.name.lower() or "dialog" in node.role.lower():
            return node.name
    return ""


def _focus_region(nodes: list[UiNode]) -> str:
    if any(node.focused and _region(node) == "modal" for node in nodes):
        return "modal"
    return "main"


def _element_type(node: UiNode) -> str:
    role = node.role.lower()
    if "button" in role:
        return "button"
    if "edit" in role:
        return "text_field"
    if "table" in role or "grid" in role:
        return "text_field"
    if "tab" in role:
        return "tab"
    if "list" in role:
        return "panel"
    if "canvas" in role or "image" in role:
        return "image"
    if "document" in role:
        return "text_field"
    return "panel"


def _region(node: UiNode) -> str:
    text = f"{node.role} {node.name}".lower()
    if "save" in text or "dialog" in text:
        return "modal"
    if "address" in text or "tab" in text:
        return "header"
    return "main"


def _semantic_label(node: UiNode) -> str:
    metadata = node.metadata or {}
    chunks = [node.name, str(metadata.get("text") or "")]
    chunks.append(str(metadata.get("value") or ""))
    chunks.append(str(metadata.get("semantic_name") or ""))
    chunks.extend(str(item) for item in list(metadata.get("selector_aliases") or []))
    return " ".join(chunk for chunk in chunks if chunk)
