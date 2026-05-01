from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentos_orchestrator.cognition.safety_gates import (
    FormalSafetyVerifier,
    SafetyPolicy,
)
from agentos_orchestrator.os_control.base import UiAction, UiNode


DEFAULT_NOTEPAD_PAYLOAD = "AgentOS live-fire Notepad smoke test."
DEFAULT_NOTEPAD_FILE_NAME = "notepad_smoke.txt"


@dataclass(slots=True)
class PollObservation:
    matched: bool
    reason: str
    node_count: int
    matched_nodes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NotepadLiveFireConfig:
    payload: str = DEFAULT_NOTEPAD_PAYLOAD
    file_name: str = DEFAULT_NOTEPAD_FILE_NAME
    dialog_timeout_seconds: float = 12.0
    file_timeout_seconds: float = 6.0
    poll_interval_seconds: float = 0.25
    stable_snapshot_count: int = 2
    editor_selector: str = "name=Text editor"
    filename_selector: str = "automation_id=1001&&class_name=Edit"
    save_button_selector: str = "automation_id=1&&class_name=Button&&name=Save"
    new_document_hotkey: str = "^n"


@dataclass(slots=True)
class NotepadLiveFireResult:
    success: bool
    target_path: str
    expected_sha256: str
    actual_sha256: str = ""
    dpi_awareness: str = "not-attempted"
    receipts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    safety_reason: str = ""
    error: str = ""
    elapsed_ms: float = 0.0


class SnapshotPollTimeout(TimeoutError):
    pass


class SnapshotPoller:
    def __init__(
        self,
        backend: Any,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.sleep_fn = sleep_fn

    def until(
        self,
        predicate: Callable[[list[UiNode]], PollObservation],
        timeout_seconds: float,
        interval_seconds: float,
        stable_count: int = 1,
    ) -> PollObservation:
        attempts = max(
            1,
            int(timeout_seconds / max(interval_seconds, 0.001)) + 1,
        )
        consecutive_matches = 0
        last_observation = PollObservation(False, "not-started", 0)
        for attempt in range(attempts):
            nodes = self.backend.snapshot()
            last_observation = predicate(nodes)
            if last_observation.matched:
                consecutive_matches += 1
                if consecutive_matches >= max(1, stable_count):
                    return last_observation
            else:
                consecutive_matches = 0
            if attempt < attempts - 1:
                self.sleep_fn(interval_seconds)
        raise SnapshotPollTimeout(last_observation.reason)


class NotepadLiveFireTrial:
    """Deterministic live Windows Notepad smoke trial.

    The trial deliberately uses the boring app first: launch Notepad, type a
    known payload, wait for the Save As dialog via UI snapshots, save under
    artifacts/live_fire, and verify the resulting file by SHA-256.
    """

    def __init__(
        self,
        backend: Any,
        workspace_root: str | Path,
        safety_verifier: FormalSafetyVerifier | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.live_fire_root = self.workspace_root / "artifacts" / "live_fire"
        self.safety_verifier = safety_verifier or FormalSafetyVerifier(
            SafetyPolicy(allowed_roots=[self.live_fire_root])
        )
        self.poller = SnapshotPoller(backend, sleep_fn=sleep_fn)

    def run(
        self,
        config: NotepadLiveFireConfig | None = None,
    ) -> NotepadLiveFireResult:
        config = config or _default_notepad_config()
        start = time.perf_counter()
        target_path = self._target_path(config)
        result = _initial_result(target_path, config)
        safety_action = self._target_path_action(config, target_path)
        if not self._verify_target_path(safety_action, result, start):
            return result

        self.live_fire_root.mkdir(parents=True, exist_ok=True)
        result.dpi_awareness = enable_windows_dpi_awareness()

        try:
            self._execute_notepad_save(
                config,
                target_path,
                safety_action,
                result,
            )
        except (
            SnapshotPollTimeout,
            FileNotFoundError,
            RuntimeError,
            OSError,
            ValueError,
        ) as exc:
            result.error = str(exc)

        result.elapsed_ms = _elapsed_ms(start)
        return result

    def _target_path(self, config: NotepadLiveFireConfig) -> Path:
        return (self.live_fire_root / config.file_name).resolve(strict=False)

    def _target_path_action(
        self,
        config: NotepadLiveFireConfig,
        target_path: Path,
    ) -> UiAction:
        return UiAction(
            action_type="set_text",
            selector=config.filename_selector,
            value=str(target_path),
            metadata={"target_path": str(target_path)},
        )

    def _verify_target_path(
        self,
        action: UiAction,
        result: NotepadLiveFireResult,
        start: float,
    ) -> bool:
        safety = self.safety_verifier.verify_action(
            action,
            objective="notepad live-fire save target",
        )
        result.safety_reason = safety.reason
        if safety.allowed:
            return True
        result.error = f"Safety gate blocked target path: {safety.reason}"
        result.elapsed_ms = _elapsed_ms(start)
        return False

    def _execute_notepad_save(
        self,
        config: NotepadLiveFireConfig,
        target_path: Path,
        safety_action: UiAction,
        result: NotepadLiveFireResult,
    ) -> None:
        self._perform(
            UiAction("launch_app", "notepad.exe", "notepad.exe"),
            result,
        )
        self._poll(
            "notepad_window",
            _notepad_window_observation,
            config,
            result,
        )
        self._perform(
            UiAction("hotkey", "app-window", config.new_document_hotkey),
            result,
        )
        self._poll(
            "notepad_window_after_new",
            _notepad_window_observation,
            config,
            result,
        )
        self._perform(
            UiAction("type", config.editor_selector, config.payload),
            result,
        )
        self._perform(UiAction("hotkey", "app-window", "^s"), result)
        self._poll(
            "save_as_dialog",
            _save_as_dialog_observation,
            config,
            result,
        )
        self._perform(safety_action, result)
        self._perform(UiAction("invoke", config.save_button_selector), result)
        result.actual_sha256 = self._wait_for_hash(target_path, config)
        result.success = result.actual_sha256 == result.expected_sha256
        if not result.success:
            result.error = "Saved file hash did not match expected payload"

    def _perform(
        self,
        action: UiAction,
        result: NotepadLiveFireResult,
    ) -> None:
        receipt = self.backend.perform(action)
        result.receipts.append(
            {
                "action_type": action.action_type,
                "selector": action.selector,
                "value": action.value,
                "metadata": dict(action.metadata),
                "receipt": _json_or_text(receipt),
            }
        )

    def _poll(
        self,
        label: str,
        predicate: Callable[[list[UiNode]], PollObservation],
        config: NotepadLiveFireConfig,
        result: NotepadLiveFireResult,
    ) -> None:
        observation = self.poller.until(
            predicate,
            timeout_seconds=config.dialog_timeout_seconds,
            interval_seconds=config.poll_interval_seconds,
            stable_count=config.stable_snapshot_count,
        )
        result.observations.append(
            {
                "label": label,
                "matched": observation.matched,
                "reason": observation.reason,
                "node_count": observation.node_count,
                "matched_nodes": observation.matched_nodes,
            }
        )

    def _wait_for_hash(
        self,
        target_path: Path,
        config: NotepadLiveFireConfig,
    ) -> str:
        attempts = max(
            1,
            int(config.file_timeout_seconds / max(config.poll_interval_seconds, 0.001))
            + 1,
        )
        for attempt in range(attempts):
            if target_path.exists():
                return _sha256_bytes(target_path.read_bytes())
            if attempt < attempts - 1:
                self.poller.sleep_fn(config.poll_interval_seconds)
        raise FileNotFoundError(f"Notepad output file was not created: {target_path}")


def enable_windows_dpi_awareness() -> str:
    if platform.system() != "Windows":
        return "not-windows"
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return "per-monitor"
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
            return "system"
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        return f"failed:{exc}"


def _notepad_window_observation(nodes: list[UiNode]) -> PollObservation:
    matches = [node for node in nodes if "notepad" in _node_text(node)]
    return PollObservation(
        matched=bool(matches),
        reason=(
            "notepad window observed" if matches else "notepad window not observed"
        ),
        node_count=len(nodes),
        matched_nodes=[_node_label(node) for node in matches[:5]],
    )


def _save_as_dialog_observation(nodes: list[UiNode]) -> PollObservation:
    dialog_nodes = _matching_nodes(nodes, _is_save_as_dialog)
    edit_nodes = _matching_nodes(nodes, _is_file_name_edit)
    save_buttons = _matching_nodes(nodes, _is_save_button)
    matched = bool(dialog_nodes and edit_nodes and save_buttons)
    reason = _save_as_reason(
        matched,
        dialog_count=len(dialog_nodes),
        edit_count=len(edit_nodes),
        save_button_count=len(save_buttons),
    )
    return PollObservation(
        matched=matched,
        reason=reason,
        node_count=len(nodes),
        matched_nodes=[
            *[_node_label(node) for node in dialog_nodes[:3]],
            *[_node_label(node) for node in edit_nodes[:3]],
            *[_node_label(node) for node in save_buttons[:3]],
        ],
    )


def _matching_nodes(
    nodes: list[UiNode],
    predicate: Callable[[UiNode], bool],
) -> list[UiNode]:
    return [node for node in nodes if predicate(node)]


def _is_save_as_dialog(node: UiNode) -> bool:
    return "save as" in _node_text(node) and _has_role(
        node,
        "window",
        "dialog",
    )


def _is_save_button(node: UiNode) -> bool:
    return "save" in _node_text(node) and (
        _has_role(node, "button") or _metadata_value(node, "class_name") == "button"
    )


def _is_file_name_edit(node: UiNode) -> bool:
    return _has_role(node, "edit") or (
        _metadata_value(node, "automation_id") == "1001"
        and _metadata_value(node, "class_name") == "edit"
    )


def _has_role(node: UiNode, *roles: str) -> bool:
    return node.role.lower() in set(roles)


def _metadata_value(node: UiNode, key: str) -> str:
    return str(node.metadata.get(key, "")).lower()


def _save_as_reason(
    matched: bool,
    dialog_count: int,
    edit_count: int,
    save_button_count: int,
) -> str:
    if matched:
        return "Save As dialog with filename field observed"
    return (
        f"waiting for Save As dialog: dialogs={dialog_count} "
        f"edits={edit_count} save_buttons={save_button_count}"
    )


def _node_text(node: UiNode) -> str:
    metadata_text = " ".join(str(value) for value in node.metadata.values())
    return f"{node.role} {node.name} {node.node_id} {metadata_text}".lower()


def _node_label(node: UiNode) -> str:
    return f"{node.role}:{node.name or node.node_id}"


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _default_notepad_config() -> NotepadLiveFireConfig:
    return NotepadLiveFireConfig(file_name=f"notepad_smoke_{int(time.time())}.txt")


def _initial_result(
    target_path: Path,
    config: NotepadLiveFireConfig,
) -> NotepadLiveFireResult:
    return NotepadLiveFireResult(
        success=False,
        target_path=str(target_path),
        expected_sha256=_sha256_text(config.payload),
    )
