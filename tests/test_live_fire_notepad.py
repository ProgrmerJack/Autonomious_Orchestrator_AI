from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from agentos_orchestrator.cli import main
from agentos_orchestrator.os_control.base import UiAction, UiNode
from agentos_orchestrator.os_control.live_fire import (
    NotepadLiveFireConfig,
    NotepadLiveFireTrial,
    _save_as_dialog_observation,
)


class FakeNotepadBackend:
    name = "fake-notepad"

    def __init__(
        self,
        workspace_root: Path,
        show_save_as: bool = True,
    ) -> None:
        self.workspace_root = workspace_root
        self.show_save_as = show_save_as
        self.actions: list[UiAction] = []
        self.payload = ""
        self.pending_path = ""
        self.notepad_open = False
        self.save_as_open = False
        self.snapshot_count = 0

    def snapshot(self) -> list[UiNode]:
        self.snapshot_count += 1
        nodes: list[UiNode] = []
        if self.notepad_open:
            nodes.extend(
                [
                    UiNode("notepad-window", "Window", "Untitled - Notepad"),
                    UiNode("notepad-editor", "Edit", "Text Editor"),
                ]
            )
        if self.save_as_open:
            nodes.extend(
                [
                    UiNode("save-as-window", "Window", "Save As"),
                    UiNode("file-name-edit", "Edit", "File name:"),
                    UiNode("save-button", "Button", "Save"),
                    UiNode("cancel-button", "Button", "Cancel"),
                ]
            )
        return nodes

    def perform(self, action: UiAction) -> str:
        self.actions.append(action)
        handler = getattr(self, f"_perform_{action.action_type}", None)
        if handler is None:
            payload = {"status": "ignored", "action": action.action_type}
        else:
            payload = handler(action)
        return json.dumps(payload)

    def _perform_launch_app(self, action: UiAction) -> dict[str, Any]:
        self.notepad_open = True
        return {"status": "launched", "launched": action.value}

    def _perform_type(self, action: UiAction) -> dict[str, Any]:
        if action.selector in {"role=Edit", "name=Text editor"}:
            self.payload = action.value or ""
            return {"status": "typed"}
        return {"status": "ignored", "selector": action.selector}

    def _perform_hotkey(self, action: UiAction) -> dict[str, Any]:
        if action.value == "^n":
            return {"status": "new-document"}
        if action.value == "^s":
            self.save_as_open = self.show_save_as
        return {"status": "hotkey-sent", "value": action.value}

    def _perform_set_text(self, action: UiAction) -> dict[str, Any]:
        if action.selector in {
            "name=File name",
            "automation_id=1001&&class_name=Edit",
        }:
            self.pending_path = action.value or ""
            return {"status": "value-set", "value": action.value}
        return {"status": "ignored", "selector": action.selector}

    def _perform_invoke(self, action: UiAction) -> dict[str, Any]:
        if (
            action.selector
            in {"name=Save", "automation_id=1&&class_name=Button&&name=Save"}
            and self.pending_path
        ):
            target_path = Path(self.pending_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(self.payload.encode("utf-8"))
            self.save_as_open = False
            return {"status": "invoked", "selector": action.selector}
        return {"status": "ignored", "selector": action.selector}


class LiveFireNotepadTests(unittest.TestCase):
    def test_cli_live_fire_requests_approval_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {
                            "actions": ["os.act"],
                            "paths": [],
                            "network_hosts": [],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": ["os.act"]},
                    }
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--policy",
                        str(policy_path),
                        "--state",
                        str(root / "state.sqlite3"),
                        "--memory",
                        str(root / "memory.sqlite3"),
                        "pc-live-fire-notepad",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertTrue(payload["requires_approval"])
            self.assertEqual(payload["approval"]["status"], "pending")

    def test_notepad_live_fire_happy_path_hash_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakeNotepadBackend(workspace_root)
            trial = NotepadLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                NotepadLiveFireConfig(
                    payload="AgentOS deterministic payload\n",
                    file_name="notepad_happy.txt",
                    poll_interval_seconds=0.001,
                    stable_snapshot_count=2,
                )
            )

            self.assertTrue(result.success, msg=result.error)
            self.assertEqual(result.actual_sha256, result.expected_sha256)
            self.assertTrue(Path(result.target_path).exists())
            self.assertEqual(
                Path(result.target_path).read_text(encoding="utf-8"),
                "AgentOS deterministic payload\n",
            )
            self.assertGreaterEqual(backend.snapshot_count, 4)
            self.assertEqual(result.safety_reason, "allowed")
            self.assertTrue(
                any(item["label"] == "save_as_dialog" for item in result.observations)
            )
            self.assertTrue(
                any(
                    item["value"] == "^n"
                    for item in result.receipts
                    if item["action_type"] == "hotkey"
                )
            )

    def test_notepad_live_fire_blocks_path_escape_before_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakeNotepadBackend(workspace_root)
            trial = NotepadLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                NotepadLiveFireConfig(
                    file_name="../escaped.txt",
                    poll_interval_seconds=0.001,
                )
            )

            self.assertFalse(result.success)
            self.assertIn("outside allowed roots", result.error)
            self.assertEqual(backend.actions, [])
            escaped_path = workspace_root / "artifacts" / "escaped.txt"
            self.assertFalse(escaped_path.exists())

    def test_notepad_live_fire_times_out_waiting_for_save_as(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakeNotepadBackend(workspace_root, show_save_as=False)
            trial = NotepadLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                NotepadLiveFireConfig(
                    file_name="timeout.txt",
                    dialog_timeout_seconds=0.002,
                    poll_interval_seconds=0.001,
                    stable_snapshot_count=1,
                )
            )

            self.assertFalse(result.success)
            self.assertIn("waiting for Save As dialog", result.error)
            self.assertFalse(Path(result.target_path).exists())

    def test_save_as_observer_requires_required_controls(self) -> None:
        incomplete = _save_as_dialog_observation(
            [UiNode("save-as-window", "Window", "Save As")]
        )
        complete = _save_as_dialog_observation(
            [
                UiNode("save-as-window", "Window", "Save As"),
                UiNode("file-name-edit", "Edit", "File name:"),
                UiNode("save-button", "Button", "Save"),
            ]
        )

        self.assertFalse(incomplete.matched)
        self.assertTrue(complete.matched)

    def test_save_as_observer_accepts_windows_11_pane_controls(self) -> None:
        observation = _save_as_dialog_observation(
            [
                UiNode("save-as-window", "Window", "Save as"),
                UiNode(
                    "file-name-edit",
                    "Pane",
                    "AgentOS.txt",
                    metadata={"automation_id": "1001", "class_name": "Edit"},
                ),
                UiNode(
                    "save-button",
                    "Pane",
                    "Save",
                    metadata={"automation_id": "1", "class_name": "Button"},
                ),
            ]
        )

        self.assertTrue(observation.matched)


if __name__ == "__main__":
    unittest.main()
