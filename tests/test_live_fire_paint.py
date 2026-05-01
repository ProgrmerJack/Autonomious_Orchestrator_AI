from __future__ import annotations

import binascii
import json
import struct
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from agentos_orchestrator.cli import main
from agentos_orchestrator.os_control.base import UiAction, UiNode
from agentos_orchestrator.os_control.paint_live_fire import (
    PaintLiveFireConfig,
    PaintLiveFireTrial,
)


class FakePaintBackend:
    name = "fake-paint"

    def __init__(
        self,
        workspace_root: Path,
        show_save_as: bool = True,
        draw_changes_pixels: bool = True,
    ) -> None:
        self.workspace_root = workspace_root
        self.show_save_as = show_save_as
        self.draw_changes_pixels = draw_changes_pixels
        self.actions: list[UiAction] = []
        self.pending_path = ""
        self.paint_open = False
        self.save_as_open = False
        self.drawn = False
        self.snapshot_count = 0

    def snapshot(self) -> list[UiNode]:
        self.snapshot_count += 1
        nodes: list[UiNode] = []
        if self.paint_open:
            nodes.extend(
                [
                    UiNode("paint-window", "Window", "Untitled - Paint"),
                    UiNode("paint-canvas", "Pane", "Canvas"),
                ]
            )
        if self.save_as_open:
            nodes.extend(
                [
                    UiNode("save-as-window", "Window", "Save As"),
                    UiNode("file-name-edit", "Edit", "File name:"),
                    UiNode("save-button", "Button", "Save"),
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
        self.paint_open = True
        return {"status": "launched", "launched": action.value}

    def _perform_draw_path(self, action: UiAction) -> dict[str, Any]:
        if action.selector in {
            "name=Paint",
            "automation_id=image&&name=Canvas",
        }:
            self.drawn = True
            return {"status": "drawn"}
        return {"status": "ignored", "selector": action.selector}

    def _perform_hotkey(self, action: UiAction) -> dict[str, Any]:
        if action.value == "^s":
            self.save_as_open = self.show_save_as
        return {"status": "hotkey-sent", "value": action.value}

    def _perform_set_text(self, action: UiAction) -> dict[str, Any]:
        if action.selector == "automation_id=1001&&class_name=Edit":
            self.pending_path = action.value or ""
            return {"status": "value-set", "value": action.value}
        return {"status": "ignored", "selector": action.selector}

    def _perform_invoke(self, action: UiAction) -> dict[str, Any]:
        if (
            action.selector == "automation_id=1&&class_name=Button&&name=Save"
            and self.pending_path
        ):
            target_path = Path(self.pending_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(
                _png_bytes(drawn=self.drawn and self.draw_changes_pixels)
            )
            self.save_as_open = False
            return {"status": "invoked", "selector": action.selector}
        return {"status": "ignored", "selector": action.selector}


class LiveFirePaintTests(unittest.TestCase):
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
                        "pc-live-fire-paint",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertTrue(payload["requires_approval"])
            self.assertEqual(payload["approval"]["status"], "pending")

    def test_paint_live_fire_happy_path_verifies_nonblank_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakePaintBackend(workspace_root)
            trial = PaintLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                PaintLiveFireConfig(
                    file_name="paint_happy.png",
                    poll_interval_seconds=0.001,
                    stable_snapshot_count=2,
                    min_non_background_width=2,
                    min_non_background_height=2,
                )
            )

            self.assertTrue(result.success, msg=result.error)
            self.assertTrue(Path(result.target_path).exists())
            self.assertEqual(result.image_width, 4)
            self.assertEqual(result.image_height, 4)
            self.assertGreater(result.distinct_pixel_count, 1)
            self.assertGreater(result.non_background_pixel_count, 0)
            self.assertEqual(result.non_background_bounds, (1, 1, 2, 2))
            self.assertGreaterEqual(backend.snapshot_count, 4)
            self.assertEqual(result.safety_reason, "allowed")
            self.assertTrue(
                any(item["action_type"] == "draw_path" for item in result.receipts)
            )

    def test_paint_live_fire_blocks_path_escape_before_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakePaintBackend(workspace_root)
            trial = PaintLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                PaintLiveFireConfig(
                    file_name="../escaped.png",
                    poll_interval_seconds=0.001,
                )
            )

            self.assertFalse(result.success)
            self.assertIn("outside allowed roots", result.error)
            self.assertEqual(backend.actions, [])

    def test_paint_live_fire_times_out_waiting_for_save_as(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakePaintBackend(workspace_root, show_save_as=False)
            trial = PaintLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                PaintLiveFireConfig(
                    file_name="timeout.png",
                    dialog_timeout_seconds=0.002,
                    poll_interval_seconds=0.001,
                    stable_snapshot_count=1,
                )
            )

            self.assertFalse(result.success)
            self.assertIn("waiting for Save As dialog", result.error)
            self.assertFalse(Path(result.target_path).exists())

    def test_paint_live_fire_rejects_blank_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            backend = FakePaintBackend(
                workspace_root,
                draw_changes_pixels=False,
            )
            trial = PaintLiveFireTrial(
                backend=backend,
                workspace_root=workspace_root,
                sleep_fn=lambda _seconds: None,
            )

            result = trial.run(
                PaintLiveFireConfig(
                    file_name="blank.png",
                    poll_interval_seconds=0.001,
                    stable_snapshot_count=1,
                )
            )

            self.assertFalse(result.success)
            self.assertEqual(result.non_background_pixel_count, 0)
            self.assertIn("2D stroke footprint", result.error)


def _png_bytes(drawn: bool) -> bytes:
    width = 4
    height = 4
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            if drawn and (x, y) in {(1, 1), (2, 1), (2, 2)}:
                row.extend((0, 0, 0))
            else:
                row.extend((255, 255, 255))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            _chunk(b"IDAT", zlib.compress(raw)),
            _chunk(b"IEND", b""),
        ]
    )


def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    return b"".join(
        [
            struct.pack(">I", len(chunk_data)),
            chunk_type,
            chunk_data,
            struct.pack(
                ">I",
                binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF,
            ),
        ]
    )


if __name__ == "__main__":
    unittest.main()
