"""Live-proof scenario: mixed browser + explorer + editor handoff.

Tests a single multi-segment objective that exercises:
  1. Browser leg       — open_url receipt is present.
  2. Explorer file-ops — copy_file receipt with source/destination via
                         FileOpsIntentAdapter sandbox-receipt metadata.
  3. Spreadsheet edit  — cell_edit receipt with cell/value via
                         SpreadsheetCellEditIntentAdapter.
  4. Clarification loop — a vague sub-task that forces requires_clarification.

Uses VirtualDesktopSandboxBackend throughout (no live OS calls).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.gateway import ChannelMessage, GatewayCommandRouter
from agentos_orchestrator.os_control import VirtualDesktopSandboxBackend

from tests.gateway_test_support import new_orchestrator


class MixedHandoffScenarioTests(unittest.TestCase):
    """Browser → file-ops → spreadsheet multi-app handoff scenarios."""

    def _desktop_orchestrator(self, root: Path):
        orchestrator = new_orchestrator(
            root,
            {
                "default": "deny",
                "allow": {
                    "actions": ["os.act", "file.write"],
                    "paths": ["runs/**", "artifacts/**"],
                    "network_hosts": [],
                },
                "forbid": {"actions": [], "paths": []},
                "require_approval": {"actions": []},
            },
        )
        orchestrator.worker.pc_backend = VirtualDesktopSandboxBackend(
            root / "virtual_desktop_sandbox.json"
        )
        return orchestrator

    # ------------------------------------------------------------------
    # Scenario 1: browser search + file copy + spreadsheet cell update
    # ------------------------------------------------------------------

    def test_browser_then_copy_then_cell_edit(self) -> None:
        """Full three-leg handoff: browser → copy → cell-edit."""
        objective = (
            "/pc search for quarterly results in browser "
            "then copy report.xlsx from Downloads to Documents "
            "then set B2 to 42"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(ChannelMessage("telegram", "99", objective))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(response.status, "completed")
        receipts = response.payload.get("receipts", [])

        # --- Browser leg ---
        url_receipts = [r for r in receipts if r["action_type"] == "open_url"]
        self.assertTrue(
            url_receipts,
            "Expected at least one open_url receipt for the browser leg.",
        )
        url_value = url_receipts[0]["receipt"]
        if isinstance(url_value, dict):
            nav_url = url_value.get("value", "")
        else:
            nav_url = str(url_value)
        self.assertIn(
            "quarterly",
            nav_url.lower() + "".join(str(r) for r in receipts),
            "Browser URL should reference the search query.",
        )

        # --- File-copy leg (FileOpsIntentAdapter) ---
        copy_receipts = [r for r in receipts if r["action_type"] == "copy_file"]
        self.assertTrue(
            copy_receipts,
            "Expected at least one copy_file receipt for the explorer leg.",
        )
        copy_receipt_payload = copy_receipts[0]["receipt"]
        if isinstance(copy_receipt_payload, dict):
            file_op = copy_receipt_payload.get("file_op", {})
            self.assertEqual(file_op.get("operation"), "copy")
            self.assertIn(
                "report.xlsx",
                str(file_op.get("source", "")),
                "file_op.source must contain the source filename.",
            )

        # --- Spreadsheet cell-edit leg (SpreadsheetCellEditIntentAdapter) ---
        cell_edit_receipts = [r for r in receipts if r["action_type"] == "cell_edit"]
        self.assertTrue(
            cell_edit_receipts,
            "Expected at least one cell_edit receipt for the spreadsheet leg.",
        )
        cell_payload = cell_edit_receipts[0]["receipt"]
        if isinstance(cell_payload, dict):
            cell_data = cell_payload.get("cell_edit", {})
            self.assertEqual(
                cell_data.get("cell"),
                "B2",
                "cell_edit.cell should be B2.",
            )
            self.assertEqual(
                str(cell_data.get("value")),
                "42",
                "cell_edit.value should be 42.",
            )

    # ------------------------------------------------------------------
    # Scenario 2: vague objective triggers clarification loop
    # ------------------------------------------------------------------

    def test_vague_mixed_command_triggers_clarification(self) -> None:
        """A vague /pc command must return clarification_required status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            response = router.handle(
                ChannelMessage("telegram", "99", "/pc do the stuff")
            )

        self.assertEqual(response.status, "clarification_required")
        plan = response.payload.get("plan", {})
        self.assertTrue(
            plan.get("requires_clarification"),
            "Plan must have requires_clarification=True for vague input.",
        )
        self.assertTrue(
            plan.get("clarification_questions"),
            "Plan must include at least one clarification question.",
        )

    # ------------------------------------------------------------------
    # Scenario 3: browser + explorer rename handoff
    # ------------------------------------------------------------------

    def test_browser_then_rename_handoff(self) -> None:
        """Browser search followed by a rename file operation."""
        objective = (
            "/pc search for project status online "
            "then rename draft.docx to final_report.docx"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(ChannelMessage("telegram", "7", objective))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(response.status, "completed")
        receipts = response.payload.get("receipts", [])

        # Browser leg present
        self.assertTrue(
            any(r["action_type"] == "open_url" for r in receipts),
            "Expected open_url receipt for browser leg.",
        )

        # Rename file-op leg present
        rename_receipts = [r for r in receipts if r["action_type"] == "rename_file"]
        self.assertTrue(
            rename_receipts,
            "Expected rename_file receipt for the rename leg.",
        )
        rename_payload = rename_receipts[0]["receipt"]
        if isinstance(rename_payload, dict):
            file_op = rename_payload.get("file_op", {})
            self.assertEqual(file_op.get("operation"), "rename")
            self.assertIn(
                "draft.docx",
                str(file_op.get("source", "")),
                "file_op.source must reference the original filename.",
            )
            self.assertIn(
                "final_report.docx",
                str(file_op.get("new_name", "")),
                "file_op.new_name must reference the new filename.",
            )

    # ------------------------------------------------------------------
    # Scenario 3b: explicit clarification loop, then browser+explorer+editor
    # ------------------------------------------------------------------

    def test_clarification_then_browser_explorer_editor_handoff(self) -> None:
        """Clarification loop followed by browser+explorer+editor execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))

            clarification = router.handle(ChannelMessage("telegram", "55", "/pc do it"))
            self.assertEqual(clarification.status, "clarification_required")
            self.assertTrue(
                clarification.payload.get("plan", {}).get("clarification_questions")
            )

            objective = (
                "/pc search for desktop automation safety in browser "
                "then move notes.txt to archive/notes.txt "
                "then draft a python script in vscode"
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(ChannelMessage("telegram", "55", objective))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(response.status, "completed")
        receipts = response.payload.get("receipts", [])

        self.assertTrue(
            any(r["action_type"] == "open_url" for r in receipts),
            "Expected open_url receipt for browser leg.",
        )

        move_receipts = [r for r in receipts if r["action_type"] == "move_file"]
        self.assertTrue(
            move_receipts,
            "Expected move_file receipt for explorer leg.",
        )
        move_payload = move_receipts[0]["receipt"]
        if isinstance(move_payload, dict):
            file_op = move_payload.get("file_op", {})
            self.assertEqual(file_op.get("operation"), "move")
            self.assertIn("notes.txt", str(file_op.get("source", "")))

        self.assertTrue(
            any(
                r["action_type"] == "type" and r["selector"] == "editor-canvas"
                for r in receipts
            ),
            "Expected editor-canvas type action for editor leg.",
        )

    def test_open_paint_and_draw_generic_scene_command(self) -> None:
        """/pc drawing task should launch Paint and execute drawing strokes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = GatewayCommandRouter(self._desktop_orchestrator(root))
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                response = router.handle(
                    ChannelMessage(
                        "telegram",
                        "88",
                        "/pc open paint app and draw a city skyline",
                    )
                )
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(response.status, "completed")
        receipts = response.payload.get("receipts", [])
        launch_receipts = [r for r in receipts if r["action_type"] == "launch_app"]
        self.assertTrue(launch_receipts)
        self.assertTrue(
            any(
                isinstance(r.get("receipt"), dict)
                and r["receipt"].get("launched") == "mspaint.exe"
                for r in launch_receipts
            ),
            "Expected launch_app receipt for mspaint.exe.",
        )
        draw_receipts = [r for r in receipts if r["action_type"] == "draw_path"]
        self.assertGreaterEqual(
            len(draw_receipts),
            1,
            "Expected at least one drawing stroke action.",
        )

    # ------------------------------------------------------------------
    # Scenario 4: sandbox receipts carry structured metadata
    # ------------------------------------------------------------------

    def test_file_ops_intent_adapter_sandbox_receipt_metadata(self) -> None:
        """FileOpsIntentAdapter step metadata must include sandbox_receipt."""
        from agentos_orchestrator.os_control.workflow.adapters import (
            FileOpsIntentAdapter,
            WorkflowContext,
        )

        adapter = FileOpsIntentAdapter()
        ctx = WorkflowContext(
            objective="copy notes.txt to archive/notes_backup.txt",
            lower="copy notes.txt to archive/notes_backup.txt",
            mode="app-task",
            app_target=None,
            artifact_path=None,
        )
        steps = adapter.steps_for(ctx)

        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertEqual(step.action_type, "copy_file")
        receipt = step.metadata.get("sandbox_receipt", {})
        self.assertEqual(receipt["operation"], "copy")
        self.assertEqual(receipt["source"], "notes.txt")
        self.assertEqual(receipt["destination"], "archive/notes_backup.txt")
        self.assertIn("timestamp", receipt)
        self.assertEqual(receipt["status"], "pending")

    def test_spreadsheet_cell_edit_intent_adapter_sandbox_receipt(self) -> None:
        """SpreadsheetCellEditIntentAdapter must carry typed receipt metadata."""
        from agentos_orchestrator.os_control.workflow.adapters import (
            SpreadsheetCellEditIntentAdapter,
            WorkflowContext,
        )

        adapter = SpreadsheetCellEditIntentAdapter()
        ctx = WorkflowContext(
            objective="update B2 to 99",
            lower="update b2 to 99",
            mode="app-task",
            app_target=None,
            artifact_path=None,
        )
        steps = adapter.steps_for(ctx)

        cell_edit_steps = [s for s in steps if s.action_type == "cell_edit"]
        self.assertEqual(len(cell_edit_steps), 1)
        step = cell_edit_steps[0]
        receipt = step.metadata.get("sandbox_receipt", {})
        self.assertEqual(receipt["operation"], "cell_edit")
        self.assertEqual(receipt["cell"], "B2")
        self.assertEqual(receipt["value"], "99")
        self.assertFalse(receipt["formula"])
        self.assertFalse(receipt["range_edit"])

    def test_spreadsheet_formula_detection(self) -> None:
        """SpreadsheetCellEditIntentAdapter detects formula values."""
        from agentos_orchestrator.os_control.workflow.adapters import (
            SpreadsheetCellEditIntentAdapter,
            WorkflowContext,
        )

        adapter = SpreadsheetCellEditIntentAdapter()
        ctx = WorkflowContext(
            objective="set C3 to =SUM(A1:B1)",
            lower="set c3 to =sum(a1:b1)",
            mode="app-task",
            app_target=None,
            artifact_path=None,
        )
        steps = adapter.steps_for(ctx)

        cell_edit_steps = [s for s in steps if s.action_type == "cell_edit"]
        self.assertTrue(cell_edit_steps)
        receipt = cell_edit_steps[0].metadata.get("sandbox_receipt", {})
        self.assertTrue(receipt["formula"], "Formula flag should be True.")


if __name__ == "__main__":
    unittest.main()
