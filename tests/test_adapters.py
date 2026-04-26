from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.os_control import (
    DirectShellBackend,
    UiAction,
    UiNode,
)
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.sandbox import SandboxManager, SandboxSpec


class AdapterTests(unittest.TestCase):
    def test_directshell_action_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "directshell.sqlite3"
            backend = DirectShellBackend(db_path)
            action_id = backend.perform(
                UiAction("click", "role=button[name='Submit']")
            )
            self.assertEqual(action_id, "1")
            connection = sqlite3.connect(db_path)
            try:
                row = connection.execute(
                    "SELECT action_type, selector FROM actions"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, ("click", "role=button[name='Submit']"))

    def test_selector_debug_ranks_accessible_candidates(self) -> None:
        report = debug_selector(
            "name=Submit",
            [
                UiNode("1", "Button", "Cancel"),
                UiNode(
                    "2",
                    "Button",
                    "Submit request",
                    metadata={"automation_id": "submitButton"},
                ),
            ],
        )

        self.assertTrue(report.ready)
        self.assertEqual(
            report.candidates[0].selector,
            "automation_id=submitButton",
        )

    def test_sandbox_manager_defaults_to_dry_run(self) -> None:
        manager = SandboxManager()
        result = manager.execute(
            SandboxSpec(provider="dry-run", image="research-vm"),
            ["python", "script.py"],
        )
        self.assertTrue(result.dry_run)
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
