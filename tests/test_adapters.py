from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.os_control import (
    UiNode,
)
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.sandbox import SandboxManager, SandboxSpec


class AdapterTests(unittest.TestCase):
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

    def test_sandbox_manager_agent_body_is_opt_in(self) -> None:
        manager = SandboxManager()
        result = manager.execute(
            SandboxSpec(provider="agent-body", image="research-vm"),
            [],
        )
        self.assertTrue(result.dry_run)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("not enabled or available", result.stdout)

    def test_sandbox_manager_agent_body_executes_control_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SandboxManager()
            root = Path(__file__).resolve().parents[1]
            result = manager.execute(
                SandboxSpec(
                    provider="agent-body",
                    image="research-vm",
                    metadata={
                        "agent_body_manifest": str(
                            root / "crates" / "agent_body" / "Cargo.toml"
                        ),
                        "state_path": str(
                            Path(temp_dir) / "agent_body_state.json"
                        ),
                        "control_request": {
                            "kind": "act",
                            "action_type": "launch_app",
                            "selector": "code",
                            "value": "code",
                        },
                    },
                ),
                [],
            )
            self.assertFalse(result.dry_run)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('"status":"launched"', result.stdout)


if __name__ == "__main__":
    unittest.main()
