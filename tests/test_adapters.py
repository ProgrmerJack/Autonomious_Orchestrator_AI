from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from agentos_orchestrator.os_control import (
    RustNativeWindowsBackend,
    UiAction,
    UiNode,
    VirtualDesktopSandboxBackend,
    WindowsUiaBackend,
)
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.sandbox import SandboxManager, SandboxSpec
from agentos_orchestrator.sandbox.agent_body_client import (
    AgentBodyClient,
    resolve_agent_body_invocation,
)


class AdapterTests(unittest.TestCase):
    def test_rust_native_backend_converts_snapshot_and_actions(self) -> None:
        class FakeAgentBody:
            def native_snapshot(self) -> dict:
                return {
                    "status": "ok",
                    "nodes": [
                        {
                            "node_id": "native-desktop",
                            "role": "Desktop",
                            "name": "Windows Desktop",
                            "bounds": [0, 0, 1920, 1080],
                            "enabled": True,
                            "focused": True,
                            "metadata": {"native": True},
                        }
                    ],
                }

            def native_act(
                self,
                action_type: str,
                selector: str,
                value: str | None,
                metadata: dict,
            ) -> dict:
                return {
                    "status": "clicked",
                    "action_type": action_type,
                    "selector": selector,
                    "value": value,
                    "metadata": metadata,
                }

        backend = RustNativeWindowsBackend(
            agent_body_client=cast(AgentBodyClient, FakeAgentBody())
        )

        nodes = backend.snapshot()
        self.assertEqual(nodes[0].node_id, "native-desktop")
        self.assertEqual(nodes[0].bounds, (0, 0, 1920, 1080))

        receipt = json.loads(
            backend.perform(UiAction("click", "10,20", metadata={"x": 10, "y": 20}))
        )
        self.assertEqual(receipt["status"], "clicked")
        self.assertEqual(receipt["backend"], "rust-native-windows")

    def test_rust_native_backend_waits_for_launched_surface(self) -> None:
        class FakeAgentBody:
            def __init__(self) -> None:
                self.snapshot_calls = 0

            def native_snapshot(self) -> dict:
                self.snapshot_calls += 1
                nodes = [
                    {
                        "node_id": "native-desktop",
                        "role": "Desktop",
                        "name": "Windows Desktop",
                        "bounds": [0, 0, 1920, 1080],
                        "enabled": True,
                        "focused": True,
                        "metadata": {"native": True},
                    },
                    {
                        "node_id": "native-cursor",
                        "role": "Pointer",
                        "name": "Mouse Cursor",
                        "bounds": [10, 10, 1, 1],
                        "enabled": True,
                        "focused": False,
                        "metadata": {"native": True},
                    },
                ]
                if self.snapshot_calls > 1:
                    nodes.append(
                        {
                            "node_id": "uia-browser-window",
                            "role": "WindowControl",
                            "name": "Browser",
                            "bounds": [100, 100, 1000, 800],
                            "enabled": True,
                            "focused": True,
                            "metadata": {
                                "native": True,
                                "automation_id": "browserWindow",
                            },
                        }
                    )
                return {"status": "ok", "nodes": nodes}

            def native_act(
                self,
                action_type: str,
                selector: str,
                value: str | None,
                metadata: dict,
            ) -> dict:
                return {
                    "status": "launched",
                    "action_type": action_type,
                    "selector": selector,
                    "value": value,
                    "metadata": metadata,
                }

        backend = RustNativeWindowsBackend(
            agent_body_client=cast(AgentBodyClient, FakeAgentBody())
        )

        receipt = json.loads(backend.perform(UiAction("launch_app", "browser")))

        self.assertTrue(receipt["launch_surface_detected"])
        self.assertIn("Browser", receipt["launch_surface_names"])
        self.assertGreaterEqual(receipt["launch_snapshot_nodes"], 3)

    def test_rust_native_backend_resolves_settings_toggle_and_proves_state(
        self,
    ) -> None:
        class FakeAgentBody:
            def __init__(self) -> None:
                self.snapshot_calls = 0
                self.last_action: dict | None = None

            def native_snapshot(self) -> dict:
                self.snapshot_calls += 1
                name = "Night light"
                if self.snapshot_calls > 1:
                    name = "Night light on"
                return {
                    "status": "ok",
                    "nodes": [
                        {
                            "node_id": "settings-night-light",
                            "role": "Button",
                            "name": name,
                            "bounds": [50, 60, 100, 20],
                            "enabled": True,
                            "focused": False,
                            "metadata": {
                                "automation_id": "nightLightToggle",
                            },
                        }
                    ],
                }

            def native_act(
                self,
                action_type: str,
                selector: str,
                value: str | None,
                metadata: dict,
            ) -> dict:
                self.last_action = {
                    "action_type": action_type,
                    "selector": selector,
                    "value": value,
                    "metadata": metadata,
                }
                return {"status": "clicked"}

        agent_body = FakeAgentBody()
        backend = RustNativeWindowsBackend(
            agent_body_client=cast(AgentBodyClient, agent_body)
        )

        receipt = json.loads(
            backend.perform(
                UiAction(
                    "click",
                    "settings-toggle",
                    "Night Light:on",
                    metadata={
                        "adapter_family": "settings",
                        "setting_name": "Night Light",
                        "setting_state": "on",
                    },
                )
            )
        )

        assert agent_body.last_action is not None
        self.assertEqual(
            agent_body.last_action["selector"],
            "name=Night light",
        )
        self.assertEqual(agent_body.last_action["metadata"]["x"], 100)
        self.assertEqual(agent_body.last_action["metadata"]["y"], 70)
        self.assertEqual(receipt["setting"]["name"], "Night Light")
        self.assertEqual(receipt["setting"]["state"], "on")

    def test_windows_uia_delegates_coordinate_action_to_rust_native(
        self,
    ) -> None:
        class ForcedUnavailableWindowsBackend(WindowsUiaBackend):
            def available(self) -> bool:
                return False

        class FakeNativeBackend:
            def available(self) -> bool:
                return True

            def perform(self, action: UiAction) -> str:
                return json.dumps(
                    {
                        "status": "clicked",
                        "action_type": action.action_type,
                        "selector": action.selector,
                        "metadata": action.metadata,
                    }
                )

        backend = ForcedUnavailableWindowsBackend(
            native_fallback=cast(RustNativeWindowsBackend, FakeNativeBackend())
        )
        receipt = json.loads(
            backend.perform(UiAction("click", "10,20", metadata={"x": 10, "y": 20}))
        )

        self.assertEqual(receipt["status"], "clicked")
        self.assertEqual(receipt["via"], "rust-native-windows")
        self.assertIn("uia_fallback_reason", receipt["metadata"])

    def test_windows_uia_backend_executes_real_file_operations(self) -> None:
        class FileOpWindowsBackend(WindowsUiaBackend):
            def __init__(self) -> None:
                super().__init__(powershell_path="pwsh")

            def available(self) -> bool:
                return True

            def snapshot(self) -> list[UiNode]:
                return []

        backend = FileOpWindowsBackend()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "report.md"
            copied = root / "archive" / "report_copy.md"
            moved = root / "published" / "report_final.md"
            source.write_text("draft", encoding="utf-8")

            copy_receipt = json.loads(
                backend.perform(
                    UiAction(
                        "copy_file",
                        "explorer-file-list",
                        metadata={
                            "source": str(source),
                            "destination": str(copied),
                        },
                    )
                )
            )
            move_receipt = json.loads(
                backend.perform(
                    UiAction(
                        "move_file",
                        "explorer-file-list",
                        metadata={
                            "source": str(source),
                            "destination": str(moved),
                        },
                    )
                )
            )
            rename_receipt = json.loads(
                backend.perform(
                    UiAction(
                        "rename_file",
                        "explorer-file-list",
                        metadata={
                            "source": str(moved),
                            "new_name": "report_ready.md",
                        },
                    )
                )
            )

            self.assertEqual(copy_receipt["status"], "file-op-executed")
            self.assertTrue(copied.exists())
            self.assertEqual(move_receipt["status"], "file-op-executed")
            self.assertEqual(
                move_receipt["file_op"]["resulting_path"],
                str(moved),
            )
            self.assertFalse(source.exists())
            self.assertEqual(rename_receipt["status"], "file-op-executed")
            self.assertTrue((root / "published" / "report_ready.md").exists())

    def test_rust_native_backend_executes_real_file_operations(self) -> None:
        class FakeAgentBody:
            def native_snapshot(self) -> dict:
                return {"status": "ok", "nodes": []}

            def native_act(
                self,
                action_type: str,
                selector: str,
                value: str | None,
                metadata: dict,
            ) -> dict:
                return {
                    "status": "unexpected-native-act",
                    "action_type": action_type,
                    "selector": selector,
                    "value": value,
                    "metadata": metadata,
                }

        backend = RustNativeWindowsBackend(
            agent_body_client=cast(AgentBodyClient, FakeAgentBody())
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "draft.txt"
            source.write_text("hello", encoding="utf-8")

            receipt = json.loads(
                backend.perform(
                    UiAction(
                        "rename_file",
                        "explorer-file-list",
                        metadata={
                            "source": str(source),
                            "new_name": "final.txt",
                        },
                    )
                )
            )

            self.assertEqual(receipt["status"], "file-op-executed")
            self.assertEqual(receipt["file_op"]["operation"], "rename")
            self.assertTrue((root / "final.txt").exists())

    def test_virtual_sandbox_records_command_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "sandbox.json"
            backend = VirtualDesktopSandboxBackend(state_path)
            receipt = json.loads(
                backend.perform(UiAction("execute_command", "python -V"))
            )

            self.assertEqual(receipt["status"], "process-executed")
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["virtual_processes"]), 1)

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
                        "state_path": str(Path(temp_dir) / "agent_body_state.json"),
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

    def test_resolve_agent_body_invocation_accepts_explicit_command(
        self,
    ) -> None:
        self.assertEqual(
            resolve_agent_body_invocation(
                {"agent_body_command": [sys.executable, "body.py"]}
            ),
            [sys.executable, "body.py"],
        )

    def test_resolve_agent_body_invocation_includes_requested_features(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "Cargo.toml"
            manifest.write_text(
                "[package]\nname='agent_body'\nversion='0.1.0'\n",
                encoding="utf-8",
            )
            with patch(
                "agentos_orchestrator.sandbox.agent_body_client.shutil.which",
                return_value="cargo",
            ):
                invocation = resolve_agent_body_invocation(
                    {
                        "agent_body_manifest": str(manifest),
                        "agent_body_features": ["uia-windows"],
                    }
                )

        self.assertEqual(
            invocation[:5],
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(manifest),
            ],
        )
        self.assertIn("--features", invocation)
        self.assertIn("uia-windows", invocation)

    def test_agent_body_client_reuses_live_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "fake_agent_body.py"
            script_path.write_text(
                "import json\n"
                "import sys\n"
                "counter = 0\n"
                "print(json.dumps({"
                "'type': 'body.started', 'status': 'ready'"
                "}), flush=True)\n"
                "for line in sys.stdin:\n"
                "    raw = line.strip()\n"
                "    if not raw:\n"
                "        continue\n"
                "    counter += 1\n"
                "    payload = json.loads(raw)\n"
                "    print(json.dumps({"
                "'sequence': counter, 'kind': payload.get('kind')"
                "}), flush=True)\n",
                encoding="utf-8",
            )
            client = AgentBodyClient(
                root / "sandbox.json",
                metadata={
                    "agent_body_command": [sys.executable, str(script_path)],
                },
            )
            try:
                first = client.request({"kind": "snapshot"})
                second = client.request({"kind": "capabilities"})
            finally:
                client.close()

            self.assertEqual(first["sequence"], 1)
            self.assertEqual(first["kind"], "snapshot")
            self.assertEqual(second["sequence"], 2)
            self.assertEqual(second["kind"], "capabilities")

    def test_agent_body_client_decodes_utf8_session_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "utf8_agent_body.py"
            script_path.write_text(
                "import json\n"
                "import sys\n"
                "sys.stdout.buffer.write((json.dumps({"
                "'type': 'body.started', 'status': 'ready'"
                "}, ensure_ascii=False) + '\\n').encode('utf-8'))\n"
                "sys.stdout.buffer.flush()\n"
                "for line in sys.stdin.buffer:\n"
                "    raw = line.strip()\n"
                "    if not raw:\n"
                "        continue\n"
                "    payload = json.loads(raw.decode('utf-8'))\n"
                "    response = {'kind': payload.get('kind'), 'name': 'Ё'}\n"
                "    sys.stdout.buffer.write((json.dumps(response, ensure_ascii=False) + '\\n').encode('utf-8'))\n"
                "    sys.stdout.buffer.flush()\n",
                encoding="utf-8",
            )
            client = AgentBodyClient(
                root / "sandbox.json",
                metadata={
                    "agent_body_command": [sys.executable, str(script_path)],
                },
            )
            try:
                response = client.request({"kind": "snapshot"})
            finally:
                client.close()

        self.assertEqual(response["kind"], "snapshot")
        self.assertEqual(response["name"], "Ё")


if __name__ == "__main__":
    unittest.main()
