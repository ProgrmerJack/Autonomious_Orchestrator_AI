from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from agentos_orchestrator.app_family_registry import sandbox_surface_for_app
from agentos_orchestrator.sandbox.agent_body_client import (
    AgentBodyClient,
    AgentBodyError,
)

from .base import UiAction, UiNode


class VirtualDesktopSandboxBackend:
    """Safe in-memory desktop simulation for full-capacity control testing."""

    name = "virtual-desktop-sandbox"

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._agent_body = AgentBodyClient(self.state_path)
        if self._agent_body.available():
            self._migrate_state_for_agent_body()
        elif not self.state_path.exists():
            self._save_state(self._default_state())

    @staticmethod
    def _default_state() -> dict:
        return {
            "focused": "window-browser",
            "last_action": None,
            "agent_history": [],
            "clipboard": "",
            "terminal_log": [],
            "virtual_processes": [],
            "modals": [],
            "virtual_files": [
                "artifacts/workflows/report.md",
                "artifacts/workflows/slides.pptx",
                "artifacts/workflows/notes.txt",
            ],
            "virtual_file_contents": {
                "artifacts/workflows/report.md": "",
                "artifacts/workflows/slides.pptx": "",
                "artifacts/workflows/notes.txt": "",
            },
            "nodes": [
                {
                    "node_id": "window-browser",
                    "role": "Window",
                    "name": "Sandbox Browser",
                    "focused": True,
                    "enabled": True,
                    "bounds": [80, 80, 1280, 900],
                    "metadata": {
                        "sandbox": True,
                        "app": "browser",
                    },
                },
                {
                    "node_id": "browser-tab-strip",
                    "role": "TabList",
                    "name": "Browser Tabs",
                    "focused": False,
                    "enabled": True,
                    "bounds": [120, 92, 900, 26],
                    "metadata": {
                        "sandbox": True,
                        "app": "browser",
                        "panel_type": "tab_strip",
                    },
                },
                {
                    "node_id": "browser-toolbar",
                    "role": "ToolBar",
                    "name": "Browser Toolbar",
                    "focused": False,
                    "enabled": True,
                    "bounds": [120, 118, 1200, 40],
                    "metadata": {
                        "sandbox": True,
                        "app": "browser",
                        "panel_type": "toolbar",
                    },
                },
                {
                    "node_id": "browser-address-bar",
                    "role": "Edit",
                    "name": "Address and search bar",
                    "focused": False,
                    "enabled": True,
                    "bounds": [120, 120, 900, 32],
                    "metadata": {
                        "sandbox": True,
                        "value": "about:blank",
                    },
                },
                {
                    "node_id": "browser-side-panel",
                    "role": "Pane",
                    "name": "Browser Side Panel",
                    "focused": False,
                    "enabled": True,
                    "bounds": [1030, 170, 290, 760],
                    "metadata": {
                        "sandbox": True,
                        "app": "browser",
                        "panel_type": "side_panel",
                        "text": "Bookmarks and research notes",
                    },
                },
                {
                    "node_id": "browser-main-doc",
                    "role": "Document",
                    "name": "Blank Page",
                    "focused": False,
                    "enabled": True,
                    "bounds": [120, 170, 1200, 760],
                    "metadata": {
                        "sandbox": True,
                        "url": "about:blank",
                        "text": "",
                    },
                },
                {
                    "node_id": "browser-status-bar",
                    "role": "StatusBar",
                    "name": "Browser Status Bar",
                    "focused": False,
                    "enabled": True,
                    "bounds": [120, 932, 1200, 24],
                    "metadata": {
                        "sandbox": True,
                        "app": "browser",
                        "panel_type": "status",
                    },
                },
            ],
        }

    def available(self) -> bool:
        return True

    def capabilities(self) -> dict:
        if self._agent_body.available():
            try:
                return self._agent_body.capabilities()
            except AgentBodyError:
                pass
        return {
            "type": "sandbox.capabilities",
            "status": "ok",
            "sandbox": True,
            "is_simulated": True,
            "rights": "simulated-virtual-rights",
            "capabilities": [
                "snapshot",
                "act",
                "exec",
                "reset",
                "adaptive-app-surfaces",
                "multi-panel-app-surfaces",
                "virtual-file-mutation",
                "virtual-filesystem",
                "virtual-processes",
                "clipboard",
                "modal-mutation",
                "simulated-sandbox-privileges",
                "stateful-control-receipts",
                "history-preserving-reset",
                "window-panel-mutation",
            ],
        }

    def reset(self) -> dict:
        if self._agent_body.available():
            try:
                return self._agent_body.reset()
            except AgentBodyError:
                pass
        previous = self._load_state() if self.state_path.exists() else {}
        fresh = self._default_state()
        fresh["agent_history"] = list(previous.get("agent_history") or [])
        self._save_state(fresh)
        return {"type": "sandbox.reset", "status": "reset", "sandbox": True}

    def snapshot(self) -> list[UiNode]:
        if self._agent_body.available():
            try:
                payload = self._agent_body.request({"kind": "snapshot"})
                return self._snapshot_from_agent_body(payload)
            except AgentBodyError:
                pass
        state = self._load_state()
        nodes: list[UiNode] = []
        for node in state.get("nodes", []):
            bounds_raw = node.get("bounds")
            bounds = None
            if isinstance(bounds_raw, list) and len(bounds_raw) == 4:
                bounds = (
                    int(bounds_raw[0]),
                    int(bounds_raw[1]),
                    int(bounds_raw[2]),
                    int(bounds_raw[3]),
                )
            nodes.append(
                UiNode(
                    node_id=str(node.get("node_id") or "unknown"),
                    role=str(node.get("role") or "Unknown"),
                    name=str(node.get("name") or "Unnamed"),
                    bounds=bounds,
                    enabled=bool(node.get("enabled", True)),
                    focused=bool(node.get("focused", False)),
                    metadata=dict(node.get("metadata") or {}),
                )
            )
        return nodes

    def perform(self, action: UiAction) -> str:
        if self._agent_body.available():
            try:
                payload = self._agent_body.request(
                    {
                        "kind": "act",
                        "action_type": action.action_type,
                        "selector": action.selector,
                        "value": action.value,
                        "metadata": dict(action.metadata or {}),
                    }
                )
                return json.dumps(
                    self._agent_body_receipt(payload, action),
                    sort_keys=True,
                )
            except AgentBodyError:
                pass
        state = self._load_state()
        nodes = list(state.get("nodes") or [])
        receipt = {
            "sandbox": True,
            "backend": self.name,
            "action": action.action_type,
            "selector": action.selector,
            "value": action.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "executed",
        }
        selected_index = self._find_node_index(nodes, action.selector)
        if selected_index is not None:
            receipt["matched_node_id"] = str(
                nodes[selected_index].get("node_id") or "unknown"
            )
        if self._apply_virtual_system_action(state, nodes, action, receipt):
            state["nodes"] = nodes
            state["last_action"] = receipt
            self._append_agent_history(state, receipt)
            self._save_state(state)
            return json.dumps(receipt, sort_keys=True)
        if self._requires_node(action.action_type) and selected_index is None:
            receipt["status"] = "selector-not-found"
            receipt["reason"] = "No UI node matched selector"
            state["last_action"] = receipt
            self._append_agent_history(state, receipt)
            self._save_state(state)
            return json.dumps(receipt, sort_keys=True)
        self._apply_launch_action(nodes, action, receipt)
        self._apply_hotkey_action(state, action, receipt)
        self._apply_focus_action(nodes, state, action, selected_index)
        self._apply_text_action(nodes, action, selected_index, receipt)
        self._apply_draw_action(nodes, action, selected_index)
        self._apply_cell_edit_action(nodes, action, selected_index, receipt)
        self._apply_form_action(nodes, action, selected_index, receipt)
        self._apply_file_operation_action(
            state,
            action,
            selected_index,
            receipt,
        )
        self._apply_window_panel_action(nodes, action, selected_index, receipt)
        self._apply_window_switch_action(nodes, state, action, receipt)
        self._apply_navigation_action(nodes, action)
        self._sync_address_bar_navigation(nodes, action, selected_index)

        state["nodes"] = nodes
        state["last_action"] = receipt
        self._append_agent_history(state, receipt)
        self._save_state(state)
        return json.dumps(receipt, sort_keys=True)

    def _migrate_state_for_agent_body(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        nodes = list(payload.get("nodes") or [])
        if payload.get("terminal_log") is not None and all(
            isinstance(node, dict) and "text" in node and "value" in node
            for node in nodes
        ):
            return
        migrated_nodes: list[dict] = []
        for node in nodes:
            metadata = dict(node.get("metadata") or {})
            migrated_nodes.append(
                {
                    "node_id": str(node.get("node_id") or "unknown"),
                    "role": str(node.get("role") or "Unknown"),
                    "name": str(node.get("name") or "Unnamed"),
                    "focused": bool(node.get("focused", False)),
                    "enabled": bool(node.get("enabled", True)),
                    "text": str(metadata.get("text") or ""),
                    "value": str(metadata.get("value") or ""),
                    "metadata": metadata,
                }
            )
        migrated = {
            "focused": str(payload.get("focused") or "window-browser"),
            "last_action": payload.get("last_action"),
            "agent_history": list(payload.get("agent_history") or []),
            "clipboard": str(payload.get("clipboard") or ""),
            "virtual_files": list(payload.get("virtual_files") or []),
            "virtual_file_contents": dict(payload.get("virtual_file_contents") or {}),
            "virtual_processes": list(payload.get("virtual_processes") or []),
            "modals": list(payload.get("modals") or []),
            "terminal_log": list(payload.get("terminal_log") or []),
            "nodes": migrated_nodes,
        }
        self.state_path.write_text(
            json.dumps(migrated, indent=2),
            encoding="utf-8",
        )

    def _snapshot_from_agent_body(self, payload: dict) -> list[UiNode]:
        nodes: list[UiNode] = []
        for node in list(payload.get("nodes") or []):
            metadata = dict(node.get("metadata") or {})
            bounds_raw = node.get("bounds") or metadata.get("bounds")
            bounds = None
            if isinstance(bounds_raw, list) and len(bounds_raw) == 4:
                bounds = (
                    int(bounds_raw[0]),
                    int(bounds_raw[1]),
                    int(bounds_raw[2]),
                    int(bounds_raw[3]),
                )
            nodes.append(
                UiNode(
                    node_id=str(node.get("node_id") or "unknown"),
                    role=str(node.get("role") or "Unknown"),
                    name=str(node.get("name") or "Unnamed"),
                    bounds=bounds,
                    enabled=bool(node.get("enabled", True)),
                    focused=bool(node.get("focused", False)),
                    metadata=metadata,
                )
            )
        return nodes

    def _agent_body_receipt(self, payload: dict, action: UiAction) -> dict:
        receipt = dict(payload)
        status = str(receipt.get("status") or "")
        if action.action_type in {"click", "invoke"} and status == "focused":
            receipt["focus_status"] = status
            receipt["status"] = "executed"
        receipt.setdefault("sandbox", True)
        receipt["backend"] = self.name
        receipt.setdefault("action", action.action_type)
        receipt.setdefault("action_type", action.action_type)
        receipt.setdefault("selector", action.selector)
        receipt.setdefault("value", action.value)
        receipt.setdefault("timestamp", datetime.now(UTC).isoformat())
        return receipt

    @staticmethod
    def _requires_node(action_type: str) -> bool:
        return action_type in {
            "focus",
            "click",
            "invoke",
            "type",
            "set_value",
            "set_text",
            "draw_path",
            "cell_edit",
            "move_window",
            "resize_window",
            "open_panel",
            "close_panel",
            "select_tab",
            "open_context_menu",
            "select_menu_item",
            "close_context_menu",
            "fill_form_field",
            "submit_form",
        }

    def _apply_launch_action(
        self,
        nodes: list[dict],
        action: UiAction,
        receipt: dict,
    ) -> None:
        if action.action_type != "launch_app":
            return
        app_name = action.value or action.selector
        if self._is_browser_app(app_name):
            self._focus_browser(nodes)
            receipt["launched"] = app_name
            return
        app_node_id = f"app-{len(nodes) + 1}"
        nodes.append(
            {
                "node_id": app_node_id,
                "role": "Window",
                "name": f"Sandbox App - {app_name}",
                "focused": True,
                "enabled": True,
                "bounds": [140, 110, 1040, 760],
                "metadata": {
                    "sandbox": True,
                    "app": app_name,
                },
            }
        )
        nodes.extend(self._app_surface_nodes(app_name, app_node_id))
        receipt["launched"] = app_name

    @staticmethod
    def _apply_hotkey_action(
        state: dict,
        action: UiAction,
        receipt: dict,
    ) -> None:
        if action.action_type != "hotkey":
            return
        value = action.value or ""
        receipt["hotkey"] = value
        state["last_hotkey"] = value

    @staticmethod
    def _apply_focus_action(
        nodes: list[dict],
        state: dict,
        action: UiAction,
        selected_index: int | None,
    ) -> None:
        if selected_index is None:
            return
        if action.action_type not in {"focus", "click", "invoke"}:
            return
        for node in nodes:
            node["focused"] = False
        nodes[selected_index]["focused"] = True
        state["focused"] = nodes[selected_index].get("node_id")

    @staticmethod
    def _apply_text_action(
        nodes: list[dict],
        action: UiAction,
        selected_index: int | None,
        receipt: dict,
    ) -> None:
        if selected_index is None:
            return
        if action.action_type not in {"type", "set_value", "set_text"}:
            return
        metadata = dict(nodes[selected_index].get("metadata") or {})
        metadata["value"] = action.value or ""
        metadata["text"] = action.value or ""
        node_id = str(nodes[selected_index].get("node_id") or "")
        if node_id == "spreadsheet-grid":
            VirtualDesktopSandboxBackend._apply_spreadsheet_cell_edit(
                metadata,
                action.value or "",
                receipt,
            )
        nodes[selected_index]["metadata"] = metadata

    @staticmethod
    def _apply_spreadsheet_cell_edit(
        metadata: dict,
        value: str,
        receipt: dict,
    ) -> None:
        cell, cell_value = VirtualDesktopSandboxBackend._parse_cell_edit(value)
        cells = dict(metadata.get("cells") or {})
        cells[cell] = cell_value
        metadata["cells"] = cells
        receipt["cell_edit"] = {
            "cell": cell,
            "value": cell_value,
        }

    @staticmethod
    def _parse_cell_edit(value: str) -> tuple[str, str]:
        chunks = value.split(":", 1)
        if len(chunks) == 2:
            return chunks[0].strip().upper(), chunks[1].strip()
        return "A1", value.strip()

    @staticmethod
    def _apply_cell_edit_action(
        nodes: list[dict],
        action: UiAction,
        selected_index: int | None,
        receipt: dict,
    ) -> None:
        """Handle the dedicated ``cell_edit`` action type.

        Emitted by ``SpreadsheetCellEditIntentAdapter``.  Produces a typed
        ``receipt["cell_edit"]`` payload so tests can assert on cell, value,
        formula, and range_edit fields directly.
        """
        if action.action_type != "cell_edit":
            return
        if selected_index is None:
            return
        meta_src = dict(action.metadata or {})
        parsed_cell, parsed_value = VirtualDesktopSandboxBackend._parse_cell_edit(
            action.value or ""
        )
        cell = str(meta_src.get("cell") or parsed_cell).upper()
        value = str(meta_src.get("value") or parsed_value)
        is_formula = bool(meta_src.get("formula", value.startswith("=")))
        is_range = bool(meta_src.get("range_edit", ":" in cell))
        metadata = dict(nodes[selected_index].get("metadata") or {})
        cells = dict(metadata.get("cells") or {})
        if is_range and ":" in cell:
            start_ref, end_ref = cell.split(":", 1)
            cells[start_ref] = value
            cells[end_ref] = value
        else:
            cells[cell] = value
        metadata["cells"] = cells
        nodes[selected_index]["metadata"] = metadata
        receipt["cell_edit"] = {
            "cell": cell,
            "value": value,
            "formula": is_formula,
            "range_edit": is_range,
        }

    @staticmethod
    def _apply_draw_action(
        nodes: list[dict],
        action: UiAction,
        selected_index: int | None,
    ) -> None:
        if selected_index is None:
            return
        if action.action_type != "draw_path":
            return
        metadata = dict(nodes[selected_index].get("metadata") or {})
        metadata["last_path"] = action.value or ""
        nodes[selected_index]["metadata"] = metadata

    @staticmethod
    def _apply_form_action(
        nodes: list[dict],
        action: UiAction,
        selected_index: int | None,
        receipt: dict,
    ) -> None:
        if selected_index is None:
            return
        if action.action_type not in {"fill_form_field", "submit_form"}:
            return
        metadata = dict(nodes[selected_index].get("metadata") or {})
        form_fields = dict(metadata.get("form_fields") or {})
        if action.action_type == "fill_form_field":
            value_raw = str(action.value or "")
            field_name = str((action.metadata or {}).get("field") or "").strip()
            field_value = str((action.metadata or {}).get("value") or "").strip()
            if not field_name and ":" in value_raw:
                left, right = value_raw.split(":", 1)
                field_name = left.strip()
                field_value = right.strip()
            if not field_name:
                field_name = "field"
            form_fields[field_name] = field_value
            metadata["form_fields"] = form_fields
            metadata["last_form_action"] = "fill"
            nodes[selected_index]["metadata"] = metadata
            receipt["form"] = {
                "action": "fill",
                "field": field_name,
                "value": field_value,
            }
            receipt["status"] = "form-field-filled"
            return

        metadata["form_submitted"] = True
        metadata["form_submitted_at"] = receipt.get("timestamp")
        metadata["last_form_action"] = "submit"
        nodes[selected_index]["metadata"] = metadata
        receipt["form"] = {
            "action": "submit",
            "field_count": len(form_fields),
        }
        receipt["status"] = "form-submitted"

    @staticmethod
    def _apply_window_switch_action(
        nodes: list[dict],
        state: dict,
        action: UiAction,
        receipt: dict,
    ) -> None:
        if action.action_type != "switch_window":
            return
        target = str(action.value or "").strip().lower()
        if not target:
            target = str((action.metadata or {}).get("window") or "").strip().lower()
        if not target:
            receipt["status"] = "selector-not-found"
            receipt["reason"] = "No target window specified"
            return
        chosen: dict | None = None
        for node in nodes:
            role = str(node.get("role") or "").lower()
            node_id = str(node.get("node_id") or "").lower()
            name = str(node.get("name") or "").lower()
            if role != "window":
                continue
            if target in node_id or target in name:
                chosen = node
                break
        if chosen is None:
            receipt["status"] = "selector-not-found"
            receipt["reason"] = "Window target not found"
            return
        for node in nodes:
            node["focused"] = False
        chosen["focused"] = True
        state["focused"] = chosen.get("node_id")
        receipt["status"] = "window-switched"
        receipt["window"] = chosen.get("name")

    def _apply_navigation_action(
        self,
        nodes: list[dict],
        action: UiAction,
    ) -> None:
        if action.action_type in {"scroll", "scroll_up", "scroll_down"}:
            delta = action.value or action.selector or "down"
            for node in nodes:
                node_id = str(node.get("node_id") or "").lower()
                role = str(node.get("role") or "").lower()
                if node_id == "browser-main-doc" or role == "document":
                    metadata = dict(node.get("metadata") or {})
                    metadata["last_scroll"] = str(delta)
                    metadata["scroll_events"] = (
                        int(metadata.get("scroll_events") or 0) + 1
                    )
                    node["metadata"] = metadata
            return
        if action.action_type not in {"navigate", "goto", "open_url"}:
            return
        url = action.value or action.selector
        self._update_browser_url(nodes, url)

    def _apply_virtual_system_action(
        self,
        state: dict,
        nodes: list[dict],
        action: UiAction,
        receipt: dict,
    ) -> bool:
        action_type = action.action_type
        if action_type in {
            "create_file",
            "write_file",
            "read_file",
            "delete_file",
            "copy_file",
            "move_file",
            "rename_file",
            "download_file",
            "upload_file",
        }:
            self._apply_virtual_file_action(state, action, receipt)
            return True
        if action_type in {"set_clipboard", "get_clipboard", "clipboard_copy"}:
            self._apply_clipboard_action(state, action, receipt)
            return True
        if action_type == "execute_command":
            self._apply_virtual_process_action(state, action, receipt)
            return True
        if action_type in {"open_modal", "close_modal"}:
            self._apply_modal_action(state, nodes, action, receipt)
            return True
        return False

    @staticmethod
    def _virtual_action_path(action: UiAction, default: str) -> str:
        metadata = dict(action.metadata or {})
        return str(
            metadata.get("path")
            or metadata.get("source")
            or action.value
            or action.selector
            or default
        ).strip()

    def _apply_virtual_file_action(
        self,
        state: dict,
        action: UiAction,
        receipt: dict,
    ) -> None:
        action_type = action.action_type
        metadata = dict(action.metadata or {})
        files = list(state.get("virtual_files") or [])
        contents = dict(state.get("virtual_file_contents") or {})
        source = self._virtual_action_path(action, "artifacts/workflows/item.txt")
        destination = str(
            metadata.get("destination") or "artifacts/workflows/copied-item.txt"
        )
        new_name = str(metadata.get("new_name") or metadata.get("name") or source)
        content_value = metadata.get("content") or metadata.get("text")
        if content_value is None and action.value and action.value != source:
            content_value = action.value
        content = str(content_value or "")
        operation = action_type.removesuffix("_file")

        if action_type in {"create_file", "write_file", "upload_file"}:
            if source not in files:
                files.append(source)
            contents[source] = content
            operation = action_type.removesuffix("_file")
        elif action_type == "read_file":
            content = str(contents.get(source) or "")
        elif action_type == "delete_file":
            files = [item for item in files if item != source]
            contents.pop(source, None)
        elif action_type == "download_file":
            destination = (
                destination if destination != source else f"downloads/{source}"
            )
            if destination not in files:
                files.append(destination)
            contents[destination] = str(contents.get(source) or content)
        elif action_type == "copy_file":
            files = self._mutate_file_list(files, "copy", source, destination, new_name)
            contents[destination] = str(contents.get(source) or content)
            operation = "copy"
        elif action_type == "move_file":
            files = self._mutate_file_list(files, "move", source, destination, new_name)
            contents[destination] = str(contents.pop(source, content))
            operation = "move"
        elif action_type == "rename_file":
            files = self._mutate_file_list(
                files, "rename", source, destination, new_name
            )
            contents[new_name] = str(contents.pop(source, content))
            operation = "rename"

        state["virtual_files"] = files
        state["virtual_file_contents"] = contents
        receipt["file_op"] = {
            "operation": operation,
            "source": source,
            "destination": destination if operation not in {"rename", "read"} else None,
            "new_name": new_name if operation == "rename" else None,
            "content": content if action_type == "read_file" else None,
            "resulting_file_count": len(files),
        }
        receipt["status"] = "file-op-executed"

    @staticmethod
    def _apply_clipboard_action(state: dict, action: UiAction, receipt: dict) -> None:
        if action.action_type in {"set_clipboard", "clipboard_copy"}:
            state["clipboard"] = action.value or action.selector or ""
        receipt["clipboard"] = state.get("clipboard") or ""
        receipt["status"] = "clipboard-updated"

    @staticmethod
    def _apply_virtual_process_action(
        state: dict,
        action: UiAction,
        receipt: dict,
    ) -> None:
        command = action.value or action.selector or ""
        processes = list(state.get("virtual_processes") or [])
        terminal_log = list(state.get("terminal_log") or [])
        process = {
            "pid": len(processes) + 1,
            "command": command,
            "status": "exited",
            "exit_code": 0,
            "sandbox": True,
            "timestamp": receipt["timestamp"],
        }
        processes.append(process)
        terminal_log.append({"command": command, "stdout": "", "exit_code": 0})
        state["virtual_processes"] = processes[-1000:]
        state["terminal_log"] = terminal_log[-5000:]
        receipt["process"] = process
        receipt["status"] = "process-executed"

    @staticmethod
    def _apply_modal_action(
        state: dict,
        nodes: list[dict],
        action: UiAction,
        receipt: dict,
    ) -> None:
        modals = list(state.get("modals") or [])
        if action.action_type == "close_modal":
            if modals:
                closed = modals.pop()
                receipt["modal"] = closed
            for node in nodes:
                if str(node.get("role") or "").lower() == "dialog":
                    node["enabled"] = False
            state["modals"] = modals
            receipt["status"] = "modal-closed"
            return
        modal_id = f"modal-{len(modals) + 1}"
        modal = {
            "modal_id": modal_id,
            "name": action.value or action.selector or "Sandbox Modal",
            "timestamp": receipt["timestamp"],
        }
        modals.append(modal)
        state["modals"] = modals
        nodes.append(
            VirtualDesktopSandboxBackend._surface_node(
                modal_id,
                "Dialog",
                str(modal["name"]),
                str(state.get("focused") or "window-browser"),
                bounds=[360, 220, 560, 320],
                panel_type="modal",
            )
        )
        receipt["modal"] = modal
        receipt["status"] = "modal-opened"

    @staticmethod
    def _apply_file_operation_action(
        state: dict,
        action: UiAction,
        selected_index: int | None,
        receipt: dict,
    ) -> None:
        if selected_index is None:
            return
        if action.action_type not in {"copy_file", "move_file", "rename_file"}:
            return
        files = list(state.get("virtual_files") or [])
        meta = dict(action.metadata or {})
        source = str(meta.get("source") or "source-item")
        destination = str(meta.get("destination") or "destination-item")
        new_name = str(meta.get("new_name") or "renamed-item")
        operation = action.action_type.replace("_file", "")
        files = VirtualDesktopSandboxBackend._mutate_file_list(
            files,
            operation,
            source,
            destination,
            new_name,
        )
        state["virtual_files"] = files
        destination_value = None
        if operation != "rename":
            destination_value = destination
        rename_value = None
        if operation == "rename":
            rename_value = new_name
        receipt["file_op"] = {
            "operation": operation,
            "source": source,
            "destination": destination_value,
            "new_name": rename_value,
            "resulting_file_count": len(files),
        }
        receipt["status"] = "file-op-executed"

    @staticmethod
    def _apply_window_panel_action(
        nodes: list[dict],
        action: UiAction,
        selected_index: int | None,
        receipt: dict,
    ) -> None:
        if action.action_type not in {
            "move_window",
            "resize_window",
            "open_panel",
            "close_panel",
            "select_tab",
            "open_context_menu",
            "select_menu_item",
            "close_context_menu",
        }:
            return
        if selected_index is None:
            return
        metadata = dict(action.metadata or {})
        node = nodes[selected_index]
        node_metadata = dict(node.get("metadata") or {})
        if action.action_type in {"move_window", "resize_window"}:
            bounds = metadata.get("bounds")
            if isinstance(bounds, list) and len(bounds) == 4:
                node["bounds"] = [int(item) for item in bounds]
                receipt["bounds"] = node["bounds"]
            receipt["status"] = "window-updated"
            return
        if action.action_type == "select_tab":
            tab = str(metadata.get("tab") or action.value or "Tab 1")
            node_metadata["selected_tab"] = tab
            node["metadata"] = node_metadata
            receipt["tab"] = tab
            receipt["status"] = "tab-selected"
            return
        if action.action_type == "close_panel":
            node["enabled"] = False
            node_metadata["visible"] = False
            node["metadata"] = node_metadata
            receipt["status"] = "panel-closed"
            return
        if action.action_type == "open_context_menu":
            context_menu_id = f"context-menu-{len(nodes) + 1}"
            bounds_raw = node.get("bounds") or [200, 200, 240, 160]
            menu_bounds = [
                int(bounds_raw[0]) + 16,
                int(bounds_raw[1]) + 16,
                260,
                220,
            ]
            nodes.append(
                VirtualDesktopSandboxBackend._surface_node(
                    context_menu_id,
                    "Menu",
                    str(action.value or "Context Menu"),
                    str(node.get("node_id") or "window"),
                    bounds=menu_bounds,
                    panel_type="context_menu",
                )
            )
            receipt["status"] = "context-menu-opened"
            receipt["context_menu_id"] = context_menu_id
            return
        if action.action_type == "select_menu_item":
            menu_item = str(action.value or metadata.get("menu_item") or "default")
            node_metadata["last_menu_item"] = menu_item
            node["metadata"] = node_metadata
            for item in nodes:
                item_meta = dict(item.get("metadata") or {})
                if str(item_meta.get("panel_type") or "") == "context_menu":
                    item["enabled"] = False
                    item_meta["visible"] = False
                    item["metadata"] = item_meta
            receipt["status"] = "menu-item-selected"
            receipt["menu_item"] = menu_item
            return
        if action.action_type == "close_context_menu":
            for item in nodes:
                item_meta = dict(item.get("metadata") or {})
                if str(item_meta.get("panel_type") or "") == "context_menu":
                    item["enabled"] = False
                    item_meta["visible"] = False
                    item["metadata"] = item_meta
            receipt["status"] = "context-menu-closed"
            return
        panel_name = str(metadata.get("panel_name") or action.value or "Agent Panel")
        panel_id = str(metadata.get("panel_id") or f"panel-{len(nodes) + 1}")
        nodes.append(
            VirtualDesktopSandboxBackend._surface_node(
                panel_id,
                str(metadata.get("role") or "Pane"),
                panel_name,
                str(node.get("node_id") or "window"),
                bounds=list(metadata.get("bounds") or [220, 180, 420, 560]),
                panel_type=str(metadata.get("panel_type") or "dynamic"),
            )
        )
        receipt["status"] = "panel-opened"
        receipt["panel_id"] = panel_id

    @staticmethod
    def _mutate_file_list(
        files: list[str],
        operation: str,
        source: str,
        destination: str,
        new_name: str,
    ) -> list[str]:
        if operation == "copy":
            return [*files, destination]
        if operation == "move":
            moved = [item for item in files if item != source]
            moved.append(destination)
            return moved
        renamed = [new_name if item == source else item for item in files]
        if source not in files and new_name not in renamed:
            renamed.append(new_name)
        return renamed

    def _sync_address_bar_navigation(
        self,
        nodes: list[dict],
        action: UiAction,
        selected_index: int | None,
    ) -> None:
        if selected_index is None:
            return
        if action.action_type not in {"type", "set_value", "set_text"}:
            return
        name = str(nodes[selected_index].get("name", "")).lower()
        if "address" not in name:
            return
        self._update_browser_url(nodes, action.value or "about:blank")

    def _load_state(self) -> dict:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"nodes": []}
        defaults = self._default_state()
        for key in (
            "agent_history",
            "clipboard",
            "terminal_log",
            "virtual_processes",
            "modals",
            "virtual_files",
            "virtual_file_contents",
        ):
            payload.setdefault(key, defaults[key])
        if not payload.get("nodes"):
            payload["nodes"] = defaults["nodes"]
        return payload

    def _save_state(self, payload: dict) -> None:
        self.state_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _find_node_index(self, nodes: list[dict], selector: str) -> int | None:
        selector_lower = selector.lower()
        selector_tokens = self._parse_selector_tokens(selector_lower)

        point = None
        if selector_lower.startswith("point="):
            point = self._parse_point(selector_lower.split("=", 1)[1])
        elif selector_tokens.get("point"):
            point = self._parse_point(selector_tokens["point"])
        if point is not None:
            point_x, point_y = point
            for index, node in enumerate(nodes):
                if self._node_contains_point(node, point_x, point_y):
                    return index

        # Tier 1: exact/semantic token match (node_id/role/name/panel_type).
        if selector_tokens:
            for index, node in enumerate(nodes):
                if self._node_matches_tokens(node, selector_tokens):
                    return index

        for index, node in enumerate(nodes):
            if self._node_matches_selector(node, selector_lower):
                return index
        if selector_lower.startswith("name="):
            target = selector_lower.split("=", 1)[1]
            for index, node in enumerate(nodes):
                if target in str(node.get("name") or "").lower():
                    return index
        return None

    @staticmethod
    def _node_matches_selector(node: dict, selector_lower: str) -> bool:
        node_id = str(node.get("node_id") or "").lower()
        name = str(node.get("name") or "").lower()
        if selector_lower in {"drawing-canvas", "design-canvas"}:
            return node_id in {"drawing-canvas", "design-canvas"} or "canvas" in name
        return selector_lower in node_id or selector_lower in name

    @staticmethod
    def _parse_selector_tokens(selector_lower: str) -> dict[str, str]:
        if "=" not in selector_lower:
            return {}
        tokens: dict[str, str] = {}
        for chunk in re.split(r"[;|,]", selector_lower):
            part = chunk.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                tokens[key] = value
        return tokens

    @staticmethod
    def _node_matches_tokens(node: dict, tokens: dict[str, str]) -> bool:
        node_id = str(node.get("node_id") or "").lower()
        role = str(node.get("role") or "").lower()
        name = str(node.get("name") or "").lower()
        metadata = dict(node.get("metadata") or {})
        panel_type = str(metadata.get("panel_type") or "").lower()
        automation_id = str(metadata.get("automation_id") or "").lower()
        class_name = str(metadata.get("class_name") or "").lower()

        target_id = tokens.get("node_id")
        if target_id and node_id != target_id:
            return False
        target_role = tokens.get("role")
        if target_role and role != target_role:
            return False
        target_name = tokens.get("name")
        if target_name and target_name not in name:
            return False
        target_panel = tokens.get("panel_type")
        if target_panel and target_panel != panel_type:
            return False
        target_automation_id = tokens.get("automation_id")
        if target_automation_id and target_automation_id != automation_id:
            return False
        target_class_name = tokens.get("class_name")
        if target_class_name and target_class_name != class_name:
            return False
        return bool(
            target_id
            or target_role
            or target_name
            or target_panel
            or target_automation_id
            or target_class_name
        )

    @staticmethod
    def _parse_point(raw: str) -> tuple[int, int] | None:
        match = re.fullmatch(r"\s*(\d{1,5})\s*[:/]\s*(\d{1,5})\s*", raw)
        if match is None:
            match = re.fullmatch(r"\s*(\d{1,5})\s*,\s*(\d{1,5})\s*", raw)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _node_contains_point(node: dict, point_x: int, point_y: int) -> bool:
        bounds = node.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            return False
        x, y, width, height = (
            int(bounds[0]),
            int(bounds[1]),
            int(bounds[2]),
            int(bounds[3]),
        )
        return x <= point_x <= (x + width) and y <= point_y <= (y + height)

    def _update_browser_url(self, nodes: list[dict], url: str) -> None:
        for node in nodes:
            node_id = str(node.get("node_id") or "")
            metadata = dict(node.get("metadata") or {})
            if node_id == "browser-address-bar":
                metadata["value"] = url
                node["metadata"] = metadata
            if node_id == "browser-main-doc":
                metadata["url"] = url
                metadata["text"] = f"Sandbox content loaded for {url}"
                node["metadata"] = metadata
                node["name"] = f"Sandbox Page - {url}"

    def hydrate_browser_page(
        self,
        url: str,
        *,
        title: str = "",
        text: str = "",
        source: str = "",
    ) -> dict:
        """Inject real fetched browser content into the sandbox document.

        The sandbox remains a safe virtual UI backend, but its browser document
        should reflect actual fetched page content when the worker has already
        retrieved it through the research engine. That turns PC navigation from
        a pure selector exercise into stateful evidence-bearing UI context.
        """
        state = self._load_state()
        nodes = list(state.get("nodes") or [])
        clean_url = str(url or "").strip() or "about:blank"
        clean_title = str(title or "").strip()
        clean_text = str(text or "")
        self._update_browser_url(nodes, clean_url)
        for node in nodes:
            node_id = str(node.get("node_id") or "")
            if node_id == "browser-address-bar":
                metadata = dict(node.get("metadata") or {})
                metadata["value"] = clean_url
                node["metadata"] = metadata
                continue
            if node_id != "browser-main-doc":
                continue
            metadata = dict(node.get("metadata") or {})
            metadata.update(
                {
                    "url": clean_url,
                    "title": clean_title,
                    "text": clean_text,
                    "content_source": source or "research-engine",
                    "content_chars": len(clean_text),
                    "hydrated_at": datetime.now(UTC).isoformat(),
                }
            )
            node["metadata"] = metadata
            node["name"] = clean_title or f"Sandbox Page - {clean_url}"
        state["nodes"] = nodes
        receipt = {
            "type": "sandbox.browser_hydrate",
            "status": "hydrated",
            "sandbox": True,
            "backend": self.name,
            "url": clean_url,
            "title": clean_title,
            "content_chars": len(clean_text),
            "source": source or "research-engine",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        state["last_action"] = receipt
        self._append_agent_history(state, receipt)
        self._save_state(state)
        return receipt

    @staticmethod
    def _is_browser_app(app_name: str) -> bool:
        lower = app_name.lower()
        return "edge" in lower or "browser" in lower

    @staticmethod
    def _focus_browser(nodes: list[dict]) -> None:
        for node in nodes:
            node["focused"] = str(node.get("node_id") or "") == "window-browser"

    @staticmethod
    def _app_surface_nodes(app_name: str, app_node_id: str) -> list[dict]:
        lower = app_name.lower()
        predefined = VirtualDesktopSandboxBackend._predefined_surface(lower)
        family = VirtualDesktopSandboxBackend._app_family_for_name(lower)
        if predefined is None:
            predefined = ("app-workspace", "Pane", "Application Workspace")
        node_id, role, name = predefined
        primary = VirtualDesktopSandboxBackend._surface_node(
            node_id,
            role,
            name,
            app_node_id,
            panel_type="primary",
            family=family,
        )
        return [
            primary,
            *VirtualDesktopSandboxBackend._support_panel_nodes(family, app_node_id),
        ]

    @staticmethod
    def _support_panel_nodes(family: str, app_node_id: str) -> list[dict]:
        specs = {
            "browser": [
                (
                    "browser-tabs",
                    "TabList",
                    "Browser Tabs",
                    [180, 118, 900, 28],
                    "tab_strip",
                ),
                (
                    "browser-main-doc",
                    "Document",
                    "Browser Document",
                    [180, 170, 860, 560],
                    "document",
                ),
                (
                    "browser-research-panel",
                    "Pane",
                    "Research Side Panel",
                    [1048, 170, 280, 560],
                    "side_panel",
                ),
            ],
            "file_explorer": [
                (
                    "explorer-navigation-tree",
                    "Tree",
                    "Explorer Navigation Tree",
                    [180, 160, 240, 620],
                    "navigation",
                ),
                (
                    "explorer-preview-pane",
                    "Pane",
                    "Explorer Preview Pane",
                    [1060, 160, 260, 620],
                    "preview",
                ),
            ],
            "terminal": [
                (
                    "terminal-toolbar",
                    "ToolBar",
                    "Terminal Toolbar",
                    [180, 160, 920, 44],
                    "toolbar",
                ),
                (
                    "terminal-input",
                    "Edit",
                    "Sandbox Terminal",
                    [180, 210, 920, 570],
                    "primary",
                ),
            ],
            "editor": [
                (
                    "editor-explorer",
                    "Tree",
                    "Editor Explorer",
                    [180, 160, 230, 620],
                    "navigation",
                ),
                (
                    "editor-outline",
                    "Tree",
                    "Editor Outline",
                    [1120, 160, 200, 620],
                    "side_panel",
                ),
            ],
            "office_form": [
                (
                    "office-ribbon",
                    "ToolBar",
                    "Office Ribbon",
                    [180, 150, 1140, 76],
                    "toolbar",
                ),
                (
                    "formula-bar",
                    "Edit",
                    "Formula Bar",
                    [180, 232, 1140, 34],
                    "formula",
                ),
            ],
            "pdf_viewer": [
                (
                    "pdf-thumbnail-pane",
                    "List",
                    "PDF Thumbnail Pane",
                    [180, 160, 220, 620],
                    "navigation",
                ),
                (
                    "pdf-document",
                    "Document",
                    "PDF Document",
                    [410, 160, 690, 620],
                    "primary",
                ),
            ],
            "chat_app": [
                (
                    "chat-thread-list",
                    "List",
                    "Chat Thread List",
                    [180, 160, 260, 620],
                    "navigation",
                ),
                (
                    "chat-history",
                    "Document",
                    "Chat History",
                    [450, 160, 640, 520],
                    "primary",
                ),
            ],
            "design_canvas": [
                (
                    "design-toolbox",
                    "ToolBar",
                    "Design Toolbox",
                    [180, 160, 90, 620],
                    "toolbar",
                ),
                (
                    "layers-panel",
                    "Pane",
                    "Layers Panel",
                    [1050, 160, 270, 620],
                    "side_panel",
                ),
            ],
            "trading_terminal": [
                (
                    "market-watchlist",
                    "Table",
                    "Market Watchlist",
                    [180, 160, 280, 620],
                    "watchlist",
                ),
                (
                    "price-chart",
                    "Chart",
                    "Price Chart",
                    [470, 160, 560, 390],
                    "chart",
                ),
                (
                    "positions-grid",
                    "Table",
                    "Positions Grid",
                    [470, 560, 850, 220],
                    "positions",
                ),
            ],
            "enterprise_grid": [
                (
                    "enterprise-filter-panel",
                    "Pane",
                    "Enterprise Filter Panel",
                    [180, 160, 260, 620],
                    "filters",
                ),
                (
                    "enterprise-detail-panel",
                    "Pane",
                    "Enterprise Detail Panel",
                    [1080, 160, 240, 620],
                    "detail",
                ),
            ],
        }
        return [
            VirtualDesktopSandboxBackend._surface_node(
                node_id,
                role,
                name,
                app_node_id,
                bounds=bounds,
                panel_type=panel_type,
                family=family,
            )
            for node_id, role, name, bounds, panel_type in specs.get(family, [])
        ]

    @staticmethod
    def _app_family_for_name(lower: str) -> str:
        if any(token in lower for token in ("browser", "edge", "chrome")):
            return "browser"
        if "explorer" in lower:
            return "file_explorer"
        if any(token in lower for token in ("powershell", "terminal", "cmd")):
            return "terminal"
        if any(token in lower for token in ("notepad", "winword", "code", "vscode")):
            return "editor"
        if any(token in lower for token in ("excel", "calc", "spreadsheet")):
            return "office_form"
        if any(token in lower for token in ("acrobat", "pdf")):
            return "pdf_viewer"
        if any(token in lower for token in ("teams", "chat")):
            return "chat_app"
        if any(
            token in lower
            for token in ("photoshop", "designer", "adobe", "paint", "gimp")
        ):
            return "design_canvas"
        if "trading" in lower:
            return "trading_terminal"
        if "enterprise" in lower:
            return "enterprise_grid"
        return "unknown"

    @staticmethod
    def _predefined_surface(lower: str) -> tuple[str, str, str] | None:
        registry_surface = sandbox_surface_for_app(lower)
        if registry_surface is not None:
            return registry_surface
        if "powerpnt" in lower:
            return "presentation-canvas", "Document", "Presentation Canvas"
        if lower == "code" or "vscode" in lower:
            return "editor-canvas", "Document", "Editor Canvas"
        return None

    @staticmethod
    def _surface_node(
        node_id: str,
        role: str,
        name: str,
        parent: str,
        bounds: list[int] | None = None,
        panel_type: str = "primary",
        family: str = "unknown",
    ) -> dict:
        return {
            "node_id": node_id,
            "role": role,
            "name": name,
            "focused": False,
            "enabled": True,
            "bounds": bounds or [180, 160, 920, 620],
            "metadata": {
                "sandbox": True,
                "parent": parent,
                "panel_type": panel_type,
                "app_family": family,
                "text": "",
            },
        }

    @staticmethod
    def _append_agent_history(state: dict, receipt: dict) -> None:
        history = list(state.get("agent_history") or [])
        history.append(receipt)
        state["agent_history"] = history[-5000:]
