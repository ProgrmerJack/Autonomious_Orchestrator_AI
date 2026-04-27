from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .base import UiAction, UiNode


class VirtualDesktopSandboxBackend:
    """Safe in-memory desktop simulation for full-capacity control testing."""

    name = "virtual-desktop-sandbox"

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._save_state(self._default_state())

    @staticmethod
    def _default_state() -> dict:
        return {
            "focused": "window-browser",
            "last_action": None,
            "virtual_files": [
                "artifacts/workflows/report.md",
                "artifacts/workflows/slides.pptx",
                "artifacts/workflows/notes.txt",
            ],
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
            ],
        }

    def available(self) -> bool:
        return True

    def snapshot(self) -> list[UiNode]:
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
        if self._requires_node(action.action_type) and selected_index is None:
            receipt["status"] = "selector-not-found"
            receipt["reason"] = "No UI node matched selector"
            state["last_action"] = receipt
            self._save_state(state)
            return json.dumps(receipt, sort_keys=True)
        self._apply_launch_action(nodes, action, receipt)
        self._apply_hotkey_action(state, action, receipt)
        self._apply_focus_action(nodes, state, action, selected_index)
        self._apply_text_action(nodes, action, selected_index, receipt)
        self._apply_draw_action(nodes, action, selected_index)
        self._apply_cell_edit_action(nodes, action, selected_index, receipt)
        self._apply_file_operation_action(
            state,
            action,
            selected_index,
            receipt,
        )
        self._apply_navigation_action(nodes, action)
        self._sync_address_bar_navigation(nodes, action, selected_index)

        state["nodes"] = nodes
        state["last_action"] = receipt
        self._save_state(state)
        return json.dumps(receipt, sort_keys=True)

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
            "copy_file",
            "move_file",
            "rename_file",
            "cell_edit",
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

    def _apply_navigation_action(
        self,
        nodes: list[dict],
        action: UiAction,
    ) -> None:
        if action.action_type not in {"navigate", "goto", "open_url"}:
            return
        url = action.value or action.selector
        self._update_browser_url(nodes, url)

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
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"nodes": []}

    def _save_state(self, payload: dict) -> None:
        self.state_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _find_node_index(self, nodes: list[dict], selector: str) -> int | None:
        selector_lower = selector.lower()
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
        return selector_lower in node_id or selector_lower in name

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
        if predefined is not None:
            node_id, role, name = predefined
            return [
                VirtualDesktopSandboxBackend._surface_node(
                    node_id,
                    role,
                    name,
                    app_node_id,
                )
            ]
        return [
            VirtualDesktopSandboxBackend._surface_node(
                "app-workspace",
                "Pane",
                "Application Workspace",
                app_node_id,
            )
        ]

    @staticmethod
    def _predefined_surface(lower: str) -> tuple[str, str, str] | None:
        if "powerpnt" in lower:
            return "presentation-canvas", "Document", "Presentation Canvas"
        if "winword" in lower or "notepad" in lower:
            return "document-canvas", "Document", "Document Canvas"
        if lower == "code" or "vscode" in lower:
            return "editor-canvas", "Document", "Editor Canvas"
        if "excel" in lower:
            return "spreadsheet-grid", "Table", "Spreadsheet Grid"
        if "explorer" in lower:
            return "explorer-file-list", "List", "Explorer File List"
        if "paint" in lower:
            return "drawing-canvas", "Canvas", "Drawing Canvas"
        return None

    @staticmethod
    def _surface_node(
        node_id: str,
        role: str,
        name: str,
        parent: str,
    ) -> dict:
        return {
            "node_id": node_id,
            "role": role,
            "name": name,
            "focused": False,
            "enabled": True,
            "bounds": [180, 160, 920, 620],
            "metadata": {
                "sandbox": True,
                "parent": parent,
                "text": "",
            },
        }
