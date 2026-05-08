from __future__ import annotations

import ctypes
import io
import json
import platform
import subprocess
import time
from typing import Any

from .base import BackendUnavailable, UiAction, UiNode
from .rust_native_windows_backend import RustNativeWindowsBackend

try:
    from comtypes import COMError  # type: ignore[import-not-found]

    _ComError = COMError
except ImportError:
    _UIA_EXCEPTIONS: tuple[type[BaseException], ...] = (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
else:
    _UIA_EXCEPTIONS = (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        _ComError,
    )


class WindowsUiaBackend:
    """Windows UI Automation backend through in-process UIA/Win32 APIs."""

    name = "windows-uia"

    _MOUSEEVENTF_MOVE = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(
        self,
        powershell_path: str | None = None,
        timeout_seconds: int = 15,
        max_depth: int = 7,
        max_nodes: int = 2000,
        native_fallback: RustNativeWindowsBackend | None = None,
        enable_native_fallback: bool = True,
    ) -> None:
        # Preserve the parameter for compatibility with older call sites and
        # tests, but the backend no longer shells out through PowerShell.
        self.powershell_path = powershell_path
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._native_fallback = native_fallback
        self.enable_native_fallback = enable_native_fallback

    def available(self) -> bool:
        if platform.system() != "Windows":
            return False
        try:
            self._automation_module()
        except BackendUnavailable:
            return False
        return True

    def snapshot(self) -> list[UiNode]:
        self._ensure_available()
        return [node for _control, node in self._live_controls()]

    def capture(self) -> bytes:
        self._ensure_available()
        payload = self._capture_png_bytes()
        if not payload:
            return b""
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BackendUnavailable("Windows screenshot payload was invalid")
        return payload

    def perform(self, action: UiAction) -> str:
        try:
            self._ensure_available()
        except BackendUnavailable as exc:
            return self._native_perform(action, str(exc))
        if action.action_type == "launch_app":
            try:
                return json.dumps(self._launch_app(action), sort_keys=True)
            except (BackendUnavailable, OSError) as exc:
                return self._native_perform(action, str(exc))
        if action.action_type == "hotkey":
            try:
                return json.dumps(self._send_hotkey(action), sort_keys=True)
            except BackendUnavailable as exc:
                return self._native_perform(action, str(exc))
            except _UIA_EXCEPTIONS as exc:
                return self._native_perform(action, str(exc))
        if action.action_type in {"move_cursor", "scroll", "wait"}:
            return self._native_perform(action, "native-only action")

        matched = self._find_target(action.selector)
        if matched is None:
            return self._native_perform(
                action,
                f"No UI element matched selector '{action.selector}'",
            )
        target, node = matched
        action = self._with_target_metadata(action, node)
        focus_error = None
        status = "matched"
        try:
            target.SetFocus()
            status = "focused"
        except _UIA_EXCEPTIONS as exc:
            focus_error = str(exc)

        draw_path: list[dict[str, int]] | None = None
        if action.action_type == "draw_path" and focus_error is not None:
            try:
                target.GetTopLevelControl().SetFocus()
            except _UIA_EXCEPTIONS:
                pass

        if action.action_type in {"invoke", "click"}:
            self._invoke_click(target)
            status = "invoked"
        elif action.action_type in {"type", "set_text", "set_value"}:
            status = self._set_text(target, str(action.value or ""))
        elif action.action_type == "open_url":
            status = self._open_url(target, str(action.value or ""))
        elif action.action_type == "cell_edit":
            status = self._edit_cell(target, action)
        elif action.action_type == "draw_path":
            draw_path = self._draw_path(target, str(action.value or ""))
            status = "drawn"
        elif action.action_type != "focus":
            return self._native_perform(
                action,
                "Windows UI Automation action "
                f"'{action.action_type}' is not supported",
            )

        receipt: dict[str, Any] = {
            "status": status,
            "action_type": action.action_type,
            "selector": action.selector,
            "matched_name": node.name,
            "matched_role": node.role,
            "focus_error": focus_error,
            "draw_path": draw_path,
        }
        if action.action_type == "cell_edit":
            sandbox_receipt = dict(
                action.metadata.get("sandbox_receipt") or {}
            )
            if sandbox_receipt:
                receipt["cell_edit"] = sandbox_receipt
        return json.dumps(receipt, sort_keys=True)

    def _native_perform(self, action: UiAction, reason: str) -> str:
        if not self._can_native_fallback(action):
            raise BackendUnavailable(reason)
        backend = self._native_backend()
        if backend is None:
            raise BackendUnavailable(reason)
        metadata = dict(action.metadata or {})
        metadata.setdefault("uia_fallback_reason", reason)
        fallback_action = UiAction(
            action_type=action.action_type,
            selector=action.selector,
            value=action.value,
            metadata=metadata,
        )
        try:
            receipt = json.loads(backend.perform(fallback_action))
        except BackendUnavailable as exc:
            raise BackendUnavailable(
                f"{reason}; Rust native fallback failed: {exc}"
            ) from exc
        receipt.setdefault("uia_fallback_reason", reason)
        receipt.setdefault("via", "rust-native-windows")
        return json.dumps(receipt, sort_keys=True)

    def _native_backend(self) -> RustNativeWindowsBackend | None:
        if not self.enable_native_fallback:
            return None
        if self._native_fallback is None:
            self._native_fallback = RustNativeWindowsBackend()
        if not self._native_fallback.available():
            return None
        return self._native_fallback

    @staticmethod
    def _can_native_fallback(action: UiAction) -> bool:
        if action.action_type in {
            "launch_app",
            "open_url",
            "hotkey",
            "move_cursor",
            "scroll",
            "wait",
        }:
            return True
        if action.action_type in {
            "click",
            "invoke",
            "type",
            "set_text",
            "set_value",
        }:
            metadata = dict(action.metadata or {})
            return bool(
                ("x" in metadata and "y" in metadata)
                or "bounds" in metadata
                or "bbox" in metadata
                or "," in str(action.selector or "")
            )
        if action.action_type == "draw_path":
            metadata = dict(action.metadata or {})
            return bool("bounds" in metadata or "bbox" in metadata)
        return False

    @staticmethod
    def _with_target_metadata(action: UiAction, node: UiNode) -> UiAction:
        metadata = dict(action.metadata or {})
        if node.bounds is not None:
            metadata.setdefault("bounds", list(node.bounds))
        metadata.setdefault("matched_name", node.name)
        metadata.setdefault("matched_role", node.role)
        return UiAction(
            action_type=action.action_type,
            selector=action.selector,
            value=action.value,
            metadata=metadata,
        )

    def _ensure_available(self) -> None:
        if not self.available():
            raise BackendUnavailable(
                "Windows UI Automation is not available; install the "
                "'uiautomation' package on Windows."
            )

    @staticmethod
    def _automation_module() -> Any:
        try:
            import uiautomation as auto  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable("uiautomation is not installed") from exc
        return auto

    def _capture_png_bytes(self) -> bytes:
        try:
            import mss  # type: ignore[import-not-found]
            from PIL import Image
        except ImportError:
            try:
                from PIL import ImageGrab
            except ImportError as exc:
                raise BackendUnavailable(
                    "No in-process screen capture backend is installed"
                ) from exc
            image = ImageGrab.grab(all_screens=True)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            shot = sct.grab(monitor)
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

    def _live_controls(self) -> list[tuple[Any, UiNode]]:
        auto = self._automation_module()
        try:
            root = auto.GetRootControl()
            top_level = list(root.GetChildren())
        except _UIA_EXCEPTIONS as exc:
            raise BackendUnavailable(
                "Could not enumerate Windows UIA controls"
            ) from exc

        controls: list[tuple[Any, UiNode]] = []
        for child in top_level:
            self._collect_controls(child, 0, "", controls)
            if len(controls) >= self.max_nodes:
                break
        return controls

    def _collect_controls(
        self,
        control: Any,
        depth: int,
        parent_id: str,
        controls: list[tuple[Any, UiNode]],
    ) -> None:
        if depth > self.max_depth or len(controls) >= self.max_nodes:
            return
        try:
            automation_id = str(getattr(control, "AutomationId", "") or "")
            node_id = f"{depth}:{len(controls)}:{automation_id}"
            node = UiNode(
                node_id=node_id,
                role=self._control_role(control),
                name=str(getattr(control, "Name", "") or ""),
                bounds=self._control_bounds(control),
                enabled=bool(getattr(control, "IsEnabled", True)),
                focused=bool(getattr(control, "HasKeyboardFocus", False)),
                metadata={
                    "automation_id": automation_id,
                    "class_name": str(getattr(control, "ClassName", "") or ""),
                    "process_id": getattr(control, "ProcessId", None),
                    "parent": parent_id,
                },
            )
        except _UIA_EXCEPTIONS:
            return

        controls.append((control, node))
        if depth >= self.max_depth or len(controls) >= self.max_nodes:
            return
        try:
            children = list(control.GetChildren())
        except _UIA_EXCEPTIONS:
            return
        for child in children:
            self._collect_controls(child, depth + 1, node.node_id, controls)
            if len(controls) >= self.max_nodes:
                return

    def _find_target(self, selector: str) -> tuple[Any, UiNode] | None:
        cleaned = str(selector or "").strip()
        if not cleaned:
            return None
        for control, node in self._live_controls():
            if self._selector_matches(cleaned, node):
                return control, node
        return None

    @staticmethod
    def _selector_matches(selector: str, node: UiNode) -> bool:
        for clause in selector.split("&&"):
            cleaned = clause.strip()
            if not cleaned:
                continue
            if not WindowsUiaBackend._selector_clause_matches(cleaned, node):
                return False
        return True

    @staticmethod
    def _selector_clause_matches(clause: str, node: UiNode) -> bool:
        values = {
            "name": node.name,
            "automation_id": str(node.metadata.get("automation_id") or ""),
            "role": node.role,
            "class_name": str(node.metadata.get("class_name") or ""),
            "process_id": str(node.metadata.get("process_id") or ""),
            "node_id": node.node_id,
        }
        if "=" in clause:
            field, expected = clause.split("=", 1)
            normalized_field = field.strip().lower()
            if normalized_field in values:
                return WindowsUiaBackend._contains_match(
                    values[normalized_field],
                    expected,
                )
        return any(
            WindowsUiaBackend._contains_match(value, clause)
            for value in values.values()
        )

    @staticmethod
    def _contains_match(actual: Any, expected: Any) -> bool:
        actual_text = str(actual or "").strip().lower()
        expected_text = str(expected or "").strip().lower()
        if not actual_text or not expected_text:
            return False
        return expected_text in actual_text

    @staticmethod
    def _control_role(control: Any) -> str:
        role = str(getattr(control, "ControlTypeName", "Unknown") or "Unknown")
        if role.endswith("Control"):
            return role[: -len("Control")]
        return role.replace("ControlType.", "")

    @staticmethod
    def _control_bounds(control: Any) -> tuple[int, int, int, int] | None:
        try:
            rect = getattr(control, "BoundingRectangle", None)
            if rect is None:
                return None
            width = int(getattr(rect, "width", 0) or 0)
            height = int(getattr(rect, "height", 0) or 0)
            if width <= 0 or height <= 0:
                return None
            return (
                int(getattr(rect, "left", 0) or 0),
                int(getattr(rect, "top", 0) or 0),
                width,
                height,
            )
        except _UIA_EXCEPTIONS:
            return None

    def _launch_app(self, action: UiAction) -> dict[str, Any]:
        launch_target = str(action.value or action.selector or "").strip()
        if not launch_target:
            raise BackendUnavailable(
                "launch_app requires an executable target"
            )
        try:
            process = subprocess.Popen(launch_target)
        except OSError:
            process = subprocess.Popen(launch_target, shell=True)
        return {
            "status": "launched",
            "action_type": action.action_type,
            "selector": action.selector,
            "launched": launch_target,
            "process_id": process.pid,
        }

    def _send_hotkey(self, action: UiAction) -> dict[str, Any]:
        auto = self._automation_module()
        hotkey = str(action.value or "")
        auto.SendKeys(hotkey, waitTime=0.05, charMode=False)
        return {
            "status": "hotkey-sent",
            "action_type": action.action_type,
            "selector": action.selector,
            "value": hotkey,
        }

    def _invoke_click(self, target: Any) -> None:
        try:
            target.Click(simulateMove=False, waitTime=0.05)
            return
        except _UIA_EXCEPTIONS:
            pass
        bounds = self._control_bounds(target)
        if bounds is None:
            raise BackendUnavailable("Target has no clickable bounds")
        left, top, width, height = bounds
        self._mouse_left_click(left + width // 2, top + height // 2)

    def _set_text(self, target: Any, value: str) -> str:
        try:
            target.GetValuePattern().SetValue(value)
            return "value-set"
        except _UIA_EXCEPTIONS:
            target.SendKeys(value, waitTime=0.05)
            return "typed"

    def _open_url(self, target: Any, value: str) -> str:
        self._set_text(target, value)
        try:
            target.SendKeys("{Enter}", waitTime=0.05, charMode=False)
        except _UIA_EXCEPTIONS:
            auto = self._automation_module()
            auto.SendKeys("{Enter}", waitTime=0.05, charMode=False)
        return "navigated"

    def _edit_cell(self, target: Any, action: UiAction) -> str:
        value = str(action.value or "")
        try:
            target.SendKeys(value, waitTime=0.05)
        except _UIA_EXCEPTIONS:
            return self._set_text(target, value)
        return "cell-edited"

    def _draw_path(self, target: Any, path_json: str) -> list[dict[str, int]]:
        bounds = self._control_bounds(target)
        if bounds is None:
            raise BackendUnavailable(
                "Cannot draw on a target with empty bounds"
            )
        left, top, width, height = bounds
        parsed = json.loads(path_json)
        points = parsed.get("points") if isinstance(parsed, dict) else parsed
        if not isinstance(points, list) or len(points) < 2:
            raise BackendUnavailable("draw_path requires at least two points")
        resolved = [
            self._resolve_point(point, left, top, width, height)
            for point in points
        ]
        first = resolved[0]
        self._set_cursor_pos(first["x"], first["y"])
        self._mouse_event(self._MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.04)
        previous = first
        for point in resolved[1:]:
            dx = point["x"] - previous["x"]
            dy = point["y"] - previous["y"]
            steps = max(1, int((max(abs(dx), abs(dy)) / 18.0) + 0.999))
            for step in range(1, steps + 1):
                x = round(previous["x"] + ((dx * step) / steps))
                y = round(previous["y"] + ((dy * step) / steps))
                self._mouse_event(
                    self._MOUSEEVENTF_MOVE,
                    dx=x - previous["x"],
                    dy=y - previous["y"],
                )
                previous = {"x": int(x), "y": int(y)}
                time.sleep(0.008)
        self._mouse_event(self._MOUSEEVENTF_LEFTUP)
        return resolved

    @staticmethod
    def _resolve_point(
        point: Any,
        origin_x: int,
        origin_y: int,
        width: int,
        height: int,
    ) -> dict[str, int]:
        if isinstance(point, dict):
            raw_x = float(point.get("x", 0.0))
            raw_y = float(point.get("y", 0.0))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            raw_x = float(point[0])
            raw_y = float(point[1])
        else:
            raise BackendUnavailable("draw_path contains an invalid point")
        return {
            "x": WindowsUiaBackend._resolve_coordinate(raw_x, origin_x, width),
            "y": WindowsUiaBackend._resolve_coordinate(
                raw_y,
                origin_y,
                height,
            ),
        }

    @staticmethod
    def _resolve_coordinate(raw: float, origin: int, size: int) -> int:
        if abs(raw) <= 1.0:
            return int(round(origin + (raw * size)))
        return int(round(raw))

    @staticmethod
    def _set_cursor_pos(x: int, y: int) -> None:
        user32 = ctypes.windll.user32
        if not user32.SetCursorPos(int(x), int(y)):
            raise BackendUnavailable("Failed to move the mouse cursor")

    def _mouse_left_click(self, x: int, y: int) -> None:
        self._set_cursor_pos(x, y)
        self._mouse_event(self._MOUSEEVENTF_LEFTDOWN)
        self._mouse_event(self._MOUSEEVENTF_LEFTUP)

    @staticmethod
    def _mouse_event(flags: int, dx: int = 0, dy: int = 0) -> None:
        ctypes.windll.user32.mouse_event(int(flags), int(dx), int(dy), 0, 0)
