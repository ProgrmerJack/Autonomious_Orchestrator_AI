"""macOS AX (Accessibility) API backend.

Reads the AXUIElement accessibility tree exposed by Cocoa, AppKit,
SwiftUI, Catalyst, and Electron apps on macOS.  Mirrors the surface of
:class:`agentos_orchestrator.os_control.windows_uia_backend.WindowsUiaBackend`.

* Available only when running on macOS *and* ``pyobjc-framework-Cocoa``
  + ``pyobjc-framework-ApplicationServices`` are installed *and* the
  current process has been granted Accessibility permission in
  System Settings → Privacy & Security → Accessibility.
* When unavailable, ``available() -> False`` and ``snapshot()`` returns
  an empty list (degrade, never refuse).
* Actions fall back to coordinate input via ``cliclick`` when present
  or via CoreGraphics ``CGEventCreateMouseEvent`` when pyobjc is loaded.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any

from .base import UiAction, UiNode


class MacOsAxBackend:
    """macOS AX API accessibility backend."""

    name = "macos-ax"

    def __init__(self) -> None:
        self._import_error: str = ""
        self._ax: Any = None
        self._app_services: Any = None
        if sys.platform == "darwin":
            try:
                import ApplicationServices  # type: ignore[import-not-found]
                import HIServices  # type: ignore[import-not-found,unused-ignore]

                self._app_services = ApplicationServices
                self._hi = HIServices
            except Exception as exc:
                self._import_error = repr(exc)
        else:
            self._import_error = f"not-darwin: {sys.platform}"

    def available(self) -> bool:
        if self._import_error:
            return False
        if self._app_services is None:
            return False
        try:
            trusted = self._app_services.AXIsProcessTrusted()
            return bool(trusted)
        except Exception:
            return False

    def snapshot(self) -> list[UiNode]:
        if not self.available():
            return []
        ax = self._app_services
        nodes: list[UiNode] = []

        try:
            system = ax.AXUIElementCreateSystemWide()
        except Exception:
            return []

        def _attr(elem: Any, name: str) -> Any:
            try:
                err, value = ax.AXUIElementCopyAttributeValue(elem, name, None)
                if err == 0:
                    return value
            except Exception:
                return None
            return None

        def _bounds_for(elem: Any) -> tuple[int, int, int, int] | None:
            pos = _attr(elem, "AXPosition")
            size = _attr(elem, "AXSize")
            if pos is None or size is None:
                return None
            try:
                return (
                    int(pos.x),
                    int(pos.y),
                    int(size.width),
                    int(size.height),
                )
            except Exception:
                return None

        def _walk(elem: Any, depth: int) -> None:
            if depth > 6 or len(nodes) >= 4096:
                return
            role = str(_attr(elem, "AXRole") or "")
            title = str(_attr(elem, "AXTitle") or _attr(elem, "AXValue") or "")
            bounds_tuple = _bounds_for(elem)
            try:
                nodes.append(
                    UiNode(
                        node_id=str(id(elem)),
                        role=role,
                        name=title,
                        bounds=bounds_tuple,
                        enabled=True,
                        focused=False,
                        metadata={"depth": str(depth), "source": "ax"},
                    ),
                )
            except Exception:
                return
            children = _attr(elem, "AXChildren")
            if children:
                try:
                    for child in list(children):
                        _walk(child, depth + 1)
                except Exception:
                    return

        # Focused application is the practical starting point
        focused_app = _attr(system, "AXFocusedApplication")
        if focused_app is not None:
            _walk(focused_app, 0)
        return nodes

    def perform(self, action: UiAction) -> str:
        if not self.available():
            return "ax-unavailable"
        atype = (action.action_type or "").lower()
        x = action.metadata.get("x")
        y = action.metadata.get("y")
        if atype in {"click", "left_click"} and x is not None and y is not None:
            if shutil.which("cliclick"):
                try:
                    subprocess.run(
                        ["cliclick", f"c:{int(x)},{int(y)}"],
                        check=False,
                        timeout=5,
                    )
                    return f"ax:click:{int(x)},{int(y)}"
                except Exception:
                    pass
            # CoreGraphics fallback
            try:
                import Quartz  # type: ignore[import-not-found]

                point = Quartz.CGPoint(int(x), int(y))
                evt_down = Quartz.CGEventCreateMouseEvent(
                    None,
                    Quartz.kCGEventLeftMouseDown,
                    point,
                    Quartz.kCGMouseButtonLeft,
                )
                evt_up = Quartz.CGEventCreateMouseEvent(
                    None,
                    Quartz.kCGEventLeftMouseUp,
                    point,
                    Quartz.kCGMouseButtonLeft,
                )
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt_down)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt_up)
                return f"ax:cg-click:{int(x)},{int(y)}"
            except Exception:
                pass
        return f"ax:unimplemented:{atype}"


def is_default_for_platform() -> bool:
    """Return True if AX should be the default backend on this host."""
    return sys.platform == "darwin"
