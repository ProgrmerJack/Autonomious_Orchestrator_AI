"""Linux AT-SPI accessibility backend.

Reads the accessibility tree exposed by GTK/Qt/Electron apps on Linux
through AT-SPI (D-Bus).  Mirrors the surface of
:class:`agentos_orchestrator.os_control.windows_uia_backend.WindowsUiaBackend`
so the rest of the stack can treat all OS backends uniformly.

* When AT-SPI is unavailable (no D-Bus, no pyatspi, non-Linux), the
  backend reports ``available() -> False`` and ``snapshot()`` returns
  an empty list.  This is the standard *degrade, never refuse* pattern.
* Actions are dispatched via ``pyatspi.Action`` interfaces when the
  element exposes them; otherwise we fall back to coordinate input via
  ``xdotool`` / ``ydotool`` when present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

from .base import UiAction, UiNode


class LinuxAtSpiBackend:
    """AT-SPI accessibility backend (Linux)."""

    name = "linux-atspi"

    def __init__(self) -> None:
        self._import_error: str = ""
        self._registry: Any = None
        if sys.platform == "linux":
            try:
                import pyatspi  # type: ignore[import-not-found]

                self._registry = pyatspi.Registry
                self._pyatspi = pyatspi
            except Exception as exc:
                self._import_error = repr(exc)
        else:
            self._import_error = f"not-linux: {sys.platform}"

    def available(self) -> bool:
        return self._registry is not None and not self._import_error

    def snapshot(self) -> list[UiNode]:
        if not self.available():
            return []
        nodes: list[UiNode] = []
        try:
            desktop = self._pyatspi.Registry.getDesktop(0)
        except Exception:
            return []

        def _walk(accessible: Any, depth: int) -> None:
            if depth > 6 or len(nodes) >= 4096:
                return
            try:
                role_name = accessible.getRoleName() or ""
                name = accessible.name or ""
                states = accessible.getState()
                enabled = (
                    bool(states.contains(self._pyatspi.STATE_ENABLED))
                    if states
                    else True
                )
                focused = (
                    bool(states.contains(self._pyatspi.STATE_FOCUSED))
                    if states
                    else False
                )
                bounds_tuple: tuple[int, int, int, int] | None = None
                try:
                    component = accessible.queryComponent()
                    extents = component.getExtents(self._pyatspi.DESKTOP_COORDS)
                    bounds_tuple = (
                        int(extents.x),
                        int(extents.y),
                        int(extents.width),
                        int(extents.height),
                    )
                except Exception:
                    bounds_tuple = None
                nodes.append(
                    UiNode(
                        node_id=str(id(accessible)),
                        role=role_name,
                        name=name,
                        bounds=bounds_tuple,
                        enabled=enabled,
                        focused=focused,
                        metadata={
                            "depth": str(depth),
                            "source": "atspi",
                        },
                    ),
                )
            except Exception:
                return
            try:
                for i in range(accessible.childCount):
                    child = accessible.getChildAtIndex(i)
                    if child is not None:
                        _walk(child, depth + 1)
            except Exception:
                return

        try:
            for i in range(desktop.childCount):
                app = desktop.getChildAtIndex(i)
                if app is not None:
                    _walk(app, 0)
        except Exception:
            pass
        return nodes

    def perform(self, action: UiAction) -> str:
        if not self.available():
            return "atspi-unavailable"
        # Coordinate-based fallback via xdotool/ydotool
        atype = (action.action_type or "").lower()
        if atype in {"click", "left_click"} and action.metadata.get("x") is not None:
            x = str(action.metadata.get("x", ""))
            y = str(action.metadata.get("y", ""))
            for tool in ("xdotool", "ydotool"):
                if shutil.which(tool):
                    try:
                        if tool == "xdotool":
                            subprocess.run(
                                [tool, "mousemove", x, y, "click", "1"],
                                check=False,
                                timeout=5,
                            )
                        else:
                            subprocess.run(
                                [tool, "mousemove", "--absolute", x, y],
                                check=False,
                                timeout=5,
                            )
                            subprocess.run(
                                [tool, "click", "1"],
                                check=False,
                                timeout=5,
                            )
                        return f"atspi:click:{x},{y}"
                    except Exception:
                        continue
        return f"atspi:unimplemented:{atype}"


def is_default_for_platform() -> bool:
    """Return True if AT-SPI should be the default backend on this host."""
    return sys.platform == "linux" and bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
