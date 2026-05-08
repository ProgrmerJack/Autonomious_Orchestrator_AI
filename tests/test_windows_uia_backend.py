from __future__ import annotations

import json
import unittest

from agentos_orchestrator.os_control.base import BackendUnavailable
from agentos_orchestrator.os_control.base import UiAction, UiNode
from agentos_orchestrator.os_control.windows_uia_backend import (
    WindowsUiaBackend,
)


class _StubWindowsUiaBackend(WindowsUiaBackend):
    def __init__(self, capture_payload: bytes = b"", controls=None) -> None:
        super().__init__(powershell_path="pwsh")
        self.capture_payload = capture_payload
        self.controls = list(controls or [])

    def available(self) -> bool:
        return True

    def _capture_png_bytes(self) -> bytes:
        return self.capture_payload

    def _live_controls(self):
        return list(self.controls)


class _FakeValuePattern:
    def __init__(self, control) -> None:
        self.control = control

    def SetValue(self, value: str) -> None:
        self.control.typed.append(value)


class _FakeControl:
    def __init__(
        self,
        *,
        name: str = "Save",
        automation_id: str = "saveButton",
        class_name: str = "Button",
        role: str = "ButtonControl",
        allow_value_pattern: bool = False,
    ) -> None:
        self.Name = name
        self.AutomationId = automation_id
        self.ClassName = class_name
        self.ControlTypeName = role
        self.ProcessId = 1234
        self.IsEnabled = True
        self.HasKeyboardFocus = False
        self.BoundingRectangle = type(
            "Rect",
            (),
            {"left": 10, "top": 20, "width": 80, "height": 40},
        )()
        self.allow_value_pattern = allow_value_pattern
        self.clicked = 0
        self.typed: list[str] = []

    def Click(self, **_kwargs) -> None:
        self.clicked += 1

    def SetFocus(self) -> bool:
        self.HasKeyboardFocus = True
        return True

    def SendKeys(self, text: str, **_kwargs) -> None:
        self.typed.append(text)

    def GetValuePattern(self):
        if not self.allow_value_pattern:
            raise RuntimeError("no value pattern")
        return _FakeValuePattern(self)

    def GetTopLevelControl(self):
        return self


class WindowsUiaBackendTests(unittest.TestCase):
    def test_capture_decodes_png_payload(self) -> None:
        backend = _StubWindowsUiaBackend(
            b"\x89PNG\r\n\x1a\nsynthetic-payload"
        )

        payload = backend.capture()

        self.assertEqual(payload, b"\x89PNG\r\n\x1a\nsynthetic-payload")

    def test_capture_rejects_invalid_payload(self) -> None:
        backend = _StubWindowsUiaBackend(b"not-png")

        with self.assertRaises(BackendUnavailable):
            backend.capture()

    def test_perform_click_uses_matched_control_in_process(self) -> None:
        control = _FakeControl()
        node = UiNode(
            node_id="0:0:saveButton",
            role="Button",
            name="Save",
            metadata={
                "automation_id": "saveButton",
                "class_name": "Button",
                "process_id": 1234,
            },
        )
        backend = _StubWindowsUiaBackend(controls=[(control, node)])

        receipt = json.loads(
            backend.perform(UiAction("click", "automation_id=saveButton"))
        )

        self.assertEqual(control.clicked, 1)
        self.assertEqual(receipt["status"], "invoked")
        self.assertEqual(receipt["matched_name"], "Save")

    def test_perform_type_falls_back_to_sendkeys_without_value_pattern(
        self,
    ) -> None:
        control = _FakeControl(name="Search", automation_id="searchBox")
        node = UiNode(
            node_id="0:0:searchBox",
            role="Edit",
            name="Search",
            metadata={
                "automation_id": "searchBox",
                "class_name": "Edit",
                "process_id": 1234,
            },
        )
        backend = _StubWindowsUiaBackend(controls=[(control, node)])

        receipt = json.loads(
            backend.perform(UiAction("type", "name=Search", "hello world"))
        )

        self.assertEqual(control.typed, ["hello world"])
        self.assertEqual(receipt["status"], "typed")
