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


class _SequencedWindowsUiaBackend(_StubWindowsUiaBackend):
    def __init__(self, snapshots) -> None:
        super().__init__(controls=[])
        self.snapshots = [list(snapshot) for snapshot in snapshots]
        self.snapshot_index = 0

    def advance_snapshot(self) -> None:
        if self.snapshot_index < len(self.snapshots) - 1:
            self.snapshot_index += 1

    def _live_controls(self):
        index = min(self.snapshot_index, len(self.snapshots) - 1)
        return list(self.snapshots[index])


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
        on_click=None,
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
        self.on_click = on_click
        self.clicked = 0
        self.typed: list[str] = []

    def Click(self, **_kwargs) -> None:
        self.clicked += 1
        if self.on_click is not None:
            self.on_click()

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
        backend = _StubWindowsUiaBackend(b"\x89PNG\r\n\x1a\nsynthetic-payload")

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

    def test_perform_type_resolves_semantic_email_selector(self) -> None:
        control = _FakeControl(
            name="Search Mailbox",
            automation_id="searchBox",
            class_name="Edit",
            role="EditControl",
        )
        node = UiNode(
            node_id="0:0:searchBox",
            role="Edit",
            name="Search Mailbox",
            bounds=(10, 20, 80, 40),
            metadata={
                "automation_id": "searchBox",
                "class_name": "Edit",
                "process_id": 1234,
            },
        )
        backend = _StubWindowsUiaBackend(controls=[(control, node)])

        receipt = json.loads(
            backend.perform(
                UiAction(
                    "type",
                    "email-search-box",
                    "quarterly forecast",
                    metadata={"adapter_family": "email"},
                )
            )
        )

        self.assertEqual(control.typed, ["quarterly forecast"])
        self.assertEqual(receipt["matched_name"], "Search Mailbox")
        self.assertEqual(receipt["selector"], "automation_id=searchBox")
        self.assertEqual(receipt["semantic_selector"], "email-search-box")

    def test_perform_type_resolves_document_canvas_for_notepad(self) -> None:
        control = _FakeControl(
            name="Text editor",
            automation_id="",
            class_name="RichEditD2DPT",
            role="DocumentControl",
        )
        node = UiNode(
            node_id="0:0:notepadDocument",
            role="Document",
            name="Text editor",
            bounds=(10, 20, 400, 200),
            metadata={
                "class_name": "RichEditD2DPT",
                "process_id": 1234,
            },
        )
        backend = _StubWindowsUiaBackend(controls=[(control, node)])

        receipt = json.loads(
            backend.perform(
                UiAction("type", "document-canvas", "hello world")
            )
        )

        self.assertEqual(control.typed, ["hello world"])
        self.assertEqual(receipt["matched_name"], "Text editor")
        self.assertEqual(receipt["semantic_selector"], "document-canvas")

    def test_perform_open_url_resolves_browser_address_bar(self) -> None:
        control = _FakeControl(
            name="Address and search bar",
            automation_id="view_1019",
            class_name="Edit",
            role="EditControl",
        )
        node = UiNode(
            node_id="0:0:browserAddress",
            role="Edit",
            name="Address and search bar",
            bounds=(10, 20, 500, 30),
            metadata={
                "automation_id": "view_1019",
                "class_name": "Edit",
                "process_id": 1234,
            },
        )
        backend = _StubWindowsUiaBackend(controls=[(control, node)])

        receipt = json.loads(
            backend.perform(
                UiAction(
                    "open_url",
                    "browser-address-bar",
                    "https://example.com",
                )
            )
        )

        self.assertEqual(control.typed, ["https://example.com", "{Enter}"])
        self.assertEqual(receipt["status"], "navigated")
        self.assertEqual(receipt["selector"], "automation_id=view_1019")
        self.assertEqual(receipt["semantic_selector"], "browser-address-bar")

    def test_perform_click_enriches_email_send_outcome(self) -> None:
        backend = _SequencedWindowsUiaBackend(
            snapshots=[
                [
                    (
                        _FakeControl(
                            name="Send",
                            automation_id="sendButton",
                            role="ButtonControl",
                        ),
                        UiNode(
                            node_id="0:0:sendButton",
                            role="Button",
                            name="Send",
                            bounds=(10, 20, 80, 40),
                            metadata={
                                "automation_id": "sendButton",
                                "class_name": "Button",
                                "process_id": 1234,
                            },
                        ),
                    )
                ],
                [
                    (
                        _FakeControl(
                            name="Message sent",
                            automation_id="statusText",
                            role="TextControl",
                        ),
                        UiNode(
                            node_id="0:0:statusText",
                            role="Text",
                            name="Message sent",
                            metadata={
                                "automation_id": "statusText",
                                "class_name": "Text",
                                "process_id": 1234,
                            },
                        ),
                    )
                ],
            ]
        )
        send_control, _ = backend.snapshots[0][0]
        send_control.on_click = backend.advance_snapshot

        receipt = json.loads(
            backend.perform(
                UiAction(
                    "click",
                    "email-send-button",
                    metadata={
                        "adapter_family": "email",
                        "recipient": "Alex",
                        "attachment": "C:/tmp/invoice.pdf",
                    },
                )
            )
        )

        self.assertEqual(receipt["email"]["status"], "sent")
        self.assertEqual(receipt["email"]["recipient"], "Alex")
        self.assertEqual(receipt["email"]["attachment"], "invoice.pdf")

    def test_perform_click_enriches_calendar_invite_outcome(self) -> None:
        backend = _SequencedWindowsUiaBackend(
            snapshots=[
                [
                    (
                        _FakeControl(
                            name="Invite Attendees",
                            automation_id="inviteButton",
                            role="ButtonControl",
                        ),
                        UiNode(
                            node_id="0:0:inviteButton",
                            role="Button",
                            name="Invite Attendees",
                            bounds=(10, 20, 80, 40),
                            metadata={
                                "automation_id": "inviteButton",
                                "class_name": "Button",
                                "process_id": 1234,
                            },
                        ),
                    )
                ],
                [
                    (
                        _FakeControl(
                            name="Invitation sent",
                            automation_id="calendarStatus",
                            role="TextControl",
                        ),
                        UiNode(
                            node_id="0:0:calendarStatus",
                            role="Text",
                            name="Invitation sent",
                            metadata={
                                "automation_id": "calendarStatus",
                                "class_name": "Text",
                                "process_id": 1234,
                            },
                        ),
                    )
                ],
            ]
        )
        invite_control, _ = backend.snapshots[0][0]
        invite_control.on_click = backend.advance_snapshot

        receipt = json.loads(
            backend.perform(
                UiAction(
                    "click",
                    "calendar-invite-button",
                    metadata={
                        "adapter_family": "calendar",
                        "event_title": "Zoom invite",
                    },
                )
            )
        )

        self.assertEqual(receipt["calendar"]["status"], "invited")
        self.assertEqual(receipt["calendar"]["event_title"], "Zoom invite")
