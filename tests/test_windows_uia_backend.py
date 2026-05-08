from __future__ import annotations

import base64
import unittest

from agentos_orchestrator.os_control.base import BackendUnavailable
from agentos_orchestrator.os_control.windows_uia_backend import WindowsUiaBackend


class _StubWindowsUiaBackend(WindowsUiaBackend):
    def __init__(self, output: str) -> None:
        super().__init__(powershell_path="pwsh")
        self.output = output
        self.last_script = ""

    def available(self) -> bool:
        return True

    def _run_text(self, script: str) -> str:
        self.last_script = script
        return self.output


class WindowsUiaBackendTests(unittest.TestCase):
    def test_capture_decodes_png_payload(self) -> None:
        backend = _StubWindowsUiaBackend(
            base64.b64encode(b"fake-png-bytes").decode("ascii")
        )

        payload = backend.capture()

        self.assertEqual(payload, b"fake-png-bytes")
        self.assertIn("CopyFromScreen", backend.last_script)

    def test_capture_rejects_invalid_payload(self) -> None:
        backend = _StubWindowsUiaBackend("not-base64")

        with self.assertRaises(BackendUnavailable):
            backend.capture()
