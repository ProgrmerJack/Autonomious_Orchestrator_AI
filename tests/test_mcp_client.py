from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from agentos_orchestrator.mcp import McpProtocolError, McpServerConfig, McpStdioClient


_SERVER_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import sys
    import threading
    import time
    from pathlib import Path

    log_path = Path(os.environ["MCP_TEST_LOG"])
    initialized = False
    cancelled = set()

    def send(payload):
        print(json.dumps(payload), flush=True)

    def delayed_tool_call(request_id):
        time.sleep(0.5)
        if request_id in cancelled:
            return
        send({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}})

    for raw in iter(input, ""):
        message = json.loads(raw)
        method = message.get("method")
        if method == "initialize":
            print("server warming", file=sys.stderr, flush=True)
            send({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"phase": "warming"}})
            send({
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"protocolVersion": message["params"]["protocolVersion"]},
            })
        elif method == "notifications/initialized":
            initialized = True
        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"initialized": initialized, "tools": [{"name": "ping"}]},
            })
        elif method == "tools/call" and message.get("params", {}).get("name") == "slow":
            threading.Thread(target=delayed_tool_call, args=(message["id"],), daemon=True).start()
        elif method == "notifications/cancelled":
            request_id = int(message.get("params", {}).get("requestId", 0))
            cancelled.add(request_id)
            log_path.write_text(json.dumps(message["params"]), encoding="utf-8")
        elif method == "tools/call":
            send({"jsonrpc": "2.0", "id": message["id"], "result": message.get("params")})
    """
)


class McpStdioClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.script_path = root / "fake_mcp_server.py"
        self.log_path = root / "cancelled.json"
        self.script_path.write_text(_SERVER_SCRIPT, encoding="utf-8")

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_initialize_sends_initialized_notification(self) -> None:
        config = McpServerConfig(
            name="fake",
            command=[sys.executable, "-u", str(self.script_path)],
            env={"MCP_TEST_LOG": str(self.log_path)},
            request_timeout_seconds=0.2,
        )
        async with McpStdioClient(config) as client:
            result = await client.initialize()
            tools = await client.list_tools()
            stderr_messages = client.stderr_messages()

        self.assertEqual(result["protocolVersion"], config.protocol_version)
        self.assertTrue(tools["initialized"])
        self.assertIn("server warming", stderr_messages)

    async def test_request_timeout_sends_cancel_notification(self) -> None:
        config = McpServerConfig(
            name="fake",
            command=[sys.executable, "-u", str(self.script_path)],
            env={"MCP_TEST_LOG": str(self.log_path)},
            request_timeout_seconds=0.1,
        )
        async with McpStdioClient(config) as client:
            await client.initialize()
            with self.assertRaises(McpProtocolError):
                await client.call_tool("slow", {})

        payload = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertIn("timed out", payload["reason"])
        self.assertGreater(payload["requestId"], 0)
