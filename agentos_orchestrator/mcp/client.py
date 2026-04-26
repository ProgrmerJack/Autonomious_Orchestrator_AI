from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from itertools import count
from typing import Any


@dataclass(slots=True)
class McpServerConfig:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


class McpProtocolError(RuntimeError):
    pass


class McpStdioClient:
    """Minimal MCP JSON-RPC client over stdio transport."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._ids = count(1)
        self._process: asyncio.subprocess.Process | None = None

    async def __aenter__(self) -> "McpStdioClient":
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._process is not None:
            return
        self._process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.config.env or None,
        )

    async def close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        await self._process.wait()
        self._process = None

    async def request(self, method: str, params: dict | None = None) -> Any:
        if self._process is None:
            await self.start()
        if self._process is None or self._process.stdin is None:
            raise McpProtocolError("MCP process is not writable")
        if self._process.stdout is None:
            raise McpProtocolError("MCP process is not readable")

        request_id = next(self._ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self._process.stdin.write(json.dumps(payload).encode() + b"\n")
        await self._process.stdin.drain()

        line = await self._process.stdout.readline()
        if not line:
            raise McpProtocolError("MCP server closed stdout")
        response = json.loads(line.decode())
        if response.get("id") != request_id:
            raise McpProtocolError("MCP response id did not match request")
        if "error" in response:
            raise McpProtocolError(json.dumps(response["error"]))
        return response.get("result")

    async def initialize(self) -> Any:
        return await self.request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {
                    "name": "agentos-orchestrator",
                    "version": "0.1.0",
                },
            },
        )

    async def list_tools(self) -> Any:
        return await self.request("tools/list")

    async def call_tool(self, name: str, arguments: dict) -> Any:
        return await self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
