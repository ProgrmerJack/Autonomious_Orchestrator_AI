from __future__ import annotations

import asyncio
from collections import deque
import contextlib
import json
import os
from dataclasses import dataclass, field
from itertools import count
from typing import Any


@dataclass(slots=True)
class McpServerConfig:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    protocol_version: str = "2025-11-25"
    request_timeout_seconds: float = 20.0


class McpProtocolError(RuntimeError):
    pass


class McpStdioClient:
    """Minimal MCP JSON-RPC client over stdio transport."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._ids = count(1)
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._initialized = False
        self._stderr_lines: deque[str] = deque(maxlen=50)

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
            env={**os.environ, **self.config.env} if self.config.env else None,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def close(self) -> None:
        if self._process is None:
            return
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
        self._fail_pending(McpProtocolError("MCP client is closing"))
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()
        self._process = None
        self._initialized = False

    async def request(
        self,
        method: str,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> Any:
        if self._process is None:
            await self.start()
        if self._process is None or self._process.stdin is None:
            raise McpProtocolError("MCP process is not writable")

        request_id = next(self._ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        self._process.stdin.write(json.dumps(payload).encode() + b"\n")
        await self._process.stdin.drain()
        effective_timeout = (
            self.config.request_timeout_seconds if timeout is None else timeout
        )
        try:
            return await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            await self._notify(
                "notifications/cancelled",
                {
                    "requestId": request_id,
                    "reason": f"timed out after {effective_timeout} seconds",
                },
            )
            raise McpProtocolError(
                self._decorate_error(f"MCP request timed out: {method}")
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict | None = None) -> None:
        if self._process is None:
            await self.start()
        if self._process is None or self._process.stdin is None:
            raise McpProtocolError("MCP process is not writable")
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        self._process.stdin.write(json.dumps(payload).encode() + b"\n")
        await self._process.stdin.drain()

    async def _read_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    raise McpProtocolError(
                        self._decorate_error("MCP server closed stdout")
                    )
                response = json.loads(line.decode())
                response_id = response.get("id")
                if response_id is None:
                    continue
                future = self._pending.get(int(response_id))
                if future is None or future.done():
                    continue
                if "error" in response:
                    future.set_exception(
                        McpProtocolError(
                            self._decorate_error(json.dumps(response["error"]))
                        )
                    )
                    continue
                future.set_result(response.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(exc)

    async def _read_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    return
                text = line.decode(errors="replace").strip()
                if text:
                    self._stderr_lines.append(text)
        except asyncio.CancelledError:
            raise

    def stderr_messages(self) -> list[str]:
        return list(self._stderr_lines)

    def _decorate_error(self, message: str) -> str:
        stderr_tail = self.stderr_messages()
        if not stderr_tail:
            return message
        return f"{message}. stderr: {' | '.join(stderr_tail[-5:])}"

    def _fail_pending(self, exc: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    async def initialize(self) -> Any:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": self.config.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": "agentos-orchestrator",
                    "version": "0.1.0",
                },
            },
        )
        await self._notify("notifications/initialized")
        self._initialized = True
        return result

    async def list_tools(self) -> Any:
        return await self.request("tools/list")

    async def call_tool(self, name: str, arguments: dict) -> Any:
        return await self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
