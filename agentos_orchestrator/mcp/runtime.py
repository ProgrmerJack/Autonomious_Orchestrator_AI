from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from .client import McpServerConfig, McpStdioClient


@dataclass(slots=True)
class McpResearchServer:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    research_tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: float = 20.0
    protocol_version: str = "2025-11-25"


@dataclass(slots=True)
class McpResearchHit:
    provider: str
    title: str
    url: str
    abstract: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_mcp_research_servers_from_env() -> list[McpResearchServer]:
    raw = os.environ.get("AGENTOS_MCP_SERVERS_JSON", "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    servers: list[McpResearchServer] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        command = [str(part).strip() for part in list(item.get("command") or [])]
        command = [part for part in command if part]
        if not name or not command:
            continue
        env = item.get("env") if isinstance(item.get("env"), dict) else {}
        arguments = (
            item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        )
        servers.append(
            McpResearchServer(
                name=name,
                command=command,
                env={str(key): str(value) for key, value in env.items()},
                research_tool=str(item.get("research_tool") or "").strip() or None,
                arguments=dict(arguments),
                request_timeout_seconds=float(
                    item.get("request_timeout_seconds") or 20.0
                ),
                protocol_version=str(item.get("protocol_version") or "2025-11-25"),
            )
        )
    return servers


def run_mcp_research_query(
    query: str,
    limit: int = 5,
) -> tuple[list[McpResearchHit], list[dict[str, Any]]]:
    return _run_async(_collect_research_hits(query, limit=limit))


def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _collect_research_hits(
    query: str,
    limit: int = 5,
) -> tuple[list[McpResearchHit], list[dict[str, Any]]]:
    hits: list[McpResearchHit] = []
    diagnostics: list[dict[str, Any]] = []
    for server in load_mcp_research_servers_from_env():
        config = McpServerConfig(
            name=server.name,
            command=server.command,
            env=server.env,
            protocol_version=server.protocol_version,
            request_timeout_seconds=server.request_timeout_seconds,
        )
        client: McpStdioClient | None = None
        try:
            async with McpStdioClient(config) as client:
                await client.initialize()
                tools_payload = await client.list_tools()
                tool_name = _resolve_tool_name(server, tools_payload)
                if not tool_name:
                    diagnostics.append(
                        {
                            "server": server.name,
                            "status": "no-research-tool",
                            "stderr": client.stderr_messages(),
                        }
                    )
                    continue
                arguments = dict(server.arguments)
                arguments.setdefault("query", query)
                arguments.setdefault("limit", limit)
                result = await client.call_tool(tool_name, arguments)
                normalized = _normalize_mcp_result(server.name, tool_name, result)
                hits.extend(normalized)
                diagnostics.append(
                    {
                        "server": server.name,
                        "status": "ok",
                        "tool": tool_name,
                        "result_count": len(normalized),
                        "stderr": client.stderr_messages(),
                    }
                )
        except Exception as exc:
            diagnostics.append(
                {
                    "server": server.name,
                    "status": "error",
                    "error": str(exc),
                    "stderr": client.stderr_messages() if client is not None else [],
                }
            )
    return hits, diagnostics


def _resolve_tool_name(server: McpResearchServer, tools_payload: Any) -> str | None:
    if server.research_tool:
        return server.research_tool
    tools = _tool_entries(tools_payload)
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if any(token in name.lower() for token in ("research", "search", "query")):
            return name
    if tools:
        return str(tools[0].get("name") or "").strip() or None
    return None


def _tool_entries(tools_payload: Any) -> list[dict[str, Any]]:
    if isinstance(tools_payload, dict) and isinstance(tools_payload.get("tools"), list):
        return [item for item in tools_payload["tools"] if isinstance(item, dict)]
    if isinstance(tools_payload, list):
        return [item for item in tools_payload if isinstance(item, dict)]
    return []


def _normalize_mcp_result(
    server_name: str,
    tool_name: str,
    payload: Any,
) -> list[McpResearchHit]:
    items = _result_items(payload)
    hits: list[McpResearchHit] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            hits.append(
                McpResearchHit(
                    provider=f"mcp:{server_name}",
                    title=f"{tool_name} result {index}",
                    url=f"mcp://{server_name}/{tool_name}/{index}",
                    abstract=text,
                    metadata={"tool": tool_name},
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        title = str(
            item.get("title") or item.get("name") or f"{tool_name} result {index}"
        ).strip()
        url = str(
            item.get("url")
            or item.get("uri")
            or f"mcp://{server_name}/{tool_name}/{index}"
        ).strip()
        abstract = str(
            item.get("abstract")
            or item.get("summary")
            or item.get("snippet")
            or item.get("text")
            or ""
        ).strip()
        hits.append(
            McpResearchHit(
                provider=str(item.get("provider") or f"mcp:{server_name}"),
                title=title,
                url=url,
                abstract=abstract,
                metadata={"tool": tool_name, "raw": item},
            )
        )
    return hits


def _result_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "items", "sources"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    content = payload.get("content")
    if isinstance(content, list):
        text_parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        if not text_parts:
            return []
        combined = "\n".join(text_parts)
        try:
            return _result_items(json.loads(combined))
        except json.JSONDecodeError:
            return [combined]
    return []
