from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any

from agentos_orchestrator.cognition.benchmark_scenarios import (
    load_golden_traces,
    replay_golden_traces,
)
from agentos_orchestrator.cognition.live_fire_eval import LiveFireEvalConfig
from agentos_orchestrator.core.types import utc_now
from agentos_orchestrator.gateway.channels import ChannelMessage
from agentos_orchestrator.os_control import (
    VirtualDesktopSandboxBackend,
    WindowsUiaBackend,
)


def _client_host(connection: Any) -> str:
    client = getattr(connection, "client", None)
    if client is None:
        return ""
    return str(getattr(client, "host", "") or "")


def _request_origin(connection: Any) -> str | None:
    headers = getattr(connection, "headers", None)
    if headers is None:
        return None
    return str(headers.get("origin") or headers.get("referer") or "") or None


def _extract_session_token(headers: Any) -> str:
    authorization = str(headers.get("authorization") or "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return str(headers.get("x-agentos-session") or "")


def _requires_unsafe_ack(path: str, method: str) -> bool:
    return method.upper() not in {"GET", "HEAD", "OPTIONS"} and path != "/auth/session"


def _pc_backend(name: str, state_path: str | Path):
    if name == "windows-uia":
        return WindowsUiaBackend()
    if name == "virtual-desktop-sandbox":
        return VirtualDesktopSandboxBackend(
            Path(state_path).with_name("virtual_desktop_sandbox.json")
        )
    raise ValueError(
        f"Unknown PC backend: {name}. Expected 'windows-uia' or "
        "'virtual-desktop-sandbox'."
    )


def _pc_backend_status(state_path: str | Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for name in ("windows-uia", "virtual-desktop-sandbox"):
        try:
            backend = _pc_backend(name, state_path)
            available = backend.available()
        except (OSError, RuntimeError, ValueError) as exc:
            statuses.append({"name": name, "available": False, "error": str(exc)})
        else:
            statuses.append({"name": name, "available": available})
    return statuses


def _record_channel_delivery(
    deliveries: deque[dict[str, Any]],
    message: ChannelMessage,
    response: dict[str, Any],
) -> None:
    deliveries.appendleft(
        {
            "created_at": utc_now(),
            "channel": message.channel,
            "sender_id": message.sender_id,
            "text": message.text,
            "status": response.get("status"),
            "response": response,
            "metadata": message.metadata,
        }
    )


def _golden_traces_payload(workspace_root: Path) -> dict[str, Any]:
    traces = load_golden_traces(workspace_root)
    return {"trace_count": len(traces), "traces": traces}


def _replay_benchmarks(workspace_root: Path, trace_id: str) -> dict[str, Any]:
    return replay_golden_traces(workspace_root, trace_id)


def _live_fire_config(payload: dict[str, Any]) -> LiveFireEvalConfig:
    config = LiveFireEvalConfig(
        run_id=str(payload.get("run_id") or ""),
        max_tasks=_optional_int(payload.get("max_tasks")),
        surfaces=tuple(_string_list(payload.get("surfaces"))),
        intents=tuple(_string_list(payload.get("intents"))),
        windows_safe_pack=bool(payload.get("windows_safe_pack", False)),
        repeat=max(1, int(payload.get("repeat") or 1)),
        promote_failures=bool(payload.get("promote_failures", True)),
        promote_after=int(payload.get("promote_after") or 1),
        replay_limit=int(payload.get("replay_limit") or 10),
        training_output=str(payload.get("training_output") or ""),
    )
    config.heldout_from = str(payload.get("heldout_from") or "")
    return config


def _workspace_root(state_path: str | Path) -> Path:
    parent = Path(state_path).resolve(strict=False).parent
    if parent.name == ".agentos":
        return parent.parent
    return Path.cwd()


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _normalize_depth(depth: object) -> str:
    allowed = {"quick", "standard", "multi-hour", "adaptive"}
    value = str(depth or "adaptive").strip().lower()
    return value if value in allowed else "adaptive"


def _objective_with_depth(objective: str, depth: str) -> str:
    if not objective:
        return objective
    cleaned = re.sub(
        r"\[(quick|standard|multi-hour|adaptive)\]\s*",
        "",
        objective,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        return objective.strip()
    return f"[{depth}] {cleaned}"


def _extract_depth_from_objective(objective: str) -> str | None:
    if not objective:
        return None
    match = re.match(
        r"\s*\[(quick|standard|multi-hour|adaptive)\]\s*",
        objective,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return _normalize_depth(match.group(1))


def _research_payload(run_id: str, workspace_root: Path) -> dict[str, Any]:
    if re.fullmatch(r"run_[A-Za-z0-9_]+", run_id) is None:
        raise FileNotFoundError("invalid run id")
    research_dir = workspace_root / "runs" / run_id / "research"
    if not research_dir.exists():
        raise FileNotFoundError("research artifacts not found")
    brief_path = research_dir / "brief.md"
    sources_path = research_dir / "sources.json"
    sources: list[dict[str, Any]] = []
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "brief": _read_optional_text(brief_path),
        "sources": sources,
        "artifacts": [
            str(path.relative_to(workspace_root))
            for path in sorted(research_dir.iterdir())
            if path.is_file()
        ],
    }


def _json_or_text(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _blocked_status(requires_approval: bool) -> str:
    return "approval_required" if requires_approval else "blocked"


def _read_optional_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
