"""Per-run heartbeat WebSocket tail.

Adds ``/ws/runs/{run_id}/heartbeat`` to the dashboard FastAPI app.
Streams JSON frames as ``runs/<run_id>/heartbeat.json`` changes on disk,
plus snapshots of any sibling event-log JSON files (e.g.
``run_progress.json``, ``pc/frontier_graph.json``).

The route is intentionally a *thin file tailer* — it does not depend on
the in-memory ``event_hub``, so it survives orchestrator restarts and
works for offline replay of completed runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Any

from agentos_orchestrator.gateway.dashboard_support import (
    _client_host,
    _request_origin,
    _workspace_root,
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def _safe_runs_root() -> Path:
    return _workspace_root() / "runs"


def _safe_run_dir(run_id: str) -> Path | None:
    if not _RUN_ID_RE.match(run_id):
        return None
    candidate = (_safe_runs_root() / run_id).resolve()
    try:
        candidate.relative_to(_safe_runs_root().resolve())
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _read_json_safe(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"_raw_excerpt": text[:1024]}


def register_run_heartbeat_route(app: Any, auth: Any) -> None:
    """Register ``/ws/runs/{run_id}/heartbeat`` on *app*.

    *app* is a FastAPI application; *auth* is the
    :class:`GatewaySecurityManager` used by the parent dashboard.
    """
    import fastapi  # local import to keep this module optional
    from agentos_orchestrator.gateway.auth import GatewaySecurityError

    async def heartbeat(websocket: Any, run_id: str) -> None:
        try:
            auth.require_session(
                session_token=str(websocket.query_params.get("session_token") or ""),
                client_host=_client_host(websocket),
                origin=_request_origin(websocket),
                csrf_token=str(websocket.query_params.get("csrf_token") or ""),
                require_csrf=True,
            )
        except GatewaySecurityError as exc:
            await websocket.close(code=4401, reason=str(exc))
            return
        run_dir = _safe_run_dir(run_id)
        if run_dir is None:
            await websocket.close(code=4404, reason="run not found")
            return
        await websocket.accept()
        watched = {
            "heartbeat": run_dir / "heartbeat.json",
            "run_progress": run_dir / "run_progress.json",
            "frontier_graph": run_dir / "pc" / "frontier_graph.json",
            "final_report": run_dir / "final_report.json",
        }
        last_mtimes: dict[str, float] = {k: 0.0 for k in watched}
        try:
            while True:
                changed: dict[str, Any] = {}
                for key, path in watched.items():
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    if st.st_mtime > last_mtimes[key]:
                        last_mtimes[key] = st.st_mtime
                        changed[key] = _read_json_safe(path)
                if changed:
                    await websocket.send_text(
                        json.dumps(
                            {"run_id": run_id, "changes": changed},
                            default=str,
                        )
                    )
                await asyncio.sleep(0.75)
        except Exception:
            with contextlib.suppress(Exception):
                await websocket.close()

    heartbeat.__annotations__["websocket"] = fastapi.WebSocket
    app.websocket("/ws/runs/{run_id}/heartbeat")(heartbeat)
