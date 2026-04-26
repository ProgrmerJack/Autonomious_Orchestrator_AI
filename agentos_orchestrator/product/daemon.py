from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentos_orchestrator.core.types import utc_now


@dataclass(slots=True)
class DaemonRecord:
    status: str
    launcher_pid: int | None
    api_url: str
    ui_url: str
    log_path: str
    started_at: str | None = None
    detail: str = ""


class DaemonManager:
    """Detached local gateway lifecycle for daily-driver operation."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        python_executable: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.python_executable = python_executable or sys.executable
        self.agent_dir = self.workspace_root / ".agentos"
        self.state_path = self.agent_dir / "daemon.json"
        self.log_dir = self.agent_dir / "logs"

    def status(self) -> DaemonRecord:
        payload = self._read_state()
        if payload is None:
            return DaemonRecord(
                status="stopped",
                launcher_pid=None,
                api_url="",
                ui_url="",
                log_path=str(self.log_dir / "daemon.log"),
            )
        pid = _int_or_none(payload.get("launcher_pid"))
        running = pid is not None and _process_running(pid)
        return DaemonRecord(
            status="running" if running else "stale",
            launcher_pid=pid,
            api_url=str(payload.get("api_url") or ""),
            ui_url=str(payload.get("ui_url") or ""),
            log_path=str(payload.get("log_path") or ""),
            started_at=payload.get("started_at"),
            detail="process is alive" if running else "state exists only",
        )

    def start(
        self,
        host: str,
        api_port: int,
        ui_port: int,
        policy: str,
        state: str,
        memory: str,
        skip_npm_install: bool = True,
        open_browser: bool = False,
    ) -> DaemonRecord:
        current = self.status()
        if current.status == "running":
            return current
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "daemon.log"
        command = [
            self.python_executable,
            "-m",
            "agentos_orchestrator",
            "--policy",
            policy,
            "--state",
            state,
            "--memory",
            memory,
            "launch",
            "--host",
            host,
            "--api-port",
            str(api_port),
            "--ui-port",
            str(ui_port),
        ]
        if skip_npm_install:
            command.append("--skip-npm-install")
        if not open_browser:
            command.append("--no-browser")
        log_handle = log_path.open("a", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            command,
            cwd=self.workspace_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        log_handle.close()
        record = DaemonRecord(
            status="running",
            launcher_pid=process.pid,
            api_url=f"http://{host}:{api_port}",
            ui_url=f"http://{host}:{ui_port}/",
            log_path=str(log_path),
            started_at=utc_now(),
            detail="detached launcher started",
        )
        self._write_state(asdict(record))
        return record

    def stop(self) -> DaemonRecord:
        current = self.status()
        if current.launcher_pid is not None:
            _terminate_tree(current.launcher_pid)
        if self.state_path.exists():
            self.state_path.unlink()
        return DaemonRecord(
            status="stopped",
            launcher_pid=None,
            api_url=current.api_url,
            ui_url=current.ui_url,
            log_path=current.log_path,
            detail="daemon stopped",
        )

    def _read_state(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_state(self, payload: dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    killpg = getattr(os, "killpg", None)
    try:
        if killpg is not None:
            killpg(pid, signal.SIGTERM)
            return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
