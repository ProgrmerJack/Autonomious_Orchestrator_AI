from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class AgentBodyError(RuntimeError):
    pass


class AgentBodyUnavailableError(AgentBodyError):
    pass


@dataclass(slots=True)
class AgentBodyClient:
    state_path: Path
    metadata: dict[str, Any]
    _session: "_AgentBodySession | None"
    _invocation: list[str] | None

    def __init__(
        self,
        state_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.metadata = dict(metadata or {})
        self._session = None
        self._invocation = None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return bool(self._resolve_invocation())

    def capabilities(self) -> dict[str, Any]:
        return self.request({"kind": "capabilities"})

    def native_snapshot(self) -> dict[str, Any]:
        return self.request({"kind": "native_snapshot"})

    def native_act(
        self,
        action_type: str,
        selector: str,
        value: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "kind": "native_act",
                "action_type": action_type,
                "selector": selector,
                "value": value,
                "metadata": dict(metadata or {}),
            }
        )

    def reset(self) -> dict[str, Any]:
        return self.request({"kind": "reset"})

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        invocation = self._resolve_invocation()
        if not invocation:
            raise AgentBodyUnavailableError("Rust agent_body was not available.")
        try:
            return self._session_request(payload, invocation)
        except AgentBodyError:
            self.close()
            return self._request_once(payload, invocation)

    def close(self) -> None:
        if self._session is None:
            return
        self._session.close()
        self._session = None

    def _resolve_invocation(self) -> list[str]:
        if self._invocation is None:
            self._invocation = resolve_agent_body_invocation(self.metadata)
        return list(self._invocation)

    def _session_request(
        self,
        payload: dict[str, Any],
        invocation: Sequence[str],
    ) -> dict[str, Any]:
        if self._session is None:
            self._session = _AgentBodySession(invocation, self.state_path)
        return self._session.request(payload)

    def _request_once(
        self,
        payload: dict[str, Any],
        invocation: Sequence[str],
    ) -> dict[str, Any]:
        completed = subprocess.run(
            [
                *invocation,
                "--state-file",
                str(self.state_path),
                "--command-json",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            message = stderr or stdout or "agent_body invocation failed"
            raise AgentBodyError(message)
        if not stdout:
            return {}
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AgentBodyError(f"agent_body returned invalid JSON: {stdout}") from exc
        if not isinstance(response, dict):
            raise AgentBodyError("agent_body returned a non-object response")
        return response

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, subprocess.SubprocessError):
            pass


@dataclass(slots=True)
class _AgentBodySession:
    invocation: tuple[str, ...]
    state_path: Path
    process: subprocess.Popen[str] | None = None

    def __init__(self, invocation: Sequence[str], state_path: Path) -> None:
        self.invocation = tuple(invocation)
        self.state_path = state_path
        self.process = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_process()
        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            raise AgentBodyError("agent_body session streams were unavailable")
        try:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()
        except OSError as exc:
            raise AgentBodyError(f"agent_body session write failed: {exc}") from exc
        response = self._read_response(stdout)
        if not isinstance(response, dict):
            raise AgentBodyError("agent_body session returned a non-object response")
        return response

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None
        stdin = process.stdin
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _ensure_process(self) -> subprocess.Popen[str]:
        process = self.process
        if process is not None and process.poll() is None:
            return process
        try:
            process = subprocess.Popen(
                [*self.invocation, "--state-file", str(self.state_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise AgentBodyError(f"agent_body session launch failed: {exc}") from exc
        self.process = process
        self._consume_startup(process)
        return process

    def _consume_startup(self, process: subprocess.Popen[str]) -> None:
        stdout = process.stdout
        if stdout is None:
            raise AgentBodyError("agent_body session stdout was unavailable")
        response = self._read_response(stdout)
        if response.get("type") != "body.started":
            raise AgentBodyError(
                "agent_body session did not emit the expected startup handshake"
            )

    def _read_response(self, stdout: Any) -> dict[str, Any]:
        while True:
            line = stdout.readline()
            if not line:
                message = "agent_body session terminated before responding"
                process = self.process
                if process is not None and process.stderr is not None:
                    try:
                        stderr = process.stderr.read().strip()
                    except OSError:
                        stderr = ""
                    if stderr:
                        message = stderr
                raise AgentBodyError(message)
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AgentBodyError(
                    f"agent_body session returned invalid JSON: {raw}"
                ) from exc
            if isinstance(payload, dict):
                return payload
            raise AgentBodyError("agent_body session returned a non-object response")


def resolve_agent_body_invocation(
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    metadata = dict(metadata or {})
    agent_body_command = metadata.get("agent_body_command")
    if isinstance(agent_body_command, (list, tuple)):
        command = [
            str(item).strip() for item in agent_body_command if str(item).strip()
        ]
        if command:
            return command
    agent_body_bin = str(metadata.get("agent_body_bin") or "").strip()
    if agent_body_bin:
        return [agent_body_bin]
    agent_body_manifest = str(metadata.get("agent_body_manifest") or "").strip()
    # Auto-running the Rust body is still an explicit opt-in for sandbox
    # providers. Real OS-control backends pass an explicit manifest path.
    if (
        not agent_body_manifest
        and os.environ.get("AGENTOS_ENABLE_AGENT_BODY", "") != "1"
    ):
        return []
    manifest = Path(
        agent_body_manifest or str(Path.cwd() / "crates" / "agent_body" / "Cargo.toml")
    )
    if not manifest.exists() or shutil.which("cargo") is None:
        return []
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(manifest),
    ]
    features = _agent_body_features(metadata)
    if features:
        command.extend(["--features", ",".join(features)])
    command.append("--")
    return command


def _agent_body_features(metadata: Mapping[str, Any]) -> list[str]:
    raw_features = metadata.get("agent_body_features")
    features: list[str] = []
    if isinstance(raw_features, str):
        features.extend(
            part.strip()
            for part in raw_features.replace(",", " ").split()
            if part.strip()
        )
    elif isinstance(raw_features, (list, tuple, set)):
        features.extend(str(item).strip() for item in raw_features if str(item).strip())
    deduped: list[str] = []
    for feature in features:
        if feature and feature not in deduped:
            deduped.append(feature)
    return deduped
