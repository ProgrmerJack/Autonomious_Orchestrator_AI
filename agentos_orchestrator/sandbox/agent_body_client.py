from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class AgentBodyError(RuntimeError):
    pass


class AgentBodyUnavailableError(AgentBodyError):
    pass


@dataclass(slots=True)
class AgentBodyClient:
    state_path: Path
    metadata: dict[str, Any]

    def __init__(
        self,
        state_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.metadata = dict(metadata or {})
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return bool(resolve_agent_body_invocation(self.metadata))

    def capabilities(self) -> dict[str, Any]:
        return self.request({"kind": "capabilities"})

    def reset(self) -> dict[str, Any]:
        return self.request({"kind": "reset"})

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        invocation = resolve_agent_body_invocation(self.metadata)
        if not invocation:
            raise AgentBodyUnavailableError("Rust agent_body was not available.")
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


def resolve_agent_body_invocation(
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    metadata = dict(metadata or {})
    agent_body_bin = str(metadata.get("agent_body_bin") or "").strip()
    if agent_body_bin:
        return [agent_body_bin]
    agent_body_manifest = str(metadata.get("agent_body_manifest") or "").strip()
    # The Rust crate is an experimental state proxy, not a native OS backend.
    # Do not auto-discover and run it just because the repo happens to contain
    # a Cargo manifest. Using it must be an explicit choice via metadata or env.
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
    return [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--",
    ]
