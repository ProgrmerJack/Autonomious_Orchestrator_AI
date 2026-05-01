from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .agent_body_client import (
    AgentBodyClient,
    AgentBodyError,
)


@dataclass(slots=True)
class SandboxSpec:
    provider: str
    image: str
    network_allowlist: list[str] = field(default_factory=list)
    mounts: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    provider: str
    command: list[str]
    stdout: str
    stderr: str
    exit_code: int
    dry_run: bool = True


class SandboxProvider(Protocol):
    name: str

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        """Execute or prepare execution inside a sandbox."""


class DryRunSandboxProvider:
    name = "dry-run"

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        return ExecutionResult(
            provider=self.name,
            command=command,
            stdout=f"Prepared sandbox '{spec.image}' without host execution.",
            stderr="",
            exit_code=0,
            dry_run=True,
        )


class CuaSandboxProvider(DryRunSandboxProvider):
    name = "cua"

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        return execute_via_agent_body(self.name, spec, command)


class FirecrackerSandboxProvider(DryRunSandboxProvider):
    name = "firecracker"

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        return execute_via_agent_body(self.name, spec, command)


class KataSandboxProvider(DryRunSandboxProvider):
    name = "kata"

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        return execute_via_agent_body(self.name, spec, command)


class AgentBodySandboxProvider(DryRunSandboxProvider):
    name = "agent-body"

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        return execute_via_agent_body(self.name, spec, command)


def execute_via_agent_body(
    provider_name: str,
    spec: SandboxSpec,
    command: list[str],
) -> ExecutionResult:
    state_path = Path(
        str(
            spec.metadata.get("state_path")
            or (Path.cwd() / ".agentos" / "sandbox" / f"{provider_name}.json")
        )
    )
    client = AgentBodyClient(state_path, metadata=spec.metadata)
    if not client.available():
        return ExecutionResult(
            provider=provider_name,
            command=command,
            stdout=(
                f"Prepared sandbox '{spec.image}' without host execution. "
                "Rust agent_body was not available."
            ),
            stderr="",
            exit_code=0,
            dry_run=True,
        )
    request = _control_request(spec, command)
    try:
        payload = client.request(request)
    except AgentBodyError as exc:
        return ExecutionResult(
            provider=provider_name,
            command=command,
            stdout="",
            stderr=str(exc),
            exit_code=1,
            dry_run=False,
        )
    return ExecutionResult(
        provider=provider_name,
        command=command,
        stdout=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        stderr="",
        exit_code=0,
        dry_run=False,
    )


class SandboxManager:
    def __init__(self) -> None:
        self.providers: dict[str, SandboxProvider] = {
            "dry-run": DryRunSandboxProvider(),
            "agent-body": AgentBodySandboxProvider(),
            "cua": CuaSandboxProvider(),
            "firecracker": FirecrackerSandboxProvider(),
            "kata": KataSandboxProvider(),
        }

    def execute(
        self,
        spec: SandboxSpec,
        command: list[str],
    ) -> ExecutionResult:
        provider = self.providers.get(spec.provider)
        if provider is None:
            names = ", ".join(sorted(self.providers))
            message = f"Unknown sandbox provider '{spec.provider}': {names}"
            raise KeyError(message)
        return provider.execute(spec, command)


def _control_request(spec: SandboxSpec, command: list[str]) -> dict:
    metadata = dict(spec.metadata or {})
    request = metadata.get("control_request")
    if isinstance(request, dict):
        return request
    return {"kind": "exec", "argv": list(command)}
