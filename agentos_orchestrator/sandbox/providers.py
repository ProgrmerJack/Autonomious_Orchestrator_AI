from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


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


class FirecrackerSandboxProvider(DryRunSandboxProvider):
    name = "firecracker"


class KataSandboxProvider(DryRunSandboxProvider):
    name = "kata"


class SandboxManager:
    def __init__(self) -> None:
        self.providers: dict[str, SandboxProvider] = {
            "dry-run": DryRunSandboxProvider(),
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
