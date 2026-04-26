from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class UiNode:
    node_id: str
    role: str
    name: str
    bounds: tuple[int, int, int, int] | None = None
    enabled: bool = True
    focused: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class UiAction:
    action_type: str
    selector: str
    value: str | None = None
    metadata: dict = field(default_factory=dict)


class OsControlBackend(Protocol):
    name: str

    def available(self) -> bool:
        """Return whether the backend can control the current environment."""

    def snapshot(self) -> list[UiNode]:
        """Return current structured UI state."""

    def perform(self, action: UiAction) -> str:
        """Execute an action and return a backend action id or receipt."""


class BackendUnavailable(RuntimeError):
    pass
