from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agentos_orchestrator.sandbox.agent_body_client import (
    AgentBodyClient,
    AgentBodyError,
)

from .base import BackendUnavailable, UiAction, UiNode


class RustNativeWindowsBackend:
    """Native Windows input bridge backed by the Rust agent_body process."""

    name = "rust-native-windows"

    def __init__(
        self,
        state_path: str | Path | None = None,
        *,
        agent_body_client: AgentBodyClient | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        manifest = root / "crates" / "agent_body" / "Cargo.toml"
        body_metadata = {
            "agent_body_manifest": str(manifest),
            **dict(metadata or {}),
        }
        body_state_path = Path(
            state_path or Path.cwd() / ".agentos" / "rust_native_body.json"
        )
        self._agent_body = agent_body_client or AgentBodyClient(
            body_state_path,
            metadata=body_metadata,
        )

    def available(self) -> bool:
        try:
            payload = self._agent_body.native_snapshot()
        except AgentBodyError:
            return False
        return payload.get("status") == "ok"

    def capabilities(self) -> dict[str, Any]:
        return {
            "type": "native.capabilities",
            "status": "ok" if self.available() else "unavailable",
            "backend": self.name,
            "native": True,
            "capabilities": [
                "native-snapshot",
                "native-act",
                "native-input",
                "launch-app",
                "open-url",
                "hotkey",
                "coordinate-click",
                "coordinate-type",
                "scroll",
                "draw-path",
            ],
        }

    def snapshot(self) -> list[UiNode]:
        try:
            payload = self._agent_body.native_snapshot()
        except AgentBodyError as exc:
            raise BackendUnavailable(str(exc)) from exc
        if payload.get("status") != "ok":
            raise BackendUnavailable(str(payload.get("error") or payload))
        return self._nodes_from_payload(payload)

    def perform(self, action: UiAction) -> str:
        try:
            payload = self._agent_body.native_act(
                action.action_type,
                action.selector,
                action.value,
                action.metadata,
            )
        except AgentBodyError as exc:
            raise BackendUnavailable(str(exc)) from exc
        if payload.get("status") == "unavailable":
            raise BackendUnavailable(str(payload.get("error") or payload))
        payload.setdefault("backend", self.name)
        payload.setdefault("action_type", action.action_type)
        payload.setdefault("selector", action.selector)
        payload.setdefault("value", action.value)
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _nodes_from_payload(payload: dict[str, Any]) -> list[UiNode]:
        nodes: list[UiNode] = []
        for node in list(payload.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            metadata = dict(node.get("metadata") or {})
            bounds = RustNativeWindowsBackend._bounds(node.get("bounds"))
            nodes.append(
                UiNode(
                    node_id=str(node.get("node_id") or "native-node"),
                    role=str(node.get("role") or "Unknown"),
                    name=str(node.get("name") or "Native Node"),
                    bounds=bounds,
                    enabled=bool(node.get("enabled", True)),
                    focused=bool(node.get("focused", False)),
                    metadata=metadata,
                )
            )
        return nodes

    @staticmethod
    def _bounds(raw: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(raw, list) or len(raw) != 4:
            return None
        return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
