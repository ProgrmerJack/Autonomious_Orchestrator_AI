from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from agentos_orchestrator.sandbox.agent_body_client import (
    AgentBodyClient,
    AgentBodyError,
)

from .base import BackendUnavailable, UiAction, UiNode
from .real_file_ops import perform_real_file_operation
from .windows_family_adapter import (
    adapt_windows_action,
    enrich_windows_receipt,
    normalize_windows_nodes,
)


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
            "agent_body_features": ["uia-windows"],
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
        return normalize_windows_nodes(self._nodes_from_payload(payload))

    def perform(self, action: UiAction) -> str:
        before_nodes = self.snapshot()
        action, matched = adapt_windows_action(action, before_nodes, self.name)
        if action.action_type in {"set_clipboard", "clipboard_copy"}:
            return json.dumps(self._set_clipboard(action), sort_keys=True)
        if action.action_type in {"copy_file", "move_file", "rename_file"}:
            try:
                return json.dumps(perform_real_file_operation(action), sort_keys=True)
            except (OSError, ValueError) as exc:
                return json.dumps(
                    {
                        "status": "failed",
                        "action_type": action.action_type,
                        "selector": action.selector,
                        "reason": str(exc),
                    },
                    sort_keys=True,
                )
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
        if (
            action.action_type in {"launch_app", "open_url"}
            and payload.get("status") == "launched"
        ):
            settled_nodes = self._settle_after_launch(before_nodes, action)
            payload.setdefault(
                "launch_surface_detected",
                self._launch_surface_detected(
                    before_nodes,
                    settled_nodes,
                    action,
                ),
            )
            payload.setdefault("launch_snapshot_nodes", len(settled_nodes))
            payload.setdefault(
                "launch_surface_names",
                [node.name for node in self._meaningful_nodes(settled_nodes)[:5]],
            )
        if matched is not None:
            payload.setdefault("matched_name", matched.name)
            payload.setdefault("matched_role", matched.role)
            payload.setdefault("matched_node_id", matched.node_id)
        if action.action_type not in {
            "launch_app",
            "hotkey",
            "move_cursor",
            "scroll",
            "wait",
        }:
            after_nodes = self.snapshot()
            payload = enrich_windows_receipt(
                action,
                payload,
                before_nodes,
                after_nodes,
                matched_node=matched,
            )
        return json.dumps(payload, sort_keys=True)

    def _settle_after_launch(
        self,
        before_nodes: list[UiNode],
        action: UiAction,
        *,
        timeout_seconds: float = 2.5,
        poll_interval: float = 0.2,
    ) -> list[UiNode]:
        deadline = time.monotonic() + timeout_seconds
        latest_nodes = before_nodes
        while time.monotonic() < deadline:
            latest_nodes = self.snapshot()
            if self._launch_surface_detected(
                before_nodes,
                latest_nodes,
                action,
            ):
                return latest_nodes
            time.sleep(poll_interval)
        return latest_nodes

    @staticmethod
    def _set_clipboard(action: UiAction) -> dict[str, Any]:
        value = str(action.value or "")
        subprocess.run("clip", input=value, text=True, check=True, shell=True)
        return {
            "status": "clipboard-updated",
            "action_type": action.action_type,
            "selector": action.selector,
            "clipboard": value,
        }

    def _launch_surface_detected(
        self,
        before_nodes: list[UiNode],
        after_nodes: list[UiNode],
        action: UiAction,
    ) -> bool:
        before_ids = {node.node_id for node in before_nodes}
        meaningful_after = RustNativeWindowsBackend._meaningful_nodes(after_nodes)
        if any(node.node_id not in before_ids for node in meaningful_after):
            return True
        if any(node.focused for node in meaningful_after):
            return True
        tokens = RustNativeWindowsBackend._launch_tokens(action)
        if not tokens:
            return bool(meaningful_after)
        for node in meaningful_after:
            haystack = " ".join(
                [
                    node.node_id,
                    node.name,
                    node.role,
                    str(node.metadata.get("automation_id") or ""),
                    str(node.metadata.get("class_name") or ""),
                ]
            ).lower()
            if any(token in haystack for token in tokens):
                return True
        return False

    @staticmethod
    def _launch_tokens(action: UiAction) -> set[str]:
        raw = " ".join(
            part
            for part in (str(action.selector or ""), str(action.value or ""))
            if part
        ).lower()
        tokens = {
            token.strip("\"'")
            for token in raw.replace("/", " ").replace(":", " ").split()
            if len(token.strip("\"'")) >= 3
        }
        return {token for token in tokens if token not in {"exe", "select"}}

    @staticmethod
    def _meaningful_nodes(nodes: list[UiNode]) -> list[UiNode]:
        return [
            node
            for node in nodes
            if node.node_id not in {"native-desktop", "native-cursor"}
        ]

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
