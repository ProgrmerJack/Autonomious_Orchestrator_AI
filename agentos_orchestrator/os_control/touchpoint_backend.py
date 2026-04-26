from __future__ import annotations

import importlib

from .base import BackendUnavailable, UiAction, UiNode


class TouchpointBackend:
    """Optional live backend for the Touchpoint accessibility library."""

    name = "touchpoint"

    def __init__(self) -> None:
        try:
            touchpoint_module = importlib.import_module("touchpoint")
        except ImportError:
            touchpoint_module = None
        self._touchpoint = touchpoint_module

    def available(self) -> bool:
        return self._touchpoint is not None

    def snapshot(self) -> list[UiNode]:
        if self._touchpoint is None:
            raise BackendUnavailable("touchpoint is not installed")
        raw_nodes = self._touchpoint.snapshot()
        nodes: list[UiNode] = []
        for index, node in enumerate(raw_nodes):
            nodes.append(
                UiNode(
                    node_id=str(getattr(node, "id", index)),
                    role=str(getattr(node, "role", "unknown")),
                    name=str(getattr(node, "name", "")),
                    bounds=getattr(node, "bounds", None),
                    enabled=bool(getattr(node, "enabled", True)),
                    focused=bool(getattr(node, "focused", False)),
                    metadata={"raw_type": type(node).__name__},
                )
            )
        return nodes

    def perform(self, action: UiAction) -> str:
        if self._touchpoint is None:
            raise BackendUnavailable("touchpoint is not installed")
        result = self._touchpoint.perform(
            action.action_type,
            selector=action.selector,
            value=action.value,
            **action.metadata,
        )
        return str(result)
