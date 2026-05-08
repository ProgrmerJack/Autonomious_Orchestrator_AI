from __future__ import annotations

import unittest

from agentos_orchestrator.core.agents import WorkerAgent
from agentos_orchestrator.os_control.base import UiNode


class WorkerSelectorTests(unittest.TestCase):
    def test_point_selector_for_node_uses_semantic_tokens(self) -> None:
        node = UiNode(
            node_id="browser-address-bar",
            role="Edit",
            name="Address and search bar",
            bounds=(10, 10, 500, 32),
            metadata={"automation_id": "41477", "class_name": "Edit"},
        )

        selector = WorkerAgent._point_selector_for_node(node)

        self.assertEqual(selector, "automation_id=41477;class_name=Edit")
        self.assertNotIn("point=", selector)
