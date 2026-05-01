from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.core.agents import WorkerAgent
from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.research import (
    DeepResearchEngine,
    ResearchBrief,
    ResearchSource,
)


class FakeResearchEngine(DeepResearchEngine):
    def run(self, objective: str, run_id: str) -> ResearchBrief:
        return ResearchBrief(
            objective=objective,
            query="test objective",
            summary="Collected deterministic test sources.",
            sources=[
                ResearchSource(
                    provider="test",
                    title="Test Source",
                    url="https://example.com/source",
                    abstract="Test evidence for orchestration.",
                    citation_count=1,
                    score=1.0,
                )
            ],
            artifacts=[f"runs/{run_id}/research/brief.md"],
            confidence=0.9,
        )


class FakePcBackend:
    name = "fake-windows-uia"

    def snapshot(self) -> list[UiNode]:
        return [
            UiNode(
                node_id="desktop-1",
                role="Window",
                name="AgentOS Dashboard - 127.0.0.1:5173",
                focused=True,
            ),
            UiNode(
                node_id="browser-1",
                role="Document",
                name="Browser research evidence",
            ),
        ]


class OrchestratorTests(unittest.TestCase):
    def test_planning_urls_support_explicit_and_generic_targets(self) -> None:
        urls = WorkerAgent._planning_urls_from_objective(
            "Review https://docs.example.org/agentos and compare benchmark safety behavior for agentos orchestration"
        )

        self.assertIn("https://docs.example.org/agentos", urls)
        self.assertTrue(
            any(url.startswith("https://html.duckduckgo.com/html/?q=") for url in urls)
        )
        self.assertTrue(
            any(
                url.startswith("https://github.com/search?type=repositories&q=")
                for url in urls
            )
        )

    def test_run_completes_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {
                            "actions": [
                                "file.write",
                                "mcp.call",
                                "mcp.list",
                                "memory.commit",
                                "network.fetch",
                            ],
                            "paths": ["runs/**", "memory://*", "mcp://*"],
                            "network_hosts": [
                                "api.openalex.org",
                                "api.semanticscholar.org",
                                "api.crossref.org",
                                "api.github.com",
                                "html.duckduckgo.com",
                                "generativelanguage.googleapis.com",
                            ],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                encoding="utf-8",
            )
            orchestrator = ResearchOrchestrator.from_paths(
                policy_path=policy_path,
                state_path=root / "state.sqlite3",
                memory_path=root / "memory.sqlite3",
            )
            orchestrator.worker.research_engine = FakeResearchEngine()
            report = orchestrator.run("test objective")
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(report.worker_results), 4)

            resumed = orchestrator.resume(report.run_id)
            self.assertEqual(resumed["checkpoint"]["stage"], "completed")
            self.assertGreaterEqual(len(resumed["events"]), 1)

    def test_pc_smoke_objective_captures_desktop_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {
                            "actions": [
                                "file.write",
                                "mcp.call",
                                "mcp.list",
                                "memory.commit",
                                "network.fetch",
                                "os.snapshot",
                            ],
                            "paths": [
                                "runs/**",
                                "memory://*",
                                "mcp://*",
                                "windows-uia://*",
                            ],
                            "network_hosts": [
                                "api.openalex.org",
                                "api.semanticscholar.org",
                                "api.crossref.org",
                                "api.github.com",
                                "html.duckduckgo.com",
                                "generativelanguage.googleapis.com",
                            ],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                encoding="utf-8",
            )
            orchestrator = ResearchOrchestrator.from_paths(
                policy_path=policy_path,
                state_path=root / "state.sqlite3",
                memory_path=root / "memory.sqlite3",
            )
            orchestrator.worker.research_engine = FakeResearchEngine()
            orchestrator.worker.pc_backend = FakePcBackend()

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                report = orchestrator.run(
                    "pc-research-smoke compare OpenClaw local PC agents"
                )
            finally:
                os.chdir(previous_cwd)

            pc_results = [
                result
                for result in report.worker_results
                if result.role == "pc-control"
            ]
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(pc_results), 1)
            self.assertEqual(pc_results[0].evidence[0]["node_count"], 2)
            artifact = root / pc_results[0].artifacts[0]
            self.assertTrue(artifact.exists())
            snapshot = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(
                snapshot[0]["name"],
                "AgentOS Dashboard - 127.0.0.1:5173",
            )

    def test_multi_hour_objective_defaults_to_deep_pass_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {
                            "actions": [
                                "file.write",
                                "mcp.call",
                                "mcp.list",
                                "memory.commit",
                                "network.fetch",
                            ],
                            "paths": ["runs/**", "memory://*", "mcp://*"],
                            "network_hosts": [
                                "api.openalex.org",
                                "api.semanticscholar.org",
                                "api.crossref.org",
                                "api.github.com",
                                "html.duckduckgo.com",
                                "generativelanguage.googleapis.com",
                            ],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                encoding="utf-8",
            )
            orchestrator = ResearchOrchestrator.from_paths(
                policy_path=policy_path,
                state_path=root / "state.sqlite3",
                memory_path=root / "memory.sqlite3",
            )
            orchestrator.worker.research_engine = FakeResearchEngine()

            previous = os.environ.pop("AGENTOS_MULTI_HOUR_MIN_SECONDS", None)
            try:
                report = orchestrator.run("[multi-hour] test objective")
            finally:
                if previous is not None:
                    os.environ["AGENTOS_MULTI_HOUR_MIN_SECONDS"] = previous

            planning_results = [
                result for result in report.worker_results if result.role == "planning"
            ]
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(planning_results), 1)

            targets = planning_results[0].evidence[0]["coverage_targets"]
            self.assertGreaterEqual(targets["max_retrieval_passes"], 48)
            self.assertGreaterEqual(targets["min_depth_passes"], 12)
            self.assertEqual(targets["min_runtime_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
