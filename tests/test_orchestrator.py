from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import unittest
from pathlib import Path

from agentos_orchestrator.core.agents import (
    SupervisorAgent,
    VerificationAgent,
    WorkerAgent,
)
from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.core.types import WorkerResult
from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.research import (
    DeepResearchEngine,
    ResearchBrief,
    ResearchSource,
)


class FakeResearchEngine(DeepResearchEngine):
    def run(
        self,
        objective: str,
        run_id: str,
        pc_context: dict[str, object] | None = None,
        planning_context: dict[str, object] | None = None,
        evidence_targets: dict[str, object] | None = None,
    ) -> ResearchBrief:
        del pc_context, planning_context, evidence_targets
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

    def _search_web_results(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        del limit
        return [
            ResearchSource(
                provider="web-search",
                title=f"Current analysis for {query}",
                url="https://example.com/current-analysis",
                abstract="Recent catalyst and risk analysis for the query.",
                year=2026,
                score=90.0,
                relevance=0.9,
                credibility_score=0.7,
                recency=0.9,
            ),
            ResearchSource(
                provider="web-search",
                title=f"Primary filing for {query}",
                url="https://example.org/filing",
                abstract="Primary filing and direct evidence for the query.",
                year=2026,
                score=85.0,
                relevance=0.82,
                credibility_score=0.8,
                recency=0.88,
            ),
        ]

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
    ) -> str:
        del accept, max_bytes, timeout_seconds
        return (
            f"<html><head><title>{url}</title></head><body>"
            "Direct website evidence with current analysis, catalysts, and downside risks."
            "</body></html>"
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
    def test_general_research_does_not_schedule_pc_steps(self) -> None:
        tasks = SupervisorAgent().plan("research market evidence with sources")

        self.assertFalse(any(task.role == "pc-control" for task in tasks))
        self.assertFalse(any(task.role == "pc-research" for task in tasks))

    def test_sandbox_objective_does_not_infer_host_pc_context(self) -> None:
        tasks = SupervisorAgent().plan(
            "Research public market catalysts using all tools in sandbox"
        )

        self.assertFalse(any(task.role == "pc-control" for task in tasks))

    def test_current_web_multi_hour_sandbox_enables_pc_research(self) -> None:
        tasks = SupervisorAgent().plan(
            "[multi-hour] Research highest-potential stocks right now using all available tools in sandbox"
        )

        self.assertTrue(any(task.role == "pc-research" for task in tasks))
        self.assertFalse(any(task.role == "pc-control" for task in tasks))

    def test_coverage_targets_prefer_literature_effective_targets(self) -> None:
        planning = WorkerResult(
            task_id="task_planning_1",
            role="planning",
            summary="planning",
            artifacts=[],
            evidence=[],
            confidence=0.9,
        )
        literature = WorkerResult(
            task_id="task_literature_1",
            role="literature",
            summary="literature",
            artifacts=[],
            evidence=[
                {
                    "source": "research-metrics",
                    "metadata": {
                        "retrieval": {
                            "targets": {
                                "min_source_count": 10,
                                "min_provider_count": 1,
                                "min_scholarly_sources": 0,
                                "min_strong_or_moderate": 3,
                                "min_novelty_rate": 0.0,
                                "max_contradiction_risk": 0.8,
                            }
                        }
                    },
                }
            ],
            confidence=0.8,
        )

        targets = ResearchOrchestrator._coverage_targets([planning, literature])

        self.assertEqual(targets.get("min_novelty_rate"), 0.0)
        self.assertEqual(targets.get("min_scholarly_sources"), 0)

    def test_current_web_gate_softens_source_and_strong_thresholds(self) -> None:
        failures = ResearchOrchestrator._coverage_failures(
            {
                "source_count": 8,
                "provider_count": 1,
                "scholarly_source_count": 0,
                "strong_or_moderate": 0,
                "max_contradiction_risk": 0.2,
                "novelty_rate": 0.0,
                "on_topic_ratio": 0.9,
            },
            {
                "min_source_count": 12,
                "min_provider_count": 1,
                "min_scholarly_sources": 0,
                "min_strong_or_moderate": 4,
                "max_contradiction_risk": 0.8,
                "min_novelty_rate": 0.0,
            },
        )

        self.assertNotIn("source_count below target", failures)
        self.assertNotIn("strong_or_moderate evidence below target", failures)

    def test_current_web_gate_softens_provider_and_source_on_partial_signal(
        self,
    ) -> None:
        failures = ResearchOrchestrator._coverage_failures(
            {
                "source_count": 2,
                "provider_count": 1,
                "scholarly_source_count": 0,
                "strong_or_moderate": 0,
                "max_contradiction_risk": 0.1,
                "novelty_rate": 0.0,
                "on_topic_ratio": 0.7,
            },
            {
                "min_source_count": 12,
                "min_provider_count": 2,
                "min_scholarly_sources": 0,
                "min_strong_or_moderate": 4,
                "max_contradiction_risk": 0.8,
                "min_novelty_rate": 0.0,
            },
        )

        self.assertNotIn("source_count below target", failures)
        self.assertNotIn("provider_count below target", failures)

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

    def test_planning_urls_distill_current_research_without_repo_noise(self) -> None:
        urls = WorkerAgent._planning_urls_from_objective(
            "Research highest-potential public companies as of now using all available evidence-gathering tools. Produce a highest-quality analyst report."
        )

        decoded_urls = [urllib.parse.unquote_plus(url) for url in urls]
        self.assertTrue(
            any("highest-potential public companies" in url for url in decoded_urls)
        )
        self.assertFalse(any("when the c" in url for url in decoded_urls))
        self.assertFalse(any("github.com/search" in url for url in urls))

    def test_supervisor_data_task_adapts_to_research_domain(self) -> None:
        tasks = SupervisorAgent().plan(
            "Research highest-potential public companies as of now"
        )
        data_task = next(task for task in tasks if task.role == "data")

        self.assertIn("structured evidence", data_task.objective)
        self.assertNotIn("implementation constraints", data_task.objective)

    def test_browser_blocked_page_signals_are_filtered(self) -> None:
        blocked = {
            "page_title": "Pardon Our Interruption",
            "page_excerpt": "As you were browsing something about your browser made us think you were a bot.",
        }
        clean = {
            "page_title": "High Upside Equities Analysis",
            "page_excerpt": "Detailed catalyst analysis with valuation assumptions and downside scenarios.",
        }

        self.assertTrue(WorkerAgent._browser_preview_is_blocked(blocked))
        self.assertFalse(WorkerAgent._browser_preview_is_blocked(clean))

    def test_verification_flags_research_quality_warnings(self) -> None:
        result = WorkerResult(
            task_id="task_literature_quality",
            role="literature",
            summary="Collected current web evidence.",
            evidence=[
                {
                    "source": "research-metrics",
                    "metadata": {
                        "coverage": {
                            "source_count": 10,
                            "provider_count": 1,
                            "strong_or_moderate": 1,
                            "perspective_ratio": 0.6,
                            "missing_perspectives": ["risk", "expert"],
                        }
                    },
                }
            ],
            confidence=0.9,
        )

        review = VerificationAgent().review("run_quality", [result])

        self.assertIn("quality warnings", review.summary)
        self.assertLessEqual(review.confidence, 0.65)
        warnings = review.evidence[0]["quality_warnings"]
        self.assertIn("research used fewer than two source providers", warnings)

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

    def test_optional_pc_research_uses_virtual_sandbox_when_approval_pending(
        self,
    ) -> None:
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
                                "sandbox.exec",
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

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                report = orchestrator.run(
                    "browser research market evidence with sources"
                )
            finally:
                os.chdir(previous_cwd)

            pc_research = [
                result
                for result in report.worker_results
                if result.role == "pc-research"
            ]
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(pc_research), 1)
            self.assertIn("sandboxed", pc_research[0].summary.lower())
            artifact = root / pc_research[0].artifacts[-1]
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "executed")
            self.assertIn("sandbox", payload["backend"].lower())
            self.assertTrue(payload["direct_urls"])
            self.assertTrue(payload["judged_results"])
            self.assertTrue(payload["search_queries"])
            self.assertFalse(
                any(
                    str(query).startswith("Perform sandboxed browser research")
                    for query in payload["search_queries"]
                )
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
