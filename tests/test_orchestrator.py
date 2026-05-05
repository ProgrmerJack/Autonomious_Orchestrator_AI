from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import unittest
from pathlib import Path

from agentos_orchestrator.cognition.frontier_api import (
    FrontierDecision,
    FrontierPrompt,
)
from agentos_orchestrator.core.agents import (
    SupervisorAgent,
    VerificationAgent,
    WorkerAgent,
)
from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.core.types import WorkerResult
from agentos_orchestrator.os_control.base import UiNode
from agentos_orchestrator.os_control.virtual_desktop_sandbox_backend import (
    VirtualDesktopSandboxBackend,
)
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


class FakeTerminalBackend:
    def __init__(self) -> None:
        self.actions = []

    def perform(self, action):
        self.actions.append(action)
        return json.dumps(
            {
                "status": "process-executed",
                "process": {"exit_code": 0},
            }
        )


class RegionAwareFrontierClient:
    def __init__(self) -> None:
        self.calls: list[FrontierPrompt] = []

    def choose_action(self, prompt: FrontierPrompt) -> FrontierDecision:
        self.calls.append(prompt)
        region = str(prompt.state_context.get("target_region") or "").strip()
        target_node_id = {
            "window": "window-browser",
            "address": "browser-address-bar",
            "content": "browser-main-doc",
        }.get(region, "")
        marks = prompt.mark_payload.get("marks") or []
        target_id = 1
        for mark in marks:
            if str(mark.get("node_id") or "") == target_node_id:
                target_id = int(mark.get("id") or 1)
                break
        return FrontierDecision(
            action="focus",
            target_id=target_id,
            rationale=f"Select {target_node_id or region}",
            confidence=0.92,
        )


class OrchestratorTests(unittest.TestCase):
    def test_general_research_adds_pc_research(self) -> None:
        tasks = SupervisorAgent().plan("research market evidence with sources")

        self.assertFalse(any(task.role == "pc-control" for task in tasks))
        self.assertTrue(any(task.role == "pc-research" for task in tasks))

    def test_sandbox_objective_does_not_infer_host_pc_context(self) -> None:
        tasks = SupervisorAgent().plan(
            "Research public market catalysts using all tools in sandbox"
        )

        self.assertFalse(any(task.role == "pc-control" for task in tasks))
        self.assertTrue(any(task.role == "pc-research" for task in tasks))

    def test_software_agent_objective_adds_pc_research(self) -> None:
        tasks = SupervisorAgent().plan(
            "Analyze why a deep research agent is not using sandbox, browser, "
            "and pc control effectively across general topics"
        )

        roles = [task.role for task in tasks]
        self.assertIn("planning", roles)
        self.assertIn("pc-research", roles)
        self.assertFalse(any(task.role == "pc-control" for task in tasks))

    def test_software_agent_objective_gets_expanded_browser_frontier_budget(
        self,
    ) -> None:
        objective = (
            "Analyze why a deep research agent is not using sandbox, browser, "
            "and pc control effectively across general topics"
        )
        browser_plan = {
            "enabled": True,
            "search_queries": [f"query {index}" for index in range(8)],
        }

        self.assertGreater(
            WorkerAgent._pc_browser_navigation_limit(
                objective,
                [f"https://example.com/{index}" for index in range(12)],
                browser_plan,
            ),
            8,
        )
        self.assertGreater(
            WorkerAgent._pc_browser_cycle_count(objective, browser_plan),
            1,
        )

    def test_current_web_multi_hour_sandbox_enables_pc_research(self) -> None:
        tasks = SupervisorAgent().plan(
            "[multi-hour] Research highest-potential stocks right now using all available tools in sandbox"
        )

        self.assertTrue(any(task.role == "pc-research" for task in tasks))
        self.assertFalse(any(task.role == "pc-control" for task in tasks))

    def test_current_web_standard_objective_adds_planning_and_pc_research(
        self,
    ) -> None:
        tasks = SupervisorAgent().plan(
            "Research semiconductor pricing and supply chain risks as of now with source-backed evidence"
        )

        roles = [task.role for task in tasks]
        self.assertIn("planning", roles)
        self.assertIn("pc-research", roles)
        self.assertLess(roles.index("planning"), roles.index("pc-research"))
        self.assertLess(roles.index("pc-research"), roles.index("literature"))

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

    def test_browser_search_queries_preserve_natural_objective_clauses(self) -> None:
        objective = (
            "[multi-hour] Research public stocks with the highest potential to soar as of now "
            "using all available tools in sandbox. Produce the deepest possible analyst and scientist report "
            "with live current-web evidence, explicit uncertainties, competing hypotheses, and critical ranking."
        )

        queries = WorkerAgent._browser_search_queries(objective, [])

        self.assertGreaterEqual(len(queries), 2)
        self.assertTrue(
            any(
                query.startswith(
                    "Research public stocks with the highest potential to soar as of now"
                )
                for query in queries
            )
        )
        self.assertTrue(
            any(
                "deepest possible analyst and scientist report" in query
                for query in queries
            )
        )

    def test_browser_search_queries_expand_for_current_web_frontier(self) -> None:
        urls = [
            f"https://html.duckduckgo.com/html/?q=semiconductor+pricing+signal+{index}"
            for index in range(14)
        ]

        queries = WorkerAgent._browser_search_queries(
            "Research semiconductor pricing and supply chain risks as of now with live current-web evidence",
            urls,
        )

        self.assertGreaterEqual(len(queries), 12)
        self.assertTrue(any("signal 13" in query for query in queries))

    def test_browser_search_queries_prioritize_planner_queries(self) -> None:
        queries = WorkerAgent._browser_search_queries(
            "Analyze deep research agent shortcomings",
            [],
            [
                "agentos sandbox browser frontier gaps",
                "pc control evidence failures in deep research agents",
            ],
        )

        self.assertEqual(queries[0], "agentos sandbox browser frontier gaps")
        self.assertIn(
            "pc control evidence failures in deep research agents",
            queries,
        )

    def test_planning_browser_urls_preserve_seed_urls_and_queries(self) -> None:
        urls = WorkerAgent._planning_browser_urls(
            {
                "browser_research": {
                    "seed_urls": [
                        "https://sec.gov",
                        "investor.example.com/filings",
                    ],
                    "search_queries": [
                        "nvidia latest 10-Q guidance",
                        "semiconductor supply chain recent pricing",
                    ],
                }
            }
        )

        decoded_urls = [urllib.parse.unquote_plus(url) for url in urls]
        self.assertIn("https://sec.gov", urls)
        self.assertIn("https://investor.example.com/filings", urls)
        self.assertTrue(
            any("nvidia latest 10-Q guidance" in url for url in decoded_urls)
        )
        self.assertTrue(
            any(
                "semiconductor supply chain recent pricing" in url
                for url in decoded_urls
            )
        )

    def test_planning_browser_urls_allow_deeper_frontier(self) -> None:
        urls = WorkerAgent._planning_browser_urls(
            {
                "browser_research": {
                    "seed_urls": [
                        "https://sec.gov",
                        "https://example.com/ir",
                    ],
                    "search_queries": [
                        f"semiconductor pricing current evidence query {index}"
                        for index in range(20)
                    ],
                }
            }
        )

        self.assertGreaterEqual(len(urls), 22)
        self.assertIn("https://sec.gov", urls)

    def test_pc_browser_navigation_limit_expands_for_current_web(self) -> None:
        urls = [f"https://example.com/report-{index}" for index in range(12)]

        limit = WorkerAgent._pc_browser_navigation_limit(
            "Research semiconductor pricing and supplier risk as of now",
            urls,
            {
                "search_queries": [
                    f"query {index} current evidence" for index in range(8)
                ]
            },
        )

        self.assertGreaterEqual(limit, 16)

    def test_run_pc_browser_actions_honors_expanded_navigation_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = VirtualDesktopSandboxBackend(Path(temp_dir) / "sandbox.json")
            worker = WorkerAgent(None, None, FakeResearchEngine())
            urls = [f"https://example.com/report-{index}" for index in range(9)]

            receipts, post_nodes, backend_name, interrupt_report, workspace = (
                worker._run_pc_browser_actions(
                    backend,
                    urls,
                    run_id="run_browser_depth",
                    navigation_limit=9,
                )
            )

            navigate_receipts = [
                receipt
                for receipt in receipts
                if receipt.get("step") == "navigate-candidate" and receipt.get("url")
            ]

            self.assertEqual(len(navigate_receipts), 9)
            self.assertEqual(len(interrupt_report.get("navigated_urls") or []), 9)
            self.assertEqual(backend_name, "virtual-desktop-sandbox")
            self.assertTrue(workspace.get("triggered"))
            self.assertTrue(post_nodes)

    def test_run_pc_browser_actions_uses_frontier_selected_browser_nodes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = VirtualDesktopSandboxBackend(Path(temp_dir) / "sandbox.json")
            frontier_client = RegionAwareFrontierClient()
            worker = WorkerAgent(
                None,
                None,
                FakeResearchEngine(),
                frontier_client=frontier_client,
            )

            receipts, post_nodes, _, _, _ = worker._run_pc_browser_actions(
                backend,
                ["https://example.com/frontier-report"],
                run_id="run_browser_frontier",
                navigation_limit=1,
                objective="Research semiconductor pricing as of now",
                frontier_state={"cycle": 1},
            )

            window_receipt = next(
                receipt
                for receipt in receipts
                if receipt.get("step") == "focus-browser-window"
            )
            navigate_receipt = next(
                receipt
                for receipt in receipts
                if receipt.get("step") == "navigate-candidate"
            )
            content_receipt = next(
                receipt
                for receipt in receipts
                if receipt.get("step") == "focus-content-region"
            )

            self.assertEqual(window_receipt["attempts"][0]["strategy"], "frontier")
            self.assertEqual(
                window_receipt["result"]["matched_node_id"],
                "window-browser",
            )
            self.assertEqual(
                navigate_receipt["result"]["matched_node_id"],
                "browser-address-bar",
            )
            self.assertEqual(
                content_receipt["result"]["matched_node_id"],
                "browser-main-doc",
            )
            self.assertGreaterEqual(len(frontier_client.calls), 3)
            self.assertTrue(post_nodes)

    def test_browser_frontier_session_persists_graph_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = VirtualDesktopSandboxBackend(root / "sandbox.json")
            frontier_client = RegionAwareFrontierClient()
            worker = WorkerAgent(
                None,
                None,
                FakeResearchEngine(),
                frontier_client=frontier_client,
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                (
                    receipts,
                    post_nodes,
                    backend_name,
                    interrupt_report,
                    workspace_report,
                    findings,
                ) = worker._run_pc_browser_frontier_session(
                    "run_frontier_graph",
                    "Research semiconductor pricing and supplier risk as of now",
                    backend,
                    ["https://example.com/seed"],
                    {
                        "seed_urls": ["https://example.com/seed"],
                        "search_queries": [
                            f"query {index} current evidence" for index in range(8)
                        ],
                    },
                    4,
                )
            finally:
                os.chdir(previous_cwd)

            graph_path = root / ".agentos/browser_frontier_graph.json"
            run_graph_path = root / "runs/run_frontier_graph/pc/frontier_graph.json"
            graph = json.loads(run_graph_path.read_text(encoding="utf-8"))

            self.assertTrue(receipts)
            self.assertTrue(post_nodes)
            self.assertEqual(backend_name, "virtual-desktop-sandbox")
            self.assertTrue(interrupt_report.get("navigated_urls"))
            self.assertTrue(workspace_report.get("triggered"))
            self.assertTrue(graph_path.exists())
            self.assertTrue(run_graph_path.exists())
            self.assertTrue(findings.get("frontier_checkpoints"))
            self.assertTrue(findings.get("worker_sessions"))
            self.assertTrue(graph.get("urls"))
            self.assertGreaterEqual(
                findings.get("frontier_graph", {})
                .get("summary", {})
                .get("url_count", 0),
                1,
            )

    def test_browser_checkpoint_contradictions_compound_in_frontier_graph(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        graph = worker._empty_frontier_graph()

        graph = worker._merge_browser_checkpoint_into_frontier_graph(
            graph,
            {
                "contradictions": [
                    "Potential contradiction between channel checks and company guidance."
                ],
                "domain_leads": ["sec.gov"],
                "url_leads": ["https://sec.gov/Archives/example-filings"],
                "missing_evidence": [
                    "Need independent verification of management guidance assumptions."
                ],
            },
            "run_contradictions",
            0,
        )

        summary = worker._frontier_graph_summary(graph)
        seed_urls = worker._frontier_graph_seed_urls(graph, limit=10)

        self.assertEqual(summary["contradiction_count"], 1)
        self.assertIn("sec.gov", summary["top_domains"])
        self.assertIn("https://sec.gov/Archives/example-filings", seed_urls)

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

    def test_browser_findings_keep_multiple_high_signal_pages_per_domain(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())

        def fake_search_results(
            query: str,
            limit: int | None = None,
        ) -> list[ResearchSource]:
            del query, limit
            return [
                ResearchSource(
                    provider="web-search",
                    title=f"Example domain page {index}",
                    url=f"https://example.com/report-{index}",
                    abstract="High-signal page from same domain.",
                    year=2026,
                    score=90.0 - index,
                )
                for index in range(5)
            ] + [
                ResearchSource(
                    provider="web-search",
                    title=f"Other domain page {index}",
                    url=f"https://other.com/post-{index}",
                    abstract="Secondary domain coverage.",
                    year=2026,
                    score=80.0 - index,
                )
                for index in range(3)
            ]

        worker.research_engine._search_web_results = fake_search_results
        worker.research_engine._dedupe_sources = lambda sources: sources
        worker.research_engine._rank_sources = lambda sources, query: sources
        worker._ai_browser_source_strategy = lambda objective: {
            "targeted_queries": ["nvidia earnings transcript", "nvidia valuation"]
        }
        worker._deep_page_read = lambda url: "signal " * 300
        worker._content_block_quality = lambda content: {"quality_score": 0.8}
        worker._browser_preview_is_blocked = lambda preview: False
        worker._extract_page_evidence = lambda content, title, query: [
            "revenue growth acceleration"
        ]
        worker._ai_page_judgment = lambda source, content, objective, core_query: (
            "high signal"
        )

        findings = worker._reasoned_browser_findings(
            "[multi-hour] NVIDIA market research",
            [],
        )

        example_hits = sum(1 for url in findings["direct_urls"] if "example.com" in url)
        self.assertEqual(findings["frontier"]["mode"], "expansive")
        self.assertGreaterEqual(findings["frontier"]["candidate_urls"], 5)
        self.assertGreaterEqual(example_hits, 2)

    def test_browser_findings_prioritize_explicit_navigation_urls(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        seed_url = "https://seed.example.com/live-brief"

        def fake_search_results(
            query: str,
            limit: int | None = None,
        ) -> list[ResearchSource]:
            del query, limit
            return [
                ResearchSource(
                    provider="web-search",
                    title="Searched result one",
                    url="https://search.example.com/result-one",
                    abstract="Search-derived page.",
                    year=2026,
                    score=90.0,
                ),
                ResearchSource(
                    provider="web-search",
                    title="Searched result two",
                    url="https://search.example.com/result-two",
                    abstract="Search-derived page.",
                    year=2026,
                    score=88.0,
                ),
            ]

        worker.research_engine._search_web_results = fake_search_results
        worker.research_engine._search_financial_portals = lambda query, limit=10: []
        worker.research_engine._search_sec_edgar = lambda query, limit=6: []
        worker.research_engine._dedupe_sources = lambda sources: sources
        worker.research_engine._rank_sources = lambda sources, query: [
            source for source in sources if "search.example.com" in source.url
        ]
        worker._ai_browser_source_strategy = lambda objective: {"targeted_queries": []}
        worker._deep_page_read = lambda url: f"signal from {url} " * 40
        worker._content_block_quality = lambda content: {"quality_score": 0.8}
        worker._browser_preview_is_blocked = lambda preview: False
        worker._extract_page_evidence = lambda content, title, query: [
            "browser seed evidence"
        ]
        worker._ai_page_judgment = lambda source, content, objective, core_query: (
            "high signal"
        )

        findings = worker._reasoned_browser_findings(
            "[multi-hour] Research live browser evidence",
            [seed_url],
        )

        self.assertEqual(findings["direct_urls"][0], seed_url)
        self.assertIn(seed_url, findings["candidate_urls"])
        self.assertEqual(findings["judged_results"][0]["url"], seed_url)
        self.assertIn(
            "browser-navigation-seed",
            findings["judged_results"][0].get("quality_flags") or [],
        )

    def test_terminal_verification_uses_terminal_selector_and_json_receipt(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        backend = FakeTerminalBackend()

        verified = worker._verify_claims_with_sandbox_terminal(
            backend,
            ["Revenue increased from 10 to 20."],
        )

        self.assertEqual(len(verified), 1)
        self.assertEqual(backend.actions[0].action_type, "execute_command")
        self.assertEqual(backend.actions[0].selector, "terminal")
        self.assertEqual(verified[0]["exit_code"], 0)

    def test_browser_findings_fallback_to_raw_sources_when_ranking_starves(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())

        def fake_search_results(
            query: str,
            limit: int | None = None,
        ) -> list[ResearchSource]:
            del query, limit
            return [
                ResearchSource(
                    provider="web-search",
                    title="Fallback candidate one",
                    url="https://example.com/fallback-one",
                    abstract="Fallback browser candidate.",
                    year=2026,
                    score=60.0,
                ),
                ResearchSource(
                    provider="web-search",
                    title="Fallback candidate two",
                    url="https://example.org/fallback-two",
                    abstract="Fallback browser candidate.",
                    year=2026,
                    score=55.0,
                ),
            ]

        worker.research_engine._search_web_results = fake_search_results
        worker.research_engine._search_financial_portals = lambda query, limit=10: []
        worker.research_engine._search_sec_edgar = lambda query, limit=6: []
        worker.research_engine._dedupe_sources = lambda sources: sources
        worker.research_engine._rank_sources = lambda sources, query: []
        worker._ai_browser_source_strategy = lambda objective: {"targeted_queries": []}
        worker._deep_page_read = lambda url: "signal " * 300
        worker._content_block_quality = lambda content: {"quality_score": 0.8}
        worker._browser_preview_is_blocked = lambda preview: False
        worker._extract_page_evidence = lambda content, title, query: [
            "candidate claim"
        ]
        worker._ai_page_judgment = lambda source, content, objective, core_query: (
            "raw fallback admitted"
        )

        findings = worker._reasoned_browser_findings(
            "[multi-hour] Research broad opportunity landscape",
            [],
        )

        self.assertGreaterEqual(len(findings["candidate_urls"]), 2)
        self.assertGreaterEqual(len(findings["direct_urls"]), 2)
        self.assertEqual(findings["frontier"]["ranked_candidates"], 0)

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
