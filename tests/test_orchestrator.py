from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.parse
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

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
from agentos_orchestrator.core.policy import PermissionPolicy
from agentos_orchestrator.core.types import TaskSpec, WorkerResult
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
    def test_software_agent_diagnostic_objective_upgrades_heuristic_profile(
        self,
    ) -> None:
        objective = (
            "Analyze why the deep research agent is not comparable to Claude, "
            "GPT, or Gemini, is not using browser and sandbox tools effectively, "
            "and fix the architectural gaps."
        )

        analysis = WorkerAgent._heuristic_objective_analysis(objective)

        self.assertGreaterEqual(analysis["complexity_score"], 8)
        self.assertTrue(analysis["profile"]["comparison"])
        self.assertTrue(analysis["profile"]["risk"])

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

    def test_expansive_browser_budget_scales_for_multi_hour_current_web(self) -> None:
        budget = WorkerAgent._browser_research_budget(
            "[multi-hour] Research highest-potential stocks right now using all available tools in sandbox",
            "highest-potential stocks right now",
            220,
        )

        self.assertEqual(budget["mode"], "expansive")
        self.assertGreaterEqual(int(budget["max_queries"]), 48)
        self.assertGreaterEqual(int(budget["max_direct_urls"]), 160)
        self.assertGreaterEqual(int(budget["candidate_urls"]), 640)
        self.assertGreaterEqual(int(budget["financial_results_per_query"]), 16)

    def test_expansive_pc_browser_frontier_budget_is_bounded(self) -> None:
        objective = (
            "[multi-hour] Research highest-potential stocks right now using all available tools in sandbox"
        )
        browser_plan = {
            "enabled": True,
            "search_queries": [f"query {index}" for index in range(12)],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = VirtualDesktopSandboxBackend(Path(temp_dir) / "sandbox.json")

            self.assertLessEqual(
                WorkerAgent._pc_browser_navigation_limit(
                    objective,
                    [f"https://example.com/{index}" for index in range(48)],
                    browser_plan,
                ),
                24,
            )
            self.assertLessEqual(
                WorkerAgent._pc_browser_cycle_count(objective, browser_plan),
                8,
            )
            self.assertLessEqual(
                WorkerAgent._pc_parallel_browser_worker_count(
                    objective,
                    backend,
                    [f"https://example.com/{index}" for index in range(48)],
                ),
                4,
            )
            self.assertLessEqual(
                WorkerAgent._pc_browser_batch_limit(objective, 64),
                12,
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

    def test_browser_search_queries_strip_instructional_prompt_fragments(
        self,
    ) -> None:
        objective = (
            "[multi-hour] As of right now, research which publicly traded companies have the highest "
            "probability-adjusted upside potential over the next 12 to 24 months. Use all available "
            "general-purpose research means, including browser-grounded web research, sandboxed exploration, "
            "current-web evidence, company filings, earnings material, product signals, independent sources, "
            "and cross-checking. Do not use finance-specific hardcoded templates or domain-specific shortcuts. "
            "Produce an analyst-grade and scientist-grade report with ranked candidates, the evidence for and "
            "against each thesis, uncertainty bounds, catalyst quality, execution risk, valuation-sensitive "
            "considerations, and clear reasons for the ranking."
        )

        queries = WorkerAgent._browser_search_queries(objective, [])

        self.assertGreaterEqual(len(queries), 1)
        self.assertTrue(
            any(
                "publicly traded companies" in query.lower()
                and "upside potential" in query.lower()
                for query in queries
            )
        )
        self.assertFalse(
            any("do not use finance-specific" in query.lower() for query in queries)
        )
        self.assertFalse(any("analyst-grade" in query.lower() for query in queries))
        self.assertFalse(
            any("all available general-purpose research means" in query.lower() for query in queries)
        )

    def test_browser_search_queries_reject_frontier_page_text_fragments(
        self,
    ) -> None:
        objective = (
            "[multi-hour] As of right now, research which publicly traded companies have the highest "
            "probability-adjusted upside potential over the next 12 to 24 months. Use all available "
            "general-purpose research means, including browser-grounded web research, sandboxed exploration, "
            "current-web evidence, company filings, earnings material, product signals, independent sources, "
            "and cross-checking. Do not use finance-specific hardcoded templates or domain-specific shortcuts. "
            "Produce an analyst-grade and scientist-grade report with ranked candidates, the evidence for and "
            "against each thesis, uncertainty bounds, catalyst quality, execution risk, valuation-sensitive "
            "considerations, and clear reasons for the ranking."
        )

        queries = WorkerAgent._browser_search_queries(
            objective,
            [
                "https://html.duckduckgo.com/html/?q="
                "bls+bureau+labor+statistics+release+calendar+subscribe+search+button+search+menu+search+button+search+release+calendar+subscribe+home"
            ],
            [
                "report with , the evidence for",
                "against each thesis",
            ],
        )

        joined = " | ".join(query.lower() for query in queries)
        self.assertTrue(any("upside potential" in query.lower() for query in queries))
        self.assertNotIn("report with", joined)
        self.assertNotIn("against each thesis", joined)
        self.assertNotIn("release calendar subscribe", joined)

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

    def test_software_agent_diagnostic_browser_plan_distills_complaint_text(
        self,
    ) -> None:
        objective = (
            "So fix the code. Analyze why the deep research agent is not "
            "comparable to Claude, GPT, or Gemini, why it is not using sandbox, "
            "browser, and pc control properly, and why retrieval breadth and "
            "useful evidence quality are weak."
        )
        worker = WorkerAgent(None, None, FakeResearchEngine())
        worker.research_engine._ai_research_strategy = lambda objective, query, depth: {}

        browser_plan = worker._planning_browser_research(
            objective,
            WorkerAgent._heuristic_objective_analysis(objective),
        )

        queries = browser_plan["search_queries"]
        self.assertGreaterEqual(len(queries), 6)
        self.assertIn(
            "deep research agent retrieval breadth crawl scaling",
            queries,
        )
        self.assertIn(
            "deep research agent browser sandbox pc control routing",
            queries,
        )
        self.assertFalse(
            any(query.lower().startswith("so fix the code") for query in queries)
        )
        self.assertIn("https://github.com", browser_plan["seed_urls"])
        self.assertIn("https://playwright.dev", browser_plan["seed_urls"])

    def test_planning_browser_research_strips_instruction_fragments_for_market_prompt(
        self,
    ) -> None:
        objective = (
            "[multi-hour] As of right now, research which publicly traded companies have the highest "
            "probability-adjusted upside potential over the next 12 to 24 months. Use all available "
            "general-purpose research means, including browser-grounded web research, sandboxed exploration, "
            "current-web evidence, company filings, earnings material, product signals, independent sources, "
            "and cross-checking. Do not use finance-specific hardcoded templates or domain-specific shortcuts. "
            "Produce an analyst-grade and scientist-grade report with ranked candidates, the evidence for and "
            "against each thesis, uncertainty bounds, catalyst quality, execution risk, valuation-sensitive "
            "considerations, and clear reasons for the ranking."
        )
        worker = WorkerAgent(None, None, FakeResearchEngine())
        worker.research_engine._ai_research_strategy = lambda objective, query, depth: {
            "reasoning_queries": [
                "scientist-grade report with ranked candidates, the evidence for",
                "against each thesis",
                "publicly traded companies upside potential 12 24 months",
            ],
            "authoritative_domains": ["sec.gov", "reuters.com"],
        }

        browser_plan = worker._planning_browser_research(
            objective,
            WorkerAgent._heuristic_objective_analysis(objective),
        )

        queries = browser_plan["search_queries"]
        self.assertTrue(
            any(
                "publicly traded companies" in query.lower()
                and "upside potential" in query.lower()
                for query in queries
            )
        )
        self.assertFalse(any("scientist-grade" in query.lower() for query in queries))
        self.assertFalse(any("against each thesis" in query.lower() for query in queries))
        self.assertIn("https://sec.gov", browser_plan["seed_urls"])
        self.assertIn("https://reuters.com", browser_plan["seed_urls"])

    def test_ai_objective_analysis_requires_llm_by_default(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        worker.research_engine._load_env_from_dotenv = lambda: None

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "No LLM provider configured for AI planning",
            ):
                worker._ai_analyze_objective("Research current market risks")

    def test_ai_objective_analysis_retries_invalid_json(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        responses = iter(
            [
                "not json at all",
                '{"complexity_score": "bad"}',
                json.dumps(
                    {
                        "complexity_score": 8,
                        "profile": {
                            "academic": False,
                            "current": True,
                            "comparison": True,
                            "risk": True,
                        },
                        "min_source_count": 12,
                        "min_provider_count": 3,
                        "min_scholarly_sources": 1,
                        "max_contradiction_risk": 0.35,
                        "max_retrieval_passes": 9,
                        "hypotheses": ["Catalyst quality and valuation will dominate."],
                    }
                ),
            ]
        )
        worker.research_engine._call_ai_text = lambda _system, _user: next(responses)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            analysis = worker._ai_analyze_objective(
                "Research current market risks and compare alternatives"
            )

        self.assertEqual(analysis["complexity_score"], 8)
        self.assertEqual(analysis["min_source_count"], 12)
        self.assertEqual(
            analysis["hypotheses"],
            ["Catalyst quality and valuation will dominate."],
        )

    def test_build_deep_plan_blocks_empty_hypotheses_without_opt_in(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        worker._ai_analyze_objective = lambda _objective: {
            "complexity_score": 6,
            "profile": {},
            "min_source_count": 8,
            "min_provider_count": 2,
            "min_scholarly_sources": 2,
            "max_contradiction_risk": 0.75,
            "max_retrieval_passes": 8,
            "hypotheses": [],
        }
        task = TaskSpec(
            task_id="task_plan",
            role="planning",
            objective="Design deep research plan for: compare live market risk signals",
            declared_actions=[],
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "returned no hypotheses",
            ):
                worker._build_deep_plan("run_plan", task)

    def test_frontier_session_seed_urls_interleave_domains_before_navigation(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())

        seed_urls = worker._frontier_session_seed_urls(
            [
                "https://finance.yahoo.com/quote/AAA",
                "https://finance.yahoo.com/quote/BBB",
                "https://finance.yahoo.com/quote/CCC",
                "https://www.sec.gov/Archives/example-filing",
                "https://www.reuters.com/markets/example-analysis",
            ],
            {
                "seed_urls": [],
                "search_queries": [
                    "publicly traded companies upside potential over next 12 24 months"
                ],
            },
            worker._empty_frontier_graph(),
            market_query=True,
        )

        self.assertEqual(seed_urls[0], "https://www.sec.gov/Archives/example-filing")
        self.assertIn("https://www.sec.gov/Archives/example-filing", seed_urls[:3])
        self.assertIn(
            "https://www.reuters.com/markets/example-analysis",
            seed_urls[:3],
        )
        self.assertFalse(any("finance.yahoo.com" in url for url in seed_urls))

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

    def test_live_pc_research_routes_as_os_act_and_bootstraps_browser(self) -> None:
        class FakeLiveBackend:
            name = "windows-uia"

            def __init__(self) -> None:
                self.launches: list[str] = []

            def available(self) -> bool:
                return True

            def snapshot(self) -> list[UiNode]:
                if self.launches:
                    return [
                        UiNode(
                            node_id="browser-1",
                            role="Window",
                            name="Microsoft Edge",
                        )
                    ]
                return [UiNode(node_id="desktop-1", role="Window", name="Desktop")]

            def perform(self, action):
                if action.action_type == "launch_app":
                    self.launches.append(str(action.value or action.selector))
                    return "launched"
                raise RuntimeError(action.action_type)

        backend = FakeLiveBackend()
        worker = WorkerAgent(None, None, FakeResearchEngine(), pc_backend=backend)

        selected_backend = worker._sandbox_pc_backend()
        self.assertEqual(
            worker._pc_research_action_surface(selected_backend),
            ("os.act", "windows-uia://browser-research"),
        )

        workspace = worker._prepare_cross_surface_workspace(
            selected_backend,
            selected_backend.snapshot(),
            "run_live_backend",
        )
        self.assertEqual(workspace.get("status"), "skipped")

        nodes, report = worker._ensure_browser_surface(
            selected_backend,
            selected_backend.snapshot(),
            "https://example.com/report",
        )
        self.assertEqual(report.get("status"), "launched")
        self.assertEqual(report.get("launch_target"), "https://example.com/report")
        self.assertTrue(worker._has_browser_surface(nodes))

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

    def test_run_pc_browser_actions_hydrates_virtual_browser_with_real_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = VirtualDesktopSandboxBackend(Path(temp_dir) / "sandbox.json")
            worker = WorkerAgent(None, None, FakeResearchEngine())

            receipts, post_nodes, _, _, _ = worker._run_pc_browser_actions(
                backend,
                ["https://example.com/current-analysis"],
                run_id="run_browser_hydrate",
                navigation_limit=1,
            )

            hydrate_receipt = next(
                receipt
                for receipt in receipts
                if receipt.get("step") == "hydrate-browser-content"
            )
            document_node = next(
                node for node in post_nodes if node.node_id == "browser-main-doc"
            )

            self.assertEqual(hydrate_receipt["result"]["status"], "hydrated")
            self.assertGreater(hydrate_receipt["result"]["content_chars"], 20)
            self.assertEqual(
                document_node.metadata.get("url"),
                "https://example.com/current-analysis",
            )
            self.assertIn(
                "Direct website evidence",
                str(document_node.metadata.get("text") or ""),
            )

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

    def test_filter_browser_findings_portals_prunes_yahoo_artifacts(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())

        findings = worker._filter_browser_findings_portals(
            {
                "candidate_urls": [
                    "https://finance.yahoo.com/quote/AAPL",
                    "https://www.sec.gov/Archives/edgar/data/320193/",
                ],
                "direct_urls": [
                    "https://finance.yahoo.com/quote/AAPL",
                    "https://www.sec.gov/Archives/edgar/data/320193/",
                ],
                "judged_results": [
                    {
                        "url": "https://finance.yahoo.com/quote/AAPL",
                        "domain": "finance.yahoo.com",
                    },
                    {
                        "url": "https://www.sec.gov/Archives/edgar/data/320193/",
                        "domain": "sec.gov",
                    },
                ],
                "discovered_domains": ["finance.yahoo.com", "sec.gov"],
            },
            market_query=True,
        )

        joined = " ".join(findings.get("candidate_urls") or [])
        self.assertNotIn("finance.yahoo.com", joined)
        self.assertEqual(
            findings["direct_urls"],
            ["https://www.sec.gov/Archives/edgar/data/320193/"],
        )
        self.assertEqual(
            [item["domain"] for item in findings["judged_results"]],
            ["sec.gov"],
        )
        self.assertEqual(findings["discovered_domains"], ["sec.gov"])

    def test_sanitize_frontier_graph_prunes_persistent_portal_urls(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        graph = {
            "version": 1,
            "urls": {
                "https://finance.yahoo.com/quote/AAPL": {
                    "url": "https://finance.yahoo.com/quote/AAPL",
                    "domain": "finance.yahoo.com",
                    "priority": 0.9,
                },
                "https://www.sec.gov/Archives/edgar/data/320193/": {
                    "url": "https://www.sec.gov/Archives/edgar/data/320193/",
                    "domain": "sec.gov",
                    "priority": 0.8,
                },
            },
            "domains": {
                "finance.yahoo.com": {
                    "urls": ["https://finance.yahoo.com/quote/AAPL"],
                    "observations": 2,
                },
                "sec.gov": {
                    "urls": ["https://www.sec.gov/Archives/edgar/data/320193/"],
                    "observations": 1,
                },
            },
            "claims": {
                "sample": {
                    "claim": "sample claim",
                    "sources": [
                        "https://finance.yahoo.com/quote/AAPL",
                        "https://www.sec.gov/Archives/edgar/data/320193/",
                    ],
                }
            },
        }

        sanitized = worker._sanitize_frontier_graph(graph, market_query=True)

        self.assertNotIn(
            "https://finance.yahoo.com/quote/AAPL",
            sanitized["urls"],
        )
        self.assertNotIn("finance.yahoo.com", sanitized["domains"])
        self.assertEqual(
            sanitized["claims"]["sample"]["sources"],
            ["https://www.sec.gov/Archives/edgar/data/320193/"],
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

    def test_frontier_graph_seed_urls_drop_market_portal_domains_for_market_runs(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())
        graph = worker._empty_frontier_graph()
        graph["urls"] = {
            "https://finance.yahoo.com/quote/AAA": {
                "url": "https://finance.yahoo.com/quote/AAA",
                "domain": "finance.yahoo.com",
                "priority": 0.95,
                "visits": 0,
                "status": "candidate",
            },
            "https://www.sec.gov/Archives/example-filing": {
                "url": "https://www.sec.gov/Archives/example-filing",
                "domain": "sec.gov",
                "priority": 0.8,
                "visits": 0,
                "status": "candidate",
            },
            "https://www.reuters.com/markets/example-analysis": {
                "url": "https://www.reuters.com/markets/example-analysis",
                "domain": "reuters.com",
                "priority": 0.79,
                "visits": 0,
                "status": "candidate",
            },
        }

        seed_urls = worker._frontier_graph_seed_urls(
            graph,
            limit=10,
            market_query=True,
        )

        self.assertIn("https://www.sec.gov/Archives/example-filing", seed_urls)
        self.assertIn(
            "https://www.reuters.com/markets/example-analysis",
            seed_urls,
        )
        self.assertFalse(any("finance.yahoo.com" in url for url in seed_urls))

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

    def test_browser_findings_recover_cross_domain_coverage_after_rank_collapse(
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
                    title=f"Yahoo market page {index}",
                    url=f"https://finance.yahoo.com/quote/TEST{index}",
                    abstract="Portal quote page.",
                    year=2026,
                    score=95.0 - index,
                )
                for index in range(5)
            ] + [
                ResearchSource(
                    provider="web-search",
                    title="SEC filing",
                    url="https://www.sec.gov/Archives/example-filing",
                    abstract="Primary filing evidence.",
                    year=2026,
                    score=82.0,
                ),
                ResearchSource(
                    provider="web-search",
                    title="Reuters market analysis",
                    url="https://www.reuters.com/markets/example-analysis",
                    abstract="Independent reporting.",
                    year=2026,
                    score=81.0,
                ),
            ]

        worker.research_engine._search_web_results = fake_search_results
        worker.research_engine._search_financial_portals = lambda query, limit=10: []
        worker.research_engine._search_sec_edgar = lambda query, limit=6: []
        worker.research_engine._dedupe_sources = lambda sources: sources
        worker.research_engine._rank_sources = lambda sources, query: [
            source for source in sources if "finance.yahoo.com" in source.url
        ][:5]
        worker._ai_browser_source_strategy = lambda objective: {"targeted_queries": []}
        worker._deep_page_read = lambda url: f"signal from {url} " * 40
        worker._content_block_quality = lambda content: {"quality_score": 0.8}
        worker._browser_preview_is_blocked = lambda preview: False
        worker._extract_page_evidence = lambda content, title, query: [
            "cross-domain evidence"
        ]
        worker._ai_page_judgment = lambda source, content, objective, core_query: (
            "high signal"
        )

        findings = worker._reasoned_browser_findings(
            "[multi-hour] Research public companies with current-web upside catalysts as of now",
            [],
        )

        discovered_domains = set(findings["discovered_domains"])
        self.assertFalse(
            any("finance.yahoo.com" in url for url in findings["candidate_urls"])
        )
        self.assertNotIn("finance.yahoo.com", discovered_domains)
        self.assertIn("sec.gov", discovered_domains)
        self.assertIn("reuters.com", discovered_domains)

    def test_browser_findings_use_preview_fallback_to_preserve_seed_domain_mix(
        self,
    ) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())

        worker.research_engine._search_web_results = lambda query, limit=None: []
        worker.research_engine._search_financial_portals = lambda query, limit=10: []
        worker.research_engine._search_sec_edgar = lambda query, limit=6: []
        worker.research_engine._dedupe_sources = lambda sources: sources
        worker.research_engine._rank_sources = lambda sources, query: sources
        worker._ai_browser_source_strategy = lambda objective: {"targeted_queries": []}
        worker._deep_page_read = lambda url: "signal " * 200 if "sec.gov" in url else ""
        worker._browser_page_preview = lambda url: {
            "page_title": f"Preview for {url}",
            "page_excerpt": "catalyst evidence valuation risk filing data " * 20,
        }
        worker._content_block_quality = lambda content: {"quality_score": 0.8}
        worker._browser_preview_is_blocked = lambda preview: False
        worker._extract_page_evidence = lambda content, title, query: [
            "cross-domain evidence"
        ]
        worker._ai_page_judgment = lambda source, content, objective, core_query: (
            "high signal"
        )

        findings = worker._reasoned_browser_findings(
            "[multi-hour] Research public companies with current-web upside catalysts as of now",
            [
                "https://www.sec.gov/Archives/example-filing",
                "https://www.reuters.com/markets/example-analysis",
                "https://fred.stlouisfed.org/series/GDPC1",
                "https://www.bls.gov/news.release/cpi.nr0.htm",
            ],
            {"search_queries": ["public companies upside catalysts"]},
        )

        discovered_domains = set(findings["discovered_domains"])
        self.assertIn("sec.gov", discovered_domains)
        self.assertIn("reuters.com", discovered_domains)
        self.assertIn("fred.stlouisfed.org", discovered_domains)
        self.assertTrue(
            any(
                "preview-only" in (item.get("quality_flags") or [])
                for item in findings["judged_results"]
                if item.get("domain") in {"reuters.com", "fred.stlouisfed.org"}
            )
        )

    def test_active_pc_research_writes_dashboard_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                worker = WorkerAgent(None, None, FakeResearchEngine())
                worker._run_pc_browser_frontier_session = lambda *args, **kwargs: (
                    [],
                    [],
                    "virtual-desktop-sandbox",
                    {"triggered": False, "reports": [], "navigated_urls": []},
                    {"triggered": False},
                    {
                        "search_queries": ["current market upside"],
                        "judged_results": [{"url": "https://example.com/report"}],
                        "direct_urls": ["https://example.com/report"],
                        "discovered_domains": ["example.com"],
                        "worker_sessions": [{"cycle": 1}],
                    },
                )
                task = TaskSpec(
                    task_id="task_pc_progress",
                    role="pc-research",
                    objective="[multi-hour] Research current market opportunities as of now",
                    declared_actions=[],
                    inputs={"optional": True, "adaptive_effort": "multi-hour"},
                )

                worker._active_pc_research("run_pc_progress", task, [])

                progress_path = root / "runs" / "run_pc_progress" / "research" / "progress.json"
                payload = json.loads(progress_path.read_text(encoding="utf-8"))

                self.assertEqual(payload["stage"], "pc-research-completed")
                self.assertEqual(payload["direct_urls"], 1)
                self.assertEqual(payload["discovered_domains"], 1)
            finally:
                os.chdir(previous_cwd)

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
        self.assertIn("example.com", findings["discovered_domains"])
        self.assertIn("example.org", findings["discovered_domains"])

    def test_browser_findings_reject_js_shell_preview_fallback_pages(self) -> None:
        worker = WorkerAgent(None, None, FakeResearchEngine())

        worker.research_engine._search_web_results = lambda query, limit=None: [
            ResearchSource(
                provider="web-search",
                title="Yahoo shell",
                url="https://finance.yahoo.com/quote/TEST",
                abstract="Script-heavy shell page.",
                year=2026,
                score=55.0,
            ),
            ResearchSource(
                provider="web-search",
                title="Yahoo shell two",
                url="https://finance.yahoo.com/quote/TEST2",
                abstract="Script-heavy shell page.",
                year=2026,
                score=54.0,
            ),
        ]
        worker.research_engine._search_financial_portals = lambda query, limit=10: []
        worker.research_engine._search_sec_edgar = lambda query, limit=6: []
        worker.research_engine._dedupe_sources = lambda sources: sources
        worker.research_engine._rank_sources = lambda sources, query: []
        worker._ai_browser_source_strategy = lambda objective: {"targeted_queries": []}
        worker._deep_page_read = lambda url: ""
        worker._browser_page_preview = lambda url: {
            "page_title": "Please enable JavaScript",
            "page_excerpt": "window.__INITIAL_STATE__ function() const document.body;",
        }

        findings = worker._reasoned_browser_findings(
            "[multi-hour] Research current market opportunities as of now",
            [],
        )

        self.assertGreaterEqual(len(findings["candidate_urls"]), 2)
        self.assertEqual(findings["direct_urls"], [])
        self.assertEqual(findings["judged_results"], [])

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

    def test_worker_tasks_respect_dependencies_parallelism_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = ResearchOrchestrator(
                policy=PermissionPolicy(
                    {
                        "default": "allow",
                        "allow": {
                            "actions": [],
                            "paths": [],
                            "network_hosts": [],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                state_path=Path(temp_dir) / "state.sqlite3",
                memory_path=Path(temp_dir) / "memory.sqlite3",
                max_parallel_workers=2,
            )
            orchestrator._authorize_task = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            orchestrator._enforce_evidence_gate = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

            barrier = threading.Barrier(2)
            attempts: dict[str, int] = {}

            def fake_run(
                run_id: str,
                task: TaskSpec,
                prior_results: list[WorkerResult],
            ) -> WorkerResult:
                del run_id
                attempts[task.task_id] = attempts.get(task.task_id, 0) + 1
                if task.task_id in {"task_a", "task_b"}:
                    try:
                        barrier.wait(timeout=1)
                    except threading.BrokenBarrierError as exc:
                        raise AssertionError(
                            "ready tasks did not execute in parallel"
                        ) from exc
                if task.task_id == "task_retry" and attempts[task.task_id] == 1:
                    raise RuntimeError("transient failure")
                if task.task_id == "task_c":
                    self.assertEqual(
                        {item.task_id for item in prior_results},
                        {"task_a", "task_b", "task_retry"},
                    )
                return WorkerResult(
                    task_id=task.task_id,
                    role=task.role,
                    summary=f"completed {task.task_id}",
                    evidence=[{"source": task.role, "claim": "ok"}],
                    confidence=0.8,
                )

            orchestrator.worker.run = fake_run  # type: ignore[method-assign]

            tasks = [
                TaskSpec(
                    task_id="task_a",
                    role="planning",
                    objective="plan",
                    declared_actions=[],
                    required_capabilities=["planning"],
                    synthesis_contract={"section": "plan", "required": True},
                ),
                TaskSpec(
                    task_id="task_b",
                    role="pc-control",
                    objective="snapshot",
                    declared_actions=[],
                    required_capabilities=["pc-control"],
                    synthesis_contract={"section": "desktop", "required": False},
                ),
                TaskSpec(
                    task_id="task_retry",
                    role="data",
                    objective="constraints",
                    declared_actions=[],
                    max_attempts=2,
                    required_capabilities=["data-extraction"],
                    synthesis_contract={
                        "section": "constraints",
                        "required": True,
                    },
                ),
                TaskSpec(
                    task_id="task_c",
                    role="synthesis",
                    objective="brief",
                    declared_actions=[],
                    depends_on=["task_a", "task_b", "task_retry"],
                    required_capabilities=["synthesis"],
                    synthesis_contract={"section": "brief", "required": True},
                ),
            ]

            orchestrator.runtime.save_manifest(
                "run_sched",
                "test scheduling objective",
                [asdict(task) for task in tasks],
            )

            results = orchestrator._execute_worker_tasks("run_sched", tasks)

            self.assertEqual(
                [result.task_id for result in results],
                ["task_a", "task_b", "task_retry", "task_c"],
            )
            self.assertEqual(attempts["task_retry"], 2)
            retry_result = next(
                result for result in results if result.task_id == "task_retry"
            )
            self.assertEqual(retry_result.attempt_count, 2)
            self.assertGreater(retry_result.provenance_score, 0.0)
            synthesis_result = next(
                result for result in results if result.task_id == "task_c"
            )
            self.assertEqual(
                synthesis_result.metadata["depends_on"],
                ["task_a", "task_b", "task_retry"],
            )

    def test_optional_task_skips_when_required_capability_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = ResearchOrchestrator(
                policy=PermissionPolicy(
                    {
                        "default": "allow",
                        "allow": {
                            "actions": [],
                            "paths": [],
                            "network_hosts": [],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                state_path=Path(temp_dir) / "state.sqlite3",
                memory_path=Path(temp_dir) / "memory.sqlite3",
            )
            orchestrator._authorize_task = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            orchestrator.worker.available_capabilities = lambda: {"planning"}  # type: ignore[method-assign]

            results = orchestrator._execute_worker_tasks(
                "run_caps",
                [
                    TaskSpec(
                        task_id="task_optional",
                        role="pc-research",
                        objective="optional research",
                        declared_actions=[],
                        inputs={"optional": True},
                        required_capabilities=["sandbox.exec"],
                    )
                ],
            )

            self.assertEqual(len(results), 1)
            self.assertIn("Skipped optional pc-research step", results[0].summary)

    def test_synthesis_contract_captures_sections_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = ResearchOrchestrator(
                policy=PermissionPolicy(
                    {
                        "default": "allow",
                        "allow": {
                            "actions": [],
                            "paths": [],
                            "network_hosts": [],
                        },
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": []},
                    }
                ),
                state_path=Path(temp_dir) / "state.sqlite3",
                memory_path=Path(temp_dir) / "memory.sqlite3",
            )
            results = [
                WorkerResult(
                    task_id="task_plan",
                    role="planning",
                    summary="planned",
                    confidence=0.8,
                    provenance_score=0.72,
                    metadata={
                        "synthesis_contract": {
                            "section": "research_plan",
                            "required": True,
                        }
                    },
                ),
                WorkerResult(
                    task_id="task_lit",
                    role="literature",
                    summary="evidence",
                    confidence=0.9,
                    provenance_score=0.83,
                    metadata={
                        "synthesis_contract": {
                            "section": "authoritative_evidence",
                            "required": True,
                        }
                    },
                ),
            ]
            verification = WorkerResult(
                task_id="verification",
                role="verification",
                summary="verified",
                confidence=0.88,
                provenance_score=0.9,
            )

            contract = orchestrator._build_synthesis_contract(
                "run_contract",
                "test objective",
                [*results, verification],
                verification,
            )

            self.assertEqual(
                contract["resolved_sections"],
                ["research_plan", "authoritative_evidence"],
            )
            self.assertEqual(contract["missing_sections"], [])
            self.assertGreater(contract["provenance"]["average"], 0.7)


if __name__ == "__main__":
    unittest.main()
