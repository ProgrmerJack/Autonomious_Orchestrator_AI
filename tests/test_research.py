from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agentos_orchestrator.product import CrawlBrokerServer, CrawlWorkerServiceRecord
from agentos_orchestrator.research import DeepResearchEngine, ResearchSource
from agentos_orchestrator.research.crawl_worker import (
    CrawlWorkerLoopConfig,
    ResearchCrawlWorker,
)


class FakeDeepResearchEngine(DeepResearchEngine):
    def _get_json(self, url: str) -> dict[str, Any]:
        if "openalex" in url:
            return {
                "results": [
                    {
                        "display_name": "Accessibility Tree Agents",
                        "publication_year": 2025,
                        "authorships": [{"author": {"display_name": "A. Researcher"}}],
                        "abstract_inverted_index": {
                            "Accessibility": [0],
                            "agents": [1],
                            "control": [2],
                        },
                        "cited_by_count": 7,
                        "id": "https://openalex.org/W1",
                    },
                    {
                        "display_name": ("Membrane Transporters in Drug Development"),
                        "publication_year": 2010,
                        "authorships": [],
                        "abstract_inverted_index": {
                            "Drug": [0],
                            "development": [1],
                            "accessibility": [2],
                        },
                        "cited_by_count": 3000,
                        "id": "https://openalex.org/W2",
                    },
                ]
            }
        if "api.github.com" in url:
            return {
                "items": [
                    {
                        "full_name": "All-Hands-AI/OpenHands",
                        "html_url": ("https://github.com/All-Hands-AI/OpenHands"),
                        "description": (
                            "OpenHands is a software agent platform for "
                            "coding workflows, OpenCode comparison, "
                            "OpenClaw comparison, local PC agents, and "
                            "computer use."
                        ),
                        "stargazers_count": 1000,
                        "updated_at": "2026-01-01T00:00:00Z",
                        "topics": ["agents", "coding", "automation"],
                        "owner": {"login": "All-Hands-AI"},
                    }
                ]
            }
        return {
            "data": [
                {
                    "title": "Desktop Agent Evaluation",
                    "abstract": "Desktop agents need grounded evaluation.",
                    "authors": [{"name": "B. Scientist"}],
                    "year": 2024,
                    "url": "https://semanticscholar.org/paper/1",
                    "citationCount": 3,
                }
            ]
        }

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        del (
            url,
            accept,
            max_bytes,
            timeout_seconds,
            range_start,
            range_end,
            extra_headers,
        )
        return ""


class FakeWebSearchResearchEngine(DeepResearchEngine):
    def _get_json(self, url: str) -> dict[str, Any]:
        del url
        return {}

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        del accept, max_bytes, timeout_seconds, range_start, range_end, extra_headers
        if "html.duckduckgo.com" in url:
            return (
                "<html><body>"
                '<a class="result__a" href="https://docs.example.org/agentos">'
                "AgentOS Safety Docs</a>"
                '<div class="result__snippet">'
                "AgentOS desktop workflow safety benchmark approvals reference."
                "</div>"
                "</body></html>"
            )
        if "docs.example.org/agentos" in url:
            return (
                "<html><head><title>AgentOS Safety Docs</title></head><body>"
                "AgentOS desktop workflow safety approvals reference and "
                "benchmark notes for autonomous desktop agent evaluation "
                "including approval gating, safety constraints, and workflow "
                "orchestration guidelines."
                "</body></html>"
            )
        return ""


class FakeBrowserPoolResearchEngine(DeepResearchEngine):
    def __init__(self, workspace_root: str | Path = ".") -> None:
        super().__init__(workspace_root=workspace_root)
        self.pool_calls: list[list[str]] = []

    def _needs_browser(self, url: str) -> bool:
        del url
        return True

    def _headless_browser_pool_fetch(
        self,
        urls: list[str],
        max_chars: int = 80_000,
        timeout_ms: int = 18_000,
    ) -> dict[str, str]:
        del max_chars, timeout_ms
        self.pool_calls.append(list(urls))
        return {
            url: (
                "Browser rendered evidence describing planner worker browser "
                "grounding, tool execution, retrieval coordination, contradiction "
                "handling, evidence index reuse, crawl queue fanout, headless "
                "rendering, benchmark safety approvals, semantic navigation, "
                "source validation, and cross run orchestration behavior across "
                "general purpose research agents."
            )
            for url in urls
        }

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        del (
            url,
            accept,
            max_bytes,
            timeout_seconds,
            range_start,
            range_end,
            extra_headers,
        )
        return ""


class SeedUrlResearchEngine(FakeDeepResearchEngine):
    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        del accept, max_bytes, timeout_seconds, range_start, range_end, extra_headers
        if "docs.example.org/agentos" in url:
            return (
                "<html><head><title>AgentOS benchmark safety approvals</title></head>"
                "<body>AgentOS benchmark safety approvals desktop workflow "
                "reliability notes and operator guidance.</body></html>"
            )
        return ""


class AutoPcContextResearchEngine(FakeDeepResearchEngine):
    def _headless_browser_pool_fetch(
        self,
        urls: list[str],
        max_chars: int = 80_000,
        timeout_ms: int = 18_000,
    ) -> dict[str, str]:
        del max_chars, timeout_ms
        return {
            url: (
                "AgentOS browser frontier evidence covers planner routing, "
                "sandbox/browser grounding, retrieval breadth, benchmark safety, "
                "and useful evidence quality for general purpose deep research "
                "agents."
            )
            for url in urls
        }


class FakeCrawlWorkerResearchEngine(DeepResearchEngine):
    def _get_json(self, url: str) -> dict[str, Any]:
        del url
        return {}

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        del accept, max_bytes, timeout_seconds, range_start, range_end, extra_headers
        if "docs.example.org/root" in url:
            return (
                "<html><body>"
                "AgentOS semantic browser research coordination covers frontier "
                "graph routing, contradiction review, crawl queue fanout, "
                "evidence reuse, and reliability controls for general purpose "
                "deep research systems. "
                '<a href="https://docs.example.org/articles/agentos-semantic-browser-worker-pool">'
                "Worker pool article</a>"
                "</body></html>"
            )
        if "agentos-semantic-browser-worker-pool" in url:
            return (
                "<html><body>"
                "AgentOS worker pools maintain browser grounded evidence, queue "
                "draining, semantic navigation, contradiction tracking, and "
                "persistent cross run observations for deep research workloads."
                "</body></html>"
            )
        return ""


class NoSnippetWebSearchResearchEngine(DeepResearchEngine):
    def _get_json(self, url: str) -> dict[str, Any]:
        del url
        return {}

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        del accept, max_bytes, timeout_seconds, range_start, range_end, extra_headers
        if "html.duckduckgo.com" in url:
            return (
                "<html><body>"
                '<a class="result__a" href="https://docs.example.org/agentos">'
                "AgentOS Safety Docs</a>"
                "</body></html>"
            )
        if "docs.example.org/agentos" in url:
            return (
                "<html><body>AgentOS desktop workflow safety approvals reference "
                "and benchmark notes.</body></html>"
            )
        return ""


class RankingProbe(DeepResearchEngine):
    @classmethod
    def rank_sources(
        cls,
        sources: list[ResearchSource],
        query: str,
    ) -> list[ResearchSource]:
        return cls._rank_sources(sources, query)


class NoisyGapAnalysisEngine(DeepResearchEngine):
    def _call_ai_text(self, system: str, user: str) -> str:
        del system, user
        return json.dumps(
            {
                "gaps": ["missing recency and catalyst evidence"],
                "follow_up_queries": [
                    "stocks right eqxmuk border-through ezyuzk",
                    "stocks catalysts earnings revisions current analysis",
                ],
            }
        )


class CapturingSynthesisEngine(FakeDeepResearchEngine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_system = ""
        self.last_user = ""

    def _call_ai_text(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return "SYNTHESIS"


class StableLowNoveltyEngine(DeepResearchEngine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.search_calls: list[str] = []

    def _search_query_across_providers(
        self,
        search_query: str,
        allowed_providers: set[str],
        per_provider_limit: int,
    ) -> list[ResearchSource]:
        del allowed_providers, per_provider_limit
        self.search_calls.append(search_query)
        return [
            ResearchSource(
                provider="google-news-rss",
                title="NVIDIA (NVDA) stock earnings guidance catalyst",
                url="https://finance.example.com/nvda-earnings",
                year=2026,
                abstract=(
                    "NVIDIA stock earnings guidance showed revenue growth, "
                    "EPS upside, valuation debate, and catalyst timing."
                ),
                score=5.0,
            )
        ]

    def _search_gemini_observation(
        self,
        query: str,
        depth: str,
    ) -> list[ResearchSource]:
        del query, depth
        return []

    def _enrich_top_sources(
        self,
        selected: list[ResearchSource],
        query: str,
    ) -> list[str]:
        del selected, query
        return []

    def _citation_chase(
        self,
        sources: list[ResearchSource],
        query: str,
        citation_depth: int = 1,
    ) -> list[ResearchSource]:
        del sources, query, citation_depth
        return []

    def _append_durable_claim_notes(
        self,
        report_path: Path,
        pass_index: int,
        sources: list[ResearchSource],
        query: str,
    ) -> None:
        del report_path, pass_index, sources, query

    def _ai_evidence_gap_analysis(
        self,
        objective: str,
        selected: list[ResearchSource],
        pass_index: int,
        force_new_domains: bool = False,
        existing_domains: list[str] | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> list[str]:
        del selected, pass_index, force_new_domains, existing_domains, coverage
        return [f"{objective} SEC filing revenue growth catalyst"]


class NoisyEnrichmentResearchEngine(StableLowNoveltyEngine):
    def _enrich_top_sources(
        self,
        selected: list[ResearchSource],
        query: str,
    ) -> list[str]:
        del selected, query
        return [
            "research agent deep research google deep vfppkd vfppkd-strngf",
            "deep research agent benchmark comparison",
        ]


class SlowMarginalYieldEngine(StableLowNoveltyEngine):
    def _search_query_across_providers(
        self,
        search_query: str,
        allowed_providers: set[str],
        per_provider_limit: int,
    ) -> list[ResearchSource]:
        del allowed_providers, per_provider_limit
        self.search_calls.append(search_query)
        source_index = len(self.search_calls)
        return [
            ResearchSource(
                provider="google-news-rss",
                title=f"NVIDIA (NVDA) catalyst update {source_index}",
                url=f"https://finance.example.com/nvda-catalyst-{source_index}",
                year=2026,
                abstract=(
                    "NVIDIA catalyst update with one incremental earnings or "
                    "valuation detail per pass."
                ),
                score=5.0,
            )
        ]

    def _refinement_variants(
        self,
        query: str,
        selected: list[ResearchSource],
        depth: str,
        pass_index: int,
        plan: dict[str, Any],
    ) -> list[str]:
        del query, selected, depth, pass_index, plan
        return []


class MixedProviderDispatchEngine(DeepResearchEngine):
    def _provider_order(
        self,
        search_query: str = "",
        allowed_providers: set[str] | None = None,
    ) -> tuple[str, ...]:
        del search_query, allowed_providers
        return ("openalex", "web-search")

    def _provider_parallel_worker_count(self, provider_count: int) -> int:
        return provider_count

    def _provider_searchers(self) -> dict[str, Any]:
        return {
            "openalex": self._search_openalex,
            "web-search": self._search_web_results,
        }

    async def _search_openalex_async(
        self,
        query: str,
        limit: int | None = None,
        client: Any | None = None,
    ) -> list[ResearchSource]:
        del limit, client
        return [
            ResearchSource(
                provider="openalex",
                title=f"async {query}",
                url="https://example.com/async",
                abstract="async provider",
                score=2.0,
            )
        ]

    def _search_web_results(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        del limit
        return [
            ResearchSource(
                provider="web-search",
                title=f"sync {query}",
                url="https://example.com/sync",
                abstract="sync provider",
                score=1.0,
            )
        ]

    def _ai_evidence_gap_analysis(
        self,
        objective: str,
        selected: list[ResearchSource],
        pass_index: int,
        force_new_domains: bool = False,
        existing_domains: list[str] | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> list[str]:
        del (
            objective,
            selected,
            pass_index,
            force_new_domains,
            existing_domains,
            coverage,
        )
        return []


class ResearchTests(unittest.TestCase):
    def test_deep_research_engine_writes_evidence_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run("accessibility tree desktop agents", "run_1")

            self.assertGreaterEqual(len(brief.sources), 2)
            self.assertGreater(brief.confidence, 0.6)
            for artifact in brief.artifacts:
                self.assertTrue((Path(temp_dir) / artifact).exists())
            self.assertIn("digest.json", " ".join(brief.artifacts))
            self.assertIn("research_plan.json", " ".join(brief.artifacts))
            self.assertIn("analysis_report.md", " ".join(brief.artifacts))
            self.assertIn("findings.json", " ".join(brief.artifacts))
            self.assertIn("claim_trace.json", " ".join(brief.artifacts))
            self.assertIn(
                "evidence_index_snapshot.json",
                " ".join(brief.artifacts),
            )
            self.assertIn(
                "crawl_queue_snapshot.json",
                " ".join(brief.artifacts),
            )
            self.assertIn(
                "provider_diagnostics.json",
                " ".join(brief.artifacts),
            )

            sources_path = Path(temp_dir) / "runs/run_1/research/sources.json"
            payload = json.loads(sources_path.read_text(encoding="utf-8"))
            by_title = {source["title"]: source for source in payload}
            self.assertIn("Accessibility Tree Agents", by_title)
            accessibility = by_title["Accessibility Tree Agents"]
            self.assertIn(
                accessibility["evidence_grade"],
                {"strong", "moderate"},
            )
            self.assertGreater(accessibility["relevance"], 0)
            self.assertNotIn(
                "Membrane Transporters in Drug Development",
                set(by_title),
            )

            claim_trace_path = Path(temp_dir) / "runs/run_1/research/claim_trace.json"
            claim_trace = json.loads(claim_trace_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(claim_trace["source_count"], 1)
            self.assertTrue(claim_trace["claims"])
            self.assertGreater(
                claim_trace["claims"][0]["support_count"],
                0,
            )

            findings_path = Path(temp_dir) / "runs/run_1/research/findings.json"
            findings = json.loads(findings_path.read_text(encoding="utf-8"))
            self.assertTrue(findings)
            self.assertIn("perspective", findings[0])

            evidence_snapshot_path = (
                Path(temp_dir) / "runs/run_1/research/evidence_index_snapshot.json"
            )
            evidence_snapshot = json.loads(
                evidence_snapshot_path.read_text(encoding="utf-8")
            )
            self.assertTrue(evidence_snapshot["claims"])

    def test_persistent_evidence_index_reuses_claims_and_seed_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            engine.run("accessibility tree desktop agents", "run_1")

            followup = FakeDeepResearchEngine(workspace_root=temp_dir)
            query = followup._query_from_objective("accessibility tree desktop agents")
            hints = followup._persistent_evidence_query_hints(
                query,
                "accessibility tree desktop agents",
                limit=8,
            )
            seeds = followup._persistent_seed_urls(
                query,
                "accessibility tree desktop agents",
                limit=8,
            )

            self.assertTrue(hints)
            self.assertTrue(seeds)

            db_path = Path(temp_dir) / ".agentos/research_state.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection:
                claim_count = connection.execute(
                    "SELECT COUNT(*) FROM evidence_claims"
                ).fetchone()[0]
                domain_count = connection.execute(
                    "SELECT COUNT(*) FROM evidence_domains"
                ).fetchone()[0]

            self.assertGreater(claim_count, 0)
            self.assertGreater(domain_count, 0)

    def test_persistent_query_hints_strip_instruction_fragments_and_portal_site_bias(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = DeepResearchEngine(workspace_root=temp_dir)
            objective = (
                "As of right now, research which publicly traded companies have the highest "
                "probability-adjusted upside potential over the next 12 to 24 months. Use all available "
                "general-purpose research means, including browser-grounded web research, sandboxed exploration, "
                "current-web evidence, company filings, earnings material, product signals, independent sources, "
                "and cross-checking. Do not use finance-specific hardcoded templates or domain-specific shortcuts."
            )
            query = engine._query_from_objective(objective)

            engine._record_crawl_observation(
                ResearchSource(
                    provider="web-search",
                    title="Yahoo quote page",
                    url="https://finance.yahoo.com/quote/TEST",
                    abstract="Portal quote page.",
                    year=2026,
                    score=60.0,
                ),
                "Current market catalyst evidence and price moves. " * 20,
                (
                    "publicly traded companies highest probability-adjusted upside potential 12 24 months "
                    "use all available general-purpose means including not"
                ),
                [
                    "site:finance.yahoo.com publicly traded companies upside potential",
                    (
                        "publicly traded companies highest probability-adjusted upside potential 12 24 months "
                        "use all available general-purpose means including not"
                    ),
                ],
                [],
                "worker-1",
                False,
            )
            engine._record_crawl_observation(
                ResearchSource(
                    provider="web-search",
                    title="SEC filing",
                    url="https://www.sec.gov/Archives/example-filing",
                    abstract="Primary filing evidence.",
                    year=2026,
                    score=80.0,
                ),
                "Company filing evidence and current catalysts. " * 20,
                query,
                ["site:sec.gov publicly traded companies upside potential"],
                [],
                "worker-1",
                False,
            )

            hints = engine._persistent_evidence_query_hints(query, objective, limit=8)

            self.assertTrue(hints)
            self.assertFalse(any("use all available" in hint.lower() for hint in hints))
            self.assertFalse(any("site:finance.yahoo.com" in hint.lower() for hint in hints))
            self.assertTrue(
                any(
                    "sec" in hint.lower() and "gov" in hint.lower()
                    for hint in hints
                )
            )

    def test_persistent_crawl_queue_claims_sources_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = DeepResearchEngine(workspace_root=temp_dir)
            engine._ensure_research_state_store()
            urls = [
                "https://docs.example.org/agentos/frontier",
                "https://research.example.org/agentos/worker-pool",
            ]
            engine._enqueue_url_batch(
                urls,
                "agentos frontier graph deep research",
                "run_1",
                source_url="https://seed.example.org/objective",
                priority=13.0,
            )

            followup = DeepResearchEngine(workspace_root=temp_dir)
            claimed = followup._claim_persistent_crawl_sources(
                "agentos frontier graph deep research",
                "agentos frontier graph deep research",
                limit=4,
            )
            claimed_urls = {source.url for source in claimed}

            self.assertEqual(claimed_urls, set(urls))

            snapshot = followup._persistent_crawl_queue_snapshot(limit=8)
            statuses = {item["url"]: item["status"] for item in snapshot["queued"]}
            self.assertEqual(statuses[urls[0]], "claimed")
            self.assertEqual(statuses[urls[1]], "claimed")

    def test_detached_crawl_worker_scaling_is_backlog_sensitive(self) -> None:
        engine = DeepResearchEngine()

        with patch("agentos_orchestrator.research.crawl_state.os.cpu_count", return_value=12):
            self.assertEqual(engine._detached_crawl_worker_count(32), 1)
            self.assertEqual(engine._detached_crawl_worker_count(160), 2)
            self.assertEqual(engine._detached_crawl_worker_count(512), 4)
            self.assertEqual(engine._detached_crawl_batch_size(32), 6)
            self.assertEqual(engine._detached_crawl_batch_size(160), 8)
            self.assertEqual(engine._detached_crawl_batch_size(512), 16)

    def test_detached_crawl_auto_start_waits_for_meaningful_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = DeepResearchEngine(workspace_root=temp_dir)

            with (
                patch.object(engine, "_auto_start_crawl_workers_enabled", return_value=True),
                patch.object(engine, "_queued_crawl_backlog_count", return_value=8),
                patch("agentos_orchestrator.product.CrawlWorkerServiceManager") as service_manager_cls,
                patch("agentos_orchestrator.product.CrawlWorkerManager") as manager_cls,
            ):
                engine._maybe_start_detached_crawl_workers()

            service_manager_cls.assert_not_called()
            manager_cls.assert_not_called()

    def test_detached_crawl_worker_processes_queue_and_records_observations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            objective = "agentos semantic browser deep research reliability"
            root_url = "https://docs.example.org/root"
            child_url = (
                "https://docs.example.org/articles/agentos-semantic-browser-worker-pool"
            )
            engine = FakeCrawlWorkerResearchEngine(workspace_root=temp_dir)
            engine._ensure_research_state_store()
            engine._enqueue_url_batch(
                [root_url],
                objective,
                "run_1",
                source_url="https://seed.example.org/objective",
                priority=12.0,
            )

            worker = ResearchCrawlWorker(
                engine,
                worker_id="worker-1",
                config=CrawlWorkerLoopConfig(
                    batch_size=2,
                    claim_ttl_seconds=60,
                    once=True,
                ),
            )
            result = worker.run_once()

            self.assertEqual(result["processed_count"], 1)
            self.assertEqual(result["failed_count"], 0)
            self.assertGreaterEqual(result["enqueued_count"], 1)

            db_path = Path(temp_dir) / ".agentos/research_state.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection:
                observation_count = connection.execute(
                    "SELECT COUNT(*) FROM crawl_observations"
                ).fetchone()[0]
                root_status = connection.execute(
                    "SELECT status FROM crawl_queue WHERE url = ?",
                    (root_url,),
                ).fetchone()[0]
                child_status = connection.execute(
                    "SELECT status FROM crawl_queue WHERE url = ?",
                    (child_url,),
                ).fetchone()[0]

            self.assertEqual(observation_count, 1)
            self.assertEqual(root_status, "processed")
            self.assertEqual(child_status, "queued")

            followup = FakeCrawlWorkerResearchEngine(workspace_root=temp_dir)
            query = followup._query_from_objective(objective)
            hints = followup._persistent_evidence_query_hints(
                query,
                objective,
                limit=8,
            )
            seeds = followup._persistent_seed_urls(query, objective, limit=8)

            self.assertTrue(hints)
            self.assertIn(root_url, seeds)
            self.assertIn(child_url, seeds)

    def test_crawl_broker_shares_queue_and_evidence_across_remote_workers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = CrawlBrokerServer(
                workspace_root=temp_dir,
                host="127.0.0.1",
                port=0,
                auth_token="secret-token",
            )
            broker.start_in_thread()
            try:
                objective = "agentos semantic browser deep research reliability"
                root_url = "https://docs.example.org/root"
                child_url = (
                    "https://docs.example.org/articles/"
                    "agentos-semantic-browser-worker-pool"
                )
                coordinator = DeepResearchEngine(
                    workspace_root=Path(temp_dir) / "client",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                coordinator._enqueue_url_batch(
                    [root_url],
                    objective,
                    "run_remote_broker",
                    source_url="https://seed.example.org/objective",
                    priority=12.0,
                )

                pre_snapshot = coordinator._persistent_crawl_queue_snapshot(limit=8)
                self.assertEqual(pre_snapshot["queued"][0]["url"], root_url)
                self.assertEqual(pre_snapshot["queued"][0]["status"], "queued")

                worker_engine = FakeCrawlWorkerResearchEngine(
                    workspace_root=Path(temp_dir) / "remote-worker",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                worker = ResearchCrawlWorker(
                    worker_engine,
                    worker_id="remote-worker-1",
                    config=CrawlWorkerLoopConfig(
                        batch_size=2,
                        claim_ttl_seconds=60,
                        once=True,
                    ),
                )
                result = worker.run_once()

                self.assertEqual(result["processed_count"], 1)
                self.assertGreaterEqual(result["enqueued_count"], 1)

                query = coordinator._query_from_objective(objective)
                hints = coordinator._persistent_evidence_query_hints(
                    query,
                    objective,
                    limit=8,
                )
                seeds = coordinator._persistent_seed_urls(
                    query,
                    objective,
                    limit=8,
                )
                post_snapshot = coordinator._persistent_crawl_queue_snapshot(limit=8)
                statuses = {
                    item["url"]: item["status"] for item in post_snapshot["queued"]
                }

                self.assertTrue(hints)
                self.assertIn(root_url, seeds)
                self.assertIn(child_url, seeds)
                self.assertEqual(statuses[root_url], "processed")
            finally:
                broker.shutdown()

    def test_crawl_broker_uses_sharded_queue_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = CrawlBrokerServer(
                workspace_root=temp_dir,
                host="127.0.0.1",
                port=0,
                auth_token="secret-token",
                shard_count=3,
            )
            broker.start_in_thread()
            try:
                urls: list[str] = []
                seen_shards: set[int] = set()
                candidate_index = 0
                while len(seen_shards) < 3 and candidate_index < 40:
                    url = f"https://docs{candidate_index}.example.org/page"
                    shard_index = broker.store._shard_index_for_key(url)
                    if shard_index not in seen_shards:
                        seen_shards.add(shard_index)
                        urls.append(url)
                    candidate_index += 1

                coordinator = DeepResearchEngine(
                    workspace_root=Path(temp_dir) / "client",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                coordinator._enqueue_url_batch(
                    urls,
                    "agentos sharded crawl broker",
                    "run_sharded_broker",
                    source_url="https://seed.example.org/objective",
                    priority=10.0,
                )

                snapshot = coordinator._persistent_crawl_queue_snapshot(limit=16)
                self.assertEqual(
                    {item["url"] for item in snapshot["queued"]}, set(urls)
                )
                status = broker.status()
                self.assertEqual(status.backend, "sqlite-sharded")
                self.assertEqual(status.shard_count, 3)
                self.assertEqual(len(broker.store.engines), 3)
                self.assertGreaterEqual(
                    sum(1 for path in broker.store._shard_paths if path.exists()),
                    2,
                )
            finally:
                broker.shutdown()

    def test_crawl_broker_routes_js_claims_to_browser_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = CrawlBrokerServer(
                workspace_root=temp_dir,
                host="127.0.0.1",
                port=0,
                auth_token="secret-token",
                shard_count=2,
            )
            broker.start_in_thread()
            try:
                objective = "agentos browser render pool routing"
                js_url = "https://www.bloomberg.com/markets/example"
                docs_url = "https://docs.example.org/browser-pool"
                coordinator = DeepResearchEngine(
                    workspace_root=Path(temp_dir) / "client",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                coordinator._enqueue_url_batch(
                    [docs_url, js_url],
                    objective,
                    "run_js_routing",
                    source_url="https://seed.example.org/objective",
                    priority=9.0,
                )

                browser_claim = coordinator._claim_crawl_queue_batch(
                    1,
                    "browser-renderer-1",
                    prefer_js_required=True,
                    max_claims_per_domain=1,
                )
                general_claim = coordinator._claim_crawl_queue_batch(
                    2,
                    "general-worker-1",
                    allow_js_required=False,
                    max_claims_per_domain=1,
                )

                self.assertEqual([row["url"] for row in browser_claim], [js_url])
                self.assertEqual([row["url"] for row in general_claim], [docs_url])
            finally:
                broker.shutdown()

    def test_crawl_broker_throttles_same_domain_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = CrawlBrokerServer(
                workspace_root=temp_dir,
                host="127.0.0.1",
                port=0,
                auth_token="secret-token",
                shard_count=2,
            )
            broker.start_in_thread()
            try:
                objective = "agentos domain throttle scheduling"
                urls = [
                    "https://docs.example.org/a",
                    "https://docs.example.org/b",
                    "https://other.example.org/c",
                ]
                coordinator = DeepResearchEngine(
                    workspace_root=Path(temp_dir) / "client",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                coordinator._enqueue_url_batch(
                    urls,
                    objective,
                    "run_domain_throttle",
                    source_url="https://seed.example.org/objective",
                    priority=11.0,
                )

                first_claim = coordinator._claim_crawl_queue_batch(
                    3,
                    "worker-1",
                    max_claims_per_domain=1,
                    default_domain_cooldown_seconds=60.0,
                    js_domain_cooldown_seconds=60.0,
                )
                second_claim = coordinator._claim_crawl_queue_batch(
                    2,
                    "worker-2",
                    max_claims_per_domain=1,
                    default_domain_cooldown_seconds=60.0,
                    js_domain_cooldown_seconds=60.0,
                )

                first_urls = [row["url"] for row in first_claim]
                self.assertEqual(len(first_urls), 2)
                self.assertEqual(
                    sum(1 for url in first_urls if "docs.example.org" in url),
                    1,
                )
                self.assertEqual(second_claim, [])
            finally:
                broker.shutdown()

    def test_crawl_broker_metrics_report_leases_and_worker_utilization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = CrawlBrokerServer(
                workspace_root=temp_dir,
                host="127.0.0.1",
                port=0,
                auth_token="secret-token",
                shard_count=2,
            )
            broker.start_in_thread()
            try:
                objective = "agentos broker observability metrics"
                docs_url = "https://docs.example.org/metrics"
                js_url = "https://www.bloomberg.com/markets/agentos-observability"
                coordinator = DeepResearchEngine(
                    workspace_root=Path(temp_dir) / "client",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                coordinator._enqueue_url_batch(
                    [docs_url, js_url],
                    objective,
                    "run_metrics",
                    source_url="https://seed.example.org/objective",
                    priority=10.0,
                )

                claimed = coordinator._claim_crawl_queue_batch(
                    1,
                    "browser-renderer-1",
                    prefer_js_required=True,
                    max_claims_per_domain=1,
                    default_domain_cooldown_seconds=5.0,
                    js_domain_cooldown_seconds=15.0,
                )
                self.assertEqual([row["url"] for row in claimed], [js_url])

                coordinator._record_crawl_observation(
                    ResearchSource(
                        provider="test",
                        title="AgentOS observability",
                        url=js_url,
                        authors=["bloomberg.com"],
                        abstract="JS-heavy source",
                        citation_count=0,
                        score=1.0,
                        quality_flags=["js-render-required"],
                    ),
                    "browser rendered content",
                    objective,
                    ["agentos metrics"],
                    [],
                    "browser-renderer-1",
                    True,
                )

                metrics = coordinator.crawl_broker_metrics()

                self.assertEqual(metrics["queue"]["total"], 2)
                self.assertEqual(metrics["queue"]["status_counts"]["claimed"], 1)
                self.assertEqual(metrics["queue"]["status_counts"]["queued"], 1)
                self.assertEqual(
                    metrics["queue"]["js_required_counts"]["claimed"],
                    1,
                )
                self.assertEqual(metrics["domain_leases"]["active_count"], 1)
                self.assertEqual(len(metrics["shards"]), 2)
                worker_stats = {
                    item["worker_id"]: item
                    for item in metrics["worker_utilization"]["items"]
                }
                self.assertIn("browser-renderer-1", worker_stats)
                self.assertEqual(
                    worker_stats["browser-renderer-1"]["active_js_claims"],
                    1,
                )
                self.assertEqual(
                    worker_stats["browser-renderer-1"]["browser_observations"],
                    1,
                )
                self.assertIn(
                    "bloomberg.com",
                    [lease["domain"] for lease in metrics["domain_leases"]["items"]],
                )
            finally:
                broker.shutdown()

    def test_crawl_broker_queue_inspect_filters_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broker = CrawlBrokerServer(
                workspace_root=temp_dir,
                host="127.0.0.1",
                port=0,
                auth_token="secret-token",
                shard_count=3,
            )
            broker.start_in_thread()
            try:
                objective = "agentos broker queue inspect"
                docs_url = "https://docs.example.org/inspect"
                js_url = "https://www.bloomberg.com/news/agentos-queue-inspect"
                coordinator = DeepResearchEngine(
                    workspace_root=Path(temp_dir) / "client",
                    crawl_broker_url=broker.url,
                    crawl_broker_token="secret-token",
                )
                coordinator._enqueue_url_batch(
                    [docs_url, js_url],
                    objective,
                    "run_inspect",
                    source_url="https://seed.example.org/objective",
                    priority=9.0,
                )

                claimed = coordinator._claim_crawl_queue_batch(
                    1,
                    "worker-1",
                    allow_js_required=False,
                    max_claims_per_domain=1,
                )
                self.assertEqual([row["url"] for row in claimed], [docs_url])

                claimed_rows = coordinator.crawl_broker_queue_inspect(
                    limit=8,
                    statuses=["claimed"],
                    worker_id="worker-1",
                )
                js_rows = coordinator.crawl_broker_queue_inspect(
                    limit=8,
                    statuses=["queued"],
                    js_required=True,
                )

                self.assertEqual(claimed_rows["total_matches"], 1)
                self.assertEqual(claimed_rows["items"][0]["url"], docs_url)
                self.assertEqual(
                    claimed_rows["items"][0]["last_claimed_by"],
                    "worker-1",
                )
                self.assertEqual(js_rows["total_matches"], 1)
                self.assertEqual(js_rows["items"][0]["url"], js_url)
                self.assertTrue(js_rows["items"][0]["js_required"])
            finally:
                broker.shutdown()

    def test_auto_start_prefers_installed_crawl_worker_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_db = Path(temp_dir) / ".agentos/research_state.sqlite3"
            engine = DeepResearchEngine(workspace_root=temp_dir)
            service_status = CrawlWorkerServiceRecord(
                status="installed",
                task_name="AgentOS Test Crawl Service",
                supported=True,
                installed=True,
                backend="windows-task-scheduler",
                workspace_root=str(Path(temp_dir).resolve()),
                config_path=str(Path(temp_dir) / ".agentos/crawl_worker_service.json"),
                worker_count=2,
                queue_db_path=str(queue_db),
                poll_interval_seconds=15.0,
                batch_size=6,
                claim_ttl_seconds=900,
                reconcile_interval_seconds=30.0,
                detail="installed",
            )

            with (
                patch.object(
                    engine,
                    "_auto_start_crawl_workers_enabled",
                    return_value=True,
                ),
                patch(
                    "agentos_orchestrator.product.CrawlWorkerServiceManager"
                ) as service_manager_cls,
                patch(
                    "agentos_orchestrator.product.CrawlWorkerManager"
                ) as worker_manager_cls,
            ):
                service_manager = service_manager_cls.return_value
                service_manager.status.return_value = service_status
                engine._maybe_start_detached_crawl_workers()

            service_manager.start.assert_called_once_with(
                task_name="AgentOS Test Crawl Service"
            )
            worker_manager_cls.return_value.start.assert_not_called()
            self.assertTrue(engine._crawl_worker_auto_started)

    def test_enrich_top_sources_uses_headless_browser_pool_for_js_sources(
        self,
    ) -> None:
        engine = FakeBrowserPoolResearchEngine()
        sources = [
            ResearchSource(
                provider="web-search",
                title="Planner worker article",
                url="https://js-heavy.example.org/article-1",
                abstract="generic web result for planner worker article",
                score=10.0,
            ),
            ResearchSource(
                provider="web-search",
                title="Browser grounding article",
                url="https://js-heavy.example.org/article-2",
                abstract="generic web result for browser grounding article",
                score=9.5,
            ),
        ]

        engine._enrich_top_sources(
            sources,
            query="browser grounded deep research agent architecture",
        )

        self.assertEqual(len(engine.pool_calls), 1)
        self.assertEqual(
            set(engine.pool_calls[0]),
            {
                "https://js-heavy.example.org/article-1",
                "https://js-heavy.example.org/article-2",
            },
        )
        self.assertTrue(
            all("Browser rendered evidence" in source.abstract for source in sources)
        )

    def test_pc_browser_frontier_seeds_expand_queries_and_sources(self) -> None:
        engine = FakeDeepResearchEngine()
        pc_context = {
            "pc_findings": {
                "search_queries": [
                    "nvidia earnings transcript data center demand",
                    "nvidia valuation gross margin outlook",
                    "nvidia sec 10-k revenue growth risks",
                    "nvidia capex supply chain commentary",
                    "nvidia analyst expectations ai demand",
                    "nvidia guidance operating margin trend",
                ],
                "judged_results": [
                    {
                        "title": f"NVIDIA judged source {index}",
                        "url": f"https://example.com/report-{index}",
                        "page_excerpt": "signal " * 40,
                        "judgment": "important source",
                        "evidence_claims": [
                            "nvidia revenue growth acceleration data center demand"
                        ],
                        "content_quality": {"quality_score": 0.9},
                    }
                    for index in range(6)
                ],
                "direct_urls": [
                    f"https://direct.com/page-{index}" for index in range(12)
                ],
                "candidate_urls": [
                    f"https://candidate.com/doc-{index}" for index in range(18)
                ],
                "frontier": {"mode": "expansive"},
            }
        }

        sources = engine._pc_finding_seed_sources(pc_context)
        queries = engine._pc_query_seeds(pc_context, "nvidia valuation outlook")

        self.assertGreaterEqual(len(sources), 20)
        self.assertGreaterEqual(len(queries), 6)
        self.assertTrue(
            any(
                "browser-frontier-candidate" in source.quality_flags
                for source in sources
            )
        )

    def test_terminal_verified_browser_findings_become_tool_observations(self) -> None:
        engine = FakeDeepResearchEngine()
        pc_context = {
            "pc_findings": {
                "judged_results": [
                    {
                        "title": "NVIDIA judged source",
                        "url": "https://example.com/nvda-report",
                        "page_excerpt": "signal " * 40,
                        "judgment": "important source",
                        "evidence_claims": [
                            "nvidia revenue growth acceleration data center demand"
                        ],
                        "content_quality": {"quality_score": 0.7},
                    }
                ],
                "terminal_verifications": [
                    {
                        "claim": "nvidia revenue growth acceleration data center demand",
                        "expression": "20/10",
                        "status": "process-executed",
                        "exit_code": 0,
                    }
                ],
            }
        }

        sources = engine._pc_finding_seed_sources(pc_context)
        ranked = DeepResearchEngine._rank_sources(
            sources,
            "nvidia revenue growth outlook",
        )

        self.assertEqual(len(sources), 1)
        self.assertIn("browser-terminal-verified", sources[0].quality_flags)
        self.assertIn("Terminal verification:", sources[0].abstract)
        self.assertEqual(ranked[0].evidence_grade, "tool-observation")

    def test_terminal_verified_claims_expand_pc_query_seeds(self) -> None:
        engine = FakeDeepResearchEngine()
        pc_context = {
            "pc_findings": {
                "judged_results": [
                    {
                        "title": "Sandbox page",
                        "url": "https://example.com/sandbox-page",
                        "page_excerpt": "signal " * 10,
                        "judgment": "important source",
                        "evidence_claims": [],
                        "content_quality": {"quality_score": 0.7},
                    }
                ],
                "terminal_verifications": [
                    {
                        "claim": "nvidia revenue growth data center demand acceleration",
                        "expression": "20/10",
                        "status": "process-executed",
                        "exit_code": 0,
                    }
                ],
            }
        }

        queries = engine._pc_query_seeds(pc_context, "nvidia valuation outlook")

        self.assertTrue(
            any(
                "revenue growth" in query.lower() and "nvidia" in query.lower()
                for query in queries
            )
        )

    def test_frontier_checkpoint_queries_expand_pc_query_seeds(self) -> None:
        engine = FakeDeepResearchEngine()
        pc_context = {
            "pc_findings": {
                "frontier_checkpoints": [
                    {
                        "follow_up_queries": [
                            "nvidia supplier concentration current evidence"
                        ],
                        "domain_leads": ["sec.gov"],
                        "contradictions": [
                            "margin compression contradiction from channel checks"
                        ],
                        "missing_evidence": [
                            "independent verification of capex and backlog assumptions"
                        ],
                    }
                ]
            }
        }

        queries = engine._pc_query_seeds(pc_context, "nvidia valuation outlook")

        self.assertTrue(
            any("supplier concentration" in query.lower() for query in queries)
        )
        self.assertTrue(any("site:sec.gov" in query.lower() for query in queries))
        self.assertTrue(
            any(
                "margin compression" in query.lower() or "capex" in query.lower()
                for query in queries
            )
        )

    def test_frontier_checkpoint_urls_become_seed_sources(self) -> None:
        engine = SeedUrlResearchEngine()
        pc_context = {
            "pc_findings": {
                "frontier_checkpoints": [
                    {"url_leads": ["https://docs.example.org/agentos"]}
                ],
                "frontier_graph": {
                    "summary": {"top_urls": ["https://docs.example.org/agentos"]}
                },
                "frontier": {"mode": "expansive"},
            }
        }

        sources = engine._pc_finding_seed_sources(pc_context)
        matching = [
            source
            for source in sources
            if source.url == "https://docs.example.org/agentos"
        ]

        self.assertTrue(matching)
        self.assertIn("browser-checkpoint-url-lead", matching[0].quality_flags)
        self.assertIn("browser-fetched-seed", matching[0].quality_flags)

    def test_standard_depth_defaults_to_multi_pass_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            engine.run("[standard] accessibility tree desktop agents", "run_1b")

            retrieval_path = (
                Path(temp_dir) / "runs/run_1b/research/retrieval_metrics.json"
            )
            metrics = json.loads(retrieval_path.read_text(encoding="utf-8"))

            self.assertGreaterEqual(len(metrics["passes"]), 2)
            self.assertNotEqual(metrics["stop_reason"], "coverage_targets_met")

    def test_pass_control_targets_do_not_count_as_coverage_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            engine.run(
                "[standard] accessibility tree desktop agents",
                "run_1c",
                evidence_targets={"max_retrieval_passes": 2},
            )

            retrieval_path = (
                Path(temp_dir) / "runs/run_1c/research/retrieval_metrics.json"
            )
            metrics = json.loads(retrieval_path.read_text(encoding="utf-8"))

            self.assertEqual(len(metrics["passes"]), 2)
            self.assertEqual(metrics["stop_reason"], "max_passes_reached")

    def test_multi_hour_depth_expands_query_plan_without_prefix_noise(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[multi-hour] protein design benchmarks",
                "run_2",
            )

            self.assertEqual(brief.objective, "protein design benchmarks")
            self.assertEqual(brief.query, "protein design benchmarks")

            plan_path = Path(temp_dir) / "runs/run_2/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["depth"], "multi-hour")
            self.assertGreaterEqual(len(plan["query_variants"]), 4)
            self.assertGreaterEqual(len(plan["subquestions"]), 3)
            self.assertGreaterEqual(len(plan["comparative_axes"]), 4)
            self.assertGreaterEqual(len(plan["perspectives"]), 4)
            self.assertIn("structured scholarly APIs", plan["token_strategy"])
            self.assertNotIn(
                "LLM agent benchmark evaluation",
                plan["query_variants"],
            )
            self.assertNotIn(
                "AI agent computer use evaluation",
                plan["query_variants"],
            )

    def test_multi_hour_run_writes_durable_report_and_pass_growth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[multi-hour] accessibility tree desktop agents",
                "run_durable",
            )

            report_path = Path(temp_dir) / "runs/run_durable/workflows/report.md"
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("## Incremental Findings", report_text)
            self.assertGreaterEqual(report_text.count("### Pass "), 2)
            self.assertIn("- [", report_text)
            self.assertTrue(
                any(
                    artifact.replace("\\", "/")
                    == "runs/run_durable/workflows/report.md"
                    for artifact in brief.artifacts
                )
            )

    def test_durable_notes_only_synthesis_excludes_abstract_payload(self) -> None:
        engine = CapturingSynthesisEngine()
        source = ResearchSource(
            provider="web-search",
            title="Durable note source",
            url="https://example.org/source",
            abstract="ABSTRACT_SENTINEL_SHOULD_NOT_APPEAR",
            evidence_grade="moderate",
        )

        summary = engine._summarize(
            objective="durable-mode objective",
            sources=[source],
            depth="multi-hour",
            plan={"subquestions": ["What is supported?"]},
            query="durable mode objective",
            durable_notes=(
                "# Durable Research Report\n\n"
                "## Incremental Findings\n\n"
                "### Pass 1\n"
                "- [moderate/web-search] Distilled finding (source: https://example.org/source)\n"
            ),
            synthesis_mode="durable-notes-only",
        )

        self.assertEqual(summary, "SYNTHESIS")
        self.assertIn("Durable report notes", engine.last_user)
        self.assertIn("Minimal source metadata", engine.last_user)
        self.assertNotIn("ABSTRACT_SENTINEL_SHOULD_NOT_APPEAR", engine.last_user)

    def test_standard_synthesis_packet_caps_prompt_sources(self) -> None:
        engine = CapturingSynthesisEngine()
        sources = [
            ResearchSource(
                provider="web-search",
                title=f"Source {index}",
                url=f"https://example.org/source-{index}",
                abstract=f"Evidence abstract {index}",
                evidence_grade="moderate",
                score=float(100 - index),
            )
            for index in range(40)
        ]
        packet = engine._build_synthesis_packet(
            "packet objective",
            "packet objective",
            sources,
            "standard",
            {"subquestions": ["What is supported?"]},
            "",
            "hybrid",
        )

        summary = engine._summarize(
            objective="packet objective",
            sources=sources,
            depth="standard",
            plan={"subquestions": ["What is supported?"]},
            query="packet objective",
            durable_notes="",
            synthesis_mode="hybrid",
            synthesis_packet=packet,
        )

        self.assertEqual(summary, "SYNTHESIS")
        self.assertEqual(packet["synthesis_source_count"], 24)
        self.assertIn("Evidence found (24 sources)", engine.last_user)
        self.assertNotIn("Source 39", engine.last_user)

    def test_multi_hour_software_agent_synthesis_packet_expands_prompt_sources(
        self,
    ) -> None:
        engine = CapturingSynthesisEngine()
        objective = (
            "Analyze why a deep research agent is not comparable to Claude, GPT, "
            "or Gemini, is not using browser and sandbox tools effectively, and "
            "fix the architectural gaps."
        )
        sources = [
            ResearchSource(
                provider="web-search",
                title=f"Source {index}",
                url=f"https://example.org/software-agent-source-{index}",
                abstract=f"Evidence abstract {index}",
                evidence_grade="moderate",
                score=float(200 - index),
            )
            for index in range(140)
        ]
        packet = engine._build_synthesis_packet(
            objective,
            objective,
            sources,
            "multi-hour",
            {"subquestions": ["Where does the evidence handoff narrow too early?"]},
            "",
            "hybrid",
        )

        summary = engine._summarize(
            objective=objective,
            sources=sources,
            depth="multi-hour",
            plan={
                "subquestions": ["Where does the evidence handoff narrow too early?"]
            },
            query=objective,
            durable_notes="",
            synthesis_mode="hybrid",
            synthesis_packet=packet,
        )

        self.assertEqual(summary, "SYNTHESIS")
        self.assertEqual(packet["synthesis_source_count"], 96)
        self.assertIn("Evidence found (96 sources)", engine.last_user)
        self.assertIn("Source 95", engine.last_user)
        self.assertNotIn("Source 120", engine.last_user)

    def test_multi_hour_synthesis_defaults_to_hybrid_with_durable_notes(self) -> None:
        with patch.dict("os.environ", {"AGENTOS_FINAL_SYNTHESIS_MODE": ""}, clear=False):
            self.assertEqual(
                DeepResearchEngine._resolve_final_synthesis_mode(
                    "multi-hour",
                    "### Pass 1\n- [strong/openalex] grounded claim",
                ),
                "hybrid",
            )

    def test_software_agent_direct_research_plan_distills_complaint_objective(
        self,
    ) -> None:
        objective = (
            "So fix the code. Analyze why the deep research agent is not "
            "comparable to Claude, GPT, or Gemini, why it is not using sandbox, "
            "browser, and pc control properly, and why retrieval breadth and "
            "useful evidence quality are weak."
        )
        engine = FakeDeepResearchEngine()
        engine._ai_research_strategy = lambda objective, query, depth: {}
        engine._ai_research_axes = lambda objective, query, depth: {}

        distilled_query = engine._query_from_objective(objective)
        plan = engine._build_research_plan(
            objective,
            distilled_query,
            "multi-hour",
            {"browser_context_detected": False},
        )

        self.assertEqual(distilled_query, "deep research agent")
        self.assertIn(
            "deep research agent browser sandbox pc control routing",
            plan["query_plan"],
        )
        self.assertIn(
            "https://github.com",
            plan["ai_authoritative_domains"],
        )
        self.assertIn(
            "https://playwright.dev",
            plan["ai_authoritative_domains"],
        )
        self.assertFalse(
            any(query.lower().startswith("so fix code") for query in plan["query_plan"])
        )
        perspective_names = {item["name"] for item in plan["perspectives"]}
        self.assertIn("browser-tooling", perspective_names)
        self.assertIn("retrieval-breadth", perspective_names)
        self.assertIn("evidence-quality", perspective_names)

    def test_current_web_research_plan_strips_instruction_fragments(self) -> None:
        objective = (
            "As of right now, research which publicly traded companies have the highest "
            "probability-adjusted upside potential over the next 12 to 24 months. Use all available "
            "general-purpose research means, including browser-grounded web research, sandboxed exploration, "
            "current-web evidence, company filings, earnings material, product signals, independent sources, "
            "and cross-checking. Do not use finance-specific hardcoded templates or domain-specific shortcuts. "
            "Produce an analyst-grade and scientist-grade report with ranked candidates, the evidence for and "
            "against each thesis, uncertainty bounds, catalyst quality, execution risk, valuation-sensitive "
            "considerations, and clear reasons for the ranking."
        )
        engine = FakeDeepResearchEngine()
        engine._ai_research_strategy = lambda objective, query, depth: {
            "reasoning_queries": [
                "scientist-grade report with ranked candidates, the evidence for",
                "against each thesis",
                "publicly traded companies upside potential 12 24 months",
            ]
        }
        engine._ai_research_axes = lambda objective, query, depth: {}

        distilled_query = engine._query_from_objective(objective)
        plan = engine._build_research_plan(
            objective,
            distilled_query,
            "multi-hour",
            {"browser_context_detected": False},
        )

        self.assertTrue(
            any(
                "publicly traded companies" in query.lower()
                and "upside potential" in query.lower()
                for query in plan["query_plan"]
            )
        )
        self.assertFalse(
            any("scientist-grade" in query.lower() for query in plan["query_plan"])
        )
        self.assertFalse(
            any("against each thesis" in query.lower() for query in plan["query_plan"])
        )

    def test_current_web_query_sanitizer_rejects_scaffold_noise(self) -> None:
        query = (
            "which publicly traded companies have highest probability-adjusted upside "
            "potential over next 12 24 months"
        )

        cleaned = DeepResearchEngine._sanitize_query_variants(
            [
                "find which",
                "use which",
                "which over employment labor demographic",
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months site finance yahoo com"
                ),
                (
                    "site arxiv org which publicly traded companies have highest "
                    "probability-adjusted upside potential over next 12 24 months"
                ),
                (
                    "site pubmed ncbi nlm nih gov which publicly traded companies "
                    "have highest probability-adjusted upside potential over next "
                    "12 24 months"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months primary evidence"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months site reuters com"
                ),
            ],
            query,
        )

        self.assertNotIn("find which", cleaned)
        self.assertNotIn("use which", cleaned)
        self.assertNotIn("which over employment labor demographic", cleaned)
        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months site finance yahoo com"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "site arxiv org which publicly traded companies have highest "
                "probability-adjusted upside potential over next 12 24 months"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "site pubmed ncbi nlm nih gov which publicly traded companies "
                "have highest probability-adjusted upside potential over next "
                "12 24 months"
            ),
            cleaned,
        )
        self.assertIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months primary evidence"
            ),
            cleaned,
        )
        self.assertIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months site reuters com"
            ),
            cleaned,
        )

    def test_current_web_query_sanitizer_rejects_runtime_retrieval_noise(
        self,
    ) -> None:
        query = (
            "which publicly traded companies have highest probability-adjusted upside "
            "potential over next 12 24 months"
        )

        cleaned = DeepResearchEngine._sanitize_query_variants(
            [
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months so fix code also"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months find authorita"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months jats title abstract jats"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months primary eviden"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months my knowledge this does"
                ),
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months veritone which ai stock"
                ),
                (
                    "site ai google dev which publicly traded companies have highest "
                    "probability-adjusted upside potential over next 12 24 months"
                ),
                "this synthesis evaluates publicly traded companies upside potential",
                "traded probability-adjusted gemini flash class attempt",
                "traded probability-adjusted class attempt theme",
                (
                    "which publicly traded companies have highest probability-adjusted "
                    "upside potential over next 12 24 months methodology"
                ),
            ],
            query,
        )

        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months so fix code also"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months find authorita"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months jats title abstract jats"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months primary eviden"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months my knowledge this does"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months veritone which ai stock"
            ),
            cleaned,
        )
        self.assertNotIn(
            (
                "site ai google dev which publicly traded companies have highest "
                "probability-adjusted upside potential over next 12 24 months"
            ),
            cleaned,
        )
        self.assertNotIn(
            "this synthesis evaluates publicly traded companies upside potential",
            cleaned,
        )
        self.assertNotIn(
            "traded probability-adjusted gemini flash class attempt",
            cleaned,
        )
        self.assertNotIn(
            "traded probability-adjusted class attempt theme",
            cleaned,
        )
        self.assertIn(
            (
                "which publicly traded companies have highest probability-adjusted "
                "upside potential over next 12 24 months methodology"
            ),
            cleaned,
        )

    def test_market_query_variants_do_not_clip_axis_suffixes(self) -> None:
        query = (
            "which publicly traded companies have highest probability-adjusted upside "
            "potential over next 12 24 months"
        )

        with patch.object(DeepResearchEngine, "_ai_query_variants", return_value=[]):
            variants = DeepResearchEngine._query_variants(query, "multi-hour")

        self.assertTrue(any(variant.endswith("primary evidence") for variant in variants))
        self.assertFalse(any(variant.endswith("primary eviden") for variant in variants))
        self.assertFalse(any(variant.endswith("counterevidenc") for variant in variants))

    def test_gemini_flash_provider_returns_no_evidence_sources(self) -> None:
        # GENERALITY GUARD: an LLM tool observation is parametric-memory
        # restatement, not a primary source.  The provider must NEVER inject
        # ResearchSource records with a synthetic ai.google.dev URL.
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = DeepResearchEngine(workspace_root=temp_dir)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "GEMINI_API_KEY": "fake-key-for-test",
                        "GOOGLE_API_KEY": "fake-key-for-test",
                    },
                ):
                    sources = engine._search_gemini_observation(
                        "any topic at all", "multi-hour"
                    )
            finally:
                pass

        self.assertEqual(sources, [])

    def test_classify_query_never_returns_gemini_flash_as_provider(self) -> None:
        # The gemini-flash *evidence* provider must be absent from every
        # classification: market, scholarly, software, current-evidence,
        # and non-academic fallbacks.
        for query in (
            "highest-potential public companies as of now",
            "site:sec.gov AAPL latest 10-K filing",
            "compare OpenHands OpenCode OpenClaw OSWorld WebArena",
            "best chocolate chip cookie recipe with brown butter",
            "transformer architecture scaling laws 2025",
            "site:fred.stlouisfed.org unemployment rate latest",
        ):
            providers = DeepResearchEngine._classify_query(query)
            self.assertNotIn(
                "gemini-flash",
                providers,
                f"gemini-flash leaked into classification for: {query!r}",
            )

    def test_pc_browser_filter_drops_yahoo_when_authoritative_seed_exists(
        self,
    ) -> None:
        from agentos_orchestrator.core.agents import WorkerAgent

        # Build a minimal worker stub; only filter logic is exercised.
        worker = WorkerAgent.__new__(WorkerAgent)

        authoritative_seed = ResearchSource(
            provider="pc-browser-research",
            title="SEC EDGAR filing",
            url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",
            year=2026,
            abstract="Primary regulatory filing.",
        )
        yahoo_source = ResearchSource(
            provider="web-search",
            title="Goldman raises Pitney Bowes",
            url="https://finance.yahoo.com/markets/stocks/articles/foo.html",
            year=2026,
            abstract="Re-publication of analyst note.",
        )
        marketwatch_source = ResearchSource(
            provider="web-search",
            title="MarketWatch market summary",
            url="https://www.marketwatch.com/story/bar",
            year=2026,
            abstract="Aggregated market wrap.",
        )
        reuters_source = ResearchSource(
            provider="web-search",
            title="Reuters primary report",
            url="https://www.reuters.com/markets/companies/ABC",
            year=2026,
            abstract="First-party reporting.",
        )

        filtered = worker._filter_market_browser_sources(
            [yahoo_source, marketwatch_source, reuters_source],
            [authoritative_seed],
        )
        urls = [str(s.url) for s in filtered]
        self.assertNotIn(yahoo_source.url, urls)
        self.assertNotIn(marketwatch_source.url, urls)
        self.assertIn(reuters_source.url, urls)

    def test_pc_browser_filter_drops_yahoo_even_when_no_seeds_if_alternative_exists(
        self,
    ) -> None:
        from agentos_orchestrator.core.agents import WorkerAgent

        worker = WorkerAgent.__new__(WorkerAgent)

        yahoo_source = ResearchSource(
            provider="web-search",
            title="Yahoo finance",
            url="https://finance.yahoo.com/quote/AAPL",
            year=2026,
            abstract="Aggregated quote page.",
        )
        sec_source = ResearchSource(
            provider="web-search",
            title="SEC filing",
            url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
            year=2026,
            abstract="Primary 10-K filing.",
        )

        filtered = worker._filter_market_browser_sources(
            [yahoo_source, sec_source],
            [],  # no navigation seeds at all
        )
        urls = [str(s.url) for s in filtered]
        self.assertNotIn(yahoo_source.url, urls)
        self.assertIn(sec_source.url, urls)

    def test_pc_browser_filter_keeps_blocked_only_as_last_resort(self) -> None:
        from agentos_orchestrator.core.agents import WorkerAgent

        worker = WorkerAgent.__new__(WorkerAgent)

        yahoo_source = ResearchSource(
            provider="web-search",
            title="Yahoo only",
            url="https://finance.yahoo.com/quote/AAPL",
            year=2026,
            abstract="Aggregator-only page.",
        )

        filtered = worker._filter_market_browser_sources(
            [yahoo_source],
            [],  # nothing authoritative anywhere
        )
        # When zero authoritative alternatives exist anywhere, the blocked
        # source is preserved (flagged) so the run does not stall.
        self.assertEqual(len(filtered), 1)
        self.assertIn("low-primary-signal-portal", filtered[0].quality_flags)

    def test_market_portal_blocklist_covers_cross_topic_aggregators(self) -> None:
        from agentos_orchestrator.core.agents import WorkerAgent

        blocklist = WorkerAgent._market_portal_browser_domains()
        # Cross-topic content farms and aggregators that re-publish content
        # rather than originate it.
        for host in (
            "finance.yahoo.com",
            "marketwatch.com",
            "msn.com",
            "businessinsider.com",
            "fool.com",
            "zacks.com",
            "medium.com",
            "substack.com",
        ):
            self.assertIn(host, blocklist, f"missing portal: {host}")

    def test_sanitize_frontier_graph_drops_portal_checkpoint_leads(self) -> None:
        from agentos_orchestrator.core.agents import WorkerAgent

        worker = WorkerAgent.__new__(WorkerAgent)
        frontier_graph = {
            "urls": {
                "https://finance.yahoo.com/quote/AAPL": {
                    "url": "https://finance.yahoo.com/quote/AAPL",
                    "domain": "finance.yahoo.com",
                },
                "https://www.reuters.com/markets": {
                    "url": "https://www.reuters.com/markets",
                    "domain": "reuters.com",
                },
            },
            "domains": {
                "finance.yahoo.com": {"urls": ["https://finance.yahoo.com/quote/AAPL"]},
                "reuters.com": {"urls": ["https://www.reuters.com/markets"]},
            },
            "reasoning_checkpoints": [
                {
                    "cycle": 1,
                    "follow_up_queries": [
                        "perform sandboxed browser research actions for: [multi-hour] AAPL upside"
                    ],
                    "domain_leads": ["finance.yahoo.com", "reuters.com"],
                    "url_leads": [
                        "https://finance.yahoo.com/quote/AAPL",
                        "https://www.reuters.com/markets",
                    ],
                    "missing_evidence": ["Need broader domain coverage."],
                    "contradictions": [],
                    "continue_research": True,
                }
            ],
            "claims": {},
        }

        sanitized = worker._sanitize_frontier_graph(frontier_graph)

        self.assertNotIn("https://finance.yahoo.com/quote/AAPL", sanitized["urls"])
        self.assertNotIn("finance.yahoo.com", sanitized["domains"])
        self.assertEqual(
            sanitized["reasoning_checkpoints"][0]["domain_leads"],
            ["reuters.com"],
        )
        self.assertEqual(
            sanitized["reasoning_checkpoints"][0]["url_leads"],
            ["https://www.reuters.com/markets"],
        )

    def test_pc_query_seeds_strip_frontier_meta_feedback_and_portal_sites(
        self,
    ) -> None:
        query = (
            "which publicly traded companies have highest probability-adjusted upside "
            "potential over next 12 24 months"
        )
        engine = FakeDeepResearchEngine()

        seeds = engine._pc_query_seeds(
            {
                "pc_findings": {
                    "frontier_checkpoints": [
                        {
                            "follow_up_queries": [
                                (
                                    "bls bureau labor statistics release calendar "
                                    "subscribe search button search menu"
                                )
                            ],
                            "domain_leads": ["finance.yahoo.com", "reuters.com"],
                            "missing_evidence": [
                                "Need broader domain coverage beyond the current browser pages.",
                                "Need more direct pages with substantive evidence extraction.",
                                "Need independent verification of browser-derived claims.",
                            ],
                        }
                    ]
                }
            },
            query,
        )

        joined = " ".join(seeds).lower()
        self.assertNotIn("finance yahoo com", joined)
        self.assertNotIn("need broader domain coverage", joined)
        self.assertNotIn("need more direct pages", joined)
        self.assertNotIn("browser-derived claims", joined)
        self.assertNotIn("release calendar subscribe", joined)
        self.assertIn("site reuters com", joined)

    def test_source_seed_urls_ignore_raw_pc_market_portal_candidates(self) -> None:
        objective = (
            "As of right now, research which publicly traded companies have the highest "
            "probability-adjusted upside potential over the next 12 to 24 months."
        )

        seed_urls = DeepResearchEngine._source_seed_urls(
            objective,
            None,
            {
                "pc_findings": {
                    "direct_urls": [
                        "https://sec.gov",
                        "https://reuters.com/markets",
                        (
                            "https://finance.yahoo.com/markets/stocks/articles/"
                            "market-chatter-rivian-automotive-working-074150543.html"
                        ),
                    ],
                    "judged_results": [
                        {"url": "https://sec.gov"},
                        {"url": "https://reuters.com/markets"},
                    ],
                    "candidate_urls": [
                        (
                            "https://finance.yahoo.com/markets/stocks/articles/"
                            "freshworks-q1-adjusted-earnings-decline-074610525.html"
                        )
                    ],
                }
            },
        )

        self.assertIn("https://sec.gov", seed_urls)
        self.assertIn("https://reuters.com/markets", seed_urls)
        self.assertFalse(any("finance.yahoo.com" in url for url in seed_urls))

    def test_software_agent_source_seed_urls_include_authoritative_roots(
        self,
    ) -> None:
        objective = (
            "Analyze why a deep research agent is not comparable to Claude, GPT, "
            "or Gemini and is not using browser and sandbox tools effectively."
        )

        seed_urls = DeepResearchEngine._source_seed_urls(objective, None, None)

        self.assertIn("https://github.com", seed_urls)
        self.assertIn("https://playwright.dev", seed_urls)
        self.assertIn("https://docs.anthropic.com", seed_urls)
        self.assertIn("https://platform.openai.com/docs", seed_urls)
        self.assertIn("https://ai.google.dev", seed_urls)

    def test_standalone_autonomous_pc_context_seeds_browser_findings(self) -> None:
        objective = (
            "So fix the code. Analyze why the deep research agent is not "
            "comparable to Claude, GPT, or Gemini, why it is not using sandbox, "
            "browser, and pc control properly, and why retrieval breadth and "
            "useful evidence quality are weak."
        )
        engine = AutoPcContextResearchEngine()

        pc_context = engine._auto_pc_context_for_run(
            objective,
            engine._query_from_objective(objective),
            "multi-hour",
            "run_auto_pc",
        )

        self.assertIsNotNone(pc_context)
        assert pc_context is not None
        pc_findings = pc_context["pc_findings"]
        self.assertIn(
            "deep research agent browser sandbox pc control routing",
            pc_findings["search_queries"],
        )
        self.assertTrue(pc_findings["judged_results"])
        self.assertIn("https://github.com", pc_findings["direct_urls"])

        sources = engine._pc_finding_seed_sources(pc_context)

        self.assertTrue(any(source.provider == "pc-browser-research" for source in sources))

    def test_static_asset_urls_are_excluded_from_persistent_url_sets(self) -> None:
        engine = FakeDeepResearchEngine()

        urls = engine._persistent_unique_urls(
            [
                "https://arxiv.org/static/browse/0.3.4/css/arxiv-html-papers.css",
                "https://arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png",
                "https://example.com/report",
            ]
        )

        self.assertEqual(urls, ["https://example.com/report"])

    def test_gibberish_query_variants_are_rejected(self) -> None:
        cleaned = DeepResearchEngine._sanitize_query_variants(
            [
                "research agent deep research google deep vfppkd vfppkd-strngf",
                "deep research agent browser sandbox pc control routing",
            ],
            "deep research agent",
        )

        self.assertNotIn(
            "research agent deep research google deep vfppkd vfppkd-strngf",
            cleaned,
        )
        self.assertIn(
            "deep research agent browser sandbox pc control routing",
            cleaned,
        )

    def test_iterative_retrieval_sanitizes_enrichment_queries_before_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = NoisyEnrichmentResearchEngine(workspace_root=temp_dir)
            retrieval = engine._iterative_retrieval(
                query="deep research agent",
                settings=DeepResearchEngine._settings_for_depth("standard"),
                plan={
                    "core_question": "deep research agent",
                    "query_plan": ["deep research agent"],
                    "perspectives": [],
                },
                targets={
                    "max_retrieval_passes": 2,
                    "min_depth_passes": 2,
                    "min_source_count": 999,
                    "min_provider_count": 1,
                    "min_scholarly_sources": 0,
                    "min_strong_or_moderate": 0,
                    "min_novelty_rate": 0.0,
                    "max_contradiction_risk": 1.0,
                },
            )

            self.assertIn(
                "deep research agent benchmark comparison",
                retrieval["query_variants"],
            )
            self.assertNotIn(
                "research agent deep research google deep vfppkd vfppkd-strngf",
                retrieval["query_variants"],
            )

    def test_select_balanced_top_preserves_browser_sources_beyond_cap(self) -> None:
        ranked = [
            ResearchSource(
                provider="crossref",
                title=f"Deep research agent study {index}",
                url=f"https://doi.org/10.1000/test-{index}",
                abstract=(
                    "Deep research agent architecture and retrieval evidence."
                ),
                score=100.0 - index,
                relevance=0.9,
                credibility_score=0.7,
            )
            for index in range(12)
        ]
        ranked.append(
            ResearchSource(
                provider="pc-browser-research",
                title="Deep research agent browser routing evidence",
                url="https://example.com/browser-evidence",
                abstract=(
                    "Browser-judged evidence about sandbox and pc control routing "
                    "for the deep research agent."
                ),
                score=12.0,
                relevance=0.8,
                credibility_score=0.55,
                evidence_grade="tool-observation",
                quality_flags=["browser-judged-source"],
            )
        )

        selected = DeepResearchEngine._select_balanced_top(
            ranked,
            max_sources=4,
            query="deep research agent browser sandbox pc control routing",
        )

        self.assertIn(
            "pc-browser-research",
            {source.provider for source in selected},
        )

    def test_run_writes_synthesis_packet_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[quick] accessibility tree desktop agents",
                "run_synthesis_packet",
            )

            packet_path = (
                Path(temp_dir)
                / "runs/run_synthesis_packet/research/synthesis_packet.json"
            )
            self.assertTrue(packet_path.exists())
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            self.assertGreater(packet["synthesis_source_count"], 0)
            self.assertTrue(
                any(
                    artifact.replace("\\", "/")
                    == "runs/run_synthesis_packet/research/synthesis_packet.json"
                    for artifact in brief.artifacts
                )
            )

    def test_depth_marker_survives_supervisor_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                (
                    "Find authoritative sources, prior systems, and gaps "
                    "for: [quick] accessibility tree desktop agents"
                ),
                "run_3",
            )

            self.assertEqual(brief.query, "accessibility tree desktop agents")
            plan_path = Path(temp_dir) / "runs/run_3/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["depth"], "quick")

    def test_adaptive_depth_scales_with_prompt_complexity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            recipe = engine.run("[adaptive] find a recipe for pesto", "run_recipe")
            engine.run(
                (
                    "comprehensive scientific literature review of long-context "
                    "GUI agents with evidence, benchmarks, risks, and limitations"
                ),
                "run_report",
            )

            recipe_plan = json.loads(
                (
                    Path(temp_dir) / "runs/run_recipe/research/research_plan.json"
                ).read_text(encoding="utf-8")
            )
            report_plan = json.loads(
                (
                    Path(temp_dir) / "runs/run_report/research/research_plan.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(recipe.objective, "find a recipe for pesto")
            self.assertEqual(recipe_plan["depth"], "quick")
            self.assertEqual(report_plan["depth"], "multi-hour")

    def test_current_evidence_queries_prefer_recency_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                (
                    "Research highest-potential public companies as of now using "
                    "all available evidence-gathering tools. Produce a rigorous "
                    "current evidence report with risks and opportunities."
                ),
                "run_current",
            )

            plan_path = Path(temp_dir) / "runs/run_current/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            metrics_path = (
                Path(temp_dir) / "runs/run_current/research/retrieval_metrics.json"
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            joined_variants = " ".join(plan["query_variants"]).lower()

            self.assertIn("highest-potential public companies", brief.query.lower())
            self.assertEqual(plan["depth"], "multi-hour")
            self.assertIn("latest", joined_variants)
            # Verify recency-axis variants are emitted directly from the
            # axis generator rather than depending on any single provider's
            # textual output.  We check the canonical recency tokens
            # (``latest`` is already asserted above; ``timeline`` and
            # ``near-term`` come from the current-evidence axes list).
            self.assertTrue(
                "timeline" in joined_variants
                or "near-term" in joined_variants
                or "current" in joined_variants,
                f"no recency axis token in variants: {joined_variants!r}",
            )
            self.assertNotIn("literature", joined_variants)
            self.assertGreaterEqual(len(plan["query_variants"]), 5)
            self.assertGreaterEqual(len(metrics["passes"]), 4)

    def test_current_evidence_queries_use_current_web_providers(self) -> None:
        providers = DeepResearchEngine._classify_query(
            "highest-potential public companies as of now"
        )

        # Current-evidence queries use real web/news providers.  The
        # ``gemini-flash`` evidence provider was intentionally removed: an
        # LLM observation is a parametric-memory restatement, not primary
        # evidence, so it must not appear in the provider rotation.
        self.assertIn("web-search", providers)
        self.assertNotIn("gemini-flash", providers)
        self.assertNotIn("openalex", providers)
        self.assertNotIn("semantic-scholar", providers)
        self.assertNotIn("seeking-alpha", providers)
        self.assertNotIn("reddit-finance", providers)

    def test_provider_order_is_topic_conditioned_not_static_template(self) -> None:
        academic_order = DeepResearchEngine._provider_order(
            "transformer scaling laws reproducibility survey",
            {"openalex", "semantic-scholar", "crossref", "web-search"},
        )
        software_order = DeepResearchEngine._provider_order(
            "compare OpenHands OpenCode OpenClaw OSWorld WebArena",
            {"github-repositories", "web-search", "bing-search", "crossref"},
        )
        market_order = DeepResearchEngine._provider_order(
            "which publicly traded companies have highest upside over next 12 months",
            {"sec-edgar", "earnings-data", "web-search", "financial-portals"},
        )

        self.assertEqual(academic_order[:3], ("openalex", "semantic-scholar", "crossref"))
        self.assertEqual(software_order[0], "github-repositories")
        self.assertEqual(market_order[:2], ("sec-edgar", "earnings-data"))

    def test_provider_dispatch_supports_mixed_async_and_sync_searchers(self) -> None:
        engine = MixedProviderDispatchEngine()

        sources = engine._search_query_across_providers(
            "async provider mix",
            {"openalex", "web-search"},
            4,
        )

        self.assertEqual(
            [source.provider for source in sources],
            ["openalex", "web-search"],
        )

    def test_provider_diagnostics_flush_live_for_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = DeepResearchEngine(workspace_root=temp_dir)
            engine._active_run_id = "run_live_diag"

            engine._record_provider_diagnostic(
                "web-search",
                "ok",
                "returned 3 results",
            )

            diagnostics_path = (
                Path(temp_dir)
                / "runs/run_live_diag/research/provider_diagnostics.json"
            )
            self.assertTrue(diagnostics_path.exists())
            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[-1]["provider"], "web-search")
            self.assertEqual(payload[-1]["status"], "ok")

    def test_current_evidence_low_diversity_expands_provider_mix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[multi-hour] accessibility tree agents as of now",
                "run_current_diversity",
            )

            providers = {source.provider for source in brief.sources}
            diagnostics_path = (
                Path(temp_dir)
                / "runs/run_current_diversity/research/provider_diagnostics.json"
            )
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

            self.assertTrue(
                any(
                    item.get("provider") == "provider-mix"
                    and item.get("status") == "expanded"
                    for item in diagnostics
                )
            )
            self.assertTrue(
                any(
                    provider in providers
                    for provider in {"openalex", "semantic-scholar", "crossref"}
                )
            )

    def test_current_web_targets_override_strict_scholarly_gates(self) -> None:
        overridden = DeepResearchEngine._current_web_target_overrides(
            {
                "min_provider_count": 6,
                "min_source_count": 18,
                "min_strong_or_moderate": 0,
                "min_scholarly_sources": 5,
                "min_novelty_rate": 0.22,
            },
            "multi-hour",
        )

        self.assertGreaterEqual(overridden["min_provider_count"], 6)
        self.assertGreaterEqual(overridden["min_source_count"], 120)
        self.assertGreaterEqual(overridden["min_strong_or_moderate"], 8)
        self.assertEqual(overridden["min_scholarly_sources"], 0)
        self.assertEqual(overridden["min_novelty_rate"], 0.0)
        self.assertEqual(overridden["max_retrieval_passes"], 48)
        self.assertGreater(
            DeepResearchEngine._effective_novelty_threshold(
                "multi-hour",
                overridden,
            ),
            0.0,
        )

    def test_multi_hour_current_web_parallelism_scales_up(self) -> None:
        engine = DeepResearchEngine()
        engine._active_objective = (
            "[multi-hour] highest-potential public companies as of now"
        )

        self.assertGreaterEqual(
            engine._query_parallel_worker_count("multi-hour", 120, 8),
            16,
        )
        self.assertGreaterEqual(
            engine._provider_parallel_worker_count(8),
            4,
        )

    def test_generic_market_words_do_not_make_source_on_topic(self) -> None:
        query = "public stocks with highest potential to soar as of now"
        generic = ResearchSource(
            provider="google-news-rss",
            title="Public market uncertainty timeline report",
            url="https://news.example.com/public-market-timeline",
            year=2026,
            abstract="Current public evidence and uncertainty timeline update.",
            score=25.0,
        )
        actionable = ResearchSource(
            provider="google-news-rss",
            title="NVIDIA (NVDA) stock earnings guidance catalyst",
            url="https://news.example.com/nvda-earnings",
            year=2026,
            abstract="Revenue growth, EPS upside, valuation, and price target revisions.",
            score=25.0,
        )
        promo = ResearchSource(
            provider="google-news-rss",
            title="3 Stocks to Buy ASAP Before a Private Company Goes Public",
            url="https://news.example.com/stocks-to-buy",
            year=2026,
            abstract="Generic stocks to buy list with no company-specific evidence.",
            score=25.0,
        )

        ranked = DeepResearchEngine._rank_sources(
            [generic, actionable, promo],
            query,
        )

        self.assertFalse(DeepResearchEngine._source_is_on_topic(generic, query))
        self.assertIn(actionable.url, {source.url for source in ranked})
        self.assertNotIn(generic.url, {source.url for source in ranked})
        self.assertNotIn(promo.url, {source.url for source in ranked})

    def test_generic_axis_query_variants_are_rejected(self) -> None:
        query = "public stocks with highest potential to soar as of now"
        cleaned = DeepResearchEngine._sanitize_query_variants(
            [
                "stocks public timeline",
                "stocks public uncertainty",
                "stocks earnings catalyst revisions",
            ],
            query,
        )

        self.assertNotIn("stocks public timeline", cleaned)
        self.assertNotIn("stocks public uncertainty", cleaned)
        self.assertIn("stocks earnings catalyst revisions", cleaned)

    def test_current_web_low_novelty_stops_before_full_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = StableLowNoveltyEngine(workspace_root=temp_dir)
            retrieval = engine._iterative_retrieval(
                query="public stocks with highest potential to soar as of now",
                settings=DeepResearchEngine._settings_for_depth("multi-hour"),
                plan={
                    "core_question": (
                        "public stocks with highest potential to soar as of now"
                    ),
                    "query_plan": ["stocks earnings catalyst revisions"],
                    "perspectives": [],
                },
                targets={
                    "max_retrieval_passes": 40,
                    "depth_pass_floor": 3,
                    "max_low_novelty_streak": 2,
                    "min_source_count": 50,
                },
                pc_context=None,
                run_id="run_low_novelty",
            )

        self.assertEqual(retrieval["stop_reason"], "novelty_below_threshold")
        self.assertLess(len(retrieval["passes"]), 40)
        self.assertIn(
            "SEC filing revenue growth catalyst".lower(),
            " ".join(retrieval["query_variants"]).lower(),
        )

    def test_current_web_low_marginal_yield_stops_before_full_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = SlowMarginalYieldEngine(workspace_root=temp_dir)
            retrieval = engine._iterative_retrieval(
                query="public stocks with highest potential to soar as of now",
                settings=DeepResearchEngine._settings_for_depth("multi-hour"),
                plan={
                    "core_question": (
                        "public stocks with highest potential to soar as of now"
                    ),
                    "query_plan": ["stocks earnings catalyst revisions"],
                    "perspectives": [],
                },
                targets={
                    "max_retrieval_passes": 24,
                    "depth_pass_floor": 3,
                    "max_low_marginal_yield_streak": 2,
                    "min_marginal_unique_url_gain": 2,
                    "min_marginal_title_gain": 2,
                    "min_source_count": 50,
                },
                pc_context=None,
                run_id="run_low_marginal_yield",
            )

        self.assertEqual(retrieval["stop_reason"], "marginal_yield_exhausted")
        self.assertLess(len(retrieval["passes"]), 24)
        self.assertEqual(retrieval["passes"][-1]["marginal_unique_url_gain"], 1)
        self.assertEqual(retrieval["passes"][-1]["marginal_title_gain"], 1)

    def test_general_complex_objective_expands_standard_settings(self) -> None:
        settings = DeepResearchEngine._settings_for_general_complex_objective(
            DeepResearchEngine._settings_for_depth("standard"),
            (
                "Analyze why a deep research agent is not using sandbox, "
                "browser, and pc control effectively across general topics"
            ),
        )

        self.assertEqual(settings.depth, "standard")
        self.assertGreaterEqual(settings.max_sources, 72)
        self.assertGreaterEqual(settings.per_provider, 24)
        self.assertGreaterEqual(settings.max_query_variants, 24)

    def test_academic_query_keeps_standard_settings(self) -> None:
        baseline = DeepResearchEngine._settings_for_depth("standard")
        settings = DeepResearchEngine._settings_for_general_complex_objective(
            baseline,
            "Compare peer-reviewed literature and citation evidence for membrane transporters",
        )

        self.assertEqual(settings, baseline)

    def test_pc_browser_urls_are_prioritized_and_fetched_as_seeds(self) -> None:
        pc_context = {
            "pc_findings": {
                "direct_urls": ["https://docs.example.org/agentos"],
                "candidate_urls": [
                    "https://html.duckduckgo.com/html/?q=agentos",
                    "https://example.com/other",
                ],
            }
        }
        urls = DeepResearchEngine._source_seed_urls(
            "Use https://example.com/requested as context",
            None,
            pc_context,
        )

        self.assertEqual(urls[0], "https://docs.example.org/agentos")
        self.assertNotIn("https://html.duckduckgo.com/html/?q=agentos", urls)

        engine = SeedUrlResearchEngine()
        sources = engine._pc_finding_seed_sources(pc_context)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].provider, "pc-browser-research")
        self.assertIn("AgentOS benchmark safety approvals", sources[0].abstract)
        self.assertIn("browser-fetched-seed", sources[0].quality_flags)

    def test_search_result_pages_are_not_seeded_as_sources(self) -> None:
        urls = DeepResearchEngine._source_seed_urls(
            "Use https://html.duckduckgo.com/html/?q=current+evidence and https://example.com/report as context",
            None,
            None,
        )

        self.assertEqual(urls, ["https://example.com/report"])

    def test_math_query_plan_ignores_repo_prefix_and_benchmark_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                (
                    "[multi-hour] Based on PROOF_STATUS.md, perform deep research "
                    "on accepted literature and plausible proof strategies for "
                    "the exact missing Collatz bridge from almost-all and finite "
                    "verification results to a universal theorem, focusing on "
                    "Lemma UB, deterministic pointwise transfer, mechanical carry "
                    "forcing, peel-chain no-crossing, Ostrowski return-block "
                    "cocycles, 2-adic conjugacy, the Lopez-Stoll critical-density "
                    "boundary, and infinity-to-finite proof methods from other "
                    "hard theorems."
                ),
                "run_5",
            )

            self.assertIn("collatz", brief.query)
            self.assertNotIn("proof_status", brief.query)

            plan_path = Path(temp_dir) / "runs/run_5/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            joined_variants = " ".join(plan["query_variants"]).lower()

            self.assertIn("collatz", joined_variants)
            self.assertIn("2-adic", joined_variants)
            self.assertNotIn("proof_status", joined_variants)
            self.assertFalse(
                any(
                    "benchmark" in variant.lower() for variant in plan["query_variants"]
                )
            )
            self.assertFalse(
                any(
                    "evaluation" in variant.lower()
                    for variant in plan["query_variants"]
                )
            )
            self.assertGreaterEqual(
                len(plan["subquestions"]),
                3,
                "Planning should produce at least 3 subquestions for a multi-hour query",
            )

    def test_software_agent_queries_include_repository_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[quick] compare OpenCode OpenClaw local PC agents",
                "run_4",
            )

            providers = {source.provider for source in brief.sources}
            self.assertIn("github-repositories", providers)
            self.assertIn("software-reference", providers)

            plan_path = Path(temp_dir) / "runs/run_4/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertIn("software repository search", plan["token_strategy"])

    def test_generic_deep_research_queries_stay_domain_specific(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[multi-hour] deep research on battery recycling economics",
                "run_4b",
            )

            self.assertIn("battery", brief.query.lower())
            self.assertIn("recycling", brief.query.lower())
            providers = {source.provider for source in brief.sources}
            self.assertNotIn("software-reference", providers)
            self.assertNotIn("github-repositories", providers)

            plan_path = Path(temp_dir) / "runs/run_4b/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertNotIn("software repository search", plan["token_strategy"])

    def test_web_search_provider_discovers_arbitrary_domain_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeWebSearchResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[quick] agentos desktop workflow safety approvals",
                "run_4c",
            )

            providers = {source.provider for source in brief.sources}
            self.assertIn("web-search", providers)
            self.assertTrue(
                any(
                    source.url == "https://docs.example.org/agentos"
                    for source in brief.sources
                )
            )

    def test_web_search_placeholder_does_not_echo_query_terms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = NoSnippetWebSearchResearchEngine(workspace_root=temp_dir)
            results = engine._search_web_results(
                "protocol-driven tool orchestration for general-purpose agents",
                limit=3,
            )

            self.assertTrue(results)
            self.assertEqual(
                results[0].abstract,
                "Generic web result. Snippet unavailable.",
            )

    def test_standard_web_search_enrichment_replaces_generic_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeWebSearchResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[standard] agentos desktop workflow safety approvals",
                "run_4c_standard",
            )

            web_sources = [
                source for source in brief.sources if source.provider == "web-search"
            ]
            self.assertTrue(web_sources)
            self.assertTrue(
                any(
                    "desktop workflow safety benchmark approvals reference"
                    in source.abstract.lower()
                    for source in web_sources
                )
            )
            self.assertTrue(
                all(
                    not source.abstract.lower().startswith("generic web result for ")
                    for source in web_sources
                )
            )

    def test_dedupe_preserves_enriched_abstract_from_duplicate_title(self) -> None:
        enriched = ResearchSource(
            provider="web-search",
            title="Microsoft Agent Framework Overview | Microsoft Learn",
            url="https://learn.microsoft.com/en-us/agent-framework/overview/",
            authors=["learn.microsoft.com"],
            abstract="Microsoft Agent Framework is a platform for orchestrating deterministic workflows and agentic interactions.",
            citation_count=1,
            score=2.0,
        )
        duplicate = ResearchSource(
            provider="web-search",
            title="Microsoft Agent Framework Overview | Microsoft Learn",
            url="https://learn.microsoft.com/en-us/agent-framework/overview/",
            authors=["learn.microsoft.com"],
            abstract="Generic web result for protocol-driven tool orchestration for general-purpose agents.",
            citation_count=3,
            score=6.0,
        )

        deduped = DeepResearchEngine._dedupe_sources([enriched, duplicate])

        self.assertEqual(len(deduped), 1)
        self.assertIn(
            "platform for orchestrating deterministic workflows", deduped[0].abstract
        )
        self.assertEqual(deduped[0].score, 6.0)
        self.assertEqual(deduped[0].citation_count, 3)

    def test_finalize_selected_sources_enriches_generic_late_web_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeWebSearchResearchEngine(workspace_root=temp_dir)
            source = ResearchSource(
                provider="web-search",
                title="AgentOS Safety Docs",
                url="https://docs.example.org/agentos",
                authors=["docs.example.org"],
                abstract="Generic web result for agentos desktop workflow safety approvals.",
                citation_count=3,
                score=6.0,
            )

            finalized = engine._finalize_selected_sources(
                [source],
                [source],
                "agentos desktop workflow safety approvals",
                5,
            )

            self.assertEqual(len(finalized), 1)
            self.assertIn(
                "desktop workflow safety approvals reference",
                finalized[0].abstract.lower(),
            )

    def test_dedupe_merges_duplicate_urls_with_different_titles(self) -> None:
        enriched = ResearchSource(
            provider="web-search",
            title="Protocol Lattice go-agent repository",
            url="https://github.com/Protocol-Lattice/go-agent",
            authors=["github.com"],
            abstract="GitHub repository for go-agent with multi-agent orchestration and UTCP-native tools.",
            citation_count=1,
            score=2.0,
        )
        duplicate = ResearchSource(
            provider="web-search",
            title="GitHub - Protocol-Lattice/go-agent: An agent framework for Go with ...",
            url="https://github.com/Protocol-Lattice/go-agent",
            authors=["github.com"],
            abstract="Generic web result for protocol-driven tool orchestration for general-purpose agents.",
            citation_count=3,
            score=6.0,
        )

        deduped = DeepResearchEngine._dedupe_sources([enriched, duplicate])

        self.assertEqual(len(deduped), 1)
        self.assertIn("multi-agent orchestration", deduped[0].abstract)
        self.assertEqual(deduped[0].score, 6.0)
        self.assertEqual(deduped[0].citation_count, 3)

    def test_explicit_urls_are_seeded_into_research(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = SeedUrlResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                (
                    "[quick] analyze https://docs.example.org/agentos "
                    "for agentos benchmark safety approvals"
                ),
                "run_4d",
            )

            seeded = [
                source for source in brief.sources if source.provider == "seed-url"
            ]
            self.assertTrue(seeded)
            self.assertEqual(seeded[0].url, "https://docs.example.org/agentos")

            plan_path = Path(temp_dir) / "runs/run_4d/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertIn("https://docs.example.org/agentos", plan["source_seeds"])

    def test_query_distillation_ignores_embedded_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                (
                    "[quick] Perform deep research on protocol-driven tool "
                    "orchestration for general-purpose agents using "
                    "https://modelcontextprotocol.io/introduction and "
                    "https://docs.python.org/3/library/urllib.parse.html as "
                    "anchor sources, then expand outward to corroborating "
                    "implementation guidance, safety constraints, and "
                    "architecture tradeoffs."
                ),
                "run_4e",
            )

            self.assertIn("protocol-driven", brief.query)
            self.assertNotIn("https://", brief.query)
            self.assertNotIn("using ht", brief.query)
            self.assertNotIn("corrob", brief.query)

    def test_balanced_selection_preserves_relevant_seed_url(self) -> None:
        query = "protocol-driven tool orchestration general-purpose agents"
        ranked = [
            ResearchSource(
                provider="web-search",
                title="Workflow orchestrations in Agent Framework",
                url="https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/",
                abstract="Official orchestration workflow overview.",
                score=52.0,
                relevance=1.0,
                credibility_score=0.53,
            ),
            ResearchSource(
                provider="web-search",
                title="AI Agent Orchestration Patterns",
                url="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns",
                abstract="Architecture patterns for agent orchestration.",
                score=49.0,
                relevance=1.0,
                credibility_score=0.45,
            ),
            ResearchSource(
                provider="seed-url",
                title="Model Context Protocol introduction",
                url="https://modelcontextprotocol.io/introduction",
                abstract=(
                    "Protocol-driven tool orchestration and safety guidance for "
                    "general-purpose agents."
                ),
                score=43.0,
                relevance=0.9,
                credibility_score=0.55,
            ),
            ResearchSource(
                provider="web-search",
                title="OpenAI Agents SDK orchestration",
                url="https://openai.github.io/openai-agents-python/multi_agent/",
                abstract="Multi-agent orchestration patterns.",
                score=41.0,
                relevance=0.9,
                credibility_score=0.43,
            ),
            ResearchSource(
                provider="crossref",
                title="Cyber Red Teaming: Overview of Sly, an Orchestration Tool",
                url="https://doi.org/10.11610/isij.5318",
                abstract="Overview of an orchestration tool.",
                score=36.0,
                relevance=0.4,
                credibility_score=0.63,
            ),
        ]

        selected = DeepResearchEngine._select_balanced_top(
            ranked,
            max_sources=3,
            query=query,
        )

        self.assertIn(
            "https://modelcontextprotocol.io/introduction",
            {source.url for source in selected},
        )

    def test_finding_ledger_prefers_perspective_specific_leads(self) -> None:
        engine = DeepResearchEngine()
        sources = [
            ResearchSource(
                provider="semantic-scholar",
                title="Deep Research Agents: A Systematic Examination And Roadmap",
                url="https://semanticscholar.org/paper/survey",
                year=2025,
                abstract=(
                    "Deep research agents are a new class of systems. "
                    "This survey maps the landscape and state of the art."
                ),
                citation_count=100,
                score=92.0,
                evidence_grade="moderate",
            ),
            ResearchSource(
                provider="semantic-scholar",
                title="WebThinker: Empowering Large Reasoning Models with Deep Research Capability",
                url="https://semanticscholar.org/paper/webthinker",
                year=2025,
                abstract=(
                    "WebThinker uses a planner worker architecture with browser "
                    "grounding for deep research tasks."
                ),
                citation_count=30,
                score=85.0,
                evidence_grade="strong",
            ),
            ResearchSource(
                provider="semantic-scholar",
                title="TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents",
                url="https://semanticscholar.org/paper/trace",
                year=2025,
                abstract=(
                    "TRACE introduces a trajectory-aware benchmark and evaluation "
                    "suite for deep research agents."
                ),
                citation_count=25,
                score=84.0,
                evidence_grade="moderate",
            ),
            ResearchSource(
                provider="semantic-scholar",
                title="DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence",
                url="https://semanticscholar.org/paper/deeptrace",
                year=2025,
                abstract=(
                    "DeepTRACE audits citation reliability, trustworthy report "
                    "generation, and safety risks in deep research systems."
                ),
                citation_count=20,
                score=83.0,
                evidence_grade="moderate",
            ),
        ]
        plan = {
            "perspectives": [
                {
                    "name": "overview",
                    "goal": "Establish the current system landscape and accepted framing.",
                    "keywords": ["survey", "overview", "landscape"],
                },
                {
                    "name": "architecture",
                    "goal": "Compare planner-worker topology, grounding, and execution design.",
                    "keywords": ["architecture", "planner", "worker", "grounding"],
                },
                {
                    "name": "evaluation",
                    "goal": "Find benchmark results, task suites, and verification evidence.",
                    "keywords": ["benchmark", "evaluation", "verification"],
                },
                {
                    "name": "safety",
                    "goal": "Find approval, safety, and trust-boundary evidence.",
                    "keywords": ["safety", "trust", "risk", "reliability"],
                },
            ]
        }

        findings = engine._finding_ledger("deep research agent", sources, plan)
        lead_titles = {
            item["perspective"]: item["supporting_sources"][0]["title"]
            for item in findings
        }

        self.assertEqual(
            lead_titles["overview"],
            "Deep Research Agents: A Systematic Examination And Roadmap",
        )
        self.assertEqual(
            lead_titles["architecture"],
            "WebThinker: Empowering Large Reasoning Models with Deep Research Capability",
        )
        self.assertEqual(
            lead_titles["evaluation"],
            "TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents",
        )
        self.assertEqual(
            lead_titles["safety"],
            "DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence",
        )

    def test_content_query_expansion_stays_anchored_to_original_query(self) -> None:
        queries = DeepResearchEngine._content_to_new_queries(
            (
                "Scitepress publication details display eds-c- padding mobile "
                "trusted display system theme share build automated system theme"
            ),
            "Scitepress styling details",
            "how to build a general-purpose deep research agent",
        )

        self.assertEqual(queries, [])

    def test_multi_hour_retains_first_passing_snapshot_after_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[multi-hour] accessibility tree desktop agents",
                "run_6",
                evidence_targets={
                    "min_source_count": 2,
                    "min_provider_count": 1,
                    "min_scholarly_sources": 1,
                    "min_strong_or_moderate": 1,
                    "min_novelty_rate": 0.1,
                    "max_retrieval_passes": 3,
                    "min_depth_passes": 3,
                },
            )

            self.assertGreaterEqual(
                brief.metadata["coverage"]["strong_or_moderate"],
                1,
            )
            self.assertGreaterEqual(
                brief.metadata["coverage"]["novelty_rate"],
                0.1,
            )

    def test_ai_gap_analysis_filters_noisy_queries(self) -> None:
        engine = NoisyGapAnalysisEngine()
        selected = [
            ResearchSource(
                provider="web-search",
                title="Stocks and catalysts overview",
                url="https://example.com/stocks-catalysts",
                abstract="Current catalysts and earnings revisions for growth stocks.",
                year=2026,
                score=80.0,
            )
        ]

        queries = engine._ai_evidence_gap_analysis(
            "stocks with highest potential right now",
            selected,
            pass_index=2,
        )
        cleaned = engine._sanitize_query_variants(
            queries,
            "stocks with highest potential right now",
        )

        self.assertNotIn("stocks right eqxmuk border-through ezyuzk", cleaned)
        self.assertIn(
            "stocks catalysts earnings revisions current analysis",
            cleaned,
        )

    def test_generic_perspectives_adapt_for_current_evidence_queries(self) -> None:
        perspectives = DeepResearchEngine._generic_perspectives(
            "stocks with highest potential to soar right now",
            "multi-hour",
        )
        names = {item.get("name") for item in perspectives}

        self.assertIn("current-signals", names)
        self.assertIn("drivers", names)
        self.assertIn("risk", names)

    def test_entity_queries_stay_grounded_for_explicit_software_entities(self) -> None:
        queries = DeepResearchEngine._entity_queries(
            "compare OpenCode OpenClaw local PC agents",
            "compare OpenCode OpenClaw local PC agents",
        )

        self.assertTrue(
            any(
                "comparison" in query.lower()
                and "opencode" in query.lower()
                and "openclaw" in query.lower()
                for query in queries
            )
        )
        self.assertFalse(
            any(
                query
                in {
                    "LLM agent benchmark evaluation",
                    "autonomous agent task planning execution",
                    "AI agent computer use evaluation",
                }
                for query in queries
            )
        )

    def test_collatz_ranking_demotes_speculative_recent_zero_citation_claims(
        self,
    ) -> None:
        query = "collatz bridge finite verification almost all"
        accepted = ResearchSource(
            provider="openalex",
            title="Almost all orbits of the Collatz map attain almost bounded values",
            url="https://openalex.org/W4289999999",
            year=2022,
            abstract="Established almost-all result for the Collatz map with quantitative bounds.",
            citation_count=135,
        )
        speculative = ResearchSource(
            provider="crossref",
            title="Collatz Conjecture Is True for All Positive Integers",
            url="https://doi.org/10.0000/example-collatz-claim",
            year=2025,
            abstract="This work proves the Collatz conjecture completely for all positive integers.",
            citation_count=0,
        )

        ranked = RankingProbe.rank_sources([speculative, accepted], query)

        self.assertEqual(ranked[0].title, accepted.title)
        accepted_ranked = next(item for item in ranked if item.title == accepted.title)
        self.assertEqual(accepted_ranked.evidence_grade, "moderate")
        speculative_ranked = next(
            (item for item in ranked if item.title == speculative.title),
            None,
        )
        if speculative_ranked is not None:
            self.assertEqual(speculative_ranked.evidence_grade, "weak")
            self.assertIn("speculative-proof-claim", speculative_ranked.quality_flags)
            self.assertLess(
                speculative_ranked.credibility_score,
                accepted.credibility_score,
            )

    def test_collatz_ranking_demotes_low_citation_proof_titles(self) -> None:
        query = "collatz bridge finite verification almost all"
        accepted = ResearchSource(
            provider="crossref",
            title="Improved verification limit for the convergence of the Collatz conjecture",
            url="https://doi.org/10.21203/rs.3.rs-3845558/v1",
            year=2025,
            abstract="Computational verification result improving the checked convergence limit for Collatz.",
            citation_count=12,
        )
        proof_title = ResearchSource(
            provider="semantic-scholar",
            title="Proof of the Collatz Conjecture Using Logical and Probabilistic Approaches",
            url="https://semanticscholar.org/paper/proof-title",
            year=2026,
            abstract="Claims a proof of the Collatz conjecture using logical and probabilistic methods.",
            citation_count=2,
        )

        ranked = RankingProbe.rank_sources([proof_title, accepted], query)

        self.assertEqual(ranked[0].title, accepted.title)
        proof_ranked = next(
            (item for item in ranked if item.title == proof_title.title),
            None,
        )
        if proof_ranked is not None:
            self.assertEqual(proof_ranked.evidence_grade, "weak")
            self.assertIn("unsupported-proof-title", proof_ranked.quality_flags)

    def test_math_refinement_variants_stay_in_domain(self) -> None:
        engine = FakeDeepResearchEngine()
        variants = engine._refinement_variants(
            "collatz bridge finite verification",
            [
                ResearchSource(
                    provider="openalex",
                    title="Almost all orbits of the Collatz map attain almost bounded values",
                    url="https://openalex.org/W4289999999",
                    year=2022,
                    abstract="Quantitative Collatz result.",
                    citation_count=135,
                )
            ],
            "multi-hour",
            2,
        )
        joined = " ".join(variants).lower()

        self.assertIn("theorem barrier", joined)
        self.assertNotIn("benchmark", joined)
        self.assertNotIn("repository architecture", joined)

    def test_low_signal_query_variants_are_rejected(self) -> None:
        self.assertTrue(
            DeepResearchEngine._is_low_signal_query_variant(
                "blur knob zenodo https records",
                "collatz bridge finite verification",
            )
        )
        self.assertFalse(
            DeepResearchEngine._is_low_signal_query_variant(
                "collatz density to pointwise transfer",
                "collatz bridge finite verification",
            )
        )

    def test_gemini_observations_survive_ranking(self) -> None:
        source = ResearchSource(
            provider="gemini-flash",
            title="Gemini observation",
            url="https://ai.google.dev/gemini-api/docs",
            year=2026,
            abstract=(
                "OpenCode, OpenClaw, OpenHands, local PC agents, browser "
                "research workflows, and operator verification should be "
                "tested together."
            ),
        )

        ranked = RankingProbe.rank_sources(
            [source],
            "OpenCode OpenClaw local PC agents browser research workflows",
        )

        self.assertEqual(ranked[0].provider, "gemini-flash")
        self.assertEqual(ranked[0].evidence_grade, "tool-observation")
        self.assertGreater(ranked[0].score, 60)

    def test_irrelevant_github_sources_are_filtered(self) -> None:
        query = "compare OpenHands OpenCode OpenClaw OSWorld WebArena"
        irrelevant = ResearchSource(
            provider="github-repositories",
            title="awesome-go",
            url="https://github.com/avelino/awesome-go",
            year=2026,
            abstract="Curated list of Go libraries and tools.",
            citation_count=100000,
        )
        relevant = ResearchSource(
            provider="github-repositories",
            title="anomalyco/opencode",
            url="https://github.com/anomalyco/opencode",
            year=2026,
            abstract="OpenCode coding agent benchmark for desktop workflows.",
            citation_count=100,
        )

        ranked = RankingProbe.rank_sources([irrelevant, relevant], query)
        titles = {item.title for item in ranked}

        self.assertIn("anomalyco/opencode", titles)
        self.assertNotIn("awesome-go", titles)

    def test_generic_research_agent_query_filters_desktop_app_lists(self) -> None:
        query = "deep research agent"
        irrelevant = ResearchSource(
            provider="github-repositories",
            title="jaywcjlove/awesome-mac",
            url="https://github.com/jaywcjlove/awesome-mac",
            year=2026,
            abstract=(
                "This project collects macOS software and desktop-app tools. "
                "Public GitHub repository evidence for software-agent research."
            ),
            citation_count=100000,
        )
        relevant = ResearchSource(
            provider="github-repositories",
            title="example/deep-research-agent",
            url="https://github.com/example/deep-research-agent",
            year=2026,
            abstract=(
                "Deep research agent benchmark and evaluation stack for "
                "browser automation workflows."
            ),
            citation_count=100,
        )

        ranked = RankingProbe.rank_sources([irrelevant, relevant], query)
        titles = {item.title for item in ranked}

        self.assertIn("example/deep-research-agent", titles)
        self.assertNotIn("jaywcjlove/awesome-mac", titles)

    def test_balanced_selection_prefers_scholarly_source(self) -> None:
        query = "compare OpenHands OpenCode OpenClaw OSWorld WebArena"
        scholarly = ResearchSource(
            provider="openalex",
            title="OSWorld Benchmark for Computer Use Agents",
            url="https://openalex.org/W123",
            year=2025,
            abstract="OSWorld benchmark evaluates desktop agent reliability.",
            citation_count=50,
        )
        github_a = ResearchSource(
            provider="github-repositories",
            title="anomalyco/opencode",
            url="https://github.com/anomalyco/opencode",
            year=2026,
            abstract="OpenCode desktop agent implementation.",
            citation_count=500,
        )
        github_b = ResearchSource(
            provider="github-repositories",
            title="openclaw/openclaw",
            url="https://github.com/openclaw/openclaw",
            year=2026,
            abstract="OpenClaw personal AI assistant.",
            citation_count=500,
        )

        # Pretend these are pre-ranked descending by score.
        selected = DeepResearchEngine._select_balanced_top(
            [github_a, github_b, scholarly],
            2,
            query,
        )
        providers = {item.provider for item in selected}

        self.assertIn("openalex", providers)
        self.assertIn("github-repositories", providers)

    def test_software_reference_sources_exclude_k_dense(self) -> None:
        sources = DeepResearchEngine._software_reference_sources(
            "OpenHands OpenCode OpenClaw"
        )
        urls = {item.url for item in sources}
        self.assertNotIn("https://www.k-dense.ai/", urls)


if __name__ == "__main__":
    unittest.main()
