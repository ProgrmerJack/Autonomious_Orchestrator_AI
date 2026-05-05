from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agentos_orchestrator.product import CrawlWorkerServiceRecord
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
            query = followup._query_from_objective(
                "accessibility tree desktop agents"
            )
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

    def test_detached_crawl_worker_processes_queue_and_records_observations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            objective = "agentos semantic browser deep research reliability"
            root_url = "https://docs.example.org/root"
            child_url = (
                "https://docs.example.org/articles/"
                "agentos-semantic-browser-worker-pool"
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

            with patch.object(
                engine,
                "_auto_start_crawl_workers_enabled",
                return_value=True,
            ), patch(
                "agentos_orchestrator.product.CrawlWorkerServiceManager"
            ) as service_manager_cls, patch(
                "agentos_orchestrator.product.CrawlWorkerManager"
            ) as worker_manager_cls:
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
                "revenue growth" in query.lower()
                and "nvidia" in query.lower()
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
            any(
                "supplier concentration" in query.lower()
                for query in queries
            )
        )
        self.assertTrue(any("site:sec.gov" in query.lower() for query in queries))
        self.assertTrue(
            any(
                "margin compression" in query.lower()
                or "capex" in query.lower()
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
                    "summary": {
                        "top_urls": ["https://docs.example.org/agentos"]
                    }
                },
                "frontier": {"mode": "expansive"},
            }
        }

        sources = engine._pc_finding_seed_sources(pc_context)
        matching = [
            source for source in sources if source.url == "https://docs.example.org/agentos"
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
            self.assertIn("current analysis", joined_variants)
            self.assertNotIn("literature", joined_variants)
            self.assertGreaterEqual(len(plan["query_variants"]), 5)
            self.assertGreaterEqual(len(metrics["passes"]), 4)

    def test_current_evidence_queries_use_current_web_providers(self) -> None:
        providers = DeepResearchEngine._classify_query(
            "highest-potential public companies as of now"
        )

        # Current-evidence queries use web-search + gemini-flash plus
        # supplementary providers for richer coverage.
        self.assertIn("web-search", providers)
        self.assertIn("gemini-flash", providers)
        self.assertNotIn("openalex", providers)
        self.assertNotIn("semantic-scholar", providers)

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
                "min_scholarly_sources": 5,
                "min_novelty_rate": 0.22,
            },
            "multi-hour",
        )

        self.assertEqual(overridden["min_provider_count"], 1)
        self.assertEqual(overridden["min_scholarly_sources"], 0)
        self.assertEqual(overridden["min_novelty_rate"], 0.0)
        self.assertEqual(overridden["max_retrieval_passes"], 120)
        self.assertGreater(
            DeepResearchEngine._effective_novelty_threshold(
                "multi-hour",
                overridden,
            ),
            0.0,
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
                query in {
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
