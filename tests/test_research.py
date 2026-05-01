from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agentos_orchestrator.research import DeepResearchEngine, ResearchSource


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
    ) -> str:
        del url, accept, max_bytes
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
    ) -> str:
        del accept, max_bytes
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
                "benchmark notes."
                "</body></html>"
            )
        return ""


class SeedUrlResearchEngine(FakeDeepResearchEngine):
    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
    ) -> str:
        del accept, max_bytes
        if "docs.example.org/agentos" in url:
            return (
                "<html><head><title>AgentOS benchmark safety approvals</title></head>"
                "<body>AgentOS benchmark safety approvals desktop workflow "
                "reliability notes and operator guidance.</body></html>"
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
    ) -> str:
        del accept, max_bytes
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
            self.assertIn(
                "Which unconditional, almost-all, or density results are already established?",
                plan["subquestions"],
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

    def test_general_research_agent_queries_are_treated_as_software_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeDeepResearchEngine(workspace_root=temp_dir)
            brief = engine.run(
                "[quick] how to build a general-purpose deep research agent",
                "run_4b",
            )

            self.assertEqual(brief.query, "deep research agent")
            providers = {source.provider for source in brief.sources}
            self.assertIn("software-reference", providers)

            plan_path = Path(temp_dir) / "runs/run_4b/research/research_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertIn("software repository search", plan["token_strategy"])

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
