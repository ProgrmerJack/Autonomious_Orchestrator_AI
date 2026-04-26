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
                        "authorships": [
                            {"author": {"display_name": "A. Researcher"}}
                        ],
                        "abstract_inverted_index": {
                            "Accessibility": [0],
                            "agents": [1],
                            "control": [2],
                        },
                        "cited_by_count": 7,
                        "id": "https://openalex.org/W1",
                    },
                    {
                        "display_name": (
                            "Membrane Transporters in Drug Development"
                        ),
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
                        "html_url": (
                            "https://github.com/All-Hands-AI/OpenHands"
                        ),
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

            claim_trace_path = (
                Path(temp_dir) / "runs/run_1/research/claim_trace.json"
            )
            claim_trace = json.loads(
                claim_trace_path.read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(claim_trace["source_count"], 1)
            self.assertTrue(claim_trace["claims"])

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

            plan_path = (
                Path(temp_dir) / "runs/run_2/research/research_plan.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["depth"], "multi-hour")
            self.assertGreaterEqual(len(plan["query_variants"]), 4)
            self.assertIn("structured scholarly APIs", plan["token_strategy"])

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
            plan_path = (
                Path(temp_dir) / "runs/run_3/research/research_plan.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["depth"], "quick")

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

            plan_path = (
                Path(temp_dir) / "runs/run_4/research/research_plan.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertIn("software repository search", plan["token_strategy"])

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


if __name__ == "__main__":
    unittest.main()
