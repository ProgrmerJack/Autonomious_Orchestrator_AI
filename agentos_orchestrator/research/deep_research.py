from __future__ import annotations

import json
import html
import ipaddress
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ResearchSource:
    provider: str
    title: str
    url: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    citation_count: int = 0
    score: float = 0.0
    relevance: float = 0.0
    recency: float = 0.0
    citation_strength: float = 0.0
    credibility_score: float = 0.0
    contradiction_risk: float = 0.0
    evidence_grade: str = "ungraded"
    quality_flags: list[str] = field(default_factory=list)

    def evidence(self) -> dict[str, Any]:
        return {
            "source": self.url,
            "provider": self.provider,
            "title": self.title,
            "year": self.year,
            "claim": self.abstract[:500] or self.title,
            "citation_count": self.citation_count,
            "evidence_grade": self.evidence_grade,
            "relevance": self.relevance,
            "credibility_score": self.credibility_score,
            "contradiction_risk": self.contradiction_risk,
            "quality_flags": self.quality_flags,
        }


@dataclass(slots=True)
class ResearchBrief:
    objective: str
    query: str
    summary: str
    sources: list[ResearchSource]
    artifacts: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def evidence(self) -> list[dict[str, Any]]:
        return [source.evidence() for source in self.sources]


@dataclass(frozen=True, slots=True)
class ResearchSettings:
    depth: str
    max_sources: int
    per_provider: int
    max_query_variants: int


class DeepResearchEngine:
    """MCP-friendly live research fallback using public scholarly APIs."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        limit_per_provider: int = 6,
        timeout_seconds: int = 20,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.limit_per_provider = limit_per_provider
        self.timeout_seconds = timeout_seconds
        self.provider_diagnostics: list[dict[str, Any]] = []

    def run(
        self,
        objective: str,
        run_id: str,
        pc_context: dict[str, Any] | None = None,
        planning_context: dict[str, Any] | None = None,
        evidence_targets: dict[str, Any] | None = None,
    ) -> ResearchBrief:
        self.provider_diagnostics = []
        depth, cleaned_objective = self._split_depth(objective)
        research_objective = self._clean_objective(cleaned_objective)
        settings = self._settings_for_depth(depth)
        query = self._query_from_objective(research_objective)
        pc_context_info = self._pc_context_summary(pc_context)
        plan = self._build_research_plan(
            research_objective,
            query,
            settings.depth,
            pc_context_info,
        )
        seed_urls = self._source_seed_urls(
            research_objective,
            planning_context,
            pc_context,
        )
        if seed_urls:
            plan["source_seeds"] = seed_urls
        merged_targets = (
            dict(planning_context.get("coverage_targets") or {})
            if planning_context
            else {}
        )
        merged_targets.update(evidence_targets or {})
        retrieval = self._iterative_retrieval(
            query=query,
            settings=settings,
            plan=plan,
            targets=merged_targets,
        )
        selected = retrieval["selected"]
        query_variants = retrieval["query_variants"]
        coverage = retrieval["coverage"]
        summary = self._summarize(
            research_objective,
            selected,
            settings.depth,
            plan,
            query,
        )
        artifacts = self._write_artifacts(
            run_id,
            research_objective,
            query,
            selected,
            summary,
            settings,
            query_variants,
            plan,
            pc_context_info,
            retrieval,
        )
        confidence = self._confidence(selected)
        return ResearchBrief(
            objective=research_objective,
            query=query,
            summary=summary,
            sources=selected,
            artifacts=artifacts,
            confidence=confidence,
            metadata={
                "coverage": coverage,
                "retrieval": {
                    "passes": retrieval["passes"],
                    "stop_reason": retrieval["stop_reason"],
                    "targets": merged_targets,
                },
            },
        )

    @classmethod
    def _source_seed_urls(
        cls,
        objective: str,
        planning_context: dict[str, Any] | None,
        pc_context: dict[str, Any] | None,
    ) -> list[str]:
        candidates: list[str] = []
        candidates.extend(cls._collect_urls(objective))
        if planning_context:
            candidates.extend(cls._collect_urls(planning_context))
        if pc_context:
            candidates.extend(cls._collect_urls(pc_context))

        deduped: list[str] = []
        seen: set[str] = set()
        for url in candidates:
            cleaned = url.rstrip(").,;]}>\"'")
            if not cls._is_safe_public_url(cleaned):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped[:12]

    @classmethod
    def _collect_urls(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return cls._urls_from_text(value)
        if isinstance(value, dict):
            results: list[str] = []
            for item in value.values():
                results.extend(cls._collect_urls(item))
            return results
        if isinstance(value, (list, tuple, set)):
            results: list[str] = []
            for item in value:
                results.extend(cls._collect_urls(item))
            return results
        return []

    @staticmethod
    def _urls_from_text(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"https?://[^\s<>()]+", text)

    def _seed_sources(self, seed_urls: list[str]) -> list[ResearchSource]:
        if not seed_urls:
            return []
        sources: list[ResearchSource] = []
        for url in seed_urls:
            source = self._seed_source(url)
            if source is not None:
                sources.append(source)
        self._record_provider_diagnostic(
            "seed-url",
            "ok" if sources else "empty",
            f"seeded {len(sources)} explicit URL sources",
        )
        return sources

    def _seed_source(self, url: str) -> ResearchSource | None:
        if not self._is_safe_public_url(url):
            return None
        content = self._fetch_page_text(url, max_bytes=40_000)
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        title = self._label_from_url(url)
        if content:
            first_sentence = re.split(r"[.!?]", content, maxsplit=1)[0].strip()
            if first_sentence and len(first_sentence.split()) <= 18:
                title = first_sentence[:160]
        abstract = (
            content
            or f"Seeded external source collected from the objective or context: {url}"
        )[:2000]
        return ResearchSource(
            provider="seed-url",
            title=title[:160],
            url=url,
            authors=[host] if host else [],
            abstract=abstract,
            score=18.0,
        )

    def _iterative_retrieval(
        self,
        query: str,
        settings: ResearchSettings,
        plan: dict[str, Any],
        targets: dict[str, Any],
    ) -> dict[str, Any]:
        all_sources: list[ResearchSource] = self._seed_sources(
            plan.get("source_seeds") or []
        )
        all_variants = plan["query_plan"][: settings.max_query_variants * 4]
        min_runtime_seconds = self._min_runtime_seconds(settings.depth, targets)
        min_depth_passes = self._min_depth_passes(settings.depth, targets)
        max_low_novelty_streak = self._max_low_novelty_streak(
            settings.depth,
            targets,
        )
        coverage_targets = self._coverage_targets(targets)
        max_passes = int(
            targets.get("max_retrieval_passes")
            or self._default_max_passes(settings.depth)
        )
        max_passes = max(max_passes, self._runtime_pass_floor(min_runtime_seconds))
        max_passes = max(max_passes, min_depth_passes)
        max_passes = max(1, min(max_passes, 240))
        started_at = time.monotonic()
        retrieval_passes: list[dict[str, Any]] = []
        previous_titles: set[str] = set()
        low_novelty_streak = 0
        stop_reason = "max_passes_reached"
        passing_snapshot: dict[str, Any] | None = None
        # Classify the query once to gate providers for every pass.
        allowed_providers = self._classify_query(query)

        for pass_index in range(max_passes):
            pass_variants = self._pass_variants(
                all_variants,
                pass_index,
                settings.max_query_variants,
            )
            if not pass_variants:
                elapsed = time.monotonic() - started_at
                if elapsed < min_runtime_seconds and all_variants:
                    pass_variants = all_variants[: settings.max_query_variants]
                else:
                    stop_reason = "no_query_variants"
                    break
            pass_sources: list[ResearchSource] = []
            for search_query in pass_variants:
                pass_sources.extend(
                    self._search_query_across_providers(
                        search_query,
                        allowed_providers,
                        settings.per_provider,
                    )
                )

            if pass_index == 0 and self._looks_like_software_agent_query(query):
                pass_sources.extend(self._software_reference_sources(query))
            if pass_index == 0:
                pass_sources.extend(
                    self._search_gemini_observation(query, settings.depth)
                )

            all_sources.extend(pass_sources)
            ranked = self._rank_sources(self._dedupe_sources(all_sources), query)
            selected = self._select_balanced_top(
                ranked,
                settings.max_sources,
                query,
            )

            # --- ENRICHMENT: fetch real content and chase citations ---
            # This is the work that makes research genuinely take time.
            # It runs after the first API pass so we enrich concrete results.
            if pass_index == 0 and settings.depth in {"standard", "multi-hour"}:
                content_queries = self._enrich_top_sources(
                    selected[: min(12, settings.max_sources)],
                    query,
                )
                for cq in content_queries:
                    if cq and cq not in all_variants:
                        all_variants.append(cq)
            if pass_index == 0 and settings.depth == "multi-hour":
                # Follow citations of the top scholarly sources (depth=1).
                cited = self._citation_chase(selected[:10], query, citation_depth=1)
                if cited:
                    all_sources.extend(cited)
                    ranked = self._rank_sources(
                        self._dedupe_sources(all_sources), query
                    )
                    selected = ranked[: settings.max_sources]
            if pass_index == 2 and settings.depth == "multi-hour":
                # Follow citations of citations (depth=2) for maximum coverage.
                cited2 = self._citation_chase(selected[:8], query, citation_depth=2)
                if cited2:
                    all_sources.extend(cited2)
                    ranked = self._rank_sources(
                        self._dedupe_sources(all_sources), query
                    )
                    selected = ranked[: settings.max_sources]
            if (
                settings.depth == "multi-hour"
                and pass_index >= 5
                and (pass_index + 1) % 4 == 0
            ):
                cited_more = self._citation_chase(
                    selected[:6],
                    query,
                    citation_depth=1,
                )
                if cited_more:
                    all_sources.extend(cited_more)
                    ranked = self._rank_sources(
                        self._dedupe_sources(all_sources), query
                    )
                    selected = ranked[: settings.max_sources]

            current_titles = {
                self._normalize_title(source.title)
                for source in selected
                if source.title
            }
            new_titles = current_titles - previous_titles
            novelty_rate = len(new_titles) / max(len(current_titles), 1)
            coverage = self._coverage_metrics(selected, novelty_rate, plan)
            retrieval_passes.append(
                {
                    "pass_index": pass_index + 1,
                    "query_variants": pass_variants,
                    "selected_count": len(selected),
                    "provider_count": coverage["provider_count"],
                    "novelty_rate": round(coverage["novelty_rate"], 3),
                    "max_contradiction_risk": round(
                        coverage["max_contradiction_risk"],
                        3,
                    ),
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                }
            )
            previous_titles = current_titles
            budget_met = (time.monotonic() - started_at) >= min_runtime_seconds
            depth_met = (pass_index + 1) >= min_depth_passes
            novelty_threshold = float(targets.get("min_novelty_rate") or 0.0)
            if coverage["novelty_rate"] < novelty_threshold and pass_index > 0:
                low_novelty_streak += 1
            else:
                low_novelty_streak = 0
            if self._meets_targets(coverage, coverage_targets):
                passing_snapshot = {
                    "selected": list(selected),
                    "coverage": dict(coverage),
                }
                if not budget_met or not depth_met:
                    continue
                stop_reason = "coverage_targets_met"
                selected = self._finalize_selected_sources(
                    selected,
                    all_sources,
                    query,
                    settings.max_sources,
                )
                coverage = self._coverage_metrics(
                    selected,
                    coverage["novelty_rate"],
                    plan,
                )
                return {
                    "selected": selected,
                    "coverage": coverage,
                    "passes": retrieval_passes,
                    "stop_reason": stop_reason,
                    "query_variants": all_variants,
                }
            if coverage["novelty_rate"] < novelty_threshold and pass_index > 0:
                if (
                    not budget_met
                    or not depth_met
                    or low_novelty_streak < max_low_novelty_streak
                ):
                    continue
                stop_reason = "novelty_below_threshold"
                break
            if coverage["max_contradiction_risk"] > float(
                targets.get("max_contradiction_risk") or 1.0
            ):
                if not budget_met or not depth_met:
                    continue
                stop_reason = "contradiction_above_threshold"
                break

            all_variants.extend(
                self._refinement_variants(
                    query,
                    selected,
                    settings.depth,
                    pass_index,
                    plan,
                )
            )

        if passing_snapshot is not None:
            selected = self._finalize_selected_sources(
                list(passing_snapshot["selected"]),
                all_sources,
                query,
                settings.max_sources,
            )
            coverage = self._coverage_metrics(
                selected,
                float(passing_snapshot["coverage"].get("novelty_rate") or 0.0),
                plan,
            )
            return {
                "selected": selected,
                "coverage": coverage,
                "passes": retrieval_passes,
                "stop_reason": (
                    "coverage_targets_met"
                    if stop_reason == "max_passes_reached"
                    else stop_reason
                ),
                "query_variants": all_variants[: settings.max_query_variants],
            }

        ranked = self._rank_sources(self._dedupe_sources(all_sources), query)
        selected = self._select_balanced_top(
            ranked,
            settings.max_sources,
            query,
        )
        selected = self._finalize_selected_sources(
            selected,
            all_sources,
            query,
            settings.max_sources,
        )
        novelty_rate = retrieval_passes[-1]["novelty_rate"] if retrieval_passes else 0.0
        coverage = self._coverage_metrics(selected, float(novelty_rate), plan)
        return {
            "selected": selected,
            "coverage": coverage,
            "passes": retrieval_passes,
            "stop_reason": stop_reason,
            "query_variants": all_variants[: settings.max_query_variants],
        }

    @staticmethod
    def _min_runtime_seconds(depth: str, targets: dict[str, Any]) -> int:
        raw_value = targets.get("min_runtime_seconds", 0)
        try:
            target_seconds = int(raw_value)
        except (TypeError, ValueError):
            target_seconds = 0
        if depth != "multi-hour":
            return 0
        return max(target_seconds, 0)

    @staticmethod
    def _min_depth_passes(depth: str, targets: dict[str, Any]) -> int:
        raw_value = targets.get("min_depth_passes", 0)
        try:
            target_passes = int(raw_value)
        except (TypeError, ValueError):
            target_passes = 0
        if depth != "multi-hour":
            return max(target_passes, 1)
        return max(target_passes, 12)

    @staticmethod
    def _default_max_passes(depth: str) -> int:
        if depth == "quick":
            return 1
        if depth == "multi-hour":
            return 12
        return 4

    @staticmethod
    def _coverage_targets(targets: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "min_source_count",
            "min_provider_count",
            "min_scholarly_sources",
            "min_strong_or_moderate",
            "min_novelty_rate",
            "max_contradiction_risk",
            "min_perspective_count",
            "min_perspective_ratio",
        )
        return {key: targets[key] for key in keys if key in targets}

    @staticmethod
    def _max_low_novelty_streak(depth: str, targets: dict[str, Any]) -> int:
        raw_value = targets.get("max_low_novelty_streak", 0)
        try:
            streak = int(raw_value)
        except (TypeError, ValueError):
            streak = 0
        if depth != "multi-hour":
            return max(streak, 1)
        return max(streak, 3)

    @staticmethod
    def _pass_variants(
        variants: list[str],
        pass_index: int,
        limit: int,
    ) -> list[str]:
        if not variants:
            return []
        stride = max(1, limit // 2)
        start = (pass_index * stride) % len(variants)
        window_size = min(limit, len(variants))
        return [
            variants[(start + index) % len(variants)] for index in range(window_size)
        ]

    @staticmethod
    def _runtime_pass_floor(min_runtime_seconds: int) -> int:
        if min_runtime_seconds <= 0:
            return 1
        # Typical provider pass cost is tens of seconds; floor upward.
        estimated_pass_seconds = 30
        return max(
            1,
            (min_runtime_seconds + estimated_pass_seconds - 1)
            // estimated_pass_seconds,
        )

    @classmethod
    def _coverage_metrics(
        cls,
        selected: list[ResearchSource],
        novelty_rate: float,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_count = len({source.provider for source in selected})
        scholarly_source_count = sum(
            1
            for source in selected
            if source.provider in {"openalex", "semantic-scholar", "crossref"}
        )
        strong_or_moderate = sum(
            1
            for source in selected
            if source.evidence_grade in {"strong", "moderate", "tool-observation"}
        )
        contradiction_max = max(
            (source.contradiction_risk for source in selected),
            default=0.0,
        )
        perspective_coverage = cls._perspective_coverage(
            selected,
            (plan or {}).get("perspectives") or [],
        )
        return {
            "source_count": len(selected),
            "provider_count": provider_count,
            "scholarly_source_count": scholarly_source_count,
            "strong_or_moderate": strong_or_moderate,
            "novelty_rate": novelty_rate,
            "max_contradiction_risk": contradiction_max,
            "perspective_count": perspective_coverage["count"],
            "perspective_total": perspective_coverage["total"],
            "perspective_ratio": perspective_coverage["ratio"],
            "covered_perspectives": perspective_coverage["covered"],
            "missing_perspectives": perspective_coverage["missing"],
        }

    @classmethod
    def _perspective_coverage(
        cls,
        selected: list[ResearchSource],
        perspectives: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not perspectives:
            return {
                "count": 0,
                "total": 0,
                "ratio": 0.0,
                "covered": [],
                "missing": [],
            }

        covered: list[str] = []
        missing: list[str] = []
        for perspective in perspectives:
            matches = cls._matched_sources_for_perspective(selected, perspective)
            if matches:
                covered.append(str(perspective["name"]))
            else:
                missing.append(str(perspective["name"]))

        total = len(perspectives)
        return {
            "count": len(covered),
            "total": total,
            "ratio": len(covered) / total if total else 0.0,
            "covered": covered,
            "missing": missing,
        }

    @classmethod
    def _matched_sources_for_perspective(
        cls,
        sources: list[ResearchSource],
        perspective: dict[str, Any],
    ) -> list[ResearchSource]:
        perspective_name = str(perspective.get("name") or "")
        keywords = [
            str(keyword).lower()
            for keyword in (perspective.get("keywords") or [])
            if str(keyword).strip()
        ]
        matched: list[ResearchSource] = []
        for source in sources:
            text = f"{source.title} {source.abstract}".lower()
            keyword_hits = sum(1 for keyword in keywords if keyword in text)
            if keyword_hits > 0:
                matched.append(source)
                continue
            if perspective_name in {"overview", "established-results"} and (
                source.provider in {"openalex", "semantic-scholar", "crossref"}
                and source.evidence_grade in {"strong", "moderate"}
            ):
                matched.append(source)
                continue
            if perspective_name in {"evidence", "evaluation", "computation"} and (
                source.evidence_grade in {"strong", "moderate", "tool-observation"}
            ):
                matched.append(source)
                continue
            if perspective_name in {
                "limitations",
                "proof-barriers",
                "failure-analysis",
                "safety",
            } and (
                source.contradiction_risk >= 0.25
                or any(
                    flag in source.quality_flags
                    for flag in (
                        "speculative-proof-claim",
                        "unsupported-proof-title",
                    )
                )
            ):
                matched.append(source)
        matched.sort(key=lambda item: item.score, reverse=True)
        return matched

    @staticmethod
    def _meets_targets(coverage: dict[str, Any], targets: dict[str, Any]) -> bool:
        if not targets:
            return False
        checks = [
            coverage["source_count"] >= int(targets.get("min_source_count", 0)),
            coverage["provider_count"] >= int(targets.get("min_provider_count", 0)),
            coverage["scholarly_source_count"]
            >= int(targets.get("min_scholarly_sources", 0)),
            coverage["strong_or_moderate"]
            >= int(targets.get("min_strong_or_moderate", 0)),
            coverage["novelty_rate"] >= float(targets.get("min_novelty_rate", 0.0)),
            coverage["max_contradiction_risk"]
            <= float(targets.get("max_contradiction_risk", 1.0)),
            coverage["perspective_count"]
            >= int(targets.get("min_perspective_count", 0)),
            coverage["perspective_ratio"]
            >= float(targets.get("min_perspective_ratio", 0.0)),
        ]
        return all(checks)

    def _refinement_variants(
        self,
        query: str,
        selected: list[ResearchSource],
        depth: str,
        pass_index: int,
        plan: dict[str, Any] | None = None,
    ) -> list[str]:
        providers = {source.provider for source in selected}
        software_mode = self._looks_like_software_agent_query(query)
        math_mode = self._looks_like_math_query(query)
        variants: list[str] = []
        if software_mode:
            variants.extend(
                [
                    f"{query} benchmark reproducibility pass {pass_index + 2}",
                    f"{query} limitations failure analysis pass {pass_index + 2}",
                ]
            )
            if "openalex" not in providers or "semantic-scholar" not in providers:
                variants.append(f"{query} survey paper evaluation methods")
            if "github-repositories" not in providers:
                variants.append(f"{query} repository architecture implementation")
        elif math_mode:
            variants.extend(
                [
                    f"{query} theorem barrier",
                    f"{query} obstruction transfer",
                    f"{query} density to pointwise",
                ]
            )
            if "openalex" not in providers or "semantic-scholar" not in providers:
                variants.append(f"{query} expository survey")
        else:
            variants.extend(
                [
                    f"{query} literature review",
                    f"{query} limitations open problems",
                ]
            )
            if "openalex" not in providers or "semantic-scholar" not in providers:
                variants.append(f"{query} survey paper")
        if plan is not None and plan.get("perspectives"):
            missing = self._perspective_coverage(
                selected,
                plan.get("perspectives") or [],
            )["missing"]
            for perspective in plan.get("perspectives") or []:
                if perspective.get("name") not in missing:
                    continue
                variants.extend((perspective.get("queries") or [])[:2])
        variants.extend(self._query_variants(query, depth))
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            if self._is_low_signal_query_variant(variant, query):
                continue
            normalized = self._normalize_title(variant)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(variant[:240])
        return deduped

    def _search_openalex(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "search": query,
                "per-page": str(limit or self.limit_per_provider),
                "select": ",".join(
                    [
                        "id",
                        "display_name",
                        "publication_year",
                        "authorships",
                        "abstract_inverted_index",
                        "cited_by_count",
                        "doi",
                        "primary_location",
                    ]
                ),
            }
        )
        payload = self._get_json(f"https://api.openalex.org/works?{params}")
        sources: list[ResearchSource] = []
        for item in payload.get("results", []):
            title = html.unescape(str(item.get("display_name") or "").strip())
            if not title:
                continue
            location = item.get("primary_location") or {}
            landing_page = location.get("landing_page_url") or item.get("doi")
            url = str(landing_page or item.get("id") or "")
            authors = [
                str(author.get("author", {}).get("display_name"))
                for author in item.get("authorships", [])[:6]
                if author.get("author", {}).get("display_name")
            ]
            citation_count = int(item.get("cited_by_count") or 0)
            sources.append(
                ResearchSource(
                    provider="openalex",
                    title=title,
                    url=url,
                    year=item.get("publication_year"),
                    authors=authors,
                    abstract=self._openalex_abstract(
                        item.get("abstract_inverted_index") or {}
                    ),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "openalex",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    def _search_github_repositories(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "q": f"{query} in:name,description,readme",
                "sort": "stars",
                "order": "desc",
                "per_page": str(limit or self.limit_per_provider),
            }
        )
        payload = self._get_json(f"https://api.github.com/search/repositories?{params}")
        if "items" not in payload:
            self._record_provider_diagnostic(
                "github-repositories",
                "empty",
                "GitHub repository search returned no items.",
            )
        sources: list[ResearchSource] = []
        for item in payload.get("items", []):
            name = str(item.get("full_name") or item.get("name") or "")
            if not name:
                continue
            description = str(item.get("description") or "")
            stars = int(item.get("stargazers_count") or 0)
            updated_at = str(item.get("updated_at") or "")
            year = _year_from_timestamp(updated_at)
            topics = ", ".join(str(topic) for topic in item.get("topics", []))
            sources.append(
                ResearchSource(
                    provider="github-repositories",
                    title=name,
                    url=str(item.get("html_url") or ""),
                    year=year,
                    authors=[str(item.get("owner", {}).get("login") or "")],
                    abstract=(
                        f"{description} Topics: {topics}. Public GitHub "
                        f"repository evidence for software-agent research."
                    ).strip(),
                    citation_count=stars,
                    score=float(stars),
                )
            )
        self._record_provider_diagnostic(
            "github-repositories",
            "ok" if sources else "empty",
            f"returned {len(sources)} repositories",
        )
        return sources

    def _search_gemini_observation(
        self,
        query: str,
        depth: str,
    ) -> list[ResearchSource]:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self._record_provider_diagnostic(
                "gemini-flash",
                "skipped",
                "GEMINI_API_KEY or GOOGLE_API_KEY was not configured.",
            )
            return []
        prompt = (
            "Act as a concise tool observer for an AgentOS smoke test. "
            "Compare the named local OS/coding/research agents only at a "
            "high level, mention uncertainty, and list concrete capabilities "
            "to verify locally. Query: "
            f"{query}. Depth: {depth}."
        )
        payload = json.dumps(
            {"contents": [{"parts": [{"text": prompt}]}]},
        ).encode("utf-8")
        request = urllib.request.Request(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-flash-latest:generateContent"
            ),
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": api_key,
                "User-Agent": "agentos-orchestrator/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - user-configured API
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = f"HTTP {exc.code}: {exc.reason}"
            try:
                body = exc.read().decode("utf-8")
            except OSError:
                body = ""
            self._record_provider_diagnostic(
                "gemini-flash",
                "error",
                f"{detail}. {body[:240]}",
            )
            return []
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self._record_provider_diagnostic(
                "gemini-flash",
                "error",
                f"{type(exc).__name__}: {exc}",
            )
            return []
        text = _gemini_text(response_payload)
        if not text:
            self._record_provider_diagnostic(
                "gemini-flash",
                "empty",
                "Gemini returned no text parts.",
            )
            return []
        self._record_provider_diagnostic(
            "gemini-flash",
            "ok",
            f"returned {len(text)} characters",
        )
        return [
            ResearchSource(
                provider="gemini-flash",
                title=f"Gemini Flash tool observation for {query[:80]}",
                url="https://ai.google.dev/gemini-api/docs",
                year=datetime.now(UTC).year,
                authors=["Google Gemini API"],
                abstract=text,
                citation_count=0,
                score=25.0,
            )
        ]

    def _search_semantic_scholar(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": str(limit or self.limit_per_provider),
                "fields": ",".join(
                    [
                        "title",
                        "abstract",
                        "authors",
                        "year",
                        "url",
                        "citationCount",
                        "openAccessPdf",
                    ]
                ),
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        payload = self._get_json(url)
        sources: list[ResearchSource] = []
        for item in payload.get("data", []):
            title = html.unescape(str(item.get("title") or "").strip())
            if not title:
                continue
            open_pdf = item.get("openAccessPdf") or {}
            source_url = str(open_pdf.get("url") or item.get("url") or "")
            citation_count = int(item.get("citationCount") or 0)
            sources.append(
                ResearchSource(
                    provider="semantic-scholar",
                    title=title,
                    url=source_url,
                    year=item.get("year"),
                    authors=[
                        str(author.get("name"))
                        for author in item.get("authors", [])[:6]
                        if author.get("name")
                    ],
                    abstract=str(item.get("abstract") or ""),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "semantic-scholar",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    def _search_crossref(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode(
            {
                "query.bibliographic": query,
                "rows": str(limit or self.limit_per_provider),
                "sort": "relevance",
                "order": "desc",
            }
        )
        payload = self._get_json(f"https://api.crossref.org/works?{params}")
        message = payload.get("message") if isinstance(payload, dict) else {}
        items = message.get("items") if isinstance(message, dict) else []
        sources: list[ResearchSource] = []
        for item in items or []:
            title_list = item.get("title") or []
            title = html.unescape(str(title_list[0] if title_list else "").strip())
            if not title:
                continue
            doi = str(item.get("DOI") or "").strip()
            source_url = str(
                item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            )
            created = item.get("created") or {}
            date_parts = created.get("date-parts") or []
            year = None
            if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                try:
                    year = int(date_parts[0][0])
                except (TypeError, ValueError):
                    year = None
            authors = []
            for author in item.get("author", [])[:6]:
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            citation_count = int(item.get("is-referenced-by-count") or 0)
            sources.append(
                ResearchSource(
                    provider="crossref",
                    title=title,
                    url=source_url,
                    year=year,
                    authors=authors,
                    abstract=str(item.get("abstract") or ""),
                    citation_count=citation_count,
                    score=float(citation_count),
                )
            )
        self._record_provider_diagnostic(
            "crossref",
            "ok" if sources else "empty",
            f"returned {len(sources)} records",
        )
        return sources

    def _search_web_results(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        params = urllib.parse.urlencode({"q": query})
        raw_html = self._get_text(
            f"https://html.duckduckgo.com/html/?{params}",
            accept="text/html,application/xhtml+xml",
            max_bytes=120_000,
        )
        sources: list[ResearchSource] = []
        if raw_html:
            for rank, match in enumerate(
                re.finditer(
                    (
                        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"'
                        r"[^>]*>(.*?)</a>"
                    ),
                    raw_html,
                    flags=re.IGNORECASE | re.DOTALL,
                ),
                start=1,
            ):
                raw_url = self._normalize_web_result_url(match.group(1))
                if not self._is_safe_public_url(raw_url):
                    continue
                title = self._html_to_text(match.group(2)) or self._label_from_url(
                    raw_url
                )
                tail = raw_html[match.end() : match.end() + 1500]
                snippet_match = re.search(
                    r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                    tail,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                snippet = (
                    self._html_to_text(snippet_match.group(1))
                    if snippet_match is not None
                    else ""
                )
                host = urllib.parse.urlparse(raw_url).netloc.lower().lstrip("www.")
                score = max((limit or self.limit_per_provider) - rank + 1, 0)
                sources.append(
                    ResearchSource(
                        provider="web-search",
                        title=title[:160],
                        url=raw_url,
                        authors=[host] if host else [],
                        abstract=(
                            snippet or "Generic web result. Snippet unavailable."
                        )[:1200],
                        citation_count=score,
                        score=float(score),
                    )
                )
                if len(sources) >= (limit or self.limit_per_provider):
                    break
        self._record_provider_diagnostic(
            "web-search",
            "ok" if sources else "empty",
            f"returned {len(sources)} results",
        )
        return sources

    # ------------------------------------------------------------------
    # Provider routing
    # ------------------------------------------------------------------

    def _search_query_across_providers(
        self,
        search_query: str,
        allowed_providers: set[str],
        per_provider_limit: int,
    ) -> list[ResearchSource]:
        provider_searchers = self._provider_searchers()
        sources: list[ResearchSource] = []
        for provider in self._provider_order():
            if provider not in allowed_providers:
                continue
            if (
                provider == "github-repositories"
                and not self._looks_like_software_agent_query(search_query)
            ):
                continue
            searcher = provider_searchers.get(provider)
            if searcher is None:
                continue
            limit = (
                min(per_provider_limit, 5)
                if provider == "github-repositories"
                else per_provider_limit
            )
            provider_results = searcher(search_query, limit)
            if not provider_results:
                self._record_provider_diagnostic(
                    provider,
                    "query-empty",
                    f"0 results for query: {search_query[:120]}",
                )
                continue
            sources.extend(provider_results)
        return sources

    @staticmethod
    def _provider_order() -> tuple[str, ...]:
        return (
            "openalex",
            "semantic-scholar",
            "crossref",
            "web-search",
            "github-repositories",
        )

    def _provider_searchers(self) -> dict[str, Any]:
        return {
            "openalex": self._search_openalex,
            "semantic-scholar": self._search_semantic_scholar,
            "crossref": self._search_crossref,
            "web-search": self._search_web_results,
            "github-repositories": self._search_github_repositories,
        }

    @classmethod
    def _classify_query(cls, query: str) -> set[str]:
        """Return the set of provider keys that are appropriate for *query*.

        The goal is to avoid calling GitHub for a recipe question, or
        sending biomedical terms to a code-repo search engine.  All
        unrecognised queries fall back to the full scholarly stack.
        """
        lower = query.lower()
        words = set(re.findall(r"\b[a-z]+\b", lower))

        # Queries about cooking, food, travel, entertainment → only a
        # general-knowledge LLM can help; scholarly APIs return nothing.
        non_academic = {
            "recipe",
            "recipes",
            "cooking",
            "cook",
            "food",
            "meal",
            "ingredient",
            "ingredients",
            "bake",
            "baking",
            "dish",
            "travel",
            "restaurant",
            "hotel",
            "weather",
            "sports",
            "movie",
            "music",
            "celebrity",
            "fashion",
        }
        if words & non_academic:
            # Scholarly providers won't return useful results. Fall back to
            # broad web search and tool observations.
            return {"web-search", "gemini-flash"}

        # Default scholarly stack is always included.
        selected: set[str] = {
            "openalex",
            "semantic-scholar",
            "crossref",
            "web-search",
        }

        # Code / software queries also warrant a GitHub search.
        software_words = {
            "github",
            "code",
            "repository",
            "repo",
            "software",
            "framework",
            "library",
            "api",
            "runtime",
            "cli",
            "sdk",
            "deploy",
            "deployment",
            "compiler",
            "programming",
            "developer",
        }
        if words & software_words or cls._looks_like_software_agent_query(query):
            selected.add("github-repositories")

        return selected

    # ------------------------------------------------------------------
    # Content enrichment and citation chasing
    # ------------------------------------------------------------------

    def _enrich_top_sources(
        self,
        sources: list[ResearchSource],
        query: str = "",
    ) -> list[str]:
        """Fetch each source's landing page, extend its abstract with real
        content, and return new query strings extracted from that content.

        This is the primary driver of genuine research runtime: every HTTP
        fetch introduces real I/O latency.  No artificial sleeps are used;
        the time cost comes entirely from network round-trips.
        """
        new_queries: list[str] = []
        for source in sources:
            if not self._is_safe_public_url(source.url):
                continue
            content = self._fetch_page_text(source.url, max_bytes=40_000)
            if len(content) > 80:
                # Extend the abstract so ranking gets real signal.
                extra = content[:600]
                if source.abstract.lower().startswith("generic web result for "):
                    source.abstract = extra[:2000]
                else:
                    source.abstract = f"{source.abstract} {extra}".strip()[:2000]
                # Derive new focused queries from the fetched content.
                new_queries.extend(
                    self._content_to_new_queries(content, source.title, query)
                )
        # Deduplicate before returning.
        seen: set[str] = set()
        result: list[str] = []
        for q in new_queries:
            if self._is_low_signal_query_variant(q, query):
                continue
            norm = self._normalize_title(q)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(q[:80])
        return result[:12]

    def _fetch_page_text(self, url: str, max_bytes: int = 40_000) -> str:
        """Fetch *url* and return stripped plain text.

        Returns an empty string on any error — callers must tolerate failure.
        """
        if not self._is_safe_public_url(url):
            return ""
        raw = self._get_text(
            url,
            accept="text/html,application/xhtml+xml,*/*",
            max_bytes=max_bytes,
        )
        if not raw:
            return ""
        return self._html_to_text(raw)

    def _finalize_selected_sources(
        self,
        selected: list[ResearchSource],
        all_sources: list[ResearchSource],
        query: str,
        max_sources: int,
    ) -> list[ResearchSource]:
        needs_enrichment = [
            source
            for source in selected
            if self._is_safe_public_url(source.url)
            and self._abstract_quality(source.abstract)[0] == 0
        ]
        if not needs_enrichment:
            return selected
        self._enrich_top_sources(needs_enrichment[: min(12, max_sources)], query)
        ranked = self._rank_sources(self._dedupe_sources(all_sources), query)
        return self._select_balanced_top(ranked, max_sources, query)

    def _get_text(
        self,
        url: str,
        accept: str = "text/html,application/xhtml+xml,*/*",
        max_bytes: int = 40_000,
    ) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "agentos-orchestrator/0.1 (research enrichment)",
            },
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - policy-gated URLs
                request,
                timeout=min(self.timeout_seconds, 15),
            ) as response:
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if content_type and not any(
                    marker in content_type
                    for marker in ("text/", "html", "xml", "json")
                ):
                    return ""
                return response.read(max_bytes).decode("utf-8", errors="replace")
        except (OSError, urllib.error.URLError):
            return ""

    @staticmethod
    def _html_to_text(raw: str) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_web_result_url(url: str) -> str:
        cleaned = html.unescape(url).strip()
        if cleaned.startswith("//"):
            cleaned = f"https:{cleaned}"
        if cleaned.startswith("/"):
            cleaned = urllib.parse.urljoin("https://html.duckduckgo.com/", cleaned)
        parsed = urllib.parse.urlparse(cleaned)
        if "duckduckgo.com" in parsed.netloc:
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [None])[0]
            if target:
                return urllib.parse.unquote(target)
        return cleaned

    @staticmethod
    def _label_from_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
        candidate = path or parsed.netloc.lower().lstrip("www.") or url
        candidate = re.sub(r"\.[a-z0-9]{1,5}$", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"[-_]+", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        return candidate[:160] or url[:160]

    @staticmethod
    def _is_safe_public_url(url: str) -> bool:
        if not url.lower().startswith(("http://", "https://")):
            return False
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        if not host or host == "localhost":
            return False
        if host.endswith((".local", ".lan", ".internal", ".home", ".localdomain")):
            return False
        if "." not in host and not re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return True
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )

    @classmethod
    def _content_to_new_queries(
        cls,
        content: str,
        source_title: str,
        query: str = "",
    ) -> list[str]:
        """Extract 2-4 focused keyword phrases from fetched page content."""
        # Pick the most frequent non-stop content words.
        words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{3,}\b", content.lower())
        stop = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "will",
            "were",
            "they",
            "their",
            "which",
            "there",
            "about",
            "also",
            "when",
            "into",
            "more",
            "some",
            "than",
            "your",
            "each",
            "other",
            "over",
            "such",
            "like",
            "only",
            "both",
            "abstract",
            "introduction",
            "conclusion",
            "references",
            "section",
            "figure",
            "table",
            "paper",
            "work",
            "using",
            "https",
            "http",
            "www",
            "doi",
            "arxiv",
            "zenodo",
            "record",
            "records",
            "download",
            "license",
            "copyright",
            "manifest",
            "version",
            "supplementary",
        }
        anchor_stop = {
            "how",
            "build",
            "building",
            "general",
            "purpose",
            "deep",
            "agent",
            "agents",
            "system",
            "systems",
            "research",
        }
        counts = Counter(w for w in words if w not in stop and len(w) > 4)
        top_terms = [w for w, _ in counts.most_common(8)]
        if not top_terms:
            return []
        anchor_terms = {
            term
            for term in cls._entity_terms_from_query(query)
            if term and len(term) >= 4
        }
        anchor_terms.update(
            word
            for word in cls._keywords(query)
            if word not in anchor_stop and len(word) >= 4
        )
        source_text = f"{source_title} {content}".lower()
        matching_anchors = [
            term for term in anchor_terms if term.lower() in source_text
        ]
        if query and not matching_anchors:
            return []
        # Combine title keywords with top content terms.
        title_words = [
            w
            for w in re.findall(r"\b[a-zA-Z]{4,}\b", source_title.lower())
            if w not in stop
        ][:3]
        anchor_prefix = " ".join(matching_anchors[:2]).strip()
        queries: list[str] = []
        if top_terms[:3]:
            candidate = " ".join(top_terms[:3])
            if anchor_prefix and not any(
                term in candidate for term in matching_anchors
            ):
                candidate = f"{anchor_prefix} {candidate}".strip()
            queries.append(candidate)
        if title_words and top_terms[:2]:
            candidate = f"{' '.join(title_words[:2])} {' '.join(top_terms[:2])}".strip()
            if anchor_prefix and not any(
                term in candidate for term in matching_anchors
            ):
                candidate = f"{anchor_prefix} {candidate}".strip()
            queries.append(candidate)
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in queries:
            normalized = cls._normalize_title(candidate)
            if not normalized or normalized in seen:
                continue
            if query and not any(
                term in candidate.lower() for term in matching_anchors
            ):
                continue
            seen.add(normalized)
            deduped.append(candidate[:80])
        return deduped[:4]

    def _citation_chase(
        self,
        sources: list[ResearchSource],
        query: str,
        citation_depth: int = 1,
    ) -> list[ResearchSource]:
        """Follow cited-works links for OpenAlex sources and return newly
        discovered papers.

        *citation_depth* determines how many hops to follow:
        - depth=1: cited works of the seeds
        - depth=2: cited works of those cited works (i.e. grandchildren)

        Each API call and enrichment fetch contributes genuine I/O latency;
        this is what makes multi-hour depth naturally take more time.
        """
        frontier = list(sources)
        all_chased: list[ResearchSource] = []
        seen_ids: set[str] = {s.url for s in sources if s.url}
        for _depth in range(citation_depth):
            next_frontier: list[ResearchSource] = []
            for source in frontier:
                if not source.url:
                    continue
                # Only OpenAlex sources expose cited-works via API.
                oa_match = re.search(r"openalex\.org/(W\d+)", source.url)
                if not oa_match:
                    continue
                work_id = oa_match.group(1)
                cited = self._fetch_openalex_cited_works(work_id, limit=6)
                for c in cited:
                    if c.url not in seen_ids:
                        seen_ids.add(c.url)
                        next_frontier.append(c)
                        all_chased.append(c)
            # Enrich the newly discovered sources before the next depth hop.
            if next_frontier and citation_depth > 1:
                self._enrich_top_sources(next_frontier[:8], query)
            frontier = next_frontier
        return all_chased

    def _fetch_openalex_cited_works(
        self,
        work_id: str,
        limit: int = 6,
    ) -> list[ResearchSource]:
        """Return ResearchSource objects for works cited by *work_id*."""
        # First get the referenced_works list.
        detail = self._get_json(
            f"https://api.openalex.org/works/{work_id}?select=referenced_works"
        )
        ref_ids = [
            r.rstrip("/").rsplit("/", 1)[-1]
            for r in detail.get("referenced_works", [])[:limit]
        ]
        if not ref_ids:
            return []
        # Batch-fetch those works.
        filter_param = "|".join(ref_ids)
        select_fields = ",".join(
            [
                "id",
                "display_name",
                "publication_year",
                "authorships",
                "abstract_inverted_index",
                "cited_by_count",
                "doi",
                "primary_location",
            ]
        )
        payload = self._get_json(
            "https://api.openalex.org/works"
            f"?filter=openalex_id:{filter_param}"
            f"&select={select_fields}"
        )
        sources: list[ResearchSource] = []
        for item in payload.get("results", []):
            title = html.unescape(str(item.get("display_name") or "").strip())
            if not title:
                continue
            location = item.get("primary_location") or {}
            landing_page = location.get("landing_page_url") or item.get("doi")
            url = str(landing_page or item.get("id") or "")
            authors = [
                str(a.get("author", {}).get("display_name"))
                for a in item.get("authorships", [])[:4]
                if a.get("author", {}).get("display_name")
            ]
            sources.append(
                ResearchSource(
                    provider="openalex",
                    title=title,
                    url=url,
                    year=item.get("publication_year"),
                    authors=authors,
                    abstract=self._openalex_abstract(
                        item.get("abstract_inverted_index") or {}
                    ),
                    citation_count=int(item.get("cited_by_count") or 0),
                )
            )
        return sources

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "agentos-orchestrator/0.1",
            },
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - policy-gated URLs
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return {}

    def _record_provider_diagnostic(
        self,
        provider: str,
        status: str,
        detail: str = "",
    ) -> None:
        self.provider_diagnostics.append(
            {
                "provider": provider,
                "status": status,
                "detail": detail[:500],
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

    def _write_artifacts(
        self,
        run_id: str,
        objective: str,
        query: str,
        sources: list[ResearchSource],
        summary: str,
        settings: ResearchSettings,
        query_variants: list[str],
        plan: dict[str, Any],
        pc_context_info: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> list[str]:
        artifact_dir = self.workspace_root / "runs" / run_id / "research"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sources_path = artifact_dir / "sources.json"
        brief_path = artifact_dir / "brief.md"
        digest_path = artifact_dir / "digest.json"
        plan_path = artifact_dir / "research_plan.json"
        claim_trace_path = artifact_dir / "claim_trace.json"
        findings_path = artifact_dir / "findings.json"
        diagnostics_path = artifact_dir / "provider_diagnostics.json"
        analysis_report_path = artifact_dir / "analysis_report.md"
        paper_report_path = artifact_dir / "paper_report.md"
        retrieval_metrics_path = artifact_dir / "retrieval_metrics.json"
        evidence_graph_path = artifact_dir / "evidence_graph.json"
        benchmark_adapters_path = artifact_dir / "benchmark_adapters.json"
        findings = self._finding_ledger(query, sources, plan)
        retrieval_payload = {
            "coverage": retrieval["coverage"],
            "passes": retrieval["passes"],
            "stop_reason": retrieval["stop_reason"],
            "query_variants": retrieval["query_variants"],
        }
        benchmark_adapters = self._benchmark_adapters(sources)

        sources_path.write_text(
            json.dumps([asdict(source) for source in sources], indent=2),
            encoding="utf-8",
        )
        brief_path.write_text(
            self._brief_markdown(
                objective,
                query,
                summary,
                sources,
                settings.depth,
            ),
            encoding="utf-8",
        )
        digest_path.write_text(
            json.dumps(
                [
                    {
                        "title": source.title,
                        "provider": source.provider,
                        "url": source.url,
                        "year": source.year,
                        "citation_count": source.citation_count,
                        "score": round(source.score, 3),
                        "quality": {
                            "relevance": round(source.relevance, 3),
                            "recency": round(source.recency, 3),
                            "citation_strength": round(
                                source.citation_strength,
                                3,
                            ),
                            "contradiction_risk": round(
                                source.contradiction_risk,
                                3,
                            ),
                            "evidence_grade": source.evidence_grade,
                        },
                        "claim": source.abstract[:700] or source.title,
                    }
                    for source in sources
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        plan_path.write_text(
            json.dumps(
                {
                    "depth": settings.depth,
                    "objective": objective,
                    "query": query,
                    "query_variants": query_variants,
                    "source_seeds": plan.get("source_seeds") or [],
                    "max_sources": settings.max_sources,
                    "per_provider": settings.per_provider,
                    "core_question": plan["core_question"],
                    "subquestions": plan["subquestions"],
                    "comparative_axes": plan["comparative_axes"],
                    "evidence_requirements": plan["evidence_requirements"],
                    "perspectives": plan.get("perspectives") or [],
                    "pc_context": pc_context_info,
                    "coverage": retrieval["coverage"],
                    "stop_reason": retrieval["stop_reason"],
                    "token_strategy": (
                        "structured scholarly APIs, broad web search, "
                        "explicit URL seeding, software repository search, "
                        "optional model observations, exact dedupe, plan-first "
                        "multi-perspective query decomposition, finding "
                        "support/conflict ledger, relevance ranking, "
                        "compressed digest artifacts"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        analysis_report_path.write_text(
            self._analysis_report_markdown(
                objective,
                summary,
                sources,
                plan,
                pc_context_info,
            ),
            encoding="utf-8",
        )
        paper_report_path.write_text(
            self._paper_report_markdown(
                objective,
                summary,
                sources,
                plan,
                retrieval,
                pc_context_info,
            ),
            encoding="utf-8",
        )
        retrieval_metrics_path.write_text(
            json.dumps(retrieval_payload, indent=2),
            encoding="utf-8",
        )
        findings_path.write_text(
            json.dumps(findings, indent=2),
            encoding="utf-8",
        )
        claim_trace_path.write_text(
            json.dumps(
                self._claim_trace(objective, summary, findings),
                indent=2,
            ),
            encoding="utf-8",
        )
        evidence_graph_path.write_text(
            json.dumps(
                self._evidence_graph(
                    objective,
                    sources,
                    retrieval,
                    pc_context_info,
                    findings,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        benchmark_adapters_path.write_text(
            json.dumps(benchmark_adapters, indent=2),
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            json.dumps(self.provider_diagnostics, indent=2),
            encoding="utf-8",
        )
        return [
            str(sources_path.relative_to(self.workspace_root)),
            str(brief_path.relative_to(self.workspace_root)),
            str(digest_path.relative_to(self.workspace_root)),
            str(plan_path.relative_to(self.workspace_root)),
            str(analysis_report_path.relative_to(self.workspace_root)),
            str(paper_report_path.relative_to(self.workspace_root)),
            str(retrieval_metrics_path.relative_to(self.workspace_root)),
            str(findings_path.relative_to(self.workspace_root)),
            str(claim_trace_path.relative_to(self.workspace_root)),
            str(evidence_graph_path.relative_to(self.workspace_root)),
            str(benchmark_adapters_path.relative_to(self.workspace_root)),
            str(diagnostics_path.relative_to(self.workspace_root)),
        ]

    def _benchmark_adapters(
        self,
        sources: list[ResearchSource],
    ) -> dict[str, Any]:
        def _extract_records(framework: str) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for source in sources:
                text = f"{source.title} {source.abstract}".lower()
                if framework not in text:
                    continue
                metric_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", source.abstract)
                success_rate = (
                    float(metric_match.group(1)) / 100.0
                    if metric_match is not None
                    else None
                )
                records.append(
                    {
                        "framework": framework,
                        "source_title": source.title,
                        "source_url": source.url,
                        "provider": source.provider,
                        "task_family": "desktop-web-multistep",
                        "success_rate": success_rate,
                        "evidence_grade": source.evidence_grade,
                        "citation_count": source.citation_count,
                    }
                )
            return records

        return {
            "osworld": {
                "schema_version": "1.0",
                "records": _extract_records("osworld"),
            },
            "webarena": {
                "schema_version": "1.0",
                "records": _extract_records("webarena"),
            },
        }

    def _paper_report_markdown(
        self,
        objective: str,
        summary: str,
        sources: list[ResearchSource],
        plan: dict[str, Any],
        retrieval: dict[str, Any],
        pc_context_info: dict[str, Any],
    ) -> str:
        lines = [
            "# Paper-Mode Research Report",
            "",
            "## Methods",
            "",
            f"Objective: {objective}",
            "",
            "Hypothesis-driven subquestions:",
        ]
        for question in plan["subquestions"]:
            lines.append(f"- {question}")
        lines.extend(
            [
                "",
                "Iterative retrieval protocol:",
                f"- Passes executed: {len(retrieval['passes'])}",
                f"- Stopping criterion: {retrieval['stop_reason']}",
                f"- Coverage snapshot: {json.dumps(retrieval['coverage'])}",
                "",
                "Local PC instrumentation:",
                (
                    f"- Snapshot available: {pc_context_info['available']}; "
                    f"nodes: {pc_context_info['node_count']}"
                ),
                "",
                "## Results",
                "",
                summary,
                "",
                "Evidence table:",
                "",
                "| Claim Source | Provider | Grade | Citation Count |",
                "|---|---|---|---|",
            ]
        )
        for source in sources:
            lines.append(
                "| "
                f"{source.title} | {source.provider} | {source.evidence_grade} | "
                f"{source.citation_count} |"
            )
        lines.extend(
            [
                "",
                "## Discussion",
                "",
                "Strengths:",
                "- Structured planning and explicit coverage gates were applied.",
                "- Evidence was linked into claim traces and graph nodes.",
                "",
                "Limitations:",
                "- Provider availability can still constrain source diversity.",
                "- Repository documentation is weaker than controlled benchmarks.",
                "",
                "Reproducibility:",
                "- Required artifacts: research_plan.json, retrieval_metrics.json, claim_trace.json, evidence_graph.json.",
                "- Each final claim must map to at least one source URL in claim_trace.json.",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    def _evidence_graph(
        self,
        objective: str,
        sources: list[ResearchSource],
        retrieval: dict[str, Any],
        pc_context_info: dict[str, Any],
        findings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = [
            {
                "id": "objective",
                "type": "objective",
                "label": objective,
            }
        ]
        edges: list[dict[str, Any]] = []
        source_ids: dict[str, str] = {}
        for index, source in enumerate(sources, start=1):
            source_id = f"source_{index}"
            source_ids[source.title] = source_id
            nodes.append(
                {
                    "id": source_id,
                    "type": "source",
                    "provider": source.provider,
                    "title": source.title,
                    "url": source.url,
                    "grade": source.evidence_grade,
                }
            )
        if findings:
            for index, finding in enumerate(findings, start=1):
                finding_id = f"finding_{index}"
                nodes.append(
                    {
                        "id": finding_id,
                        "type": "finding",
                        "perspective": finding.get("perspective"),
                        "label": finding.get("finding"),
                        "confidence": finding.get("confidence"),
                        "support_count": finding.get("support_count"),
                    }
                )
                edges.append(
                    {
                        "from": "objective",
                        "to": finding_id,
                        "relation": "answered-by",
                    }
                )
                for supporting in finding.get("supporting_sources") or []:
                    source_id = source_ids.get(str(supporting.get("title") or ""))
                    if source_id is None:
                        continue
                    edges.append(
                        {
                            "from": finding_id,
                            "to": source_id,
                            "relation": "supported-by",
                        }
                    )
        else:
            for source_id in source_ids.values():
                edges.append(
                    {
                        "from": "objective",
                        "to": source_id,
                        "relation": "supported-by",
                    }
                )
        nodes.append(
            {
                "id": "retrieval",
                "type": "retrieval",
                "label": retrieval["stop_reason"],
                "coverage": retrieval["coverage"],
            }
        )
        edges.append(
            {
                "from": "objective",
                "to": "retrieval",
                "relation": "evaluated-by",
            }
        )
        nodes.append(
            {
                "id": "pc-context",
                "type": "pc-context",
                "available": pc_context_info.get("available"),
                "node_count": pc_context_info.get("node_count"),
            }
        )
        edges.append(
            {
                "from": "objective",
                "to": "pc-context",
                "relation": "grounded-by",
            }
        )
        return {
            "nodes": nodes,
            "edges": edges,
        }

    def _finding_ledger(
        self,
        query: str,
        sources: list[ResearchSource],
        plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        perspectives = (plan or {}).get("perspectives") or self._research_perspectives(
            query,
            query,
            "standard",
            self._looks_like_software_agent_query(query),
            self._looks_like_math_query(query),
        )
        findings: list[dict[str, Any]] = []
        used_titles: set[str] = set()
        for index, perspective in enumerate(perspectives, start=1):
            matched = self._matched_sources_for_perspective(sources, perspective)
            if not matched:
                continue
            lead = self._finding_lead_source(matched, perspective, used_titles)
            if lead is None:
                continue
            lead_key = self._normalize_title(lead.title)
            if lead_key:
                used_titles.add(lead_key)
            lead_identity = lead.url or lead_key
            ordered_support = [lead]
            ordered_support.extend(
                source
                for source in matched
                if (source.url or self._normalize_title(source.title)) != lead_identity
            )
            support_count = len(matched)
            provider_count = len({source.provider for source in matched})
            contradiction_count = sum(
                1 for source in matched if source.contradiction_risk >= 0.25
            )
            findings.append(
                {
                    "finding_id": f"finding_{index}",
                    "perspective": perspective["name"],
                    "goal": perspective.get("goal") or "",
                    "finding": self._finding_text(lead, perspective),
                    "support_count": support_count,
                    "provider_count": provider_count,
                    "contradiction_count": contradiction_count,
                    "confidence": self._finding_confidence(
                        matched,
                        contradiction_count,
                        provider_count,
                    ),
                    "supporting_sources": [
                        {
                            "title": source.title,
                            "url": source.url,
                            "provider": source.provider,
                            "evidence_grade": source.evidence_grade,
                        }
                        for source in ordered_support[:4]
                    ],
                }
            )
        findings.sort(
            key=lambda item: (
                self._finding_confidence_rank(str(item.get("confidence") or "")),
                int(item.get("support_count") or 0),
                int(item.get("provider_count") or 0),
            ),
            reverse=True,
        )
        return findings[:6]

    @classmethod
    def _finding_lead_source(
        cls,
        matched: list[ResearchSource],
        perspective: dict[str, Any],
        used_titles: set[str],
    ) -> ResearchSource | None:
        if not matched:
            return None
        unused = [
            source
            for source in matched
            if cls._normalize_title(source.title) not in used_titles
        ]
        pool = unused or matched
        return max(
            pool,
            key=lambda source: cls._perspective_lead_score(source, perspective),
        )

    @classmethod
    def _perspective_lead_score(
        cls,
        source: ResearchSource,
        perspective: dict[str, Any],
    ) -> tuple[int, int, int, int, float, int]:
        focus_terms = cls._perspective_focus_terms(perspective)
        title = source.title.lower()
        abstract = source.abstract.lower()
        title_hits = sum(1 for term in focus_terms if term in title)
        abstract_hits = sum(1 for term in focus_terms if term in abstract)
        sentence_hits = max(
            (
                sum(1 for term in focus_terms if term in sentence.lower())
                for sentence in _sentences(source.abstract)
            ),
            default=0,
        )
        return (
            title_hits,
            sentence_hits,
            abstract_hits,
            cls._evidence_grade_rank(source.evidence_grade),
            source.score,
            source.citation_count,
        )

    @classmethod
    def _perspective_focus_terms(
        cls,
        perspective: dict[str, Any],
    ) -> list[str]:
        perspective_name = str(perspective.get("name") or "").lower()
        focus_terms = [
            str(keyword).lower()
            for keyword in (perspective.get("keywords") or [])
            if str(keyword).strip()
        ]
        focus_terms.extend(
            token for token in re.split(r"[-\s]+", perspective_name) if len(token) >= 4
        )
        focus_terms.extend(
            term
            for term in cls._keywords(str(perspective.get("goal") or ""))
            if len(term) >= 4
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for term in focus_terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    @classmethod
    def _finding_text(
        cls,
        source: ResearchSource,
        perspective: dict[str, Any] | None = None,
    ) -> str:
        sentence = _sentences(source.abstract)
        if perspective is not None and sentence:
            focus_terms = cls._perspective_focus_terms(perspective)
            ranked_sentences = sorted(
                sentence,
                key=lambda item: (
                    sum(1 for term in focus_terms if term in item.lower()),
                    len(item),
                ),
                reverse=True,
            )
            best = ranked_sentences[0]
            if any(term in best.lower() for term in focus_terms):
                return best[:220]
        if sentence:
            return sentence[0][:220]
        return source.title[:220]

    @staticmethod
    def _finding_confidence(
        matched: list[ResearchSource],
        contradiction_count: int,
        provider_count: int,
    ) -> str:
        strong_count = sum(
            1
            for source in matched
            if source.evidence_grade in {"strong", "tool-observation"}
        )
        moderate_or_better = sum(
            1
            for source in matched
            if source.evidence_grade in {"strong", "moderate", "tool-observation"}
        )
        if moderate_or_better >= 3 and provider_count >= 2 and contradiction_count == 0:
            return "high"
        if moderate_or_better >= 2 and contradiction_count <= 1:
            return "medium"
        if strong_count >= 1 or moderate_or_better >= 1:
            return "low"
        return "needs-verification"

    @staticmethod
    def _evidence_grade_rank(evidence_grade: str) -> int:
        return {
            "strong": 4,
            "tool-observation": 3,
            "moderate": 2,
            "weak": 1,
            "ungraded": 0,
        }.get(evidence_grade, 0)

    @staticmethod
    def _finding_confidence_rank(confidence: str) -> int:
        return {
            "high": 4,
            "medium": 3,
            "low": 2,
            "needs-verification": 1,
        }.get(confidence, 0)

    def _brief_markdown(
        self,
        objective: str,
        query: str,
        summary: str,
        sources: list[ResearchSource],
        depth: str,
    ) -> str:
        lines = [
            "# Deep Research Brief",
            "",
            f"Objective: {objective}",
            "",
            f"Depth: {depth}",
            "",
            f"Query: {query}",
            "",
            "## Synthesis",
            "",
            summary,
            "",
            "## Evidence Quality",
            "",
            self._quality_summary(sources),
            "",
            "## Sources",
            "",
        ]
        for index, source in enumerate(sources, start=1):
            authors = ", ".join(source.authors[:3]) or "Unknown authors"
            year = source.year or "n.d."
            lines.extend(
                [
                    f"{index}. {source.title}",
                    f"   Provider: {source.provider}",
                    f"   Authors: {authors}",
                    f"   Year: {year}",
                    f"   Grade: {source.evidence_grade}",
                    (
                        "   Quality: "
                        f"relevance {source.relevance:.2f}, "
                        f"recency {source.recency:.2f}, "
                        f"citations {source.citation_strength:.2f}, "
                        f"contradiction risk {source.contradiction_risk:.2f}"
                    ),
                    f"   URL: <{source.url}>",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _summarize(
        self,
        objective: str,
        sources: list[ResearchSource],
        depth: str = "standard",
        plan: dict[str, Any] | None = None,
        query: str = "",
    ) -> str:
        if not sources:
            return (
                "Live research did not return sources from configured public "
                f"providers for: {objective}. Check network policy, API "
                "availability, or attach MCP research servers."
            )
        findings = self._finding_ledger(query or objective, sources, plan)
        perspective_coverage = self._perspective_coverage(
            sources,
            (plan or {}).get("perspectives") or [],
        )
        subquestion_count = len((plan or {}).get("subquestions", []))
        leading_findings = "; ".join(
            f"{finding['perspective']}: {finding['finding']}"
            for finding in findings[:3]
        ) or "; ".join(source.title for source in sources[:3])
        missing_perspectives = (
            ", ".join(perspective_coverage["missing"][:3])
            or "no major uncovered perspectives detected"
        )
        conflict_count = sum(
            int(finding.get("contradiction_count") or 0) for finding in findings
        )
        return (
            f"Collected {len(sources)} evidence-backed sources in {depth} "
            "mode for: "
            f"{objective}. "
            f"The research plan tracked {subquestion_count} subquestions. "
            "Perspective coverage reached "
            f"{perspective_coverage['count']}/{perspective_coverage['total']} "
            "planned angles. "
            f"Most-supported findings are: {leading_findings}. "
            f"Open gaps remain in: {missing_perspectives}. "
            f"Contradiction signals were observed in {conflict_count} finding clusters."
        )

    def _build_research_plan(
        self,
        objective: str,
        query: str,
        depth: str,
        pc_context_info: dict[str, Any],
    ) -> dict[str, Any]:
        software_mode = self._looks_like_software_agent_query(query)
        math_mode = self._looks_like_math_query(query)
        perspectives = self._research_perspectives(
            query,
            objective,
            depth,
            software_mode,
            math_mode,
        )
        if software_mode:
            subquestions = [
                "What problem scope and evaluation claims are being made?",
                "Which benchmark suites and task categories are reported?",
                "What failure modes, safety limits, and uncertainty statements exist?",
            ]
            comparative_axes = [
                "task success rate",
                "latency and token efficiency",
                "desktop/browser action reliability",
                "approval and safety model",
                "checkpoint and recovery design",
                "operator observability",
            ]
            evidence_requirements = [
                "peer-reviewed or benchmark-style evidence",
                "official project documentation and release notes",
                "reproducible implementation details",
                "explicitly stated limitations and risks",
            ]
            subquestions.extend(
                [
                    "How do OpenClaw/OpenCode/OpenHands differ in planner-worker topology?",
                    "How does accessibility-tree control compare with vision-only control?",
                    "Which systems expose approval, trust, and durable replay primitives?",
                ]
            )
        elif math_mode:
            subquestions = [
                "Which unconditional, almost-all, or density results are already established?",
                "What exact barrier blocks transfer from finite verification or almost-all control to a universal theorem?",
                "Which proof frameworks engage parity vectors, 2-adic dynamics, return maps, or well-quasi-order arguments?",
                "Which infinity-to-finite proof strategies succeeded in other hard theorems, and what structure made them work?",
            ]
            comparative_axes = [
                "strength of theorem",
                "assumptions and prerequisites",
                "proof mechanism",
                "finite-to-infinite transfer method",
                "explicit barrier statement",
                "constructive certificate leverage",
            ]
            evidence_requirements = [
                "peer-reviewed mathematics sources or primary technical references",
                "explicit theorem statements and hypotheses",
                "identified barriers, obstructions, or impossibility statements",
                "proof sketches or mechanisms tied to the queried dynamics",
            ]
        else:
            subquestions = [
                "What exact problem statement and accepted prior results define the topic?",
                "Which methods, reductions, or constructions recur across the literature?",
                "What explicit limitations, barriers, or open questions remain?",
            ]
            comparative_axes = [
                "strength of result",
                "assumptions and prerequisites",
                "method and mechanism",
                "scope of evidence",
                "stated limitations",
                "independent verification",
            ]
            evidence_requirements = [
                "primary or peer-reviewed evidence",
                "explicit methods and assumptions",
                "clear limitations or open questions",
                "independent corroboration when available",
            ]
        if pc_context_info.get("browser_context_detected"):
            subquestions.append(
                "How does live browser/app context from the local PC alter the evidence collection sequence?"
            )

        # Entity-focused short queries come FIRST so they are not cut by
        # max_query_variants when the list is later sliced.
        plan_queries = self._entity_queries(query, objective)
        for perspective in perspectives:
            plan_queries.extend(perspective.get("queries") or [])
        # Then add short keyword variants of the core query.
        plan_queries.extend(self._query_variants(query, depth))
        # Subquestions are turned into short keyword phrases, NOT appended
        # verbatim as full sentence strings (those confuse API search).
        for question in subquestions:
            kw = self._question_to_keywords(question, query)
            if kw:
                plan_queries.append(kw)

        deduped_queries: list[str] = []
        seen: set[str] = set()
        for candidate in plan_queries:
            normalized = self._normalize_title(candidate)
            if not normalized or normalized in seen:
                continue
            deduped_queries.append(candidate)
            seen.add(normalized)

        return {
            "core_question": objective[:300],
            "subquestions": subquestions,
            "comparative_axes": comparative_axes,
            "evidence_requirements": evidence_requirements,
            "perspectives": perspectives,
            "query_plan": deduped_queries,
        }

    @staticmethod
    def _research_perspectives(
        query: str,
        objective: str,
        depth: str,
        software_mode: bool,
        math_mode: bool,
    ) -> list[dict[str, Any]]:
        del objective
        if software_mode:
            return [
                {
                    "name": "overview",
                    "goal": "Establish the current system landscape and accepted framing.",
                    "keywords": ["survey", "overview", "state of the art", "landscape"],
                    "queries": [query, f"{query} overview", f"{query} landscape"],
                },
                {
                    "name": "architecture",
                    "goal": "Compare planner-worker topology, grounding, and execution design.",
                    "keywords": [
                        "architecture",
                        "planner",
                        "worker",
                        "grounding",
                        "control",
                    ],
                    "queries": [f"{query} architecture", f"{query} planner worker"],
                },
                {
                    "name": "evaluation",
                    "goal": "Find benchmark results, task suites, and verification evidence.",
                    "keywords": [
                        "benchmark",
                        "evaluation",
                        "success rate",
                        "verification",
                        "osworld",
                        "webarena",
                    ],
                    "queries": [f"{query} benchmark", f"{query} evaluation"],
                },
                {
                    "name": "safety",
                    "goal": "Find approval, safety, and trust-boundary evidence.",
                    "keywords": [
                        "safety",
                        "approval",
                        "trust",
                        "policy",
                        "risk",
                        "guardrail",
                    ],
                    "queries": [
                        f"{query} safety approval",
                        f"{query} risk limitations",
                    ],
                },
                {
                    "name": "failure-analysis",
                    "goal": "Surface failure modes, repairs, and uncertainty statements.",
                    "keywords": [
                        "failure",
                        "limitation",
                        "repair",
                        "recovery",
                        "uncertainty",
                    ],
                    "queries": [f"{query} failure analysis", f"{query} limitations"],
                },
                {
                    "name": "operations",
                    "goal": "Check deployment, observability, and operator workflow fit.",
                    "keywords": [
                        "deployment",
                        "observability",
                        "workflow",
                        "operator",
                        "production",
                    ],
                    "queries": [f"{query} deployment", f"{query} operator workflow"],
                },
            ]
        if math_mode:
            return [
                {
                    "name": "established-results",
                    "goal": "Anchor the search in accepted unconditional results.",
                    "keywords": [
                        "theorem",
                        "result",
                        "almost all",
                        "density",
                        "orbits",
                        "verification",
                    ],
                    "queries": [
                        query,
                        f"{query} established results",
                        f"{query} unconditional results",
                    ],
                },
                {
                    "name": "proof-barriers",
                    "goal": "Identify exact obstructions that block the missing bridge.",
                    "keywords": [
                        "barrier",
                        "obstruction",
                        "limitation",
                        "cannot",
                        "open problem",
                    ],
                    "queries": [f"{query} theorem barrier", f"{query} obstruction"],
                },
                {
                    "name": "proof-mechanisms",
                    "goal": "Collect mechanisms that could transfer local control into global structure.",
                    "keywords": [
                        "proof",
                        "mechanism",
                        "transfer",
                        "conjugacy",
                        "return map",
                        "2-adic",
                    ],
                    "queries": [f"{query} mechanism", f"{query} transfer method"],
                },
                {
                    "name": "computation",
                    "goal": "Track finite verification, computation, and certificate leverage.",
                    "keywords": [
                        "computation",
                        "verification",
                        "finite",
                        "certificate",
                        "algorithm",
                    ],
                    "queries": [f"{query} finite verification", f"{query} computation"],
                },
                {
                    "name": "analogies",
                    "goal": "Search for analogous infinity-to-finite proof strategies in other theorems.",
                    "keywords": [
                        "analogy",
                        "related theorem",
                        "transfer",
                        "well-quasi-order",
                        "compactness",
                    ],
                    "queries": [
                        f"{query} analogous theorems",
                        f"{query} infinity to finite",
                    ],
                },
            ]
        perspectives = [
            {
                "name": "overview",
                "goal": "Establish accepted framing, definitions, and prior work.",
                "keywords": [
                    "overview",
                    "survey",
                    "review",
                    "background",
                    "state of the art",
                ],
                "queries": [query, f"{query} overview", f"{query} survey"],
            },
            {
                "name": "mechanisms",
                "goal": "Collect recurring methods, models, or mechanisms.",
                "keywords": [
                    "method",
                    "approach",
                    "framework",
                    "mechanism",
                    "model",
                    "process",
                ],
                "queries": [f"{query} methods", f"{query} approach"],
            },
            {
                "name": "evidence",
                "goal": "Identify direct studies, experiments, benchmarks, or measurements.",
                "keywords": [
                    "evaluation",
                    "experiment",
                    "study",
                    "trial",
                    "measurement",
                    "result",
                ],
                "queries": [f"{query} evidence", f"{query} evaluation"],
            },
            {
                "name": "limitations",
                "goal": "Surface risks, critiques, and open problems.",
                "keywords": [
                    "limitation",
                    "risk",
                    "challenge",
                    "critique",
                    "uncertainty",
                    "failure",
                ],
                "queries": [f"{query} limitations", f"{query} criticism"],
            },
            {
                "name": "applications",
                "goal": "Check practical impact, adoption, or deployment relevance.",
                "keywords": [
                    "application",
                    "deployment",
                    "impact",
                    "adoption",
                    "workflow",
                    "use case",
                ],
                "queries": [f"{query} applications", f"{query} adoption"],
            },
            {
                "name": "alternatives",
                "goal": "Compare alternatives, baselines, or competing explanations.",
                "keywords": [
                    "alternative",
                    "comparison",
                    "baseline",
                    "trade-off",
                    "versus",
                    "competitor",
                ],
                "queries": [f"{query} comparison", f"{query} alternatives"],
            },
        ]
        if depth == "quick":
            return perspectives[:4]
        return perspectives

    @staticmethod
    def _entity_queries(query: str, objective: str) -> list[str]:
        lower = f"{query} {objective}".lower()
        if DeepResearchEngine._looks_like_math_query(lower):
            focus_terms = DeepResearchEngine._math_focus_terms(lower)
            focused: list[str] = []
            if any("collatz" in term.lower() for term in focus_terms):
                focused.extend(
                    [
                        "Collatz conjecture",
                        "Collatz almost all",
                        "Collatz finite verification",
                        "Collatz universal theorem",
                    ]
                )
            for term in focus_terms:
                if term == "Collatz conjecture":
                    continue
                if "collatz" in lower and "collatz" not in term.lower():
                    focused.append(f"Collatz {term}")
                else:
                    focused.append(term)
            return focused
        entities = []
        for name in ("openclaw", "opencode", "openhands", "agentos"):
            if name in lower:
                entities.append(name)
        # Entity-specific queries come first so they are not cut by
        # max_query_variants when the list is later sliced.
        focused: list[str] = []
        for entity in entities:
            focused.extend(
                [
                    f"{entity} architecture",
                    f"{entity} benchmark evaluation",
                    f"{entity} safety approval",
                    f"{entity} implementation",
                ]
            )
        if len(entities) > 1:
            joined = " vs ".join(item.title() for item in entities)
            focused.extend(
                [
                    f"{joined} comparison",
                    f"{joined} benchmark",
                ]
            )
        if "osworld" in lower or "webarena" in lower:
            focused.extend(
                [
                    "OSWorld benchmark",
                    "WebArena benchmark",
                    "OSWorld WebArena computer use evaluation",
                ]
            )
        if DeepResearchEngine._looks_like_software_agent_query(lower):
            focused.extend(
                [
                    "LLM agent benchmark evaluation",
                    "autonomous agent task planning execution",
                    "AI agent computer use evaluation",
                ]
            )
            if not entities and "research" in lower:
                focused.extend(
                    [
                        "AI research agent evaluation",
                        "automated literature review agent",
                        "agentic technical due diligence system",
                    ]
                )
        return focused

    @staticmethod
    def _question_to_keywords(question: str, query: str) -> str:
        """Convert a full subquestion sentence into a short keyword phrase
        suitable for API search (≤60 chars)."""
        # Drop stop words and common filler.
        stop_words = {
            "how",
            "do",
            "does",
            "does",
            "which",
            "what",
            "where",
            "when",
            "are",
            "is",
            "the",
            "a",
            "an",
            "in",
            "of",
            "to",
            "and",
            "or",
            "for",
            "with",
            "from",
            "that",
            "this",
            "their",
            "its",
            "differ",
            "compare",
            "comparisons",
            "vs",
            "system",
            "systems",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", question.lower())
        keywords = [w for w in words if w not in stop_words]
        phrase = " ".join(keywords[:6])
        return phrase[:60].strip() if len(phrase) >= 6 else ""

    @staticmethod
    def _clean_objective(objective: str) -> str:
        cleaned = re.sub(r"\s+", " ", objective).strip()
        prefixes = (
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
            "Merge worker outputs into a verified research brief for:",
        )
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned

    def _pc_context_summary(
        self,
        pc_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not pc_context:
            return {
                "available": False,
                "snapshot_path": None,
                "node_count": 0,
                "browser_context_detected": False,
                "top_labels": [],
            }

        snapshot_path = Path(str(pc_context.get("snapshot_path") or ""))
        top_labels: list[str] = []
        node_count = 0
        browser_context = False
        if snapshot_path.exists():
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                node_count = len(payload)
                for node in payload:
                    if not isinstance(node, dict):
                        continue
                    name = str(node.get("name") or "").strip()
                    if name:
                        top_labels.append(name)
                    if any(
                        marker in name.lower()
                        for marker in ("browser", "chrome", "edge", "firefox")
                    ):
                        browser_context = True
                    if len(top_labels) >= 8:
                        break
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        return {
            "available": snapshot_path.exists(),
            "snapshot_path": str(snapshot_path).replace("\\", "/"),
            "node_count": node_count,
            "browser_context_detected": browser_context,
            "top_labels": top_labels,
        }

    def _analysis_report_markdown(
        self,
        objective: str,
        summary: str,
        sources: list[ResearchSource],
        plan: dict[str, Any],
        pc_context_info: dict[str, Any],
    ) -> str:
        lines = [
            "# Deep Research Analysis Report",
            "",
            "## Objective",
            "",
            objective,
            "",
            "## Research Design",
            "",
            f"Core question: {plan['core_question']}",
            "",
            "Subquestions:",
        ]
        for item in plan["subquestions"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "Comparative axes:",
            ]
        )
        for axis in plan["comparative_axes"]:
            lines.append(f"- {axis}")

        lines.extend(
            [
                "",
                "Evidence requirements:",
            ]
        )
        for requirement in plan["evidence_requirements"]:
            lines.append(f"- {requirement}")

        lines.extend(
            [
                "",
                "## Live PC Context",
                "",
                (
                    f"Snapshot available: {pc_context_info['available']}; "
                    f"nodes: {pc_context_info['node_count']}; "
                    "browser context detected: "
                    f"{pc_context_info['browser_context_detected']}"
                ),
                "",
            ]
        )
        if pc_context_info["top_labels"]:
            lines.append("Observed UI labels:")
            for label in pc_context_info["top_labels"]:
                lines.append(f"- {label}")
            lines.append("")

        lines.extend(
            [
                "## Comparative Evidence Matrix",
                "",
                "| Source | Provider | Grade | Key claim |",
                "|---|---|---|---|",
            ]
        )
        for source in sources:
            claim = (source.abstract or source.title).replace("|", " ").strip()
            lines.append(
                "| "
                f"{source.title} | {source.provider} | {source.evidence_grade} | "
                f"{claim[:160]} |"
            )

        lines.extend(
            [
                "",
                "## Synthesis",
                "",
                summary,
                "",
                "## Limitations",
                "",
                "- Provider coverage may vary due to API availability and query drift.",
                "- Repository metadata is not equivalent to peer-reviewed evidence.",
                "- Local PC context was read-only unless explicit act approvals are granted.",
                "",
                "## Next Experiments",
                "",
                "- Run the same plan with controlled query slices per competitor (one system at a time).",
                "- Add explicit benchmark extraction for OSWorld/WebArena task families.",
                "- Add claim-level contradiction checks across providers before final ranking.",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _query_from_objective(objective: str) -> str:
        cleaned = re.sub(r"\s+", " ", objective).strip()
        prefixes = (
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
        )
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        distilled = DeepResearchEngine._query_core_terms(cleaned)
        return distilled[:240].strip() or cleaned[:240].strip() or objective[:240]

    @staticmethod
    def _split_depth(objective: str) -> tuple[str, str]:
        match = re.search(r"\[(quick|standard|multi-hour)\]\s*", objective)
        if match is None:
            return "standard", objective.strip()
        cleaned = f"{objective[: match.start()]}{objective[match.end() :]}"
        return match.group(1), cleaned.strip()

    @staticmethod
    def _settings_for_depth(depth: str) -> ResearchSettings:
        if depth == "quick":
            return ResearchSettings(
                depth="quick",
                max_sources=5,
                per_provider=4,
                max_query_variants=2,
            )
        if depth == "multi-hour":
            return ResearchSettings(
                depth="multi-hour",
                max_sources=48,
                per_provider=12,
                max_query_variants=12,
            )
        return ResearchSettings(
            depth="standard",
            max_sources=10,
            per_provider=6,
            max_query_variants=5,
        )

    @classmethod
    def _query_variants(cls, query: str, depth: str = "standard") -> list[str]:
        """Return short (≤80 char) keyword-phrase query variants suitable for
        API search calls.  Long essay-style objectives are first decomposed
        into focused terms before expansion."""
        core = cls._query_core_terms(query)
        lower = core.lower()
        software_mode = cls._looks_like_software_agent_query(query)
        math_mode = cls._looks_like_math_query(query)
        variants: list[str] = []
        # Always start with the distilled core phrase.
        if core:
            variants.append(core)
        if math_mode:
            math_focus = cls._math_focus_terms(query)
            variants.extend(
                [
                    f"{core} theorem",
                    f"{core} proof barrier",
                ]
            )
            if depth == "multi-hour":
                variants.extend(
                    [
                        f"{core} almost all",
                        f"{core} finite verification",
                        f"{core} 2-adic",
                        f"{core} density results",
                        "Collatz conjecture unconditional results",
                    ]
                )
            for term in math_focus:
                if term.lower() == "collatz conjecture":
                    continue
                variants.append(
                    f"Collatz {term}" if "collatz" in query.lower() else term
                )
        elif software_mode:
            if depth in {"standard", "multi-hour"} and core:
                variants.extend(
                    [
                        f"{core} evaluation",
                        f"{core} benchmark",
                    ]
                )
            if depth == "multi-hour" and core:
                variants.extend(
                    [
                        f"{core} survey",
                        f"{core} state of the art",
                        f"{core} limitations safety",
                        f"{core} implementation framework",
                        f"{core} systematic review",
                    ]
                )
        else:
            if depth in {"standard", "multi-hour"} and core:
                variants.append(f"{core} literature")
                variants.append(f"{core} prior work")
            if depth == "multi-hour" and core:
                variants.extend(
                    [
                        f"{core} survey",
                        f"{core} limitations",
                        f"{core} open problems",
                        f"{core} review",
                    ]
                )
        if any(t in lower for t in ("desktop", "computer use", "gui")):
            variants.append("GUI agent computer use evaluation")
            variants.append("LLM agent desktop control benchmark")
        if software_mode:
            variants.append("LLM autonomous agent WebArena OSWorld")
            variants.append("autonomous agent planning execution evaluation")
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = cls._normalize_title(variant)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(variant[:80])
        return deduped[:8]

    @staticmethod
    def _query_core_terms(query: str) -> str:
        """Distil a (potentially long) query/objective down to a short
        keyword phrase suitable as an API search string (≤60 chars)."""
        # Strip known preamble prefixes.
        prefixes = (
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
            "Merge worker outputs into a verified research brief for:",
            "Produce a research dossier covering",
            "Produce a rigorous",
        )
        cleaned = re.sub(r"\s+", " ", query).strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()
        cleaned = re.sub(
            (
                r"\busing\s+https?://[^\s<>()]+"
                r"(?:\s+and\s+https?://[^\s<>()]+)*"
                r"\s+as anchor sources\b"
            ),
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"https?://[^\s<>()]+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b[a-z0-9_./\\-]+\.(?:md|txt|json|ya?ml)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        boilerplate_patterns = (
            r"\bbased on\b",
            r"\bperform(?:ing)? deep research on\b",
            r"\bperform research on\b",
            r"\baccepted literature\b",
            r"\bplausible proof strategies\b",
            r"\bfocusing on\b",
            r"\bthe exact missing\b",
            r"\bas anchor sources\b",
            r"\bthen expand outward to\b",
            r"\bcorroborat\w*\b",
            r"^how to build (?:a|an|the)\b",
            r"^how to\b",
            r"^build (?:a|an|the)\b",
        )
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:-")
        if DeepResearchEngine._looks_like_math_query(cleaned):
            focus_terms = DeepResearchEngine._math_focus_terms(cleaned)
            if focus_terms:
                return " ".join(term.lower() for term in focus_terms[:4])[:70].strip()
        if DeepResearchEngine._looks_like_software_agent_query(cleaned):
            software_focus: list[str] = []
            lower = cleaned.lower()
            if "deep research" in lower:
                software_focus.append("deep research agent")
            elif "research" in lower and "agent" in lower:
                software_focus.append("research agent")
            if software_focus:
                return " ".join(software_focus[:2])[:70].strip()
        # If still long, take the first sentence or first 60 chars of words.
        if len(cleaned) > 70:
            first_sentence = re.split(r"[.!?;]", cleaned)[0].strip()
            if 10 <= len(first_sentence) <= 70:
                cleaned = first_sentence
            else:
                # Take the first 10 words.
                words = cleaned.split()
                cleaned = " ".join(words[:8])
        return cleaned[:70].strip()

    @staticmethod
    def _openalex_abstract(inverted_index: dict[str, list[int]]) -> str:
        if not inverted_index:
            return ""
        positions: dict[int, str] = {}
        for word, indexes in inverted_index.items():
            for index in indexes:
                positions[int(index)] = word
        return " ".join(positions[index] for index in sorted(positions))

    @staticmethod
    def _dedupe_sources(sources: list[ResearchSource]) -> list[ResearchSource]:
        by_identity: dict[str, ResearchSource] = {}
        deduped: list[ResearchSource] = []
        for source in sources:
            keys = DeepResearchEngine._source_identity_keys(source)
            existing = next(
                (by_identity[key] for key in keys if key in by_identity),
                None,
            )
            if existing is None:
                deduped.append(source)
                for key in keys:
                    by_identity[key] = source
                continue

            DeepResearchEngine._merge_source_records(existing, source)
            for key in keys:
                by_identity[key] = existing
            for key in DeepResearchEngine._source_identity_keys(existing):
                by_identity[key] = existing
        return deduped

    @staticmethod
    def _merge_source_records(existing: ResearchSource, source: ResearchSource) -> None:
        source_wins = source.score > existing.score
        if source_wins:
            existing.provider = source.provider
            existing.title = source.title
            existing.url = source.url
            existing.year = source.year
            existing.authors = list(source.authors)
        elif not existing.url and source.url:
            existing.url = source.url
        elif not existing.authors and source.authors:
            existing.authors = list(source.authors)
        elif existing.year is None and source.year is not None:
            existing.year = source.year

        if DeepResearchEngine._abstract_quality(
            source.abstract
        ) > DeepResearchEngine._abstract_quality(existing.abstract):
            existing.abstract = source.abstract
        existing.citation_count = max(existing.citation_count, source.citation_count)
        existing.score = max(existing.score, source.score)

    @staticmethod
    def _source_identity_keys(source: ResearchSource) -> list[str]:
        keys: list[str] = []
        if source.url:
            parsed = urllib.parse.urlsplit(source.url.strip())
            if parsed.scheme and parsed.netloc:
                normalized_url = urllib.parse.urlunsplit(
                    (
                        parsed.scheme.lower(),
                        parsed.netloc.lower(),
                        parsed.path.rstrip("/"),
                        "",
                        "",
                    )
                )
                keys.append(f"url:{normalized_url}")
        title_key = DeepResearchEngine._normalize_title(source.title)
        if title_key:
            keys.append(f"title:{title_key}")
        return keys

    @staticmethod
    def _abstract_quality(text: str) -> tuple[int, int]:
        cleaned = (text or "").strip()
        if not cleaned:
            return (0, 0)
        generic = cleaned.lower().startswith("generic web result for ")
        return (0 if generic else 1, len(cleaned))

    # Maximum proportion of final selected sources from any single provider.
    _MAX_PROVIDER_FRACTION = 0.5

    @classmethod
    def _rank_sources(
        cls,
        sources: list[ResearchSource],
        query: str,
    ) -> list[ResearchSource]:
        query_terms = set(cls._keywords(query))
        entity_terms = cls._entity_terms_from_query(query)
        if not query_terms:
            query_terms = set(re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", query))
        generic_terms = {
            "how",
            "agent",
            "agents",
            "build",
            "building",
            "deep",
            "general",
            "purpose",
            "model",
            "models",
            "system",
            "systems",
            "research",
            "using",
            "rigorous",
            "dossier",
            "comparative",
            "covering",
            "concrete",
            "adoption",
            "recommendation",
            "covering",
        }
        distinctive_terms = (query_terms | entity_terms) - generic_terms
        scored: list[tuple[float, ResearchSource]] = []
        for source in sources:
            scored.append(
                (
                    cls._score_source(source, distinctive_terms, entity_terms, query),
                    source,
                )
            )
        scored.sort(key=lambda t: t[0], reverse=True)
        # Exclude sources with zero relevance score — they failed all
        # relevance checks and should not appear in the final set.
        filtered = [source for score, source in scored if score > 0.0]
        return cls._enforce_provider_diversity(filtered)

    @classmethod
    def _score_source(
        cls,
        source: ResearchSource,
        distinctive_terms: set[str],
        entity_terms: set[str],
        query: str,
    ) -> float:
        haystack = f"{source.title} {source.abstract}".lower()
        entity_hits = sum(1 for t in entity_terms if t in haystack)
        entity_relevance = entity_hits / max(len(entity_terms), 1)
        distinctive_hits = sum(1 for t in distinctive_terms if t in haystack)
        term_relevance = distinctive_hits / max(len(distinctive_terms), 1)
        relevance = max(entity_relevance, term_relevance)
        recency = cls._recency_score(source.year)
        citation_strength = min(source.citation_count, 1000) / 1000
        credibility_score, credibility_penalty, quality_flags = cls._source_credibility(
            source,
            query,
        )
        contradiction = cls._contradiction_risk(source.abstract)
        # Mutate the source in place (existing pattern).
        source.relevance = relevance
        source.recency = recency
        source.citation_strength = citation_strength
        source.credibility_score = credibility_score
        source.contradiction_risk = contradiction
        source.quality_flags = quality_flags
        if source.provider == "gemini-flash":
            source.relevance = max(relevance, 0.65)
            source.credibility_score = max(credibility_score, 0.7)
            source.evidence_grade = "tool-observation"
            base = 80.0 + recency * 6.0 - contradiction * 4.0
            source.score = base
            return base
        if source.provider in {"openalex", "semantic-scholar", "crossref"}:
            # Scholarly sources: citation strength counts heavily; relevance
            # is softened so a partially relevant paper isn't ejected.
            if relevance <= 0 and not entity_hits:
                source.score = 0.0
                return 0.0
            # When there are multiple distinctive terms (≥3), require at least
            # 2 hits to avoid false positives (e.g. biomedical paper that
            # coincidentally contains one word from the query).
            if len(distinctive_terms) >= 3 and not entity_hits:
                hits = sum(1 for t in distinctive_terms if t in haystack)
                if hits < 2:
                    source.score = 0.0
                    return 0.0
            effective_relevance = max(relevance, 0.1 if entity_hits else 0.0)
            base = (
                effective_relevance * 54.0
                + citation_strength * 26.0
                + recency * 8.0
                + credibility_score * 18.0
                - contradiction * 6.0
                - credibility_penalty
            )
            if source.provider == "semantic-scholar":
                base += 4.0
            source.score = max(base, 0.0)
            source.evidence_grade = cls._evidence_grade(source)
            return source.score
        if source.provider == "github-repositories":
            benchmark_terms = {
                "osworld",
                "webarena",
                "benchmark",
                "evaluation",
                "computer use",
                "desktop agent",
                "desktop control",
                "browser agent",
                "browser automation",
            }
            benchmark_hits = sum(1 for t in benchmark_terms if t in haystack)
            if entity_terms and entity_hits == 0 and benchmark_hits == 0:
                source.score = 0.0
                return 0.0
            # Repos get a lower base ceiling than scholarly sources so they
            # don't crowd out papers; they still win when they are directly
            # about the queried entity.
            entity_boost = entity_relevance * 25.0
            base = (
                28.0
                + entity_boost
                + term_relevance * 12.0
                + benchmark_hits * 4.0
                + recency * 5.0
                + credibility_score * 4.0
                + min(citation_strength, 0.35) * 6.0
                - contradiction * 2.0
            )
            source.score = base
            source.evidence_grade = cls._evidence_grade(source)
            return base
        # software-reference and other providers
        source.evidence_grade = cls._evidence_grade(source)
        base = 20.0 + relevance * 20.0 + credibility_score * 6.0
        source.score = base
        return base

    @classmethod
    def _source_credibility(
        cls,
        source: ResearchSource,
        query: str,
    ) -> tuple[float, float, list[str]]:
        current_year = datetime.now(UTC).year
        title = source.title.lower()
        abstract = source.abstract.lower()
        url = source.url.lower()
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        credibility = 0.35
        penalty = 0.0
        flags: list[str] = []

        if source.provider in {"openalex", "semantic-scholar", "crossref"}:
            credibility += 0.15
        if any(
            host in url
            for host in (
                "doi.org/",
                "acm.org",
                "springer",
                "sciencedirect",
                "wiley",
                "nature.com",
            )
        ):
            credibility += 0.08
        if source.citation_count >= 50:
            credibility += 0.25
        elif source.citation_count >= 10:
            credibility += 0.16
        elif source.citation_count >= 3:
            credibility += 0.08
        if source.year is not None and source.year <= current_year - 3:
            credibility += 0.05
        if (
            source.year is not None
            and source.year >= current_year - 1
            and source.citation_count == 0
        ):
            penalty += 4.0
            flags.append("recent-uncited")

        if host.endswith(".gov") or host.endswith(".edu"):
            credibility += 0.18
        if (
            host.startswith("docs.")
            or host.startswith("developer.")
            or host
            in {
                "learn.microsoft.com",
                "developer.mozilla.org",
                "docs.python.org",
            }
            or host.endswith(".readthedocs.io")
        ):
            credibility += 0.1
        if "wikipedia.org" in host:
            credibility += 0.05
        if any(
            marker in url
            for marker in ("/docs", "/documentation", "/manual", "/reference")
        ):
            credibility += 0.06

        if any(
            host in url
            for host in (
                "arxiv.org",
                "zenodo.org",
                "figshare.com",
                "osf.io",
                "biorxiv.org",
                "medrxiv.org",
            )
        ):
            penalty += 3.0
            flags.append("preprint-or-repository")

        if re.search(r"\bv\d+(?:\.\d+){0,3}\b", title):
            penalty += 6.0
            flags.append("versioned-release")

        packaging_markers = (
            "proof package",
            "review manuscript",
            "lemma stock",
            "manifest",
            "demo",
            "aux",
            "workflow package",
        )
        if any(marker in title or marker in abstract for marker in packaging_markers):
            penalty += 8.0
            flags.append("package-like-source")

        if cls._looks_like_math_query(query):
            accepted_markers = (
                "almost all",
                "bounded values",
                "stopping time",
                "stopping times",
                "2-adic",
                "p-adic",
                "de bruijn",
                "verification limit",
                "finite verification",
                "density",
                "orbits",
                "conjugacy",
            )
            if any(
                marker in title or marker in abstract for marker in accepted_markers
            ):
                credibility += 0.12
            speculative_markers = (
                "is true for all positive integers",
                "complete resolution",
                "conquered by",
                "complete hand-verification",
                "proof-completion",
                "final gate",
                "active-route",
                "blur-knob",
                "settles the collatz conjecture",
                "we prove the collatz conjecture",
                "proof of the collatz conjecture",
                "complete proof of the collatz conjecture",
                "human-llm collaboration",
            )
            unsupported_proof_title_markers = (
                "proof of the collatz conjecture",
                "solution to the collatz conjecture",
                "solves the collatz conjecture",
                "collatz conjecture is true",
                "resolution of the collatz conjecture",
                "block format solves the collatz conjecture",
                "human-llm collaboration",
            )
            if (
                "collatz" in query.lower()
                and any(marker in title for marker in unsupported_proof_title_markers)
                and source.citation_count <= 2
            ):
                penalty += 22.0
                flags.append("unsupported-proof-title")
            if (
                source.citation_count == 0
                and source.year is not None
                and source.year >= current_year - 1
                and any(
                    marker in title or marker in abstract
                    for marker in speculative_markers
                )
            ):
                penalty += 26.0
                flags.append("speculative-proof-claim")

        credibility = max(0.0, min(credibility, 1.0))
        return credibility, penalty, flags

    @classmethod
    def _enforce_provider_diversity(
        cls,
        ranked: list[ResearchSource],
    ) -> list[ResearchSource]:
        """Prevent any single provider from holding more than
        _MAX_PROVIDER_FRACTION of the final set."""
        total = len(ranked)
        if total == 0:
            return ranked
        cap = max(3, int(total * cls._MAX_PROVIDER_FRACTION))
        counts: dict[str, int] = {}
        result: list[ResearchSource] = []
        overflow: list[ResearchSource] = []
        for source in ranked:
            count = counts.get(source.provider, 0)
            if count < cap:
                counts[source.provider] = count + 1
                result.append(source)
            else:
                overflow.append(source)
        result.extend(overflow)
        return result

    @staticmethod
    def _select_balanced_top(
        ranked: list[ResearchSource],
        max_sources: int,
        query: str,
    ) -> list[ResearchSource]:
        if not ranked:
            return []
        capped = ranked[: max(max_sources * 3, max_sources)]
        by_provider: dict[str, list[ResearchSource]] = {}
        for source in capped:
            by_provider.setdefault(source.provider, []).append(source)

        selected: list[ResearchSource] = []

        def append_preferred(
            provider: str,
            predicate: Any | None = None,
        ) -> bool:
            provider_sources = by_provider.get(provider) or []
            for index, source in enumerate(provider_sources):
                if predicate is not None and not predicate(source):
                    continue
                selected.append(source)
                del provider_sources[index]
                return True
            return False

        # Preserve at least one explicit user/context anchor when it remains
        # relevant enough to rank, otherwise explicit sources disappear behind
        # generic search hits.
        append_preferred(
            "seed-url",
            lambda source: source.relevance >= 0.2 or source.credibility_score >= 0.35,
        )

        # Prefer at least one scholarly source when available.
        scholarly_order = ("openalex", "semantic-scholar", "crossref")
        for provider in scholarly_order:
            if append_preferred(provider):
                break

        # Prefer at least one code/provider source for software comparisons.
        if DeepResearchEngine._looks_like_software_agent_query(query):
            append_preferred("github-repositories")

        # Fill remaining slots by global ranking while avoiding duplicates.
        selected_urls = {s.url for s in selected}
        for source in capped:
            if len(selected) >= max_sources:
                break
            if source.url in selected_urls:
                continue
            selected.append(source)
            selected_urls.add(source.url)
        return selected[:max_sources]

    @staticmethod
    def _entity_terms_from_query(query: str) -> set[str]:
        lower = query.lower()
        software_mode = DeepResearchEngine._looks_like_software_agent_query(query)
        entities = {
            "openclaw",
            "opencode",
            "openhands",
            "agentos",
            "osworld",
            "webarena",
            "webagent",
            "computeruse",
            "windows",
        }
        matched = {e for e in entities if e in lower}
        if "research" in lower and "agent" in lower:
            matched.add("research agent")
        if "deep research" in lower:
            matched.add("deep research")
        if not software_mode and "literature" in lower and "review" in lower:
            matched.add("literature review")
        if not software_mode and "technical" in lower and "diligence" in lower:
            matched.add("technical due diligence")
        if not software_mode and "market" in lower and "intelligence" in lower:
            matched.add("market intelligence")
        if not software_mode and "safety" in lower and "critical" in lower:
            matched.add("safety critical")
        return matched

    @staticmethod
    def _quality_summary(sources: list[ResearchSource]) -> str:
        if not sources:
            return "No evidence was available to grade."
        grades = Counter(source.evidence_grade for source in sources)
        strongest = ", ".join(
            f"{grade}: {count}" for grade, count in sorted(grades.items())
        )
        average_relevance = sum(source.relevance for source in sources) / len(sources)
        average_credibility = sum(source.credibility_score for source in sources) / len(
            sources
        )
        risk = max(
            (source.contradiction_risk for source in sources),
            default=0.0,
        )
        return (
            f"Evidence grades: {strongest}. Average relevance is "
            f"{average_relevance:.2f}; average credibility is "
            f"{average_credibility:.2f}; maximum contradiction risk is "
            f"{risk:.2f}."
        )

    @staticmethod
    def _claim_trace(
        objective: str,
        summary: str,
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unique_sources = {
            source["url"]
            for finding in findings
            for source in (finding.get("supporting_sources") or [])
            if source.get("url")
        }
        return {
            "objective": objective,
            "summary": summary,
            "claims": [
                {
                    "claim_id": finding.get("finding_id") or f"claim_{index}",
                    "claim": finding.get("finding") or "",
                    "perspective": finding.get("perspective") or "",
                    "confidence": finding.get("confidence") or "needs-verification",
                    "support_count": finding.get("support_count") or 0,
                    "provider_count": finding.get("provider_count") or 0,
                    "contradiction_count": finding.get("contradiction_count") or 0,
                    "supporting_sources": finding.get("supporting_sources") or [],
                }
                for index, finding in enumerate(findings, start=1)
            ],
            "source_count": len(unique_sources),
            "minimum_confidence": min(
                (
                    DeepResearchEngine._finding_confidence_rank(
                        str(finding.get("confidence") or "")
                    )
                    for finding in findings
                ),
                default=0,
            ),
        }

    @staticmethod
    def _recency_score(year: int | None) -> float:
        if year is None:
            return 0.2
        current_year = datetime.now(UTC).year
        age = max(current_year - int(year), 0)
        return max(0.1, 1.0 - min(age, 20) / 20)

    @staticmethod
    def _contradiction_risk(text: str) -> float:
        lower = text.lower()
        markers = (
            "conflicting",
            "contradict",
            "inconsistent",
            "mixed evidence",
            "debated",
            "controvers",
        )
        matches = sum(1 for marker in markers if marker in lower)
        return min(matches / 4, 1.0)

    @staticmethod
    def _evidence_grade(source: ResearchSource) -> str:
        if source.provider == "gemini-flash":
            return "tool-observation"
        if (
            source.credibility_score < 0.25
            or "speculative-proof-claim" in source.quality_flags
            or "unsupported-proof-title" in source.quality_flags
        ):
            return "weak"
        if (
            source.relevance >= 0.7
            and source.citation_strength >= 0.2
            and source.credibility_score >= 0.55
        ):
            return "strong"
        if source.relevance >= 0.45 and source.credibility_score >= 0.35:
            return "moderate"
        if (
            source.relevance >= 0.25
            and source.credibility_score >= 0.72
            and source.citation_strength >= 0.02
        ):
            return "moderate"
        return "weak"

    @classmethod
    def _is_low_signal_query_variant(cls, variant: str, query: str = "") -> bool:
        lower = variant.lower().strip()
        if not lower:
            return True
        if "http://" in lower or "https://" in lower or "www." in lower:
            return True
        if cls._looks_like_math_query(query):
            if any(
                marker in lower
                for marker in (
                    "benchmark",
                    "evaluation",
                    "failure analysis",
                    "repository architecture",
                    "implementation",
                )
            ):
                return True
        noise_tokens = {
            "https",
            "http",
            "www",
            "display",
            "record",
            "records",
            "download",
            "license",
            "copyright",
            "manifest",
            "mobile",
            "padding",
            "sha",
            "share",
            "theme",
            "toggle",
            "blur",
            "knob",
            "aux",
            "demo",
        }
        words = re.findall(r"\b[a-z0-9.-]+\b", lower)
        if len(words) < 2:
            return True
        noise_hits = sum(1 for word in words if word in noise_tokens)
        return noise_hits >= 2 and noise_hits >= max(2, len(words) // 2)

    @staticmethod
    def _looks_like_software_agent_query(query: str) -> bool:
        lower = query.lower()
        markers = (
            "agentos",
            "analyst system",
            "browser research",
            "computer use",
            "desktop agent",
            "deep research",
            "github",
            "literature review agent",
            "local pc",
            "market intelligence",
            "openclaw",
            "opencode",
            "openhands",
            "orchestrator",
            "pc agent",
            "research agent",
            "safety-critical",
            "software agent",
            "technical due diligence",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    def _looks_like_math_query(query: str) -> bool:
        lower = query.lower()
        markers = (
            "collatz",
            "conjecture",
            "theorem",
            "lemma",
            "proof",
            "2-adic",
            "number theory",
            "density",
            "ostrowski",
            "residue",
            "well-quasi-order",
            "wqo",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    def _math_focus_terms(query: str) -> list[str]:
        lower = query.lower()
        focus_map = (
            ("collatz conjecture", "Collatz conjecture"),
            ("collatz bridge", "Collatz bridge"),
            ("collatz", "Collatz conjecture"),
            ("almost-all", "almost all"),
            ("almost all", "almost all"),
            ("finite verification", "finite verification"),
            ("universal theorem", "universal theorem"),
            ("lemma ub", "Lemma UB"),
            ("2-adic conjugacy", "2-adic conjugacy"),
            ("2-adic", "2-adic"),
            ("deterministic pointwise transfer", "pointwise transfer"),
            ("mechanical carry forcing", "mechanical carry forcing"),
            ("peel-chain no-crossing", "peel-chain no-crossing"),
            ("ostrowski return-block cocycles", "Ostrowski return-block cocycles"),
            ("ostrowski", "Ostrowski"),
            ("critical-density boundary", "critical density"),
            ("critical density", "critical density"),
            ("lopez-stoll", "Lopez-Stoll"),
            ("infinity-to-finite", "infinity-to-finite transfer"),
            ("hard theorems", "hard theorems"),
        )
        focused: list[str] = []
        seen: set[str] = set()
        for marker, label in focus_map:
            if marker not in lower or label in seen:
                continue
            focused.append(label)
            seen.add(label)
        return focused

    @staticmethod
    def _software_reference_sources(query: str) -> list[ResearchSource]:
        year = datetime.now(UTC).year
        encoded_query = urllib.parse.quote_plus(query)
        return [
            ResearchSource(
                provider="software-reference",
                title=f"GitHub repository search for {query[:80]}",
                url=(f"https://github.com/search?type=repositories&q={encoded_query}"),
                year=year,
                authors=["GitHub"],
                abstract=(
                    "Live software-agent research should inspect public "
                    "repository search results, project READMEs, issues, "
                    "release notes, and docs for exact capabilities."
                ),
                citation_count=0,
                score=18.0,
            ),
        ]

    @staticmethod
    def _normalize_title(value: str) -> str:
        return re.sub(r"\W+", "", value.lower())

    @staticmethod
    def _keywords(text: str) -> list[str]:
        stopwords = {
            "about",
            "across",
            "after",
            "also",
            "and",
            "analysis",
            "because",
            "between",
            "could",
            "for",
            "from",
            "have",
            "into",
            "research",
            "that",
            "the",
            "their",
            "these",
            "this",
            "through",
            "using",
            "were",
            "with",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        counts = Counter(word for word in words if word not in stopwords)
        return [word for word, _count in counts.most_common(12)]

    @staticmethod
    def _confidence(sources: list[ResearchSource]) -> float:
        if not sources:
            return 0.35
        provider_count = len({source.provider for source in sources})
        citation_total = sum(source.citation_count for source in sources)
        citation_bonus = min(citation_total, 500)
        confidence = 0.55 + min(len(sources), 10) * 0.025
        confidence += provider_count * 0.05
        confidence += citation_bonus / 5000
        return min(confidence, 0.92)


def _sentences(text: str) -> list[str]:
    sentences = [
        item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()
    ]
    return sentences[:6] or ([text.strip()] if text.strip() else [])


def _year_from_timestamp(value: str) -> int | None:
    match = re.match(r"(\d{4})-", value)
    if match is None:
        return None
    return int(match.group(1))


def _gemini_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = str(part.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()
