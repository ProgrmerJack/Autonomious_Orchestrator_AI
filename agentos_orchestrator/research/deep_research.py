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


def _extract_ticker_candidates(text: str) -> list[str]:
    blocked = {
        "THE",
        "AND",
        "FOR",
        "NOW",
        "BEST",
        "TOP",
        "LIST",
        "AI",
        "API",
        "ETF",
        "UTF",
        "FFF",
        "STOCK",
        "STOCKS",
        "LIVE",
        "ANY",
        "LIST",
        "BEST",
        "RIGHT",
        "NOW",
        "BULLISH",
        "USA",
        "US",
        "GDP",
    }
    tokens = re.findall(r"\b[A-Z]{1,5}\b", text or "")
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in blocked:
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result[:8]


def _sanitize_evidence_claim_text(title: str, abstract: str, url: str) -> str:
    raw = (abstract or "").strip()
    lower = raw.lower()
    if (
        not raw
        or lower.startswith("generic web result")
        or "snippet unavailable" in lower
    ):
        tickers = _extract_ticker_candidates(f"{title} {abstract}")
        if tickers:
            return f"Ticker candidates mentioned by source: {', '.join(tickers)}."
        return (title or url)[:240]

    cleaned = re.sub(r"\s+", " ", raw)
    sentence = re.split(r"[.!?]", cleaned, maxsplit=1)[0].strip()
    if sentence and len(sentence) >= 30:
        cleaned = sentence
    tickers = _extract_ticker_candidates(f"{title} {cleaned}")
    if tickers and all(t not in cleaned for t in tickers):
        cleaned = f"{cleaned}. Tickers referenced: {', '.join(tickers)}"
    return cleaned[:500] or (title or url)[:240]


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
            "claim": _sanitize_evidence_claim_text(self.title, self.abstract, self.url),
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
        self._durable_note_urls: set[str] = set()
        self._durable_note_passes: set[int] = set()
        self._dotenv_loaded = False
        self._semantic_gate_cache: dict[str, bool] = {}

    def run(
        self,
        objective: str,
        run_id: str,
        pc_context: dict[str, Any] | None = None,
        planning_context: dict[str, Any] | None = None,
        evidence_targets: dict[str, Any] | None = None,
    ) -> ResearchBrief:
        self.provider_diagnostics = []
        self._durable_note_urls = set()
        self._durable_note_passes = set()
        self._semantic_gate_cache = {}
        self._load_env_from_dotenv()
        self._record_provider_preflight()
        depth, cleaned_objective = self._split_depth(objective)
        research_objective = self._clean_objective(cleaned_objective)
        settings = self._settings_for_depth(depth)
        query = self._query_from_objective(research_objective)
        self._write_progress_checkpoint(
            run_id,
            {
                "run_id": run_id,
                "depth": settings.depth,
                "stage": "research-initialized",
                "pass_index": 0,
                "stop_reason": None,
                "recent_queries": [],
                "passes": [],
                "last_updated": datetime.now(UTC).isoformat(),
            },
        )
        current_web_mode = self._looks_like_current_evidence_query(
            research_objective
        ) and not self._looks_like_academic_query(research_objective)
        if current_web_mode:
            settings = self._settings_for_current_web(settings)
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
        if current_web_mode:
            merged_targets = self._current_web_target_overrides(
                merged_targets,
                settings.depth,
            )
        self._write_progress_checkpoint(
            run_id,
            {
                "run_id": run_id,
                "depth": settings.depth,
                "stage": "retrieval-starting",
                "pass_index": 0,
                "stop_reason": None,
                "recent_queries": list(plan.get("query_plan") or [])[:3],
                "passes": [],
                "last_updated": datetime.now(UTC).isoformat(),
            },
        )
        retrieval = self._iterative_retrieval(
            query=query,
            settings=settings,
            plan=plan,
            targets=merged_targets,
            run_id=run_id,
        )
        selected = retrieval["selected"]
        query_variants = retrieval["query_variants"]
        coverage = retrieval["coverage"]
        durable_notes = self._load_durable_notes(run_id)
        synthesis_mode = self._resolve_final_synthesis_mode(
            settings.depth,
            durable_notes,
        )
        summary = self._summarize(
            research_objective,
            selected,
            settings.depth,
            plan,
            query,
            durable_notes,
            synthesis_mode,
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
            run_id,
            synthesis_mode,
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

    def _write_progress_checkpoint(self, run_id: str, payload: dict[str, Any]) -> None:
        if not run_id:
            return
        try:
            progress_dir = self.workspace_root / "runs" / run_id / "research"
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_path = progress_dir / "progress.json"
            progress_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _write_retrieval_heartbeat(
        self,
        run_id: str,
        depth: str,
        pass_index: int,
        search_query: str,
        query_index: int,
        query_total: int,
        retrieval_passes: list[dict[str, Any]],
        started_at: float,
    ) -> None:
        if not run_id:
            return
        self._write_progress_checkpoint(
            run_id,
            {
                "run_id": run_id,
                "depth": depth,
                "stage": "retrieval-query-active",
                "pass_index": pass_index + 1,
                "query_index": query_index,
                "query_total": query_total,
                "active_query": search_query[:160],
                "stop_reason": None,
                "recent_queries": [search_query[:160]],
                "passes": retrieval_passes,
                "elapsed_seconds": round(time.monotonic() - started_at, 1),
                "last_updated": datetime.now(UTC).isoformat(),
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
            if cls._is_search_result_url(cleaned):
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

    @staticmethod
    def _is_search_result_url(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower().lstrip("www.")
        path = parsed.path.lower()
        query = urllib.parse.parse_qs(parsed.query)
        search_routes = {
            "duckduckgo.com": bool(query.get("q")),
            "html.duckduckgo.com": bool(query.get("q")),
            "google.com": path.startswith("/search"),
            "bing.com": path.startswith("/search"),
            "github.com": path == "/search",
        }
        return search_routes.get(host, False)

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
        run_id: str = "",
    ) -> dict[str, Any]:
        durable_report_path = self._initialize_durable_report(
            run_id,
            settings.depth,
            plan.get("core_question") or query,
        )
        all_sources: list[ResearchSource] = self._seed_sources(
            plan.get("source_seeds") or []
        )
        # Seed AI-reasoned authoritative domains as sources so they are
        # considered from the very first pass.
        ai_domains = plan.get("ai_authoritative_domains") or []
        if ai_domains:
            all_sources.extend(
                self._ai_domain_seed_sources(
                    plan.get("core_question") or query,
                    ai_domains,
                )
            )
        all_variants = self._sanitize_query_variants(
            plan["query_plan"][: settings.max_query_variants * 4],
            query,
        )
        # Artificial runtime floors removed — depth is driven by
        # information exhaustion (enrichment, citation chasing, gap analysis)
        # not wall-clock timers.
        min_runtime_seconds = 0
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
        # No runtime-based pass floor — passes are information-driven.
        max_passes = max(max_passes, min_depth_passes)
        max_passes = max(1, min(max_passes, 240))
        started_at = time.monotonic()
        retrieval_passes: list[dict[str, Any]] = []
        previous_titles: set[str] = set()
        selected_domain_counts: dict[str, int] = {}
        low_novelty_streak = 0
        starvation_streak = 0
        stop_reason = "max_passes_reached"
        passing_snapshot: dict[str, Any] | None = None
        # Classify the query once to gate providers for every pass.
        # Classify the query once to gate providers for every pass.
        # Use the full original question from the plan when available so that
        # stop-word stripping in _query_core_terms does not erase cues like
        # "as of now" → "as now", which would cause _classify_query to fall
        # back to the default scholarly stack and include OpenAlex for live
        # market / current-evidence queries.
        classify_input = plan.get("core_question") or query
        allowed_providers = self._classify_query(classify_input)

        for pass_index in range(max_passes):
            pass_variants = self._pass_variants(
                all_variants,
                pass_index,
                settings.max_query_variants,
            )
            if not pass_variants:
                if all_variants:
                    pass_variants = all_variants[: settings.max_query_variants]
                else:
                    stop_reason = "no_query_variants"
                    break
            pass_sources: list[ResearchSource] = []
            for query_index, search_query in enumerate(pass_variants, start=1):
                self._write_retrieval_heartbeat(
                    run_id,
                    settings.depth,
                    pass_index,
                    search_query,
                    query_index,
                    len(pass_variants),
                    retrieval_passes,
                    started_at,
                )
                pass_sources.extend(
                    self._search_query_across_providers(
                        search_query,
                        allowed_providers,
                        settings.per_provider,
                    )
                )

            if pass_index == 0 and self._looks_like_software_agent_query(query):
                pass_sources.extend(self._software_reference_sources(query))
            # Call gemini observation on pass 0 and, for multi-hour depth, on
            # every 4th subsequent pass (passes 4, 8, 12…).  This ensures
            # gemini-flash remains a live provider throughout long runs,
            # preventing structural provider monoculture after pass 0.
            _is_periodic_gemini_pass = (
                settings.depth == "multi-hour"
                and pass_index > 0
                and pass_index % 4 == 0
            )
            if pass_index == 0 or _is_periodic_gemini_pass:
                pass_sources.extend(
                    self._search_gemini_observation(query, settings.depth)
                )

            all_sources.extend(pass_sources)
            ranked = self._rank_sources(self._dedupe_sources(all_sources), query)
            reranked = self._rerank_for_domain_diversity(
                ranked,
                selected_domain_counts,
                low_novelty_streak,
                pass_index,
            )
            selected = self._select_balanced_top(
                reranked,
                settings.max_sources,
                query,
            )
            self._accumulate_domain_counts(selected_domain_counts, selected)

            # --- ENRICHMENT: deep-read high-value sources on EVERY pass ---
            # A scientist reads full papers, not just abstracts.  We enrich
            # top sources every pass so the engine accumulates genuine
            # understanding through real I/O, not timers.
            if settings.depth in {"standard", "multi-hour"}:
                enrich_count = (
                    min(12, settings.max_sources)
                    if pass_index == 0
                    else min(8, settings.max_sources)
                )
                content_queries = self._enrich_top_sources(
                    selected[:enrich_count],
                    query,
                )
                self._append_durable_claim_notes(
                    durable_report_path,
                    pass_index + 1,
                    selected[:enrich_count],
                    query,
                )
                for cq in content_queries:
                    if cq and cq not in all_variants:
                        all_variants.append(cq)
                # Flush transient enrichment text from local loop context after
                # durable write so synthesis does not rely on large in-memory blobs.
                del content_queries

            # --- CITATION CHASING: follow references every pass for multi-hour ---
            # Real scientists follow footnotes and build literature trees.
            if settings.depth == "multi-hour":
                chase_depth = 2 if pass_index <= 2 else 1
                chase_count = min(10, len(selected))
                cited = self._citation_chase(
                    selected[:chase_count], query, citation_depth=chase_depth
                )
                if cited:
                    all_sources.extend(cited)
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
            coverage = self._coverage_metrics(selected, novelty_rate, plan, query)
            domain_count = len(
                {
                    self._source_domain(source.url)
                    for source in selected
                    if self._source_domain(source.url)
                }
            )
            min_provider_target = max(
                1,
                int(coverage_targets.get("min_provider_count") or 1),
            )
            min_domain_target = max(2, min_provider_target)
            unreachable_count = sum(
                1
                for source in selected
                if "unreachable-paywalled" in (source.quality_flags or [])
            )
            source_count = max(len(selected), 1)
            source_starved = (
                int(coverage.get("provider_count") or 0) < min_provider_target
                or domain_count < min_domain_target
                or unreachable_count >= max(2, source_count // 3)
            )
            if source_starved and pass_index > 0:
                starvation_streak += 1
            else:
                starvation_streak = 0
            if settings.depth in {"standard", "multi-hour"}:
                expanded = self._expand_provider_mix_for_diversity(
                    query,
                    allowed_providers,
                    coverage,
                )
                if expanded:
                    self._record_provider_diagnostic(
                        "provider-mix",
                        "expanded",
                        "expanded provider set after low-diversity pass",
                    )
                    allowed_providers = expanded
            pass_record = {
                "pass_index": pass_index + 1,
                "query_variants": pass_variants,
                "selected_count": len(selected),
                "provider_count": coverage["provider_count"],
                "domain_count": domain_count,
                "novelty_rate": round(coverage["novelty_rate"], 3),
                "on_topic_ratio": round(
                    float(coverage.get("on_topic_ratio") or 0.0), 3
                ),
                "weak_ratio": round(float(coverage.get("weak_ratio") or 0.0), 3),
                "max_contradiction_risk": round(
                    coverage["max_contradiction_risk"],
                    3,
                ),
                "source_starved": source_starved,
                "starvation_streak": starvation_streak,
                "unreachable_count": unreachable_count,
                "elapsed_seconds": round(time.monotonic() - started_at, 1),
            }
            retrieval_passes.append(pass_record)
            if starvation_streak >= 2:
                pivot_queries = self._domain_diversification_queries(
                    plan.get("core_question") or query,
                    selected,
                    pass_index,
                    coverage,
                )
                if pivot_queries:
                    self._record_provider_diagnostic(
                        "starvation-pivot",
                        "triggered",
                        (
                            f"pass {pass_index + 1}: provider_count="
                            f"{coverage.get('provider_count')} domain_count={domain_count}"
                        ),
                    )
                    all_variants.extend(
                        self._sanitize_query_variants(
                            pivot_queries,
                            query,
                        )
                    )
                expanded = self._expand_provider_mix_for_diversity(
                    query,
                    allowed_providers,
                    {
                        "provider_count": 0,
                    },
                )
                if expanded:
                    allowed_providers = expanded
                starvation_streak = 0
            # Write live progress so external monitors can track the run.
            if run_id:
                try:
                    progress_dir = self.workspace_root / "runs" / run_id / "research"
                    progress_dir.mkdir(parents=True, exist_ok=True)
                    progress_path = progress_dir / "progress.json"
                    progress_history: list[dict[str, Any]] = []
                    if progress_path.exists():
                        try:
                            current_progress = json.loads(
                                progress_path.read_text(encoding="utf-8")
                            )
                            loaded_history = current_progress.get("passes")
                            if isinstance(loaded_history, list):
                                progress_history = [
                                    item
                                    for item in loaded_history
                                    if isinstance(item, dict)
                                ]
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            progress_history = []
                    progress_history.append(dict(pass_record))
                    progress_path.write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "depth": settings.depth,
                                "pass_index": pass_index + 1,
                                "max_passes": max_passes,
                                "sources_found": len(selected),
                                "elapsed_seconds": pass_record["elapsed_seconds"],
                                "novelty_rate": pass_record["novelty_rate"],
                                "domain_count": pass_record["domain_count"],
                                "stage": "retrieval-active",
                                "stop_reason": None,
                                "recent_queries": pass_variants[:3],
                                "passes": progress_history,
                                "last_updated": datetime.now(UTC).isoformat(),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
            previous_titles = current_titles
            # No artificial budget — depth is information-driven.
            budget_met = True
            depth_met = (pass_index + 1) >= min_depth_passes
            novelty_threshold = float(targets.get("min_novelty_rate") or 0.0)
            if coverage["novelty_rate"] < novelty_threshold and pass_index > 0:
                low_novelty_streak += 1
            else:
                low_novelty_streak = 0
            _raw_weak = coverage.get("weak_ratio")
            quality_gate_passed = (
                float(coverage.get("on_topic_ratio") or 0.0) >= 0.7
                and float(1.0 if _raw_weak is None else _raw_weak) <= 0.45
            )
            if self._meets_targets(coverage, coverage_targets) and quality_gate_passed:
                passing_snapshot = {
                    "selected": list(selected),
                    "coverage": dict(coverage),
                }
                if not depth_met:
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
                    query,
                )
                return {
                    "selected": selected,
                    "coverage": coverage,
                    "passes": retrieval_passes,
                    "stop_reason": stop_reason,
                    "query_variants": all_variants,
                }
            if coverage["novelty_rate"] < novelty_threshold and pass_index > 0:
                if not depth_met or low_novelty_streak < max_low_novelty_streak:
                    continue
                stop_reason = "novelty_below_threshold"
                break
            if coverage["max_contradiction_risk"] > float(
                targets.get("max_contradiction_risk") or 1.0
            ):
                if not depth_met:
                    continue
                stop_reason = "contradiction_above_threshold"
                break

            all_variants.extend(
                self._sanitize_query_variants(
                    self._refinement_variants(
                        query,
                        selected,
                        settings.depth,
                        pass_index,
                        plan,
                    ),
                    query,
                )
            )

            # --------------------------------------------------------
            # AI GAP ANALYSIS: run EVERY pass to reason about what's
            # missing and generate targeted follow-up queries based on
            # the actual evidence found — not pre-baked templates.
            # A real scientist re-evaluates gaps after each evidence round.
            # --------------------------------------------------------
            if pass_index >= 1:
                gap_queries = self._ai_evidence_gap_analysis(
                    plan.get("core_question") or query,
                    selected,
                    pass_index,
                )
                all_variants.extend(
                    self._sanitize_query_variants(
                        gap_queries,
                        query,
                    )
                )
            all_variants = self._sanitize_query_variants(all_variants, query)

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
                query,
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
        coverage = self._coverage_metrics(selected, float(novelty_rate), plan, query)
        return {
            "selected": selected,
            "coverage": coverage,
            "passes": retrieval_passes,
            "stop_reason": stop_reason,
            "query_variants": all_variants[: settings.max_query_variants],
        }

    @staticmethod
    def _source_domain(url: str) -> str:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return host

    @classmethod
    def _accumulate_domain_counts(
        cls,
        domain_counts: dict[str, int],
        selected: list[ResearchSource],
    ) -> None:
        for source in selected:
            domain = cls._source_domain(source.url)
            if not domain:
                continue
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    @classmethod
    def _rerank_for_domain_diversity(
        cls,
        ranked: list[ResearchSource],
        domain_counts: dict[str, int],
        low_novelty_streak: int,
        pass_index: int,
    ) -> list[ResearchSource]:
        if not ranked:
            return ranked
        # Do not force diversity early; apply pressure only after novelty has
        # started collapsing so the loop can still converge on strong evidence.
        if low_novelty_streak <= 0 and pass_index < 2:
            return ranked

        intensity = min(
            0.4,
            0.08 * max(low_novelty_streak, 1) + 0.02 * max(pass_index - 1, 0),
        )
        rescored: list[tuple[float, ResearchSource]] = []
        for source in ranked:
            domain = cls._source_domain(source.url)
            repeats = domain_counts.get(domain, 0)
            novelty_bonus = 0.07 if repeats == 0 else 0.0
            penalty = intensity * repeats
            adjusted = float(source.score) + novelty_bonus - penalty
            rescored.append((adjusted, source))

        rescored.sort(
            key=lambda item: (
                item[0],
                item[1].relevance,
                item[1].credibility_score,
                item[1].recency,
            ),
            reverse=True,
        )
        return [source for _score, source in rescored]

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
        raw_floor = targets.get("depth_pass_floor", 12)
        try:
            target_passes = int(raw_value)
        except (TypeError, ValueError):
            target_passes = 0
        try:
            depth_floor = int(raw_floor)
        except (TypeError, ValueError):
            depth_floor = 12
        if depth != "multi-hour":
            return max(target_passes, 1)
        return max(target_passes, max(depth_floor, 1))

    @staticmethod
    def _default_max_passes(depth: str) -> int:
        if depth == "quick":
            return 1
        if depth == "multi-hour":
            # 48 passes guarantees naturally long runtimes through real I/O:
            # each pass triggers network fetches, enrichment, and citation
            # chasing that accumulate genuine elapsed time.
            return 48
        return 6

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
            "min_on_topic_ratio",
            "max_weak_ratio",
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
        # Give multi-hour runs more tolerance for low-novelty passes so that
        # the gap-analysis mechanism has time to generate fresh query angles
        # before the streak limit fires.
        return max(streak, 6)

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

    @classmethod
    def _sanitize_query_variants(
        cls,
        variants: list[str],
        query: str,
    ) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            text = str(variant or "").strip()
            if not text:
                continue
            # Truncate FIRST so variants that differ only beyond 240 chars
            # are treated as duplicates, preventing 4 identical shortened
            # strings from polluting the query pool.
            text = text[:240]
            if cls._is_low_signal_query_variant(text, query):
                continue
            if cls._is_noisy_query_variant(text, query):
                continue
            normalized = cls._normalize_title(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(text)
        return deduped

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
        query: str = "",
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
        weak_count = sum(1 for source in selected if source.evidence_grade == "weak")
        on_topic_count = sum(
            1
            for source in selected
            if cls._objective_alignment_score(
                f"{source.title} {source.abstract}",
                query,
            )
            >= 0.35
        )
        on_topic_ratio = on_topic_count / max(len(selected), 1)
        weak_ratio = weak_count / max(len(selected), 1)
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
            "weak_count": weak_count,
            "weak_ratio": weak_ratio,
            "novelty_rate": novelty_rate,
            "on_topic_count": on_topic_count,
            "on_topic_ratio": on_topic_ratio,
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
            coverage.get("on_topic_ratio", 0.0)
            >= float(targets.get("min_on_topic_ratio", 0.0)),
            coverage.get("weak_ratio", 1.0)
            <= float(targets.get("max_weak_ratio", 1.0)),
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
        variants: list[str] = []
        core = self._query_core_terms(query) or query
        providers = {source.provider for source in selected}

        # Generate follow-ups from strongest current evidence terms.
        evidence_terms: list[str] = []
        for source in selected[:10]:
            text = f"{source.title} {source.abstract}"
            for token in self._keywords(text):
                if len(token) < 4:
                    continue
                if token in evidence_terms:
                    continue
                evidence_terms.append(token)
                if len(evidence_terms) >= 12:
                    break
            if len(evidence_terms) >= 12:
                break

        math_mode = self._looks_like_math_query(query)
        axes = [
            "primary evidence",
            "counterevidence",
            "methodology quality",
            "uncertainty bounds",
            "independent replication",
            "failure modes",
            "causal factors",
        ]
        if self._looks_like_current_evidence_query(query):
            axes = [
                "latest evidence",
                "current analysis",
                "timeline",
                "near-term scenarios",
                "risk factors",
                "counterevidence",
                "independent verification",
            ]
        if math_mode:
            axes = [
                "theorem barrier",
                "transfer mechanism",
                "counterexample search",
                "formal verification",
                "independent verification",
                "limitations",
            ]

        variants.append(core)
        if math_mode:
            for focus in self._math_focus_terms(query):
                variants.append(str(focus))
        for term in evidence_terms[:5]:
            for axis in axes[: 4 if depth == "quick" else 6]:
                variants.append(f"{term} {axis}")

        if plan is not None and plan.get("perspectives"):
            missing = self._perspective_coverage(
                selected,
                plan.get("perspectives") or [],
            )["missing"]
            for perspective in plan.get("perspectives") or []:
                if perspective.get("name") not in missing:
                    continue
                variants.extend((perspective.get("queries") or [])[:3])

        if "web-search" not in providers:
            variants.append(f"{core} independent sources")

        variants.extend(self._query_variants(query, depth))

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            candidate = str(variant or "").strip()[:240]
            if not candidate:
                continue
            if self._is_low_signal_query_variant(candidate, query):
                continue
            if self._is_noisy_query_variant(candidate, query):
                continue
            normalized = self._normalize_title(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate)
        max_items = 6 if depth == "quick" else 12 if depth == "standard" else 20
        return deduped[:max_items]

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
            self._load_env_from_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        if not api_key:
            self._record_provider_diagnostic(
                "gemini-flash",
                "skipped",
                (
                    "GEMINI_API_KEY or GOOGLE_API_KEY was not configured "
                    "in process env or .env file."
                ),
            )
            return []
        # Build a query-aware prompt so the observation is on-topic for
        # non-software queries (e.g. market research, scientific topics).
        if self._looks_like_software_agent_query(query):
            prompt = (
                "Act as a concise tool observer for an AgentOS smoke test. "
                "Compare the named local OS/coding/research agents only at a "
                "high level, mention uncertainty, and list concrete capabilities "
                "to verify locally. Query: "
                f"{query}. Depth: {depth}."
            )
        else:
            prompt = (
                "You are an expert research analyst. Provide a concise, "
                "factual synthesis on the following research query. Include "
                "key evidence, quantitative data where available, important "
                "uncertainties, and your analytical assessment. Do not pad "
                "with marketing language. Query: "
                f"{query}. Research depth: {depth}."
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

    # ------------------------------------------------------------------
    # AI REASONING LAYER
    # The engine must THINK about where to look and why — not just rotate
    # through templates.  These helpers call a lightweight text-only AI
    # endpoint (Gemini Flash when available, graceful no-op otherwise).
    # ------------------------------------------------------------------

    def _call_ai_text(self, system: str, user: str) -> str:
        """Call a text-only AI endpoint and return the raw text response.

        Uses Gemini Flash when a key is configured, falls back to an empty
        string so callers can degrade gracefully.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self._load_env_from_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        if not api_key:
            return ""
        payload = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash-lite:generateContent"
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
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:  # noqa: S310
                return _gemini_text(json.loads(resp.read().decode("utf-8")))
        except Exception:
            return ""

    def _ai_research_strategy(
        self,
        objective: str,
        query: str,
        depth: str,
    ) -> dict[str, Any]:
        """Ask AI to THINK about the best research strategy for this objective.

        Returns a dict with keys:
          causal_connections  – list of "A affects B because ..." strings
          authoritative_domains – specific domains/URLs worth targeting
          reasoning_queries – search queries derived from causal thinking
          subquestions – specific sub-questions to answer

        If AI is unavailable, returns empty lists so callers fall back to
        template-based defaults.
        """
        system = (
            "You are a research strategy advisor for a deep research engine. "
            "Think carefully about the objective and reason about (1) what "
            "entities and causal relationships are at play, (2) which specific "
            "authoritative domains or pages would have the best evidence, "
            "(3) what events or factors could influence the answer, and (4) "
            "what targeted search queries (not generic expansions) would uncover "
            "the strongest evidence. Respond ONLY with valid JSON."
        )
        user = (
            f"Research objective: {objective}\n"
            f"Core query: {query}\n"
            f"Depth: {depth}\n\n"
            "Reason step-by-step, then produce JSON with these exact keys:\n"
            "{\n"
            '  "causal_connections": ["<A> affects <B> because <reason>", ...],\n'
            '  "authoritative_domains": ["example.gov", "domain.org/path", ...],\n'
            '  "reasoning_queries": ["specific query derived from causal thinking", ...],\n'
            '  "subquestions": ["precise sub-question to answer", ...]\n'
            "}\n"
            "reasoning_queries must be 3-10 concrete search phrases (not generic "
            "variations). authoritative_domains should be real domains likely to "
            "have primary-source evidence."
        )
        raw = self._call_ai_text(system, user)
        try:
            # Find the first JSON object in the response.
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                return {
                    "causal_connections": list(parsed.get("causal_connections") or [])[
                        :8
                    ],
                    "authoritative_domains": list(
                        parsed.get("authoritative_domains") or []
                    )[:10],
                    "reasoning_queries": list(parsed.get("reasoning_queries") or [])[
                        :10
                    ],
                    "subquestions": list(parsed.get("subquestions") or [])[:8],
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {
            "causal_connections": [],
            "authoritative_domains": [],
            "reasoning_queries": [],
            "subquestions": [],
        }

    def _ai_evidence_gap_analysis(
        self,
        objective: str,
        selected: list[ResearchSource],
        pass_index: int,
        force_new_domains: bool = False,
        existing_domains: list[str] | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> list[str]:
        """After a retrieval pass, ask AI what's missing and what to pursue next.

        Returns a list of follow-up search queries derived from reasoning about
        gaps, causal connections, recency, and missing angles — NOT templates.
        Falls back to an empty list when AI is unavailable.
        """
        if not selected:
            return []
        # Build a compact summary of what was found.
        source_lines: list[str] = []
        for src in selected[:15]:
            year = f" ({src.year})" if src.year else ""
            snippet = (src.abstract or src.title)[:120].replace("\n", " ")
            source_lines.append(f"- [{src.provider}] {src.title}{year}: {snippet}")
        source_summary = "\n".join(source_lines)
        domain_hint = ", ".join(existing_domains[:12]) if existing_domains else "none"
        coverage_hint = coverage or {}
        system = (
            "You are a research gap analyst. Given what a research engine has "
            "found so far, reason about what important angles are missing, what "
            "causal connections implied by the findings should be followed up, "
            "whether the evidence is current enough, and what specific search "
            "queries would best fill the gaps. Respond ONLY with valid JSON."
        )
        user = (
            f"Research objective: {objective}\n"
            f"Retrieval pass: {pass_index + 1}\n\n"
            f"Current coverage: {json.dumps(coverage_hint, ensure_ascii=True)}\n"
            f"Current domains consulted: {domain_hint}\n\n"
            f"Sources found so far:\n{source_summary}\n\n"
            "Analyze the above and produce JSON with these exact keys:\n"
            "{\n"
            '  "gaps": ["missing angle or unanswered aspect", ...],\n'
            '  "follow_up_queries": ["specific search query to fill gap", ...]\n'
            "}\n"
            "follow_up_queries must be 3-6 targeted search phrases derived "
            "from your gap analysis — not generic expansions of the core query. "
            "Think: what causal factor is implied but not yet sourced? What "
            "recent event could change the picture? Which authority has not yet "
            "been consulted?"
        )
        if force_new_domains:
            user += (
                "\n\nCritical constraint: the retrieval loop is starved. "
                "Generate follow_up_queries that force evidence from entirely new "
                "authoritative domains/providers than those already listed."
            )
        raw = self._call_ai_text(system, user)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                queries = list(parsed.get("follow_up_queries") or [])
                # Relevance guard: reject any AI-generated query that shares
                # ZERO domain terms with the original objective.  This prevents
                # off-topic sources (Planck papers, chatgpt, etc.) from poisoning
                # the gap analysis when the retrieval engine found wrong results.
                obj_terms = set(
                    re.findall(r"\b[a-z][a-z0-9-]{3,}\b", objective.lower())
                ) - {
                    "what",
                    "that",
                    "this",
                    "have",
                    "been",
                    "will",
                    "were",
                    "they",
                    "their",
                    "which",
                    "about",
                    "also",
                    "into",
                    "with",
                    "from",
                    "then",
                    "than",
                    "some",
                    "such",
                    "both",
                    "each",
                    "more",
                    "most",
                    "just",
                    "does",
                    "other",
                }
                filtered: list[str] = []
                for q in queries:
                    candidate = str(q or "").strip()[:240]
                    if not candidate:
                        continue
                    if self._is_low_signal_query_variant(candidate, objective):
                        continue
                    if self._is_noisy_query_variant(candidate, objective):
                        continue
                    q_lower = candidate.lower()
                    overlap = sum(1 for term in obj_terms if term in q_lower)
                    if overlap == 0:
                        continue
                    if (
                        len(obj_terms) >= 4
                        and overlap < 2
                        and self._objective_alignment_score(candidate, objective) < 0.4
                    ):
                        continue
                    if self._objective_alignment_score(candidate, objective) < 0.35:
                        continue
                    filtered.append(candidate)
                return filtered[:6]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return []

    def _domain_diversification_queries(
        self,
        objective: str,
        selected: list[ResearchSource],
        pass_index: int,
        coverage: dict[str, Any],
    ) -> list[str]:
        existing_domains = sorted(
            {
                self._source_domain(source.url)
                for source in selected
                if self._source_domain(source.url)
            }
        )
        ai_queries = self._ai_evidence_gap_analysis(
            objective,
            selected,
            pass_index,
            force_new_domains=True,
            existing_domains=existing_domains,
            coverage=coverage,
        )
        if ai_queries:
            return ai_queries[:8]

        anchors = sorted(self._entity_terms_from_query(objective))[:3]
        if not anchors:
            anchors = self._keywords(objective)[:3]
        if not anchors:
            anchors = [self._query_core_terms(objective)[:40] or "objective"]

        templates = [
            "{anchor} independent primary source dataset",
            "{anchor} regulatory filing primary evidence",
            "{anchor} counterevidence bear case independent analyst",
            "{anchor} methodology critique data limitations",
            "{anchor} competing viewpoint contradictory analysis",
            "{anchor} official statistics longitudinal evidence",
        ]
        queries: list[str] = []
        for anchor in anchors[:3]:
            for template in templates[:3]:
                queries.append(template.format(anchor=anchor))

        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            normalized = self._normalize_title(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item[:120])
        return deduped[:8]

    def _ai_research_axes(
        self,
        objective: str,
        query: str,
        depth: str,
    ) -> dict[str, Any]:
        """Ask AI for comparative axes and evidence requirements specific to this objective.

        Returns a dict with keys ``comparative_axes`` and ``evidence_requirements``.
        Returns an empty dict when AI is unavailable so callers fall back to templates.
        """
        system = (
            "You are a research design expert. Given a research objective, generate "
            "the specific comparative axes and evidence quality requirements needed to "
            "evaluate findings for THAT topic. Be concrete and topic-specific — not "
            "generic placeholders. Respond ONLY with valid JSON."
        )
        user = (
            f"Research objective: {objective}\n"
            f"Core query: {query}\n"
            f"Depth: {depth}\n\n"
            "Produce JSON with these exact keys:\n"
            "{\n"
            '  "comparative_axes": ["specific dimension 1", ...],\n'
            '  "evidence_requirements": ["quality criterion 1", ...]\n'
            "}\n"
            "comparative_axes: 4-6 dimensions specific to THIS topic for comparing/evaluating evidence.\n"
            "evidence_requirements: 3-5 concrete quality criteria the evidence must meet for THIS topic.\n"
            "Both lists must be grounded in the actual subject matter."
        )
        raw = self._call_ai_text(system, user)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                axes = [
                    str(a)[:120]
                    for a in (parsed.get("comparative_axes") or [])
                    if str(a).strip()
                ][:6]
                reqs = [
                    str(r)[:120]
                    for r in (parsed.get("evidence_requirements") or [])
                    if str(r).strip()
                ][:5]
                if axes and reqs:
                    return {"comparative_axes": axes, "evidence_requirements": reqs}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {}

    def _ai_domain_seed_sources(
        self,
        objective: str,
        authoritative_domains: list[str],
    ) -> list[ResearchSource]:
        """Convert AI-reasoned authoritative domains into seed ResearchSources.

        This lets the engine consult specific authoritative pages that the AI
        identified through causal reasoning, not just search-API results.
        """
        sources: list[ResearchSource] = []
        for domain in authoritative_domains[:6]:
            domain = domain.strip()
            if not domain:
                continue
            # Validate it looks like a domain/URL (basic safety check).
            if not re.match(r"^[a-zA-Z0-9._\-/:%?=&#]+$", domain):
                continue
            url = domain if domain.startswith("http") else f"https://{domain}"
            # Validate it's not a private/local address.
            try:
                hostname = urllib.parse.urlparse(url).hostname or ""
                if hostname:
                    ipaddress.ip_address(hostname)
                    continue  # skip raw IP addresses
            except ValueError:
                pass  # not an IP, fine
            sources.append(
                ResearchSource(
                    provider="ai-reasoned-domain",
                    title=f"AI-identified authoritative source: {domain}",
                    url=url,
                    year=datetime.now(UTC).year,
                    abstract=(
                        f"Authoritative domain identified by AI reasoning for: "
                        f"{objective[:120]}"
                    ),
                    score=30.0,
                )
            )
        return sources

    def _ai_generate_perspectives(
        self,
        query: str,
        objective: str,
        depth: str,
    ) -> list[dict[str, Any]]:
        """Generate research perspectives tailored to this objective via AI.

        Asks the frontier model what distinct research angles matter for this
        specific topic — no hardcoded mode flags. Falls back to
        ``_generic_perspectives`` when the AI is unavailable or returns
        invalid JSON.
        """
        n = 6 if depth == "multi-hour" else (5 if depth == "standard" else 3)
        system = (
            "You are a research design specialist. Given a research query, "
            "generate search perspectives that each reveal complementary evidence "
            "a single broad query would miss. Each perspective must target a "
            "distinct evidence type. Respond with a valid JSON array only — "
            "no prose, no markdown fences."
        )
        user = (
            f"Query: {query}\n"
            f"Objective: {objective}\n\n"
            f"Generate {n} research perspectives as a JSON array. "
            "Each element must have: "
            '"name" (short lowercase-hyphenated slug), '
            '"goal" (one sentence describing what evidence this perspective collects), '
            '"keywords" (list of 4-6 relevant search keywords specific to the topic), '
            '"queries" (list of 1-3 short search phrases derived directly from the '
            "actual query terms — NOT generic placeholders like '{query} methodology').\n"
            "All keywords and queries must be grounded in the specific topic, "
            "not copy-paste templates."
        )
        try:
            raw = self._call_ai_text(system, user)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and len(parsed) >= 2:
                    valid: list[dict[str, Any]] = []
                    for p in parsed:
                        if (
                            isinstance(p, dict)
                            and isinstance(p.get("name"), str)
                            and isinstance(p.get("goal"), str)
                            and isinstance(p.get("keywords"), list)
                            and isinstance(p.get("queries"), list)
                            and len(p.get("queries") or []) >= 1
                        ):
                            valid.append(p)
                    if len(valid) >= 2:
                        return valid
        except Exception:
            pass
        return self._generic_perspectives(query, depth)

    @staticmethod
    def _generic_perspectives(query: str, depth: str) -> list[dict[str, Any]]:
        """Fallback perspectives that stay domain-agnostic and evidence-driven."""
        core = DeepResearchEngine._query_core_terms(query) or query
        seed_terms = [
            token for token in DeepResearchEngine._keywords(query) if len(token) >= 4
        ]
        if not seed_terms:
            seed_terms = [
                token for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{3,}\b", core)
            ]
        seed_terms = list(dict.fromkeys(seed_terms))[:8]

        axis_specs: list[tuple[str, str, list[str]]]
        if DeepResearchEngine._looks_like_current_evidence_query(query):
            axis_specs = [
                (
                    "current-signals",
                    "Track how evidence changed over time and what is newest.",
                    ["latest evidence", "current analysis", "timeline"],
                ),
                (
                    "drivers",
                    "Identify strongest causal drivers and catalysts.",
                    ["causal factors", "drivers", "mechanisms"],
                ),
                (
                    "counterevidence",
                    "Capture competing claims and disconfirming evidence.",
                    ["counterevidence", "alternative explanation", "disagreement"],
                ),
                (
                    "risk",
                    "Quantify uncertainty, scenario spread, and confidence limits.",
                    ["uncertainty", "risk scenarios", "confidence bounds"],
                ),
            ]
        else:
            axis_specs = [
                (
                    "baseline",
                    "Establish the factual baseline and scope.",
                    ["baseline", "scope", "definitions"],
                ),
                (
                    "evidence",
                    "Gather primary evidence and independent validation.",
                    ["primary evidence", "independent verification", "data sources"],
                ),
                (
                    "mechanisms",
                    "Explain causal mechanisms and constraints.",
                    ["causal factors", "mechanisms", "constraints"],
                ),
                (
                    "limitations",
                    "Evaluate limitations, edge cases, and uncertainty.",
                    ["limitations", "failure modes", "uncertainty"],
                ),
                (
                    "counterevidence",
                    "Collect contradictory evidence and alternative interpretations.",
                    ["counterevidence", "alternative interpretation", "disagreement"],
                ),
            ]

        if depth == "quick":
            axis_specs = axis_specs[:3]
        elif depth == "standard":
            axis_specs = axis_specs[:4]

        perspectives: list[dict[str, Any]] = []
        for name, goal, axis_keywords in axis_specs:
            keywords = [core, *axis_keywords, *seed_terms[:3]]
            keywords = list(dict.fromkeys(keywords))[:6]
            queries = [
                f"{core} {axis_keywords[0]}",
                f"{core} {axis_keywords[1]}",
            ]
            if seed_terms:
                queries.append(f"{seed_terms[0]} {axis_keywords[0]}")
            perspectives.append(
                {
                    "name": name,
                    "goal": goal,
                    "keywords": keywords,
                    "queries": queries[:3],
                }
            )
        return perspectives

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
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        page_stride = 20
        max_pages = max(1, min(8, (target_limit + page_stride - 1) // page_stride + 1))
        seen_urls: set[str] = set()
        seen_pages: set[str] = set()
        global_rank = 0
        preview_fetch_budget = (
            12 if self._looks_like_current_evidence_query(query) else 6
        )
        preview_fetches = 0

        for page_index in range(max_pages):
            page_start = page_index * page_stride
            params = {"q": query}
            if page_start > 0:
                params["s"] = str(page_start)
                params["dc"] = str(page_start)
            search_url = (
                f"https://html.duckduckgo.com/html/?{urllib.parse.urlencode(params)}"
            )
            try:
                raw_html = self._get_text(
                    search_url,
                    accept="text/html,application/xhtml+xml",
                    max_bytes=120_000,
                    timeout_seconds=6,
                )
            except TypeError:
                raw_html = self._get_text(
                    search_url,
                    accept="text/html,application/xhtml+xml",
                    max_bytes=120_000,
                )
            if not raw_html:
                break

            page_fingerprint = self._normalize_title(raw_html[:2400])
            if page_fingerprint in seen_pages:
                break
            seen_pages.add(page_fingerprint)

            added_this_page = 0
            for match in re.finditer(
                (
                    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"'
                    r"[^>]*>(.*?)</a>"
                ),
                raw_html,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                raw_url = self._normalize_web_result_url(match.group(1))
                if not self._is_safe_public_url(raw_url):
                    continue
                if raw_url in seen_urls:
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
                quality_flags: list[str] = []
                if not snippet.strip() and preview_fetches < preview_fetch_budget:
                    preview_fetches += 1
                    try:
                        preview = self._get_text(
                            raw_url,
                            accept="text/html,application/xhtml+xml,*/*",
                            max_bytes=30_000,
                            timeout_seconds=4,
                        )
                    except TypeError:
                        preview = self._get_text(
                            raw_url,
                            accept="text/html,application/xhtml+xml,*/*",
                            max_bytes=30_000,
                        )
                    preview_text = self._html_to_text(preview)
                    tickers = _extract_ticker_candidates(f"{title} {preview_text}")
                    if tickers:
                        snippet = f"Ticker candidates mentioned: {', '.join(tickers)}."
                    elif len(preview_text) >= 80:
                        snippet = re.sub(r"\s+", " ", preview_text)[:320]
                    else:
                        quality_flags.append("snippet-unavailable")
                elif not snippet.strip():
                    quality_flags.append("snippet-unavailable")

                seen_urls.add(raw_url)
                added_this_page += 1
                global_rank += 1
                host = urllib.parse.urlparse(raw_url).netloc.lower().lstrip("www.")
                score = max(target_limit - global_rank + 1, 0)
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
                        quality_flags=quality_flags,
                    )
                )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit or added_this_page == 0:
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

        if cls._looks_like_current_evidence_query(
            query
        ) and not cls._looks_like_academic_query(query):
            if cls._looks_like_market_query(
                query
            ) and not cls._looks_like_quant_finance_query(query):
                # Current market tasks should prioritize real-time web/provider
                # evidence. Scholarly providers are admitted only for explicit
                # quant-finance modeling objectives.
                return {"web-search", "gemini-flash", "software-reference"}
            # Include software-reference for price/product data, and crossref for
            # reports/whitepapers that may index financial analyses.
            return {"web-search", "gemini-flash", "software-reference", "crossref"}

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

    @classmethod
    def _expand_provider_mix_for_diversity(
        cls,
        query: str,
        allowed_providers: set[str],
        coverage: dict[str, Any],
    ) -> set[str] | None:
        # Only expand when provider diversity is genuinely low relative to what
        # is available (fewer than 3 distinct providers delivering results).
        if int(coverage.get("provider_count") or 0) >= 3:
            return None
        expanded = set(allowed_providers)
        # Respect the original classification domain: only expand to scholarly
        # providers when the query was already classified as needing scholarly
        # coverage.  Market and current-evidence queries must not be poisoned
        # with academic literature — scholarly providers are added only when
        # the original allowed set already contains at least one of them.
        scholarly = {"openalex", "semantic-scholar", "crossref"}
        if allowed_providers & scholarly:
            expanded.update(scholarly)
        expanded.add("web-search")
        if cls._looks_like_software_agent_query(query):
            expanded.add("github-repositories")
        return expanded if expanded != allowed_providers else None

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
            # Deterministically page and stitch long-form sources so we do not
            # over-index on metadata-heavy front matter.
            raw_html = self._get_text_stitched(
                source.url,
                accept="text/html,application/xhtml+xml,*/*",
                page_bytes=60_000,
                max_pages=4,
                overlap_bytes=1_600,
                query=query,
            )
            if not raw_html:
                continue

            # Interrupt-handler state machine: if signal collapses and overlay
            # markers are present, enter a bounded modal-dismissal routine.
            # If still blocked after two attempts, tag as unreachable-paywalled.
            signal_before = self._text_signal_score(self._html_to_text(raw_html))
            overlay_markers = self._overlay_marker_count(raw_html)
            if signal_before < 0.16 and overlay_markers > 0:
                resolved_html, status = self._interrupt_resolve_overlays(raw_html)
                if status != "resolved":
                    if "unreachable-paywalled" not in source.quality_flags:
                        source.quality_flags.append("unreachable-paywalled")
                    self.provider_diagnostics.append(
                        {
                            "provider": source.provider,
                            "query": query,
                            "status": "unreachable-paywalled",
                            "url": source.url,
                        }
                    )
                    continue
                raw_html = resolved_html

            content = self._html_to_text(raw_html)
            if len(content) > 80:
                # Extend the abstract so ranking gets real signal.
                extra = content[:1200]
                if source.abstract.lower().startswith("generic web result for "):
                    source.abstract = extra[:3000]
                else:
                    source.abstract = f"{source.abstract} {extra}".strip()[:3000]
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

    @staticmethod
    def _text_signal_score(text: str) -> float:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        if not words:
            return 0.0
        noise_tokens = {
            "cookie",
            "consent",
            "privacy",
            "terms",
            "subscribe",
            "newsletter",
            "sign",
            "login",
            "close",
            "accept",
            "decline",
        }
        lower_words = [word.lower() for word in words]
        noise_hits = sum(1 for word in lower_words if word in noise_tokens)
        density = min(1.0, len(words) / 260.0)
        noise_ratio = noise_hits / max(len(lower_words), 1)
        return max(0.0, min(1.0, density * (1.0 - min(noise_ratio * 6.0, 1.0))))

    @staticmethod
    def _overlay_marker_count(raw_html: str) -> int:
        lower = raw_html.lower()
        markers = (
            'role="dialog"',
            "role='dialog'",
            'role="alert"',
            "position:fixed",
            "position: fixed",
            "cookie",
            "consent",
            "paywall",
            "subscribe",
            "newsletter",
            "modal",
            "overlay",
        )
        return sum(1 for marker in markers if marker in lower)

    @classmethod
    def _interrupt_resolve_overlays(cls, raw_html: str) -> tuple[str, str]:
        html_candidate = raw_html
        for _ in range(2):
            html_candidate = cls._strip_known_overlays(html_candidate)
            signal = cls._text_signal_score(cls._html_to_text(html_candidate))
            if signal >= 0.2:
                return html_candidate, "resolved"
        return raw_html, "unreachable-paywalled"

    @staticmethod
    def _strip_known_overlays(raw_html: str) -> str:
        patterns = [
            r"<div[^>]*role=['\"](?:dialog|alert)['\"][^>]*>.*?</div>",
            r"<div[^>]*(?:cookie|consent|newsletter|subscribe|paywall|modal|overlay)[^>]*>.*?</div>",
            r"<aside[^>]*(?:cookie|consent|newsletter|subscribe|paywall|modal|overlay)[^>]*>.*?</aside>",
            r"<section[^>]*(?:cookie|consent|newsletter|subscribe|paywall|modal|overlay)[^>]*>.*?</section>",
        ]
        cleaned = raw_html
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned

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
        timeout_seconds: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        headers = {
            "Accept": accept,
            "User-Agent": "agentos-orchestrator/0.1 (research enrichment)",
        }
        if range_start is not None:
            bounded_start = max(0, range_start)
            bounded_end = max(
                bounded_start,
                range_end if range_end is not None else bounded_start + max_bytes - 1,
            )
            headers["Range"] = f"bytes={bounded_start}-{bounded_end}"
        if extra_headers:
            headers.update(
                {str(key): str(value) for key, value in extra_headers.items()}
            )
        request = urllib.request.Request(
            url,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - policy-gated URLs
                request,
                timeout=min(timeout_seconds or self.timeout_seconds, 15),
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

    def _get_text_stitched(
        self,
        url: str,
        accept: str,
        page_bytes: int,
        max_pages: int,
        overlap_bytes: int,
        query: str,
    ) -> str:
        chunks: list[str] = []
        seen_fingerprints: set[str] = set()
        for index in range(max_pages):
            start = index * max(1, page_bytes - overlap_bytes)
            end = start + page_bytes - 1
            raw = self._get_text(
                url,
                accept=accept,
                max_bytes=page_bytes,
                range_start=start,
                range_end=end,
                timeout_seconds=10,
            )
            if not raw:
                break
            fingerprint = self._normalize_title(raw[:1800])
            if fingerprint in seen_fingerprints:
                break
            seen_fingerprints.add(fingerprint)

            chunk = self._extract_signal_text_chunk(raw, query)
            if not chunk:
                if index == 0:
                    continue
                break
            chunks.append(chunk)

            if len(raw) < int(page_bytes * 0.85):
                break

        if not chunks:
            return ""
        stitched = self._stitch_text_chunks(chunks, overlap_chars=140)
        return stitched[:24_000]

    def _extract_signal_text_chunk(self, raw_html: str, query: str) -> str:
        signal_pattern = re.compile(
            (
                r"<(?:h1|h2|h3|h4)[^>]*>[^<]*(?:Conclusion|Results|Discussion|"
                r"Financial Summary|Risk Factors|Outlook|Guidance)[^<]*</(?:h1|h2|h3|h4)>"
            ),
            re.IGNORECASE | re.DOTALL,
        )
        signal_hits = signal_pattern.findall(raw_html)
        if signal_hits:
            candidate = self._strip_dom_noise_tokens(
                self._html_to_text(" ".join(signal_hits[:8]))
            )
            if self._text_signal_score(candidate) >= 0.08:
                if query and not self._passes_semantic_binary_gate(candidate, query):
                    return ""
                return candidate

        candidate = self._strip_dom_noise_tokens(self._html_to_text(raw_html))
        if self._text_signal_score(candidate) < 0.05:
            return ""
        if query and not self._passes_semantic_binary_gate(candidate, query):
            return ""
        if query:
            anchors = set(self._keywords(query)) | set(
                self._entity_terms_from_query(query)
            )
            if anchors and not any(anchor in candidate.lower() for anchor in anchors):
                return ""
        return candidate

    @staticmethod
    def _stitch_text_chunks(chunks: list[str], overlap_chars: int = 120) -> str:
        if not chunks:
            return ""
        stitched = chunks[0]
        for chunk in chunks[1:]:
            suffix = stitched[-overlap_chars:]
            prefix = chunk[:overlap_chars]
            overlap = 0
            max_window = min(len(suffix), len(prefix))
            for width in range(max_window, 24, -1):
                if suffix[-width:] == prefix[:width]:
                    overlap = width
                    break
            stitched += chunk[overlap:]
        return stitched

    @staticmethod
    def _looks_like_market_query(query: str) -> bool:
        lower = query.lower()
        market_tokens = {
            "stock",
            "stocks",
            "equity",
            "equities",
            "market",
            "markets",
            "invest",
            "investing",
            "investor",
            "upside",
            "downside",
            "earnings",
            "valuation",
            "price target",
            "wall street",
            "bull",
            "bear",
            "portfolio",
            "ticker",
        }
        return any(token in lower for token in market_tokens)

    @staticmethod
    def _looks_like_quant_finance_query(query: str) -> bool:
        lower = query.lower()
        quant_tokens = {
            "factor model",
            "asset pricing",
            "fama",
            "carhart",
            "garch",
            "stochastic volatility",
            "black-scholes",
            "heston",
            "value at risk",
            "expected shortfall",
            "portfolio optimization",
            "mean variance",
            "cointegration",
            "statistical arbitrage",
            "market microstructure",
        }
        return any(token in lower for token in quant_tokens)

    @staticmethod
    def _strip_dom_noise_tokens(text: str) -> str:
        cleaned = text
        cleaned = re.sub(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}\b", " ", cleaned)
        cleaned = re.sub(r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b", " ", cleaned)
        cleaned = re.sub(
            (
                r"\b(?:hover|focus|active|visited|disabled|aria-|role=|"
                r"min-width|max-width|font-size|font-weight|line-height|"
                r"padding|margin|display|overflow|grid|flex|text-headline|"
                r"text-title|className|querySelector|innerText|"
                r"storywithleadvideo|storywith|leadvideo|flexi-page|"
                r"nimbus|progressive-advanced)\b"
            ),
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _has_dom_noise_pattern(text: str) -> bool:
        lower = text.lower()
        if re.search(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}\b", lower):
            return True
        if re.search(r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b", text):
            return True
        return any(
            token in lower
            for token in (
                "min-width",
                "max-width",
                "font-size",
                "font-weight",
                "text-headline",
                "text-title",
                "queryselector",
                "innertext",
            )
        )

    def _passes_semantic_binary_gate(self, text: str, query: str) -> bool:
        sample = self._strip_dom_noise_tokens((text or "")[:4000])
        if not sample.strip():
            return False
        cache_key = (
            f"{self._normalize_title(query)}::{self._normalize_title(sample[:900])}"
        )
        if cache_key in self._semantic_gate_cache:
            return self._semantic_gate_cache[cache_key]

        deterministic = self._passes_deterministic_semantic_gate(sample, query)
        if deterministic:
            self._semantic_gate_cache[cache_key] = True
            return True
        anchors = self._objective_anchor_terms(query)
        words = set(re.findall(r"\b[a-z][a-z0-9-]{2,}\b", sample.lower()))
        overlap = len(anchors & words) if anchors else 0
        if anchors and overlap == 0:
            self._semantic_gate_cache[cache_key] = False
            return False

        # Optional LLM arbitration for ambiguous edge cases.
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            prompt = (
                "Binary relevance gate. Answer only TRUE or FALSE. "
                "TRUE only if the text explicitly addresses the objective. "
                f"Objective: {query}\nText: {sample[:1200]}"
            )
            decision = (
                self._call_ai_text(
                    "You are a strict relevance gate. Return only TRUE or FALSE.",
                    prompt,
                )
                .strip()
                .upper()
            )
            allowed = decision.startswith("TRUE")
            self._semantic_gate_cache[cache_key] = allowed
            return allowed

        allowed = overlap >= 1 and not self._has_dom_noise_pattern(sample)
        self._semantic_gate_cache[cache_key] = allowed
        return allowed

    @classmethod
    def _passes_deterministic_semantic_gate(cls, text: str, query: str) -> bool:
        sample = cls._strip_dom_noise_tokens(text)
        anchors = cls._objective_anchor_terms(query)
        words = set(re.findall(r"\b[a-z][a-z0-9-]{2,}\b", sample.lower()))
        overlap = len(anchors & words) if anchors else 0
        if cls._looks_like_market_query(query):
            market_vocab = {
                "stock",
                "stocks",
                "equity",
                "equities",
                "market",
                "markets",
                "earnings",
                "valuation",
                "price",
                "target",
                "revenue",
                "margin",
                "growth",
                "upside",
                "downside",
                "ticker",
                "guidance",
            }
            offdomain_vocab = {
                "radiocarbon",
                "paleoenvironmental",
                "antarctica",
                "hepatocellular",
                "osteoarthritis",
                "clinical",
                "blood pressure",
                "tumour",
                "trial",
            }
            has_market = any(token in sample.lower() for token in market_vocab)
            has_offdomain = any(token in sample.lower() for token in offdomain_vocab)
            if has_offdomain and not has_market:
                return False
            if not has_market and overlap < 2:
                return False
        return overlap >= 2 and not cls._has_dom_noise_pattern(sample)

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
        # Detect JS-blocked / error pages — these produce garbage queries.
        _js_block_signals = (
            "enable javascript",
            "javascript is required",
            "javascript is disabled",
            "please enable javascript",
            "this site requires javascript",
            "pardon our interruption",
            "access denied",
            "403 forbidden",
            "404 not found",
            "cloudflare ray id",
            "captcha",
            "you are being rate limited",
            "your request has been blocked",
        )
        content = cls._strip_dom_noise_tokens(content)
        source_title = cls._strip_dom_noise_tokens(source_title)
        _content_lower = content.lower()[:2000]
        if any(sig in _content_lower for sig in _js_block_signals):
            return []
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
            # JS/browser error page tokens — these appear when pages are blocked
            "javascript",
            "function",
            "return",
            "window",
            "pardon",
            "captcha",
            "cloudflare",
            "browser",
            "cookies",
            "forbidden",
            "interruption",
            "enable",
            "script",
            "loading",
            "redirect",
            # Common web-nav / financial-site UI noise — these words appear
            # in sidebars, menus, and ticker widgets and produce garbage n-gram
            # queries like "marketbeat stock stocks stock" or "english investing".
            "english",
            "investing",
            "financial",
            "markets",
            "market",
            "today",
            "movers",
            "gainers",
            "shares",
            "ticker",
            "click",
            "search",
            "login",
            "register",
            "subscribe",
            "newsletter",
            "privacy",
            "terms",
            "contact",
            "homepage",
            "sidebar",
            "widget",
            "footer",
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
            if cls._has_dom_noise_pattern(candidate):
                continue
            if query and cls._objective_alignment_score(candidate, query) < 0.35:
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

    def _load_env_from_dotenv(self) -> None:
        if self._dotenv_loaded:
            return
        self._dotenv_loaded = True
        dotenv_path = self.workspace_root / ".env"
        if not dotenv_path.exists():
            return
        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env_key = key.strip()
                if not env_key:
                    continue
                cleaned = value.strip().strip('"').strip("'")
                if env_key not in os.environ and cleaned:
                    os.environ[env_key] = cleaned
        except OSError:
            return

    def _record_provider_preflight(self) -> None:
        gemini_present = bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self._record_provider_diagnostic(
            "provider-preflight",
            "ok" if gemini_present else "warning",
            (
                "gemini key configured"
                if gemini_present
                else "gemini key not found in env/.env; gemini provider can be skipped"
            ),
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
        run_id_for_durable_notes: str,
        synthesis_mode: str,
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
        durable_notes_path = self._durable_report_path(run_id_for_durable_notes)
        findings = self._finding_ledger(query, sources, plan)
        retrieval_payload = {
            "coverage": retrieval["coverage"],
            "passes": retrieval["passes"],
            "stop_reason": retrieval["stop_reason"],
            "query_variants": retrieval["query_variants"],
        }
        benchmark_adapters = self._benchmark_adapters(sources)
        token_strategy_parts = [
            "structured scholarly APIs",
            "broad web search",
            "explicit URL seeding",
        ]
        if self._looks_like_software_agent_query(f"{query} {objective}"):
            token_strategy_parts.append("software repository search")
        token_strategy_parts.extend(
            [
                "optional model observations",
                "exact dedupe",
                "plan-first multi-perspective query decomposition",
                "finding support/conflict ledger",
                "relevance ranking",
                "compressed digest artifacts",
            ]
        )

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
                    "final_synthesis_mode": synthesis_mode,
                    "token_strategy": ", ".join(token_strategy_parts),
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
        artifacts = [
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
        if durable_notes_path is not None and durable_notes_path.exists():
            artifacts.append(str(durable_notes_path.relative_to(self.workspace_root)))
        return artifacts

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
        durable_notes: str = "",
        synthesis_mode: str = "hybrid",
    ) -> str:
        if not sources:
            return (
                "Live research did not return sources from configured public "
                f"providers for: {objective}. Check network policy, API "
                "availability, or attach MCP research servers."
            )
        if synthesis_mode == "durable-notes-only" and durable_notes.strip():
            ai_synthesis = self._ai_durable_notes_synthesis(
                objective,
                sources,
                durable_notes,
                plan,
            )
            if ai_synthesis:
                return ai_synthesis
            note_lines = [
                line
                for line in durable_notes.splitlines()
                if line.strip().startswith("- [")
            ]
            return (
                f"Durable-notes synthesis mode used for: {objective}. "
                f"Integrated {len(note_lines)} distilled claims from workflows/report.md "
                "with minimal source metadata."
            )
        findings = self._finding_ledger(query or objective, sources, plan)
        perspective_coverage = self._perspective_coverage(
            sources,
            (plan or {}).get("perspectives") or [],
        )
        subquestion_count = len((plan or {}).get("subquestions", []))

        # ------------------------------------------------------------------
        # SCIENTIST SYNTHESIS: ask AI to reason about what the evidence
        # actually says — forming conclusions, noting contradictions, and
        # flagging gaps — rather than generating a boilerplate summary.
        # ------------------------------------------------------------------
        ai_synthesis = self._ai_scientist_synthesis(
            objective,
            sources,
            findings,
            plan,
            perspective_coverage,
            durable_notes,
            synthesis_mode,
        )
        if ai_synthesis:
            return ai_synthesis

        # Fallback template summary when AI is unavailable.
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

    def _ai_scientist_synthesis(
        self,
        objective: str,
        sources: list[ResearchSource],
        findings: list[dict[str, Any]],
        plan: dict[str, Any] | None,
        perspective_coverage: dict[str, Any],
        durable_notes: str = "",
        synthesis_mode: str = "hybrid",
    ) -> str:
        """Ask AI to synthesize evidence like a scientist: form conclusions,
        note contradictions and gaps, and calibrate confidence.

        Returns an empty string when AI is unavailable.
        """
        if not sources:
            return ""
        # Build compact evidence digest. In durable-notes-only mode we only use
        # minimal metadata and never include source abstracts/snippets.
        source_lines: list[str]
        if synthesis_mode == "durable-notes-only":
            source_lines = self._minimal_source_metadata_lines(sources)
        else:
            source_lines = []
            for src in sources[:20]:
                year = f" ({src.year})" if src.year else ""
                grade = src.evidence_grade
                snippet = (src.abstract or src.title)[:160].replace("\n", " ")
                source_lines.append(
                    f"[{src.provider}/{grade}] {src.title}{year}: {snippet}"
                )
        finding_lines: list[str] = []
        for f in findings[:6]:
            finding_lines.append(
                f"  {f['perspective']} ({f['confidence']}, "
                f"{f['support_count']} sources, "
                f"{f['contradiction_count']} contradictions): {f['finding']}"
            )
        missing = ", ".join(perspective_coverage.get("missing") or []) or "none"
        subquestions = (
            "\n".join(f"  - {sq}" for sq in (plan or {}).get("subquestions", [])[:6])
            or "  (none recorded)"
        )
        system = (
            "You are a senior research scientist writing an evidence synthesis. "
            "Your task is NOT to summarize — it is to ANALYZE. You must:\n"
            "1. WEIGH contradictory evidence: when sources disagree, explain "
            "WHY they disagree (methodology, scope, recency, bias) and which "
            "position is better supported.\n"
            "2. GRADE credibility: for each major claim, state whether it is "
            "supported by high-credibility sources (peer-reviewed, gov data, "
            "authoritative institutions) or low-credibility ones (preprints, "
            "blogs, uncited papers).\n"
            "3. FORM conclusions: don't just list findings — reason about what "
            "the aggregate evidence means, what is robust, and what is speculative.\n"
            "4. IDENTIFY gaps: explicitly state what evidence is missing and "
            "what additional investigation would change the conclusions.\n"
            "5. CALIBRATE confidence: assign explicit confidence levels "
            "(high/moderate/low/speculative) to each conclusion with justification.\n"
            "Be specific, cite evidence types, and never use generic filler phrases."
        )
        user = (
            f"Research objective: {objective}\n\n"
            f"Subquestions investigated:\n{subquestions}\n\n"
            f"Evidence found ({len(sources)} sources):\n"
            + "\n".join(source_lines)
            + (
                "\n\nDurable distilled report notes (primary synthesis input):\n"
                + durable_notes[:12000]
                if durable_notes
                else ""
            )
            + f"\n\nPer-perspective findings:\n"
            + "\n".join(finding_lines or ["  (none yet)"])
            + f"\n\nUncovered perspectives: {missing}\n\n"
            "Synthesize this evidence as a senior research analyst would:\n"
            "1. What does the evidence most strongly support? Grade each conclusion "
            "(high/moderate/low/speculative confidence) with justification.\n"
            "2. Where do sources contradict? For EACH contradiction, explain the "
            "likely cause (methodology, recency, scope, bias) and which side "
            "the weight of evidence favors.\n"
            "3. For each major claim, explicitly grade the supporting sources' "
            "credibility (peer-reviewed > government data > industry reports > "
            "preprints > blogs > uncited papers).\n"
            "4. What important gaps remain? What specific evidence would be "
            "needed to move speculative conclusions to moderate confidence?\n"
            "5. What is the overall synthesis confidence and why?\n"
            "Write a coherent 4-8 paragraph analysis. Be substantive and specific — "
            "a reader should be able to make informed decisions based on your synthesis."
        )
        return self._call_ai_text(system, user)

    def _ai_durable_notes_synthesis(
        self,
        objective: str,
        sources: list[ResearchSource],
        durable_notes: str,
        plan: dict[str, Any] | None,
    ) -> str:
        """Synthesize using only durable report notes plus minimal metadata."""
        if not durable_notes.strip():
            return ""
        metadata_lines = "\n".join(self._minimal_source_metadata_lines(sources))
        subquestions = (
            "\n".join(f"  - {sq}" for sq in (plan or {}).get("subquestions", [])[:6])
            or "  (none recorded)"
        )
        system = (
            "You are a senior research scientist. Build the final synthesis using "
            "ONLY the provided durable notes and minimal source metadata. "
            "Do not request or infer hidden abstract text. "
            "Explicitly weigh contradictions, confidence, and missing evidence."
        )
        user = (
            f"Research objective: {objective}\n\n"
            f"Subquestions investigated:\n{subquestions}\n\n"
            "Durable report notes:\n"
            f"{durable_notes[:16000]}\n\n"
            "Minimal source metadata:\n"
            f"{metadata_lines}\n\n"
            "Write a 4-8 paragraph final synthesis with confidence levels and "
            "contradiction analysis."
        )
        return self._call_ai_text(system, user)

    @staticmethod
    def _minimal_source_metadata_lines(sources: list[ResearchSource]) -> list[str]:
        lines: list[str] = []
        for src in sources[:40]:
            year = str(src.year) if src.year else "n.d."
            lines.append(
                (
                    f"- [{src.provider}/{src.evidence_grade}] {src.title} "
                    f"(year: {year}) url: {src.url}"
                )[:320]
            )
        return lines

    @staticmethod
    def _resolve_final_synthesis_mode(depth: str, durable_notes: str) -> str:
        configured = (
            str(os.environ.get("AGENTOS_FINAL_SYNTHESIS_MODE") or "").strip().lower()
        )
        if configured in {"hybrid", "durable-notes-only"}:
            return configured
        if depth == "multi-hour" and durable_notes.strip():
            return "durable-notes-only"
        return "hybrid"

    def _initialize_durable_report(
        self,
        run_id: str,
        depth: str,
        objective: str,
    ) -> Path | None:
        if not run_id:
            return None
        report_path = self._durable_report_path(run_id)
        if report_path is None:
            return None
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not report_path.exists():
            report_path.write_text(
                (
                    "# Durable Research Report\n\n"
                    f"Depth: {depth}\n\n"
                    f"Objective: {objective}\n\n"
                    "## Incremental Findings\n\n"
                ),
                encoding="utf-8",
            )
        else:
            try:
                existing = report_path.read_text(encoding="utf-8")
                self._durable_note_passes = {
                    int(match.group(1))
                    for match in re.finditer(r"^###\s+Pass\s+(\d+)\b", existing, re.M)
                }
            except (OSError, ValueError):
                self._durable_note_passes = set()
        return report_path

    def _append_durable_claim_notes(
        self,
        report_path: Path | None,
        pass_index: int,
        sources: list[ResearchSource],
        query: str,
    ) -> None:
        if report_path is None or not sources:
            return
        if pass_index in self._durable_note_passes:
            return
        lines: list[str] = [f"### Pass {pass_index}"]
        wrote_any = False
        for source in sources:
            if not source.url or source.url in self._durable_note_urls:
                continue
            if source.evidence_grade not in {"strong", "moderate", "tool-observation"}:
                continue
            claim = self._compressed_claim(source, query)
            if not claim:
                continue
            wrote_any = True
            self._durable_note_urls.add(source.url)
            lines.append(
                "- "
                f"[{source.evidence_grade}/{source.provider}] {claim} "
                f"(source: {source.url})"
            )
        if not wrote_any:
            lines.append("- [info/system] no-new-distilled-claims this pass")
        lines.append("")
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._durable_note_passes.add(pass_index)

    @staticmethod
    def _compressed_claim(source: ResearchSource, query: str = "") -> str:
        text = (source.abstract or source.title or "").strip()
        if not text:
            return ""
        lower_raw = text.lower()
        if (
            query
            and DeepResearchEngine._looks_like_market_query(query)
            and source.provider == "gemini-flash"
        ):
            return ""
        if query and DeepResearchEngine._looks_like_market_query(query):
            tickers = _extract_ticker_candidates(f"{source.title} {source.abstract}")
            if len(tickers) >= 2:
                return f"Ticker candidates mentioned: {', '.join(tickers)}"
        if (
            lower_raw.startswith("generic web result")
            or "snippet unavailable" in lower_raw
        ):
            return ""
        text = DeepResearchEngine._html_to_text(text)
        if re.search(r"\{.*\}|\[.*\]|\"[a-z0-9_-]+\"\s*:\s*", text[:500], re.I):
            return ""
        text = re.sub(
            r"\b(?:jats|xml|xmlns|sec-type|content-type|article-meta)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text or DeepResearchEngine._text_signal_score(text) < 0.07:
            return ""
        if len(re.findall(r"[{}\[\]<>]", text)) >= 4:
            return ""
        promotional_markers = (
            "skip to content",
            "top rated",
            "trading signals",
            "subscribe",
            "newsletter",
            "market intelligence",
        )
        if sum(1 for marker in promotional_markers if marker in text.lower()) >= 2:
            return ""
        if query:
            anchors = set(DeepResearchEngine._keywords(query)) | set(
                DeepResearchEngine._entity_terms_from_query(query)
            )
            if anchors and not any(anchor in text.lower() for anchor in anchors):
                return ""
            if DeepResearchEngine._objective_alignment_score(text, query) < 0.22:
                return ""
        sentence = re.split(r"[.!?]", text, maxsplit=1)[0].strip()
        if sentence and DeepResearchEngine._text_signal_score(sentence) >= 0.08:
            claim = sentence
        else:
            # Short headline-like first sentences often have low token density.
            # Fall back to the cleaned full text so valid web evidence can still
            # be distilled into durable notes.
            claim = text
        claim = re.sub(r"\s+", " ", claim)
        if DeepResearchEngine._text_signal_score(claim) < 0.07:
            return ""
        return claim[:260]

    def _load_durable_notes(self, run_id: str) -> str:
        report_path = self._durable_report_path(run_id)
        if report_path is None or not report_path.exists():
            return ""
        try:
            return report_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _durable_report_path(self, run_id: str) -> Path | None:
        if not run_id:
            return None
        return self.workspace_root / "runs" / run_id / "workflows" / "report.md"

    def _build_research_plan(
        self,
        objective: str,
        query: str,
        depth: str,
        pc_context_info: dict[str, Any],
    ) -> dict[str, Any]:
        # ----------------------------------------------------------------
        # THINKING STEP: Ask AI to reason about this specific objective —
        # what entities are involved, what causal relationships matter, what
        # authoritative sources exist, and what queries would expose the
        # strongest evidence.
        # ----------------------------------------------------------------
        ai_strategy = self._ai_research_strategy(objective, query, depth)

        # ------------------------------------------------------------------
        # ADAPTIVE PLANNING: derive perspectives, comparative axes, and
        # evidence requirements from AI reasoning about THIS specific
        # objective, not from domain-type templates.
        # ------------------------------------------------------------------
        perspectives = self._research_perspectives(query, objective, depth)

        ai_axes = self._ai_research_axes(objective, query, depth)
        comparative_axes = ai_axes.get("comparative_axes") or []
        evidence_requirements = ai_axes.get("evidence_requirements") or []

        # AI-derived subquestions from strategy call above.
        ai_subquestions = list(ai_strategy.get("subquestions") or [])

        if not ai_subquestions:
            # Absolute minimal fallback if AI strategy generation failed.
            ai_subquestions = [
                "What exact problem statement defines the topic?",
                "Which causal drivers and mechanisms recur across the evidence?",
                "What explicit limitations or uncertainties remain?",
            ]

        if not comparative_axes:
            comparative_axes = [
                "source credibility and recency",
                "methodological rigor",
                "stated limitations and uncertainties",
                "independent corroboration",
            ]

        if not evidence_requirements:
            evidence_requirements = [
                "primary or authoritative evidence",
                "explicit causal or methodological data",
                "independent corroboration when available",
                "clear risk, uncertainty, or limitation statements",
            ]

        if pc_context_info.get("browser_context_detected"):
            ai_subquestions.append(
                "How does live browser/app context from the local PC alter the evidence collection sequence?"
            )

        # Deduplicate subquestions (AI-derived only — template_subquestions
        # was removed when planning was made fully AI-first).
        merged_subquestions: list[str] = []
        seen_sq: set[str] = set()
        for sq in list(ai_subquestions):
            key = sq.lower().strip()[:80]
            if key and key not in seen_sq:
                merged_subquestions.append(sq)
                seen_sq.add(key)
        subquestions = merged_subquestions

        # Entity-focused short queries come FIRST so they are not cut by
        # max_query_variants when the list is later sliced.
        plan_queries = self._entity_queries(query, objective)

        # AI-reasoned queries come next — these are derived from causal
        # thinking, not generic template expansions.
        for rq in ai_strategy.get("reasoning_queries") or []:
            if rq and rq.strip():
                plan_queries.append(rq.strip()[:240])
        # Short keyword variants come BEFORE perspectives so that recency /
        # domain-specific variants (e.g. "latest", "2-adic") are not pushed
        # past the max_query_variants cutoff by the larger perspective lists.
        plan_queries.extend(self._query_variants(query, depth))
        for perspective in perspectives:
            plan_queries.extend(perspective.get("queries") or [])
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
            # AI-reasoned authoritative domains are stored so that
            # _iterative_retrieval can seed the source list with them.
            "ai_authoritative_domains": ai_strategy.get("authoritative_domains") or [],
            "ai_causal_connections": ai_strategy.get("causal_connections") or [],
        }

    def _research_perspectives(
        self,
        query: str,
        objective: str,
        depth: str,
    ) -> list[dict[str, Any]]:
        """Generate perspectives for this research objective via AI.

        The ``software_mode`` and ``math_mode`` flags are retained in the
        signature for backward compatibility but are no longer used to select
        a hardcoded list — the AI derives what angles are relevant instead.
        """
        return self._ai_generate_perspectives(query, objective, depth)

    @staticmethod
    def _entity_queries(query: str, objective: str) -> list[str]:
        lower = f"{query} {objective}".lower()
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
                "judged_site_count": 0,
                "direct_urls": [],
                "discovered_domains": [],
            }

        snapshot_path = Path(str(pc_context.get("snapshot_path") or ""))
        pc_findings = pc_context.get("pc_findings") or {}
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

        direct_urls = [
            url
            for url in self._collect_urls(pc_findings)
            if self._is_safe_public_url(url) and not self._is_search_result_url(url)
        ]
        discovered_domains = [
            str(domain).strip()
            for domain in (pc_findings.get("discovered_domains") or [])
            if str(domain).strip()
        ]
        judged_results = pc_findings.get("judged_results") or []
        if direct_urls or discovered_domains or judged_results:
            browser_context = True
        if not top_labels:
            top_labels = [
                str(label).strip()
                for label in (pc_findings.get("post_snapshot_labels") or [])
                if str(label).strip()
            ][:8]

        return {
            "available": snapshot_path.exists() or bool(pc_findings),
            "snapshot_path": str(snapshot_path).replace("\\", "/"),
            "node_count": node_count,
            "browser_context_detected": browser_context,
            "top_labels": top_labels,
            "judged_site_count": len(judged_results),
            "direct_urls": direct_urls[:6],
            "discovered_domains": discovered_domains[:6],
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
            "Find authoritative sources, direct evidence, and major uncertainties for:",
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
        )
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        distilled = DeepResearchEngine._query_core_terms(cleaned)
        return distilled[:240].strip() or cleaned[:240].strip() or objective[:240]

    @classmethod
    def _split_depth(cls, objective: str) -> tuple[str, str]:
        match = re.search(r"\[(quick|standard|multi-hour|adaptive)\]\s*", objective)
        if match is None:
            cleaned = objective.strip()
            return cls.adaptive_depth_for_objective(cleaned), cleaned
        cleaned = f"{objective[: match.start()]}{objective[match.end() :]}".strip()
        marker = match.group(1)
        if marker == "adaptive":
            return cls.adaptive_depth_for_objective(cleaned), cleaned
        return marker, cleaned

    @classmethod
    def research_depth_for_objective(cls, objective: str) -> str:
        depth, _cleaned = cls._split_depth(objective)
        return depth

    @classmethod
    def adaptive_depth_for_objective(cls, objective: str) -> str:
        """Infer research effort from task complexity using AI.

        Analyzes the objective to decide if it needs a quick lookup,
        standard research, or multi-hour deep investigation.
        """
        system = (
            "You are a research effort estimator. Analyze the research objective "
            "and decide which depth category it requires:\n"
            "- 'quick': simple lookups, single facts, or basic definitions.\n"
            "- 'standard': topics requiring cross-referencing multiple sources "
            "or basic market/technical analysis.\n"
            "- 'multi-hour': deep scientific, financial, or academic research "
            "requiring citation chasing, evidence weighing, and exhaustive foraging.\n"
            "Respond ONLY with one of the three strings: quick, standard, multi-hour."
        )
        try:
            # We use a static-like call here; in practice the orchestrator
            # would pass a client.
            raw = cls()._call_ai_text(system, f"Objective: {objective}")
            raw = raw.lower().strip()
            for depth in ("multi-hour", "standard", "quick"):
                if depth in raw:
                    return depth
        except Exception:
            pass

        # Minimal heuristic fallback if AI is unavailable.
        lower = objective.lower()
        if any(
            c in lower for c in ("research", "literature", "systematic", "exhaustive")
        ):
            return "multi-hour"
        if any(c in lower for c in ("compare", "analyze", "benchmark")):
            return "standard"
        return "quick"

    @staticmethod
    def _looks_like_simple_lookup(lower: str) -> bool:
        if len(lower.split()) <= 10 and any(
            cue in lower
            for cue in (
                "recipe",
                "how many",
                "what is",
                "who is",
                "when is",
                "weather",
                "definition",
                "syntax",
                "quick lookup",
            )
        ):
            return True
        return any(
            phrase in lower
            for phrase in (
                "find a recipe",
                "search for a recipe",
                "quick recipe",
                "one source",
                "single source",
            )
        )

    @staticmethod
    def _looks_like_comprehensive_research(lower: str) -> bool:
        comprehensive_cues = (
            "comprehensive",
            "systematic review",
            "scientific literature",
            "literature review",
            "meta-analysis",
            "full report",
            "market report",
            "s&p 500",
            "sp 500",
            "all companies",
            "exhaustive",
            "deep research",
            "state of the art",
            "regulatory landscape",
        )
        if any(cue in lower for cue in comprehensive_cues):
            return True
        return (
            sum(
                1
                for cue in (
                    "compare",
                    "rank",
                    "sources",
                    "evidence",
                    "risks",
                    "limitations",
                    "opportunities",
                    "benchmarks",
                )
                if cue in lower
            )
            >= 3
        )

    @staticmethod
    def _settings_for_depth(depth: str) -> ResearchSettings:
        if depth == "quick":
            return ResearchSettings(
                depth="quick",
                max_sources=6,
                per_provider=4,
                max_query_variants=2,
            )
        if depth == "multi-hour":
            return ResearchSettings(
                depth="multi-hour",
                max_sources=72,
                per_provider=24,
                max_query_variants=18,
            )
        return ResearchSettings(
            depth="standard",
            max_sources=18,
            per_provider=10,
            max_query_variants=8,
        )

    @staticmethod
    def _settings_for_current_web(settings: ResearchSettings) -> ResearchSettings:
        if settings.depth == "multi-hour":
            return ResearchSettings(
                depth=settings.depth,
                max_sources=max(settings.max_sources, 72),
                per_provider=max(settings.per_provider, 24),
                max_query_variants=max(settings.max_query_variants, 18),
            )
        if settings.depth == "standard":
            return ResearchSettings(
                depth=settings.depth,
                max_sources=max(settings.max_sources, 24),
                per_provider=max(settings.per_provider, 12),
                max_query_variants=max(settings.max_query_variants, 10),
            )
        return ResearchSettings(
            depth=settings.depth,
            max_sources=max(settings.max_sources, 12),
            per_provider=max(settings.per_provider, 8),
            max_query_variants=max(settings.max_query_variants, 4),
        )

    @staticmethod
    def _current_web_targets(depth: str) -> dict[str, int | float]:
        if depth == "multi-hour":
            return {
                "max_retrieval_passes": 28,
                "depth_pass_floor": 10,
                "max_low_novelty_streak": 6,
                "min_perspective_count": 5,
                "min_perspective_ratio": 0.75,
            }
        if depth == "standard":
            return {
                "max_retrieval_passes": 8,
                "depth_pass_floor": 4,
                "min_perspective_count": 4,
                "min_perspective_ratio": 0.7,
            }
        return {
            "max_retrieval_passes": 3,
            "depth_pass_floor": 2,
        }

    @classmethod
    def _current_web_target_overrides(
        cls,
        targets: dict[str, Any],
        depth: str,
    ) -> dict[str, Any]:
        merged = dict(targets)
        for key, value in cls._current_web_targets(depth).items():
            if key == "max_retrieval_passes":
                # Preserve a higher planning-derived pass budget — only raise,
                # never lower.  Planning may have set 48 for multi-hour jobs;
                # the current-web override should be treated as a floor, not a cap.
                merged[key] = max(int(merged.get(key) or 0), int(value))
            else:
                merged[key] = value

        # Current-web tasks often rely on news/market/web signals where
        # strict scholarly thresholds can be unattainable and cause false
        # coverage-gate failures.  gemini-flash is a quality enhancer for
        # current-evidence queries, but when it is rate-limited the run should
        # still complete on web-search alone — so min_provider_count is always
        # capped at 1, making diversity best-effort rather than a hard gate.
        provider_cap = 1
        merged["min_provider_count"] = min(
            int(merged.get("min_provider_count") or provider_cap),
            provider_cap,
        )
        merged["min_provider_count"] = max(int(merged["min_provider_count"]), 1)
        # Current-web retrieval commonly yields fewer high-signal, on-topic
        # sources than broad literature mode. Keep source-count thresholds
        # ambitious but attainable so coverage gates do not become impossible.
        source_floor = 18 if depth == "multi-hour" else 8 if depth == "standard" else 4
        merged["min_source_count"] = min(
            int(merged.get("min_source_count") or source_floor),
            source_floor,
        )
        merged["min_source_count"] = max(int(merged["min_source_count"]), 1)
        merged["min_scholarly_sources"] = 0
        merged["min_novelty_rate"] = 0.0
        return merged

    @classmethod
    def _query_variants(cls, query: str, depth: str = "standard") -> list[str]:
        """Return query variants generated from objective terms, not fixed templates."""
        ai_variants = cls._ai_query_variants(query, depth)
        if ai_variants:
            return ai_variants

        core = cls._query_core_terms(query)
        if not core:
            return []

        max_variants = 4 if depth == "quick" else 8 if depth == "standard" else 14
        axes = [
            "primary evidence",
            "methodology",
            "counterevidence",
            "uncertainty analysis",
            "independent verification",
            "limitations",
            "comparative analysis",
            "longitudinal data",
        ]
        math_mode = cls._looks_like_math_query(query)
        if cls._looks_like_current_evidence_query(query):
            axes = [
                "latest evidence",
                "current analysis",
                "timeline",
                "near-term drivers",
                "risk scenarios",
                "independent verification",
                "counterevidence",
                "uncertainty analysis",
            ]
        if math_mode:
            axes = [
                "theorem barrier",
                "transfer mechanism",
                "counterexample search",
                "formal verification",
                "independent verification",
                "limitations",
            ]

        anchors: list[str] = []
        anchors.extend(sorted(cls._entity_terms_from_query(query)))
        for keyword in cls._keywords(query):
            if len(keyword) < 4:
                continue
            if keyword in anchors:
                continue
            anchors.append(keyword)
            if len(anchors) >= 8:
                break
        if math_mode:
            for focus in cls._math_focus_terms(query):
                focus_term = str(focus).strip().lower()
                if not focus_term or focus_term in anchors:
                    continue
                anchors.append(focus_term)
                if len(anchors) >= 12:
                    break
        if core not in anchors:
            anchors.insert(0, core)

        variants: list[str] = [core]
        for anchor in anchors[:4]:
            for axis in axes:
                variants.append(f"{anchor} {axis}")
                if len(variants) >= max_variants * 3:
                    break
            if len(variants) >= max_variants * 3:
                break

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            candidate = variant[:120].strip()
            if not candidate:
                continue
            if cls._is_low_signal_query_variant(candidate, query):
                continue
            if cls._is_noisy_query_variant(candidate, query):
                continue
            normalized = cls._normalize_title(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate)
            if len(deduped) >= max_variants:
                break
        return deduped

    @classmethod
    def _ai_query_variants(cls, query: str, depth: str = "standard") -> list[str]:
        """Optional AI variant generator.

        This class-level helper intentionally returns an empty list by default,
        allowing instance-level refinement and evidence-gap analysis to remain
        the primary adaptive path when no external model is available.
        """
        del query, depth
        return []

    @staticmethod
    def _query_core_terms(query: str) -> str:
        """Distill long prompts into domain terms while removing orchestration boilerplate."""
        prefixes = (
            "Find authoritative sources, direct evidence, and major uncertainties for:",
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
            r"\bperform(?:ing)? deep research on\b",
            r"\bperform research on\b",
            r"\busing all available [^.;]+",
            r"\bbrowser or pc evidence [^.;]+",
            r"\bdurable artifacts\b",
            r"\bproduce (?:a|an) [^.;]+ report\b",
            r"\bdo not use (?:a|an) fixed template\b",
            r"\badapt depth and effort [^.;]+",
            r"\baccepted literature\b",
            r"\bplausible proof strategies\b",
            r"\bfocusing on\b",
            r"\bthe exact missing\b",
            r"\bas anchor sources\b",
            r"\bthen expand outward to\b",
            r"\bcorroborat\w*\b",
        )
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:-")
        if not cleaned:
            return ""

        if len(cleaned) > 140:
            stop = {
                "what",
                "which",
                "when",
                "where",
                "why",
                "how",
                "the",
                "a",
                "an",
                "and",
                "or",
                "for",
                "to",
                "with",
                "from",
                "into",
                "about",
                "using",
                "need",
                "please",
                "make",
                "build",
                "create",
                "do",
                "does",
                "did",
                "can",
                "could",
                "should",
                "would",
                "have",
                "has",
                "had",
                "being",
                "been",
            }
            words = [
                token.strip("?.,!:;")
                for token in cleaned.split()
                if token.strip("?.,!:;") and token.strip("?.,!:;").lower() not in stop
            ]
            if words:
                cleaned = " ".join(words)

        return cleaned[:240].strip().lower()

    @staticmethod
    def _looks_like_current_evidence_query(query: str) -> bool:
        lower = query.lower()
        return any(
            cue in lower
            for cue in (
                "as of now",
                "right now",
                "current",
                "currently",
                "latest",
                "today",
                "recent",
                "near-term",
                "newest",
                "this week",
                "this month",
                "this year",
                "market",
            )
        )

    @staticmethod
    def _looks_like_academic_query(query: str) -> bool:
        lower = query.lower()
        return any(
            cue in lower
            for cue in (
                "scientific literature",
                "literature review",
                "systematic review",
                "meta-analysis",
                "peer-reviewed",
                "scholarly",
                "paper",
                "papers",
                "pubmed",
                "openalex",
            )
        )

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
    # Maximum proportion of final selected sources from any single provider.
    _MAX_PROVIDER_FRACTION = 0.5

    # Scoring weights for scholarly sources (openalex/semantic-scholar/crossref).
    # Named so they can be understood and adjusted without hunting for magic numbers.
    _SCORE_W_RELEVANCE: float = 54.0
    _SCORE_W_CITATION: float = 26.0
    _SCORE_W_RECENCY: float = 8.0
    _SCORE_W_CREDIBILITY: float = 18.0
    _SCORE_W_CONTRADICTION: float = 6.0

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
        combined = cls._strip_dom_noise_tokens(f"{source.title} {source.abstract}")
        if source.provider == "web-search":
            lower_combined = combined.lower()
            if any(
                marker in lower_combined
                for marker in (
                    "javascript is disabled",
                    "verify that you're not a robot",
                    "verify you are not a robot",
                    "captcha",
                    "access denied",
                    "pardon our interruption",
                    "--wp--preset--aspect-ratio",
                    "@charset",
                )
            ):
                source.quality_flags = [*(source.quality_flags or []), "bot-wall"]
                source.score = 0.0
                return 0.0
            unavailable = "snippet-unavailable" in (source.quality_flags or [])
            ticker_hits = len(
                _extract_ticker_candidates(f"{source.title} {source.abstract}")
            )
            if unavailable and ticker_hits == 0:
                source.quality_flags = [*(source.quality_flags or []), "low-signal-web"]
                source.score = 0.0
                return 0.0
        if cls._has_dom_noise_pattern(source.title) and cls._has_dom_noise_pattern(
            source.abstract
        ):
            source.quality_flags = [*(source.quality_flags or []), "dom-noise"]
            source.score = 0.0
            return 0.0
        # Allow sources that directly contain an entity term even when the
        # deterministic anchor-overlap gate would reject them (e.g. a repo
        # that is named after one of the queried systems but whose abstract
        # doesn't also contain a second anchor word).
        entity_terms_pre = cls._entity_terms_from_query(query)
        haystack_pre = combined.lower()
        entity_hits_pre = sum(1 for t in entity_terms_pre if t in haystack_pre)
        gate_allowed = cls._passes_deterministic_semantic_gate(combined, query) or (
            entity_hits_pre >= 1
        )
        if not gate_allowed:
            source.quality_flags = [*(source.quality_flags or []), "off-topic"]
            source.score = 0.0
            return 0.0
        haystack = combined.lower()
        objective_alignment = cls._objective_alignment_score(haystack, query)
        entity_hits = sum(1 for t in entity_terms if t in haystack)
        entity_relevance = entity_hits / max(len(entity_terms), 1)
        distinctive_hits = sum(1 for t in distinctive_terms if t in haystack)
        term_relevance = distinctive_hits / max(len(distinctive_terms), 1)
        relevance = max(entity_relevance, term_relevance, objective_alignment)
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
        if objective_alignment < 0.22:
            source.quality_flags.append("off-topic")
        if source.provider == "gemini-flash":
            source.relevance = max(relevance, 0.65)
            source.credibility_score = max(credibility_score, 0.7)
            source.evidence_grade = "tool-observation"
            base = 80.0 + recency * 6.0 - contradiction * 4.0
            source.score = base
            return base
        if source.provider == "web-search":
            # Web sources: relevance + recency are the primary quality signals.
            # Citation count is meaningless for news / market / general-web
            # pages, so we exclude it from the formula.  Recency weight is
            # doubled relative to the scholarly formula because timeliness is
            # web-search's unique contribution that academic sources cannot
            # provide.  The resulting score range is competitive with scholarly
            # so that a highly relevant news article is not automatically
            # outranked by a tangentially related academic paper.
            if relevance <= 0.0 and objective_alignment < 0.15:
                source.score = 0.0
                return 0.0
            base = (
                relevance * cls._SCORE_W_RELEVANCE
                + recency * cls._SCORE_W_RECENCY * 2.0
                + credibility_score * cls._SCORE_W_CREDIBILITY
                - contradiction * cls._SCORE_W_CONTRADICTION
                - credibility_penalty
            )
            source.score = max(base, 0.0)
            source.evidence_grade = cls._evidence_grade(source)
            return source.score
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
            if objective_alignment < 0.22 and entity_hits == 0:
                source.score = 0.0
                return 0.0
            effective_relevance = max(relevance, 0.1 if entity_hits else 0.0)
            base = (
                effective_relevance * cls._SCORE_W_RELEVANCE
                + citation_strength * cls._SCORE_W_CITATION
                + recency * cls._SCORE_W_RECENCY
                + credibility_score * cls._SCORE_W_CREDIBILITY
                - contradiction * cls._SCORE_W_CONTRADICTION
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
            if objective_alignment < 0.2 and entity_hits == 0:
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
        if objective_alignment < 0.12:
            base = max(base - 12.0, 0.0)
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
            and source.provider not in {"web-search", "gemini-flash"}
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

        # Detect speculative proof claims: zero-citation recent papers whose
        # title or abstract claims a complete proof of a known open problem.
        speculative_proof_patterns = (
            "proof of the",
            "proves the",
            "proves that",
            "conjecture is true",
            "conjecture is solved",
            "conjecture is proved",
            "conjecture is proven",
            "we prove the conjecture",
            "we have proved",
            "has been proved completely",
            "completely for all positive integers",
            "completely for all integers",
        )
        if source.citation_count == 0 and any(
            p in title or p in abstract for p in speculative_proof_patterns
        ):
            penalty += 10.0
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
        for provider_sources in by_provider.values():
            provider_sources.sort(key=lambda source: source.score, reverse=True)

        selected: list[ResearchSource] = []
        provider_minimum = 3 if max_sources >= 18 else 2

        # Stabilize provider diversity early with a round-robin pass.
        provider_order = sorted(
            by_provider,
            key=lambda provider: (-(by_provider[provider][0].score), provider),
        )
        while len(selected) < max_sources:
            progressed = False
            represented = {item.provider for item in selected}
            for provider in provider_order:
                bucket = by_provider.get(provider) or []
                if not bucket:
                    continue
                if len(represented) >= provider_minimum and provider not in represented:
                    continue
                selected.append(bucket.pop(0))
                represented.add(provider)
                progressed = True
                if len(selected) >= max_sources:
                    break
            if not progressed:
                break

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

        # For current-evidence tasks, preserve at least one tool observation
        # when available so runs do not collapse into a single-provider web
        # monoculture.
        if DeepResearchEngine._looks_like_current_evidence_query(
            query
        ) and not DeepResearchEngine._looks_like_academic_query(query):
            append_preferred(
                "gemini-flash",
                lambda source: (
                    DeepResearchEngine._objective_alignment_score(
                        f"{source.title} {source.abstract}",
                        query,
                    )
                    >= 0.22
                ),
            )

        # Fill remaining slots by global ranking while avoiding duplicates.
        selected_urls = {s.url for s in selected}
        for source in capped:
            if len(selected) >= max_sources:
                break
            if source.url in selected_urls:
                continue
            if "off-topic" in (source.quality_flags or []):
                continue
            if (
                DeepResearchEngine._objective_alignment_score(
                    f"{source.title} {source.abstract}",
                    query,
                )
                < 0.22
            ):
                continue
            if source.relevance < 0.1 and source.credibility_score < 0.3:
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
        # For web-search sources on current-evidence queries the primary
        # signal is relevance alone (no citation count).  Lower the threshold
        # so that on-topic news / market pages are not universally "weak".
        if (
            source.provider == "web-search"
            and source.relevance >= 0.35
            and source.credibility_score >= 0.35
        ):
            return "moderate"
        return "weak"

    @staticmethod
    def _objective_anchor_terms(query: str) -> set[str]:
        stopwords = {
            "about",
            "across",
            "after",
            "also",
            "analysis",
            "and",
            "best",
            "current",
            "for",
            "from",
            "highest",
            "latest",
            "now",
            "potential",
            "research",
            "review",
            "soar",
            "that",
            "the",
            "their",
            "these",
            "this",
            "through",
            "today",
            "using",
            "what",
            "with",
        }
        return {
            token
            for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", query.lower())
            if token not in stopwords
        }

    @classmethod
    def _objective_alignment_score(cls, text: str, query: str) -> float:
        anchors = cls._objective_anchor_terms(query)
        if not anchors:
            return 1.0
        words = {token for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", text.lower())}
        if not words:
            return 0.0
        overlap = len(anchors & words)
        # Reward overlap but stay conservative unless there are multiple matches.
        return min(overlap / max(min(len(anchors), 4), 1), 1.0)

    @classmethod
    def _is_low_signal_query_variant(cls, variant: str, query: str = "") -> bool:
        lower = variant.lower().strip()
        if not lower:
            return True
        if cls._has_dom_noise_pattern(variant):
            return True
        if "http://" in lower or "https://" in lower or "www." in lower:
            return True
        # Reject malformed search-operator fragments produced by AI hallucination
        # e.g. "stocks right site- first content", "best undervalued site- first"
        if re.search(r"\bsite[-:]\s", lower):
            return True
        # Reject scraped navigation/UI noise suffixes
        if re.search(r"\b(before content|first content|site-first)\b", lower):
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
            # JS/browser error page tokens
            "javascript",
            "function",
            "pardon",
            "captcha",
            "cloudflare",
            "cookies",
            "forbidden",
            "interruption",
            "redirect",
            # CSS layout/style tokens observed in scraped page noise
            "wrapper",
            "nreum",
            "prototype",
            "exports",
            "font-weight",
            "font-size",
            "overflow",
            "margin",
        }
        words = re.findall(r"\b[a-z0-9.-]+\b", lower)
        if len(words) < 2:
            return True
        if query and cls._objective_alignment_score(lower, query) < 0.2:
            return True
        noise_hits = sum(1 for word in words if word in noise_tokens)
        return noise_hits >= 2 and noise_hits >= max(2, len(words) // 2)

    @classmethod
    def _is_noisy_query_variant(cls, variant: str, query: str = "") -> bool:
        lower = variant.lower().strip()
        if cls._has_dom_noise_pattern(variant):
            return True
        words = re.findall(r"\b[a-z][a-z0-9-]{1,}\b", lower)
        if len(words) < 2:
            return True

        software_mode = cls._looks_like_software_agent_query(query)
        if not software_mode:
            code_noise_tokens = {
                "const",
                "navigator",
                "document",
                "window",
                "javascript",
                "typescript",
                "react",
                "css",
                "html",
                "webpack",
                "npm",
                "node",
                "function",
                "pardon",
                "captcha",
                "cloudflare",
                "forbidden",
                "browser",
                # CSS layout/style tokens seen in scraped page noise
                "wrapper",
                "font-weight",
                "font-size",
                "display",
                "height",
                "width",
                "padding",
                "margin",
                "overflow",
                # JS analytics / error page tokens
                "nreum",
                "prototype",
                "exports",
                "molluscum",  # medical spam that appeared in scraped content
                "lesions",
                "optomechanics",  # irrelevant physics from scraped abstracts
                # Yahoo Finance / CMS layout artifacts seen in runtime contamination
                "storywithleadvideo",
                "storywith",
                "leadvideo",
                "flexi",
                "nimbus",
                "calendar",
            }
            if sum(1 for word in words if word in code_noise_tokens) >= 1:
                return True

        query_tokens = set(re.findall(r"\b[a-z][a-z0-9-]{2,}\b", query.lower()))
        rare_letters = set("qxzjkvwy")
        for word in words:
            if len(word) < 6 or word in query_tokens:
                continue
            rare_ratio = sum(1 for ch in word if ch in rare_letters) / len(word)
            if rare_ratio >= 0.5:
                return True

        anchors = cls._objective_anchor_terms(query)
        if anchors and not any(anchor in lower for anchor in anchors):
            return True
        return False

    @staticmethod
    def _looks_like_software_agent_query(query: str) -> bool:
        lower = query.lower()
        explicit_markers = (
            "agentos",
            "computer use",
            "desktop agent",
            "github",
            "local pc",
            "openclaw",
            "opencode",
            "openhands",
            "orchestrator",
            "pc agent",
            "research agent",
            "software agent",
            "deep research agent",
        )
        if any(marker in lower for marker in explicit_markers):
            return True
        return bool(
            re.search(
                (
                    r"\b(build|implement|implementation|architecture|runtime|"
                    r"framework|sdk|api|repository|repositories|open[- ]source|"
                    r"code|package|library)\b"
                ),
                lower,
            )
            and re.search(
                r"\b(agent|orchestrator|desktop|browser|workflow|tool)\b",
                lower,
            )
        )

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
        """Extract high-signal mathematical focus terms from the objective text."""
        tokens = re.findall(r"\b[a-zA-Z0-9-]{3,}\b", query.lower())
        keepers: list[str] = []
        for token in tokens:
            if token in {
                "collatz",
                "conjecture",
                "theorem",
                "lemma",
                "proof",
                "bridge",
                "density",
                "verification",
                "transfer",
                "ostrowski",
                "residue",
                "2-adic",
            }:
                if token not in keepers:
                    keepers.append(token)
        compounds = (
            "2-adic conjugacy",
            "critical density",
            "pointwise transfer",
            "finite verification",
            "almost all",
            "return-block cocycles",
        )
        lower = query.lower()
        for phrase in compounds:
            if phrase in lower and phrase not in keepers:
                keepers.append(phrase)
        return keepers[:12]

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
        weak_ratio = sum(
            1 for source in sources if source.evidence_grade == "weak"
        ) / max(
            len(sources),
            1,
        )
        off_topic_ratio = sum(
            1 for source in sources if "off-topic" in (source.quality_flags or [])
        ) / max(len(sources), 1)
        avg_credibility = sum(source.credibility_score for source in sources) / max(
            len(sources),
            1,
        )
        avg_relevance = sum(source.relevance for source in sources) / max(
            len(sources),
            1,
        )
        contradiction = max(
            (source.contradiction_risk for source in sources), default=0.0
        )
        confidence = 0.44 + min(len(sources), 12) * 0.018
        confidence += min(provider_count, 6) * 0.035
        confidence += citation_bonus / 7000
        confidence += max(avg_credibility - 0.45, 0.0) * 0.16
        confidence += max(avg_relevance - 0.4, 0.0) * 0.12
        confidence -= weak_ratio * 0.28
        confidence -= off_topic_ratio * 0.24
        confidence -= contradiction * 0.1
        return max(0.2, min(confidence, 0.89))


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
