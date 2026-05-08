from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from .browser_pool import HeadlessBrowserWorkerPool as _HeadlessBrowserWorkerPool
from .models import (
    ResearchSettings,
    ResearchSource,
    extract_ticker_candidates as _extract_ticker_candidates,
)


class ResearchRetrievalMixin:
    def _iterative_retrieval(
        self,
        query: str,
        settings: ResearchSettings,
        plan: dict[str, Any],
        targets: dict[str, Any],
        pc_context: dict[str, Any] | None = None,
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
        all_sources.extend(self._pc_finding_seed_sources(pc_context))
        all_sources.extend(
            self._claim_persistent_crawl_sources(
                query,
                plan.get("core_question") or query,
                limit=min(24, settings.max_sources),
                exclude_urls=list(plan.get("source_seeds") or []),
            )
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
        min_depth_passes = self._min_depth_passes(settings.depth, targets)
        max_low_novelty_streak = self._max_low_novelty_streak(
            settings.depth,
            targets,
        )
        max_low_marginal_yield_streak = self._max_low_marginal_yield_streak(
            settings.depth,
            targets,
        )
        marginal_yield_floor = self._marginal_yield_floor(
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
        pass_cap = 480 if settings.depth == "multi-hour" else 240
        max_passes = max(1, min(max_passes, pass_cap))
        effective_novelty_threshold = self._effective_novelty_threshold(
            settings.depth,
            targets,
        )
        started_at = time.monotonic()
        retrieval_passes: list[dict[str, Any]] = []
        previous_titles: set[str] = set()
        previous_unique_url_count = 0
        selected_domain_counts: dict[str, int] = {}
        low_novelty_streak = 0
        low_marginal_yield_streak = 0
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

            # --- PARALLEL SEARCH: run all queries in this pass concurrently ---
            # This mirrors how Claude/Gemini deep research fires many searches
            # in parallel rather than waiting for each one sequentially.
            # Write the first heartbeat for UX feedback before diving in.
            if pass_variants:
                self._write_retrieval_heartbeat(
                    run_id,
                    settings.depth,
                    pass_index,
                    pass_variants[0],
                    1,
                    len(pass_variants),
                    retrieval_passes,
                    started_at,
                )
            pass_sources: list[ResearchSource] = []
            parallel_workers = self._query_parallel_worker_count(
                settings.depth,
                len(pass_variants),
                len(allowed_providers),
            )
            if parallel_workers > 1 and len(pass_variants) > 1:

                def _search_one(sq: str) -> list[ResearchSource]:
                    previous = getattr(
                        self._provider_parallelism_context,
                        "disable_nested_provider_parallelism",
                        False,
                    )
                    setattr(
                        self._provider_parallelism_context,
                        "disable_nested_provider_parallelism",
                        True,
                    )
                    try:
                        return self._search_query_across_providers(
                            sq,
                            allowed_providers,
                            settings.per_provider,
                        )
                    finally:
                        setattr(
                            self._provider_parallelism_context,
                            "disable_nested_provider_parallelism",
                            previous,
                        )

                with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                    for batch_result in pool.map(_search_one, pass_variants):
                        pass_sources.extend(batch_result)
            else:
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
            all_sources.extend(pass_sources)

            # --- INJECT URL-CHAINED SOURCES from previous enrichment pass ---
            # After _enrich_top_sources runs, it deposits outbound links into
            # self._chained_sources.  Pull them in here so they get scored and
            # potentially enriched on the next pass — this is the core engine
            # of the 1000+ URL fetch mechanism.
            chained = getattr(self, "_chained_sources", [])
            if chained:
                all_sources.extend(chained)
                self._chained_sources = []

                # ── RECURSIVE CHAIN EXPANSION ─────────────────────────────────
                # This is the mechanism that takes 10 seed pages to 100k+ URLs:
                # each fetched page spawns 40-100 outbound links, each of which
                # spawns 40-100 more.  Without this step, chained sources have
                # score=0.1 and never reach the main enrichment loop.
                # We immediately enrich priority-domain chained sources so they:
                #  (a) get real content and re-score above the 0.1 floor, and
                #  (b) generate their own sub-chains added to _chained_sources
                #      for the *next* pass — creating a self-feeding expansion.
                if settings.depth == "multi-hour":
                    _priority_finance_hosts = {
                        "sec.gov",
                        "edgar.sec.gov",
                        "finance.yahoo.com",
                        "marketwatch.com",
                        "bloomberg.com",
                        "reuters.com",
                        "stockanalysis.com",
                        "macrotrends.net",
                        "morningstar.com",
                        "seekingalpha.com",
                        "wsj.com",
                        "ft.com",
                        "cnbc.com",
                        "investing.com",
                        "finviz.com",
                        "barrons.com",
                        "wisesheets.io",
                        "simplywall.st",
                        "gurufocus.com",
                        "tradingeconomics.com",
                        "multpl.com",
                        "bea.gov",
                        "federalreserve.gov",
                        "fred.stlouisfed.org",
                        "imf.org",
                        "ssrn.com",
                        "nber.org",
                        # New Wall Street analyst additions
                        "openinsider.com",
                        "finra.org",
                        "bls.gov",
                        "treasury.gov",
                        "census.gov",
                        "worldbank.org",
                        "oecd.org",
                        "statista.com",
                        "zacks.com",
                        "tipranks.com",
                        "barchart.com",
                        "thestreet.com",
                        "businessinsider.com",
                        "fool.com",
                        "kiplinger.com",
                        "alphavantage.co",
                        "roic.ai",
                        "alphaquery.com",
                    }
                    priority_chain = [
                        s
                        for s in chained
                        if any(
                            h in (s.url or "").lower() for h in _priority_finance_hosts
                        )
                    ][:80]
                    # Also include non-priority chained sources for breadth.
                    seen_chain_urls = {s.url for s in priority_chain}
                    other_chain = [s for s in chained if s.url not in seen_chain_urls][
                        :40
                    ]
                    # 120 chains per pass × 120 passes = up to 14,400 chain fetches.
                    # This is the primary mechanism for 100k+ URL coverage.
                    chain_expansion_batch = (priority_chain + other_chain)[:120]
                    if chain_expansion_batch:
                        chain_eq = self._enrich_top_sources(
                            chain_expansion_batch, query
                        )
                        for cq in self._sanitize_query_variants(chain_eq, query):
                            if cq and cq not in all_variants:
                                all_variants.append(cq)

            queued_sources = self._claim_persistent_crawl_sources(
                query,
                plan.get("core_question") or query,
                limit=min(12 if pass_index > 0 else 16, settings.max_sources),
                exclude_urls=[source.url for source in all_sources if source.url],
            )
            if queued_sources:
                all_sources.extend(queued_sources)

            unique_url_count = self._unique_source_url_count(all_sources)

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
            # For multi-hour runs, enrich aggressively — parallel enrichment
            # with URL chaining will discover new sources organically.
            if settings.depth in {"standard", "multi-hour"}:
                enrich_count = (
                    min(60, settings.max_sources)
                    if settings.depth == "multi-hour"
                    else (
                        min(12, settings.max_sources)
                        if pass_index == 0
                        else min(8, settings.max_sources)
                    )
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
                for cq in self._sanitize_query_variants(content_queries, query):
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
            marginal_unique_url_gain = max(
                0,
                unique_url_count - previous_unique_url_count,
            )
            marginal_title_gain = len(new_titles)
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
                "unique_url_count": unique_url_count,
                "marginal_unique_url_gain": marginal_unique_url_gain,
                "marginal_title_gain": marginal_title_gain,
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
            previous_unique_url_count = unique_url_count
            # No artificial budget — depth is information-driven.
            depth_met = (pass_index + 1) >= min_depth_passes
            novelty_threshold = effective_novelty_threshold
            low_novelty_pass = (
                coverage["novelty_rate"] < novelty_threshold and pass_index > 0
            )
            low_marginal_yield_pass = (
                pass_index > 0
                and marginal_unique_url_gain < marginal_yield_floor["unique_urls"]
                and marginal_title_gain < marginal_yield_floor["titles"]
            )
            if low_novelty_pass:
                low_novelty_streak += 1
            else:
                low_novelty_streak = 0
            if low_marginal_yield_pass:
                low_marginal_yield_streak += 1
            else:
                low_marginal_yield_streak = 0
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
                min_unique_urls = int(targets.get("min_unique_urls") or 0)
                if min_unique_urls > 0 and unique_url_count < min_unique_urls:
                    continue
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
            if (
                low_novelty_pass
                and depth_met
                and low_novelty_streak >= max_low_novelty_streak
            ):
                stop_reason = "novelty_below_threshold"
                break
            if (
                low_marginal_yield_pass
                and depth_met
                and low_marginal_yield_streak >= max_low_marginal_yield_streak
            ):
                stop_reason = "marginal_yield_exhausted"
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
            if low_novelty_pass:
                pivot_queries = self._domain_diversification_queries(
                    plan.get("core_question") or query,
                    selected,
                    pass_index,
                    coverage,
                )
                if pivot_queries:
                    self._record_provider_diagnostic(
                        "low-novelty-pivot",
                        "triggered",
                        (
                            f"pass {pass_index + 1}: "
                            f"novelty_rate={coverage['novelty_rate']:.3f}"
                        ),
                    )
                    all_variants.extend(
                        self._sanitize_query_variants(
                            pivot_queries,
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
    def _max_low_marginal_yield_streak(
        depth: str,
        targets: dict[str, Any],
    ) -> int:
        raw_value = targets.get("max_low_marginal_yield_streak", 0)
        try:
            streak = int(raw_value)
        except (TypeError, ValueError):
            streak = 0
        if streak > 0:
            return max(streak, 1)
        if depth == "multi-hour":
            return 8
        if depth == "standard":
            return 3
        return 1

    @staticmethod
    def _marginal_yield_floor(
        depth: str,
        targets: dict[str, Any],
    ) -> dict[str, int]:
        raw_unique = targets.get("min_marginal_unique_url_gain", 0)
        raw_titles = targets.get("min_marginal_title_gain", 0)
        try:
            unique_urls = int(raw_unique)
        except (TypeError, ValueError):
            unique_urls = 0
        try:
            titles = int(raw_titles)
        except (TypeError, ValueError):
            titles = 0
        if depth == "multi-hour":
            return {
                "unique_urls": max(unique_urls, 2),
                "titles": max(titles, 2),
            }
        if depth == "standard":
            return {
                "unique_urls": max(unique_urls, 1),
                "titles": max(titles, 1),
            }
        return {
            "unique_urls": max(unique_urls, 1),
            "titles": max(titles, 1),
        }

    @classmethod
    def _fallback_research_axes(cls, query: str, depth: str) -> list[str]:
        core = cls._query_core_terms(query) or query
        max_axes = 5 if depth == "quick" else 8 if depth == "standard" else 10
        axes: list[str] = []
        seen: set[str] = set()

        for perspective in cls._generic_perspectives(query, depth):
            candidates = [
                str(perspective.get("name") or "").replace("-", " "),
                *[str(item) for item in (perspective.get("keywords") or [])],
            ]
            for candidate in candidates:
                cleaned = cls._trim_query_variant_text(candidate)
                if not cleaned:
                    continue
                normalized = cls._normalize_title(cleaned)
                if not normalized or normalized == cls._normalize_title(core):
                    continue
                if normalized in seen:
                    continue
                seen.add(normalized)
                axes.append(cleaned[:120])
                if len(axes) >= max_axes:
                    return axes

        return axes or ["primary evidence", "independent verification"]
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
            # broad web search.  Note: the gemini-flash *evidence* provider
            # is intentionally excluded — LLM parametric memory is a
            # synthesizer, not a source.
            return {"web-search", "bing-search", "google-news-rss"}

        if cls._looks_like_current_evidence_query(
            query
        ) and not cls._looks_like_academic_query(query):
            if cls._looks_like_market_query(
                query
            ) and not cls._looks_like_quant_finance_query(query):
                # Current market tasks: full Wall Street analyst stack.
                # Prioritize real-time financial data, SEC filings, earnings,
                # insider activity, short interest, macro, news feeds.
                # Bing + Google News adds multi-engine news coverage.
                # Keep crowd-sentiment/newsletter providers out of the default
                # rotation so weak wrappers do not dominate current-web runs.
                return {
                    "web-search",
                    "bing-search",
                    "financial-portals",
                    "sec-edgar",
                    "earnings-data",
                    "insider-transactions",
                    "short-interest",
                    "macrotrends",
                    "stockanalysis",
                    "fed-macro",
                    "google-news-rss",
                }
            # Include financial-portals for price/product data, and crossref for
            # reports/whitepapers that may index financial analyses.
            return {
                "web-search",
                "bing-search",
                "financial-portals",
                "sec-edgar",
                "google-news-rss",
                "crossref",
            }

        # Default scholarly stack is always included.
        selected: set[str] = {
            "openalex",
            "semantic-scholar",
            "crossref",
            "web-search",
            "bing-search",
            "google-news-rss",
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
        expanded.add("bing-search")
        expanded.add("google-news-rss")
        if cls._looks_like_market_query(query):
            expanded.update(
                {
                    "financial-portals",
                    "sec-edgar",
                    "earnings-data",
                    "macrotrends",
                    "stockanalysis",
                    "short-interest",
                    "fed-macro",
                }
            )
        if cls._looks_like_software_agent_query(query):
            expanded.add("github-repositories")
        return expanded if expanded != allowed_providers else None
    def _enrich_top_sources(
        self,
        sources: list[ResearchSource],
        query: str = "",
    ) -> list[str]:
        """Fetch each source's landing page, extend its abstract with real
        content, and return new query strings extracted from that content.

        Also chains out to outbound links found in the page — this is how
        Claude/Gemini deep research hits 1000+ URL fetches: each fetched page
        becomes a seed for more pages.  Outbound links are added as candidate
        sources so they can be scored and possibly enriched in later passes.

        This is the primary driver of genuine research runtime: every HTTP
        fetch introduces real I/O latency.  No artificial sleeps are used;
        the time cost comes entirely from network round-trips.
        """
        new_queries: list[str] = []
        self._chained_sources: list[ResearchSource] = getattr(
            self, "_chained_sources", []
        )
        existing_chained_urls = {
            str(source.url or "").strip()
            for source in self._chained_sources
            if str(source.url or "").strip()
        }
        browser_prefetch = self._headless_browser_pool_fetch(
            self._persistent_unique_urls(
                [
                    str(source.url or "")
                    for source in sources
                    if self._needs_browser(str(source.url or ""))
                ]
            ),
            max_chars=80_000,
            timeout_ms=18_000,
        )

        def _enrich_one(
            source: ResearchSource,
        ) -> tuple[list[str], list[ResearchSource], str]:
            content, raw_html, status, _ = self._fetch_source_content(
                source,
                query,
                browser_prefetch,
            )
            if status != "processed":
                return [], [], status
            extra_queries: list[str] = []
            chained: list[ResearchSource] = []
            if len(content) > 80:
                extra = content[:1200]
                if source.abstract.lower().startswith("generic web result for "):
                    source.abstract = extra[:3000]
                else:
                    source.abstract = f"{source.abstract} {extra}".strip()[:3000]
                extra_queries.extend(
                    self._content_to_new_queries(content, source.title, query)
                )
                # URL CHAINING: extract outbound links from the fetched page.
                # For browser-rendered pages, content is plain text so we pass
                # raw_html (which may also be the browser text); the link extractor
                # gracefully handles plain text (href regex won't match, returning []).
                # For HTTP-fetched pages, raw_html is the full HTML document.
                chained = self._extract_outbound_source_candidates(
                    raw_html, query, source.url
                )
            return extra_queries, chained, "processed"

        # Run enrichment in parallel, but keep the default worker count bounded
        # because each item may perform network I/O and queue writes.
        if not sources:
            return []
        worker_count = self._enrichment_parallel_worker_count(len(sources))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_enrich_one, src): src for src in sources}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    extra_qs, chained, status = future.result()
                    if status == "processed":
                        self._update_crawl_queue_status(source.url, "processed")
                    elif status not in {"", "skipped"}:
                        self._update_crawl_queue_status(source.url, "failed", status)
                    new_queries.extend(extra_qs)
                    if chained:
                        self._enqueue_url_batch(
                            [candidate.url for candidate in chained],
                            query,
                            self._active_run_id,
                            source_url=source.url,
                            priority=max(6.0, float(source.score or 0.0) + 1.0),
                        )
                    for candidate in chained:
                        candidate_url = str(candidate.url or "").strip()
                        if not candidate_url or candidate_url in existing_chained_urls:
                            continue
                        existing_chained_urls.add(candidate_url)
                        self._chained_sources.append(candidate)
                except Exception:
                    self._update_crawl_queue_status(
                        source.url,
                        "failed",
                        "enrichment-exception",
                    )

        seen: set[str] = set()
        result: list[str] = []
        for q in new_queries:
            if self._is_low_signal_query_variant(q, query):
                continue
            norm = self._normalize_title(q)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(q[:80])
        # Return up to 40 new query strings — double the previous 24 cap.
        return result[:40]
    def _extract_outbound_source_candidates(
        self,
        raw_html: str,
        query: str,
        source_url: str,
    ) -> list[ResearchSource]:
        """Extract outbound links from a fetched HTML page and return them as
        candidate ResearchSource objects for the next retrieval pass.

        This is the URL-chaining mechanism that lets the engine scale from
        dozens of sources to hundreds in a multi-hour run — exactly how
        Claude/Gemini deep research expands its source pool.
        """
        candidates: list[ResearchSource] = []
        seen_urls: set[str] = set()
        source_host = urllib.parse.urlparse(source_url).netloc.lower()

        # Finance/research-grade domains worth following outbound links into.
        priority_outbound_hosts = {
            # Government / Regulatory
            "sec.gov",
            "edgar.sec.gov",
            "investor.gov",
            "finra.org",
            "federalreserve.gov",
            "bls.gov",
            "census.gov",
            "irs.gov",
            "bea.gov",
            "treasury.gov",
            "cftc.gov",
            "fdic.gov",
            "occ.gov",
            # Major financial news
            "wsj.com",
            "ft.com",
            "reuters.com",
            "bloomberg.com",
            "cnbc.com",
            "marketwatch.com",
            "barrons.com",
            "businessinsider.com",
            "thestreet.com",
            "investopedia.com",
            "fool.com",
            "kiplinger.com",
            # Investment research / data
            "seekingalpha.com",
            "finance.yahoo.com",
            "investing.com",
            "morningstar.com",
            "macrotrends.net",
            "tradingeconomics.com",
            "multpl.com",
            "simplywall.st",
            "stockanalysis.com",
            "wisesheets.io",
            "gurufocus.com",
            "finviz.com",
            "barchart.com",
            "zacks.com",
            "tipranks.com",
            "alphaquery.com",
            "roic.ai",
            # Macroeconomics / data
            "statista.com",
            "worldbank.org",
            "imf.org",
            "oecd.org",
            "federalreserve.gov",
            # Academic / research finance
            "ssrn.com",
            "nber.org",
            "arxiv.org",
            "nature.com",
            "science.org",
            "pubmed.ncbi.nlm.nih.gov",
            # Company IR pages (will match via is_same_domain_article below)
            "ir.",
            "investors.",
        }

        for match in re.finditer(
            r'href=["\']([^"\'<>\s]+)["\']',
            raw_html,
            flags=re.IGNORECASE,
        ):
            raw_href = match.group(1).strip()
            if raw_href.startswith("//"):
                raw_href = "https:" + raw_href
            elif raw_href.startswith("/"):
                parsed_source = urllib.parse.urlparse(source_url)
                raw_href = f"{parsed_source.scheme}://{parsed_source.netloc}{raw_href}"
            if not raw_href.startswith(("http://", "https://")):
                continue
            normalized = self._normalize_web_result_url(raw_href)
            if not self._is_safe_public_url(normalized):
                continue
            if normalized in seen_urls:
                continue
            link_host = urllib.parse.urlparse(normalized).netloc.lower().lstrip("www.")
            # Only follow links to:
            # 1. Priority research/finance domains
            # 2. Non-navigation outbound links from the same domain
            is_priority = any(h in link_host for h in priority_outbound_hosts)
            is_same_domain_article = (
                source_host in link_host
                and len(urllib.parse.urlparse(normalized).path) > 20
            )
            if not (is_priority or is_same_domain_article):
                continue
            # Skip links that look like site navigation (short paths, auth pages)
            path = urllib.parse.urlparse(normalized).path
            if any(
                nav in path.lower()
                for nav in [
                    "/login",
                    "/signin",
                    "/register",
                    "/cart",
                    "/checkout",
                    "/category",
                    "/tag/",
                    "/page/1",
                    "/author/",
                ]
            ):
                continue
            seen_urls.add(normalized)
            label = self._label_from_url(normalized)
            # Score chained sources meaningfully:
            # Priority finance/research domains start at 20.0 so they rank
            # high enough to enter the main enrichment loop and generate
            # further sub-chains — this is the key to the 100k+ URL tree.
            # Other domains start at 5.0 (still above the 0.1 floor).
            chain_score = 20.0 if is_priority else 5.0
            candidates.append(
                ResearchSource(
                    provider="web-search",
                    title=label,
                    url=normalized,
                    authors=[link_host] if link_host else [],
                    abstract=f"Chained from {source_url}: {label}",
                    citation_count=0,
                    score=chain_score,
                    quality_flags=["url-chained"],
                )
            )
            if len(candidates) >= 100:
                break
        return candidates
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
        # Rotate realistic browser User-Agents so financial sites don't block us.
        # A Wall Street analyst's Bloomberg terminal doesn't announce itself as a bot.
        _ua_pool = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
                "Gecko/20100101 Firefox/124.0"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.4 Safari/605.1.15"
            ),
        ]
        ua = _ua_pool[hash(url) % len(_ua_pool)]
        headers = {
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "User-Agent": ua,
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
        effective_timeout = min(timeout_seconds or self.timeout_seconds, 15)

        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError:
            return ""

        try:
            with httpx.Client(
                follow_redirects=True,
                http2=True,
                verify=True,
            ) as client:
                with client.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=effective_timeout,
                ) as response:
                    content_type = str(response.headers.get("Content-Type") or "").lower()
                    if content_type and not any(
                        marker in content_type
                        for marker in ("text/", "html", "xml", "json")
                    ):
                        return ""
                    raw = bytearray()
                    for chunk in response.iter_bytes(chunk_size=8192):
                        raw.extend(chunk)
                        if len(raw) >= max_bytes:
                            break
                    return bytes(raw[:max_bytes]).decode("utf-8", errors="replace")
        except Exception:
            return ""
    _JS_REQUIRED_HOSTS: frozenset[str] = frozenset(
        {
            "bloomberg.com",
            "wsj.com",
            "barrons.com",
            "ft.com",
            "seekingalpha.com",
            "thestreet.com",
            "businessinsider.com",
            "kiplinger.com",
            "fool.com",
            "investopedia.com",
            "nasdaq.com",
            "nypost.com",
            "cnbc.com",
            "marketwatch.com",
            "msn.com",
            "statista.com",
        }
    )
    def _needs_browser(self, url: str) -> bool:
        """Return True if this URL belongs to a JS-rendered finance/news site."""
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return any(h in host for h in self._JS_REQUIRED_HOSTS)
    def _headless_browser_pool_size(self, url_count: int) -> int:
        if url_count <= 0:
            return 0
        configured = os.environ.get(
            "AGENTOS_HEADLESS_BROWSER_WORKERS",
            "",
        ).strip()
        if configured:
            try:
                return max(1, min(int(configured), url_count))
            except ValueError:
                pass
        depth = self.research_depth_for_objective(self._active_objective)
        if depth == "multi-hour":
            return max(2, min(6, url_count))
        if self._looks_like_current_evidence_query(self._active_objective):
            return max(2, min(4, url_count))
        return max(1, min(3, url_count))
    def _enrichment_parallel_worker_count(self, source_count: int) -> int:
        if source_count <= 0:
            return 0
        configured = os.environ.get("AGENTOS_ENRICHMENT_WORKERS", "").strip()
        if configured:
            try:
                return max(1, min(int(configured), source_count))
            except ValueError:
                pass
        depth = self.research_depth_for_objective(self._active_objective)
        if depth == "multi-hour":
            return max(1, min(16, source_count))
        if self._looks_like_current_evidence_query(self._active_objective):
            return max(1, min(12, source_count))
        return max(1, min(8, source_count))
    def _new_headless_browser_bundle(self) -> dict[str, Any] | None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--window-size=1280,900",
                ],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                },
            )
            return {
                "playwright": playwright,
                "browser": browser,
                "context": context,
            }
        except Exception:
            return None
    @staticmethod
    def _close_headless_browser_bundle(bundle: dict[str, Any]) -> None:
        context = bundle.get("context")
        browser = bundle.get("browser")
        playwright = bundle.get("playwright")
        try:
            if context is not None:
                context.close()
        finally:
            try:
                if browser is not None:
                    browser.close()
            finally:
                if playwright is not None:
                    playwright.stop()
    def _render_browser_page_with_context(
        self,
        context: Any,
        url: str,
        max_chars: int,
        timeout_ms: int,
    ) -> str:
        page = context.new_page()
        page.route(
            "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,mp3,webm}",
            lambda route: route.abort(),
        )
        page.route(
            "**/{ads,analytics,tracking,doubleclick,googlesyndication}**",
            lambda route: route.abort(),
        )
        try:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_load_state(
                    "networkidle",
                    timeout=min(timeout_ms, 12_000),
                )
            except Exception:
                pass
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            try:
                text = page.inner_text("body")
            except Exception:
                try:
                    text = page.evaluate("() => document.body.innerText")
                except Exception:
                    text = ""
            return (text or "")[:max_chars]
        finally:
            try:
                page.close()
            except Exception:
                pass
    def _headless_browser_pool_fetch(
        self,
        urls: list[str],
        max_chars: int = 80_000,
        timeout_ms: int = 18_000,
    ) -> dict[str, str]:
        safe_urls = self._persistent_unique_urls(urls)
        if not safe_urls:
            return {}
        worker_count = self._headless_browser_pool_size(len(safe_urls))
        if worker_count <= 0:
            return {}
        pool = _HeadlessBrowserWorkerPool(
            worker_count=worker_count,
            bundle_factory=self._new_headless_browser_bundle,
            render_with_context=self._render_browser_page_with_context,
            bundle_cleanup=self._close_headless_browser_bundle,
        )
        return pool.render_many(safe_urls, max_chars, timeout_ms)
    def _get_text_browser(
        self,
        url: str,
        max_chars: int = 80_000,
        timeout_ms: int = 18_000,
    ) -> str:
        """Render *url* with a headless Chromium browser and return visible text.

        Uses Playwright sync API. Falls back gracefully to empty string if
        Playwright is not installed or the page cannot be rendered.

        This is the core mechanism for extracting content from JS-heavy finance
        sites (Bloomberg, WSJ, SeekingAlpha, FT, Barron's) that return blank
        or paywalled HTML to plain HTTP requests.
        """
        bundle = self._new_headless_browser_bundle()
        if not bundle:
            return ""
        try:
            return self._render_browser_page_with_context(
                bundle["context"],
                url,
                max_chars,
                timeout_ms,
            )
        except Exception:
            return ""
        finally:
            self._close_headless_browser_bundle(bundle)
    def _get_text_with_browser_fallback(
        self,
        url: str,
        max_bytes: int = 60_000,
        timeout_seconds: int | None = None,
    ) -> str:
        """Fetch url using browser if JS rendering is needed, otherwise requests.

        This is the unified entry point that every enrichment call should use
        for finance/news URLs. It guarantees real content from JS-heavy sites.
        """
        if self._needs_browser(url):
            content = self._get_text_browser(url, max_chars=max_bytes)
            if content and self._text_signal_score(content) >= 0.1:
                return content
        # Standard HTTP fetch (also used as fallback when browser yields nothing).
        return self._get_text(
            url,
            accept="text/html,application/xhtml+xml,*/*",
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )
    def _should_retry_with_browser(
        self,
        url: str,
        raw_html: str,
        query: str,
    ) -> bool:
        if self._needs_browser(url):
            return True
        lower = raw_html.lower()
        blocked_markers = (
            "please enable javascript",
            "javascript is required",
            "this site requires javascript",
            "cloudflare",
            "captcha",
            "access denied",
            "forbidden",
            "pardon our interruption",
            "are you a bot",
        )
        if any(marker in lower for marker in blocked_markers):
            return True
        text = self._html_to_text(raw_html)
        signal = self._text_signal_score(text)
        if signal < 0.08:
            return True
        if query:
            anchors = set(self._keywords(query)) | set(
                self._entity_terms_from_query(query)
            )
            if anchors and not any(anchor in text.lower() for anchor in anchors):
                return len(re.findall(r"\b[a-z]{4,}\b", text.lower())) < 90
        return False
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
    def _looks_like_public_security_query(query: str) -> bool:
        lower = query.lower()
        security_tokens = {
            "stock",
            "stocks",
            "share",
            "shares",
            "equity",
            "equities",
            "ticker",
            "price target",
            "wall street",
            "public company",
            "public companies",
            "public securities",
            "portfolio",
            "upside",
            "downside",
        }
        return any(token in lower for token in security_tokens)
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
    def _has_market_identifiers(text: str) -> bool:
        if _extract_ticker_candidates(text):
            return True
        return bool(
            re.search(
                r"\b[A-Z][A-Za-z&.\-]{1,}\s+(?:Inc|Corp|Corporation|Ltd|PLC|Group|Holdings|Technologies|Energy|Pharma|Bank)\b",
                text or "",
            )
        )
    @classmethod
    def _has_actionable_market_signal(cls, text: str) -> bool:
        lower = (text or "").lower()
        if cls._has_market_identifiers(text):
            return True
        actionable_markers = (
            "earnings",
            "revenue",
            "margin",
            "guidance",
            "valuation",
            "price target",
            "free cash flow",
            "cash flow",
            "ev/ebitda",
            "ebitda",
            "p/e",
            "eps",
            "10-k",
            "10-q",
            "8-k",
            "sec filing",
            "analyst rating",
            "analyst estimate",
            "short interest",
            "insider buying",
            "insider selling",
            "institutional ownership",
            "catalyst",
            "buyback",
            "dividend",
        )
        return any(marker in lower for marker in actionable_markers)
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
                r"nimbus|progressive-advanced|window\.initiali18nstore|"
                r"app\.account\.recovery|check your spam folder)\b"
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
                "window.initiali18nstore",
                "app.account.recovery",
                "check your spam folder",
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
        entity_hits = cls._entity_hit_count(sample, query)
        if entity_hits:
            return not cls._has_dom_noise_pattern(sample)
        if cls._looks_like_market_query(query):
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
            has_market = cls._has_market_signal(sample)
            actionable_market = cls._has_actionable_market_signal(sample)
            has_offdomain = any(token in sample.lower() for token in offdomain_vocab)
            if has_offdomain and not has_market:
                return False
            if not has_market and overlap < 2:
                return False
            if (
                cls._looks_like_public_security_query(query)
                and not actionable_market
                and overlap < 2
            ):
                return False
            return (
                overlap >= 1 or actionable_market
            ) and not cls._has_dom_noise_pattern(sample)
        if not anchors:
            return False
        return overlap >= min(2, len(anchors)) and not cls._has_dom_noise_pattern(
            sample
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
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError:
            return {}

        try:
            with httpx.Client(
                follow_redirects=True,
                http2=True,
                verify=True,
            ) as client:
                response = client.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "agentos-orchestrator/0.1",
                    },
                    timeout=self.timeout_seconds,
                )
                return json.loads(response.text)
        except (Exception, json.JSONDecodeError):
            return {}
