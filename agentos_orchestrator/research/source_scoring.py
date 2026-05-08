from __future__ import annotations

import re
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from .models import ResearchSource, extract_ticker_candidates as _extract_ticker_candidates
from .query_policy import generic_query_terms as _generic_query_terms_policy


class ResearchSourceScoringMixin:
    @staticmethod
    def _looks_like_current_evidence_query(query: str) -> bool:
        lower = query.lower()
        return any(
            cue in lower
            for cue in (
                "as of now",
                "as now",
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
    @classmethod
    def _dedupe_sources(cls, sources: list[ResearchSource]) -> list[ResearchSource]:
        by_identity: dict[str, ResearchSource] = {}
        deduped: list[ResearchSource] = []
        for source in sources:
            keys = cls._source_identity_keys(source)
            existing = next(
                (by_identity[key] for key in keys if key in by_identity),
                None,
            )
            if existing is None:
                deduped.append(source)
                for key in keys:
                    by_identity[key] = source
                continue

            cls._merge_source_records(existing, source)
            for key in keys:
                by_identity[key] = existing
            for key in cls._source_identity_keys(existing):
                by_identity[key] = existing
        return deduped
    @classmethod
    def _merge_source_records(cls, existing: ResearchSource, source: ResearchSource) -> None:
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

        if cls._abstract_quality(
            source.abstract
        ) > cls._abstract_quality(existing.abstract):
            existing.abstract = source.abstract
        existing.citation_count = max(existing.citation_count, source.citation_count)
        existing.score = max(existing.score, source.score)
    @classmethod
    def _source_identity_keys(cls, source: ResearchSource) -> list[str]:
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
        title_key = cls._normalize_title(source.title)
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
    _MAX_PROVIDER_FRACTION = 0.5
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
        filtered = [
            source
            for score, source in scored
            if score > 0.0
            and "off-topic" not in (source.quality_flags or [])
            and cls._source_is_on_topic(source, query)
        ]
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
                    "window.initiali18nstore",
                    "app.account.recovery",
                    "check your spam folder",
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
            if cls._looks_like_market_query(query) and not cls._has_market_identifiers(
                f"{source.title} {source.abstract}"
            ):
                if cls._objective_alignment_score(combined, query) < 0.30:
                    source.quality_flags = [
                        *(source.quality_flags or []),
                        "market-nonspecific-web",
                    ]
                    source.score = 0.0
                    return 0.0
            if cls._looks_like_market_query(query) and any(
                flag in (source.quality_flags or [])
                for flag in (
                    "promo-market-listicle",
                    "low-signal-market-host",
                    "missing-market-identifiers",
                )
            ):
                if cls._objective_alignment_score(combined, query) < 0.45:
                    source.quality_flags = [
                        *(source.quality_flags or []),
                        "low-signal-market-web",
                    ]
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
        if cls._looks_like_public_security_query(
            query
        ) and cls._has_actionable_market_signal(combined):
            relevance = max(relevance, 0.35)
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
        source.quality_flags = list(
            dict.fromkeys([*(source.quality_flags or []), *quality_flags])
        )
        if (
            objective_alignment < 0.22
            and entity_hits == 0
            and not (
                cls._looks_like_public_security_query(query)
                and cls._has_actionable_market_signal(combined)
            )
        ):
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
            if cls._looks_like_market_query(query) and not cls._has_market_identifiers(
                f"{source.title} {source.abstract}"
            ):
                base -= 8.0
            source.score = max(base, 0.0)
            source.evidence_grade = cls._evidence_grade(source)
            if cls._looks_like_market_query(query) and any(
                flag in (source.quality_flags or [])
                for flag in (
                    "missing-market-identifiers",
                    "promo-market-listicle",
                    "low-signal-market-host",
                )
            ):
                source.evidence_grade = "weak"
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
        if source.provider == "pc-browser-research":
            if "browser-judged-source" in (source.quality_flags or []):
                credibility += 0.1
            if "browser-navigation-seed" in (source.quality_flags or []):
                credibility += 0.06
            if "browser-terminal-verified" in (source.quality_flags or []):
                credibility += 0.16
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

        if cls._looks_like_market_query(query):
            tier_one_finance_hosts = {
                "sec.gov",
                "reuters.com",
                "bloomberg.com",
                "wsj.com",
                "ft.com",
                "spglobal.com",
                "morningstar.com",
            }
            tier_two_finance_hosts = {
                "marketwatch.com",
                "finance.yahoo.com",
                "cnbc.com",
                "fred.stlouisfed.org",
                "bea.gov",
                "bls.gov",
            }
            low_signal_market_hosts = {
                "stocktwits.com",
                "pinterest.com",
                "reddit.com",
                "quora.com",
            }
            if host in tier_one_finance_hosts:
                credibility += 0.24
            elif host in tier_two_finance_hosts:
                credibility += 0.15
            if host in low_signal_market_hosts:
                penalty += 7.0
                flags.append("low-signal-market-host")

            if source.provider == "web-search" and not cls._has_market_identifiers(
                f"{source.title} {source.abstract}"
            ):
                penalty += 4.0
                flags.append("missing-market-identifiers")

            if any(
                marker in title or marker in abstract
                for marker in (
                    "top stocks",
                    "best stocks",
                    "stocks to buy",
                    "stock to buy",
                    "buy these stocks",
                    "analyst bets",
                    "buy now",
                    "hot picks",
                    "10 stocks",
                )
            ):
                penalty += 6.0
                flags.append("promo-market-listicle")

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
    @classmethod
    def _select_balanced_top(
        cls,
        ranked: list[ResearchSource],
        max_sources: int,
        query: str,
    ) -> list[ResearchSource]:
        if not ranked:
            return []
        capped_limit = max(max_sources * 3, max_sources)
        capped = ranked[:capped_limit]
        capped_identity_keys = {
            key
            for source in capped
            for key in cls._source_identity_keys(source)
        }
        preserved_browser_sources = 0
        for source in ranked[capped_limit:]:
            if preserved_browser_sources >= 3:
                break
            if source.provider != "pc-browser-research":
                continue
            if "off-topic" in (source.quality_flags or []):
                continue
            if not any(
                flag in (source.quality_flags or [])
                for flag in (
                    "browser-terminal-verified",
                    "browser-navigation-seed",
                    "browser-judged-source",
                    "browser-fetched-seed",
                )
            ):
                continue
            if not cls._source_is_on_topic(source, query):
                continue
            source_keys = cls._source_identity_keys(source)
            if source_keys and any(key in capped_identity_keys for key in source_keys):
                continue
            capped.append(source)
            for key in source_keys:
                capped_identity_keys.add(key)
            preserved_browser_sources += 1
        capped = [
            source
            for source in capped
            if "off-topic" not in (source.quality_flags or [])
            and (
                float(source.score or 0.0) <= 0.0
                or cls._source_is_on_topic(source, query)
            )
        ]
        if not capped:
            return []
        by_provider: dict[str, list[ResearchSource]] = {}
        for source in capped:
            by_provider.setdefault(source.provider, []).append(source)
        if cls._looks_like_market_query(query):
            for provider, provider_sources in list(by_provider.items()):
                filtered_sources = [
                    source
                    for source in provider_sources
                    if not (
                        source.provider == "web-search"
                        and source.evidence_grade == "weak"
                        and "missing-market-identifiers" in (source.quality_flags or [])
                    )
                ]
                if filtered_sources:
                    by_provider[provider] = filtered_sources
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
        if cls._looks_like_software_agent_query(query):
            append_preferred("github-repositories")

        append_preferred(
            "pc-browser-research",
            lambda source: (
                (
                    "browser-terminal-verified" in (source.quality_flags or [])
                    or "browser-navigation-seed" in (source.quality_flags or [])
                    or "browser-judged-source" in (source.quality_flags or [])
                    or "browser-fetched-seed" in (source.quality_flags or [])
                )
                and cls._source_is_on_topic(source, query)
            ),
        )

        # For current-evidence tasks, preserve at least one tool observation
        # when available so runs do not collapse into a single-provider web
        # monoculture.
        if cls._looks_like_current_evidence_query(
            query
        ) and not cls._looks_like_academic_query(query):
            append_preferred(
                "gemini-flash",
                lambda source: (
                    cls._objective_alignment_score(
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
                cls._objective_alignment_score(
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
    @classmethod
    def _entity_terms_from_query(cls, query: str) -> set[str]:
        raw_query = query or ""
        lower = query.lower()
        software_mode = cls._looks_like_software_agent_query(query)
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
        generic = cls._generic_query_terms()
        for ticker in _extract_ticker_candidates(raw_query):
            matched.add(ticker.lower())
        for quoted in re.findall(r"[\"'“”]([^\"'“”]{3,80})[\"'“”]", raw_query):
            tokens = [
                token.lower()
                for token in re.findall(
                    r"\b[A-Za-z][A-Za-z0-9&.\-]{2,}\b",
                    quoted,
                )
                if token.lower() not in generic
            ]
            if tokens:
                matched.add(" ".join(tokens[:5]))
        for phrase in re.findall(
            (
                r"\b(?:[A-Z][A-Za-z0-9&.\-]{2,})"
                r"(?:\s+[A-Z][A-Za-z0-9&.\-]{2,}){0,4}\b"
            ),
            raw_query,
        ):
            tokens = [
                token.lower().strip("&.-")
                for token in phrase.split()
                if token.lower().strip("&.-") not in generic
            ]
            if not tokens:
                continue
            candidate = " ".join(tokens)
            if len(candidate) >= 3:
                matched.add(candidate)
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
    def _market_signal_snapshot(sources: list[ResearchSource]) -> list[str]:
        by_ticker: dict[str, dict[str, Any]] = {}
        for source in sources[:30]:
            tickers = _extract_ticker_candidates(f"{source.title} {source.abstract}")
            if not tickers:
                continue
            for ticker in tickers[:3]:
                record = by_ticker.setdefault(
                    ticker,
                    {
                        "score": 0.0,
                        "count": 0,
                        "providers": set(),
                        "sample_title": source.title,
                    },
                )
                record["score"] += float(source.score or 0.0)
                record["count"] += 1
                record["providers"].add(source.provider)

        ranked = sorted(
            by_ticker.items(),
            key=lambda item: (
                item[1]["score"],
                item[1]["count"],
                len(item[1]["providers"]),
            ),
            reverse=True,
        )
        lines: list[str] = []
        for ticker, payload in ranked[:8]:
            providers = ", ".join(sorted(payload["providers"]))
            title = str(payload["sample_title"] or "")[:120]
            lines.append(
                f"- {ticker}: seen in {payload['count']} sources across {providers}; example source: {title}"
            )
        return lines
    @classmethod
    def _claim_trace(
        cls,
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
                    cls._finding_confidence_rank(
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
            source.provider == "pc-browser-research"
            and any(
                flag in (source.quality_flags or [])
                for flag in (
                    "browser-judged-source",
                    "browser-navigation-seed",
                    "browser-terminal-verified",
                )
            )
            and "off-topic" not in (source.quality_flags or [])
        ):
            return "tool-observation"
        if (
            source.credibility_score < 0.25
            or "off-topic" in source.quality_flags
            or "market-nonspecific-web" in source.quality_flags
            or "low-signal-web" in source.quality_flags
            or "speculative-proof-claim" in source.quality_flags
            or "unsupported-proof-title" in source.quality_flags
        ):
            return "weak"
        if source.provider in {
            "web-search",
            "bing-search",
            "google-news-rss",
            "financial-portals",
            "reddit-finance",
            "pc-browser-research",
            "seed-url",
        }:
            if source.relevance >= 0.62 and source.credibility_score >= 0.55:
                return "strong"
            if source.relevance >= 0.48 and source.credibility_score >= 0.42:
                return "moderate"
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
    def _generic_query_terms() -> set[str]:
        return _generic_query_terms_policy()
    @classmethod
    def _objective_anchor_terms(cls, query: str) -> set[str]:
        stopwords = cls._generic_query_terms()
        return {
            token
            for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", query.lower())
            if token not in stopwords
        }
    @classmethod
    def _objective_alignment_score(cls, text: str, query: str) -> float:
        anchors = cls._objective_anchor_terms(query)
        entity_terms = cls._entity_terms_from_query(query)
        lower_text = text.lower()
        words = {token for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", lower_text)}
        if not words and not lower_text.strip():
            return 0.0
        overlap = len(anchors & words)
        overlap += sum(1 for term in entity_terms if term and term in lower_text)
        denominator = len(anchors) + len(entity_terms)
        if denominator <= 0:
            return 0.0
        # Reward overlap but stay conservative unless there are multiple matches.
        return min(overlap / max(min(denominator, 4), 1), 1.0)
