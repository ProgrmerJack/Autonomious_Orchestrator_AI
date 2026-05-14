from __future__ import annotations

import re
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from .models import (
    ResearchIntentSpec,
    ResearchSource,
    extract_ticker_candidates as _extract_ticker_candidates,
)
from .query_policy import generic_query_terms as _generic_query_terms_policy


class ResearchSourceScoringMixin:
    @staticmethod
    def _market_signal_strength(text: str) -> int:
        lower = (text or "").lower()
        signal_terms = (
            "stock",
            "stocks",
            "share",
            "shares",
            "equity",
            "ticker",
            "analyst",
            "price target",
            "earnings",
            "guidance",
            "revenue",
            "valuation",
            "market cap",
            "eps",
            "consensus",
            "upgrade",
            "downgrade",
        )
        return sum(1 for term in signal_terms if term in lower)

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

    @staticmethod
    def _source_host(url: str) -> str:
        return urllib.parse.urlparse(url or "").netloc.lower().lstrip("www.")

    @staticmethod
    def _is_sec_filing_url(url: str) -> bool:
        path = (urllib.parse.urlparse(url or "").path or "").lower()
        return any(
            token in path
            for token in (
                "/archives/edgar/",
                "/ixviewer/",
                "/cgi-bin/browse-edgar",
                "/edgar/search",
                "/search-filings/company-search",
            )
        )

    @classmethod
    def _is_market_navigation_page(cls, source: ResearchSource, text: str) -> bool:
        host = cls._source_host(source.url)
        path = (urllib.parse.urlparse(source.url or "").path or "").lower().rstrip("/")
        lower = (text or "").lower()
        if not host:
            return False

        title_markers = (
            "highest upside",
            "top stocks",
            "stock list",
            "stock lists",
            "stock screener",
            "stock ratings",
            "market headlines",
            "breaking stock market news",
            "investing ideas",
            "undervalued stocks",
            "discover",
            "screen",
            "screener",
            "watchlist",
        )
        path_markers = (
            "/list",
            "/discover",
            "/screener",
            "/markets/stocks",
            "/stocks",
            "/analysts/top-",
        )
        if host == "stockanalysis.com" and path in {"", "/", "/list"}:
            return True
        if host == "cnbc.com" and path == "/stocks":
            return True
        if host == "reuters.com" and path == "/markets/stocks":
            return True
        if host == "simplywall.st" and path.startswith("/discover"):
            return True
        if any(marker in path for marker in path_markers) and any(
            marker in lower for marker in title_markers
        ):
            return True
        if any(marker in lower for marker in title_markers) and any(
            nav_term in lower
            for nav_term in (
                "download",
                "symbol",
                "market cap",
                "price change",
                "watchlist",
                "indicators",
            )
        ):
            return True
        return False

    @staticmethod
    def _extract_company_binding(text: str) -> str:
        raw = text or ""
        blocked_bindings = {
            "public company",
            "public companies",
            "private company",
            "private companies",
            "listed company",
            "listed companies",
        }
        blocked_binding_terms = {
            "account",
            "affiliate",
            "analysis",
            "article",
            "calendar",
            "chart",
            "chartmill",
            "community",
            "contact",
            "download",
            "filing",
            "filings",
            "free",
            "home",
            "info",
            "insights",
            "investing",
            "learn",
            "list",
            "lists",
            "login",
            "market",
            "markets",
            "monitor",
            "more",
            "news",
            "overview",
            "page",
            "performance",
            "plans",
            "portfolio",
            "potential",
            "privacy",
            "probability",
            "public",
            "publicly",
            "quote",
            "ratings",
            "research",
            "screen",
            "screener",
            "search",
            "statistics",
            "stock",
            "stocks",
            "subscription",
            "symbol",
            "target",
            "tools",
            "trading",
            "traded",
            "upside",
            "watchlist",
            "why",
            "companies",
            "company",
            "highest",
            "months",
            "month",
            "next",
        }
        trailing_binding_markers = {
            "analyst",
            "earnings",
            "forecast",
            "guidance",
            "revenue",
            "shares",
            "stock",
            "valuation",
        }

        def _normalize_binding_candidate(candidate: str) -> str:
            normalized = candidate.strip(" -,:;.")
            words = normalized.split()
            while words and words[-1].lower() in trailing_binding_markers:
                words.pop()
            return " ".join(words).strip()

        def _binding_allowed(candidate: str) -> bool:
            normalized = _normalize_binding_candidate(candidate)
            if not normalized:
                return False
            lowered = normalized.lower()
            if lowered in blocked_bindings:
                return False
            words = re.findall(r"\b[A-Za-z][A-Za-z&.\-]{1,}\b", normalized)
            if not words:
                return False
            blocked_hits = sum(
                1 for word in words if word.lower() in blocked_binding_terms
            )
            if blocked_hits >= max(1, len(words) // 2):
                return False
            if len(words) >= 3 and blocked_hits > 0:
                return False
            return True

        tickers = _extract_ticker_candidates(raw)
        if tickers:
            return tickers[0].upper()
        suffix_match = re.search(
            (
                r"\b([A-Z][A-Za-z&.\-]{1,}"
                r"(?:\s+[A-Z][A-Za-z&.\-]{1,}){0,3}\s+"
                r"(?:Inc|Corp|Corporation|Ltd|PLC|Group|Holdings|"
                r"Technologies|Technology|Energy|Pharma|Bank|Co|Company))\b"
            ),
            raw,
        )
        if suffix_match:
            candidate = _normalize_binding_candidate(suffix_match.group(1))
            if _binding_allowed(candidate):
                return candidate
        contextual_match = re.search(
            (
                r"\b([A-Z][A-Za-z&.\-]{2,}"
                r"(?:\s+[A-Z][A-Za-z&.\-]{2,}){0,2})\s+"
                r"(?:stock|shares|earnings|guidance|revenue|valuation|"
                r"price target|analyst)\b"
            ),
            raw,
        )
        if contextual_match:
            candidate = _normalize_binding_candidate(contextual_match.group(1))
            if _binding_allowed(candidate):
                return candidate
        lowercase_contextual_match = re.search(
            (
                r"\b([a-z][a-z&.\-]{2,}"
                r"(?:\s+[a-z][a-z&.\-]{2,}){0,2})\s+"
                r"(?:stock|shares|earnings|guidance|revenue|valuation|"
                r"price target|analyst|forecast)\b"
            ),
            raw,
        )
        if lowercase_contextual_match:
            candidate = _normalize_binding_candidate(
                lowercase_contextual_match.group(1)
            ).title()
            if _binding_allowed(candidate):
                return candidate
        return ""

    @classmethod
    def _intent_spec(cls, query: str) -> ResearchIntentSpec:
        if cls._looks_like_public_security_query(query):
            return ResearchIntentSpec(
                mode="public-company-market",
                accepted_subject_kinds=("public-company", "market-basket"),
                accepted_evidence_kinds=(
                    "filing",
                    "earnings-news",
                    "analyst-coverage",
                    "market-analysis",
                    "market-data",
                    "company-profile",
                    "current-news",
                ),
                rejected_evidence_kinds=(
                    "app-listing",
                    "map-story",
                    "developer-docs",
                    "academic-paper",
                    "forum",
                    "social",
                    "macro-series",
                    "market-navigation",
                    "regulatory-navigation",
                ),
                required_context_terms=(
                    "stock",
                    "stocks",
                    "share",
                    "shares",
                    "equity",
                    "equities",
                    "ticker",
                    "public company",
                    "public companies",
                    "publicly traded",
                    "listed company",
                    "analyst",
                    "earnings",
                    "revenue",
                    "guidance",
                    "valuation",
                    "price target",
                    "filing",
                    "10-k",
                    "10-q",
                    "8-k",
                ),
            )
        if cls._looks_like_market_query(query):
            return ResearchIntentSpec(
                mode="market",
                accepted_subject_kinds=(
                    "public-company",
                    "market-basket",
                    "macro-series",
                ),
                accepted_evidence_kinds=(
                    "filing",
                    "earnings-news",
                    "analyst-coverage",
                    "market-analysis",
                    "market-data",
                    "company-profile",
                    "current-news",
                    "macro-series",
                ),
                rejected_evidence_kinds=(
                    "app-listing",
                    "map-story",
                    "developer-docs",
                    "forum",
                    "social",
                    "market-navigation",
                    "regulatory-navigation",
                ),
                required_context_terms=(
                    "market",
                    "markets",
                    "stock",
                    "stocks",
                    "equity",
                    "equities",
                    "earnings",
                    "valuation",
                    "ticker",
                    "investor",
                ),
            )
        if cls._looks_like_software_agent_query(query):
            return ResearchIntentSpec(
                mode="software-agent",
                accepted_subject_kinds=(
                    "software-project",
                    "software-product",
                    "benchmark",
                    "documentation",
                    "academic-topic",
                ),
                accepted_evidence_kinds=(
                    "repository",
                    "benchmark",
                    "documentation",
                    "developer-docs",
                    "issue",
                    "release-note",
                    "technical-blog",
                    "academic-paper",
                ),
                rejected_evidence_kinds=("app-listing", "map-story"),
                required_context_terms=(
                    "agent",
                    "browser",
                    "desktop",
                    "workflow",
                    "benchmark",
                    "repository",
                    "documentation",
                ),
            )
        if cls._looks_like_academic_query(query):
            return ResearchIntentSpec(
                mode="academic",
                accepted_subject_kinds=("academic-topic",),
                accepted_evidence_kinds=("academic-paper",),
            )
        return ResearchIntentSpec()

    @classmethod
    def _assign_source_semantics(
        cls,
        source: ResearchSource,
        query: str = "",
    ) -> ResearchSource:
        combined = cls._strip_dom_noise_tokens(f"{source.title} {source.abstract}")
        lower = combined.lower()
        host = cls._source_host(source.url)
        entity_binding = cls._extract_company_binding(
            f"{source.title} {source.abstract}"
        )
        evidence_kind = "unknown"
        subject_kind = "unknown"
        document_kind = "unknown"

        app_hosts = {
            "apps.apple.com",
            "play.google.com",
            "apps.microsoft.com",
        }
        academic_hosts = {
            "arxiv.org",
            "openalex.org",
            "semanticscholar.org",
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
        }
        developer_hosts = {
            "learn.microsoft.com",
            "developer.mozilla.org",
            "docs.python.org",
        }
        forum_hosts = {
            "reddit.com",
            "quora.com",
            "stackoverflow.com",
            "stackexchange.com",
        }
        social_hosts = {
            "twitter.com",
            "x.com",
            "facebook.com",
            "linkedin.com",
            "stocktwits.com",
        }
        macro_hosts = {"fred.stlouisfed.org", "bea.gov", "bls.gov"}
        market_hosts = {
            "sec.gov",
            "finance.yahoo.com",
            "marketwatch.com",
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "spglobal.com",
            "morningstar.com",
            "cnbc.com",
            "seekingalpha.com",
            "investing.com",
        }

        if source.provider == "github-repositories" or host in {
            "github.com",
            "gitlab.com",
        }:
            evidence_kind = "repository"
            subject_kind = "software-project"
            document_kind = "repository"
        elif host in app_hosts or any(
            marker in lower
            for marker in (
                "app store",
                "google play",
                "microsoft store",
                "download on the app store",
                "get it on google play",
            )
        ):
            evidence_kind = "app-listing"
            subject_kind = "software-product"
            document_kind = "app-store-listing"
        elif (
            "storymaps" in host
            or host.endswith("arcgis.com")
            or any(
                marker in lower for marker in ("storymaps", "story map", "arcgis story")
            )
        ):
            evidence_kind = "map-story"
            subject_kind = "place-location"
            document_kind = "story-map"
        elif host in forum_hosts:
            evidence_kind = "forum"
            subject_kind = "discussion-thread"
            document_kind = "forum-thread"
        elif host in social_hosts:
            evidence_kind = "social"
            subject_kind = "social-post"
            document_kind = "social-post"
        elif (
            source.provider in {"openalex", "semantic-scholar", "crossref"}
            or host in academic_hosts
            or any(
                marker in lower
                for marker in (
                    "peer reviewed",
                    "systematic review",
                    "meta-analysis",
                    "journal article",
                    "conference paper",
                )
            )
        ):
            evidence_kind = "academic-paper"
            subject_kind = "academic-topic"
            document_kind = "paper"
        elif (
            host.startswith("docs.")
            or host.startswith("developer.")
            or host in developer_hosts
            or host.endswith(".readthedocs.io")
            or any(
                marker in lower
                for marker in (
                    "api reference",
                    "sdk reference",
                    "developer documentation",
                    "installation guide",
                )
            )
        ):
            evidence_kind = "developer-docs"
            subject_kind = "documentation"
            document_kind = "documentation"
        elif (host == "sec.gov" and cls._is_sec_filing_url(source.url)) or any(
            marker in lower
            for marker in (
                "primary filing",
                "regulatory filing",
                "company filing",
                "filed with the sec",
                "10-k",
                "10-q",
                "8-k",
                "form 10-k",
                "form 10-q",
                "form 8-k",
                "sec filing",
                "annual report",
                "quarterly report",
                "investor relations",
            )
        ):
            evidence_kind = "filing"
            subject_kind = "public-company"
            document_kind = "regulatory-filing"
        elif host == "sec.gov":
            evidence_kind = "regulatory-navigation"
            subject_kind = "regulator"
            document_kind = "regulatory-portal"
        elif cls._is_market_navigation_page(source, combined):
            evidence_kind = "market-navigation"
            subject_kind = "market-basket"
            document_kind = "navigation-page"
        elif host in macro_hosts or any(
            marker in lower
            for marker in (
                "consumer price index",
                "nonfarm payroll",
                "gross domestic product",
                "treasury yield",
                "federal funds rate",
                "unemployment rate",
            )
        ):
            evidence_kind = "macro-series"
            subject_kind = "macro-series"
            document_kind = "economic-data"
        elif cls._has_market_signal(combined) or host in market_hosts:
            if any(
                marker in lower
                for marker in (
                    "analyst",
                    "analysts",
                    "analyst rating",
                    "analyst estimate",
                    "price target",
                    "upgrade",
                    "downgrade",
                    "outperform",
                    "underperform",
                    "overweight",
                    "underweight",
                    "bullish",
                    "bearish",
                )
            ):
                evidence_kind = "analyst-coverage"
                document_kind = "analysis"
            elif any(
                marker in lower
                for marker in (
                    "earnings",
                    "revenue",
                    "guidance",
                    "quarterly results",
                    "fiscal q",
                    "eps",
                    "cash flow",
                )
            ):
                evidence_kind = "earnings-news"
                document_kind = "news-article"
            elif any(
                marker in lower
                for marker in (
                    "market cap",
                    "shares outstanding",
                    "52-week",
                    "dividend yield",
                    "quote",
                    "p/e",
                    "eps estimate",
                    "enterprise value",
                )
            ):
                evidence_kind = "market-data"
                document_kind = "market-data-page"
            elif any(
                marker in lower
                for marker in (
                    "announced",
                    "reported",
                    "breaking news",
                    "company news",
                )
            ):
                evidence_kind = "current-news"
                document_kind = "news-article"
            else:
                evidence_kind = "market-analysis"
                document_kind = "analysis"
            if (
                entity_binding
                or cls._has_market_identifiers(combined)
                or any(
                    marker in lower
                    for marker in (
                        "public company",
                        "public companies",
                        "publicly traded",
                        "listed company",
                        "listed companies",
                        "shares of",
                    )
                )
            ):
                subject_kind = "public-company"
            else:
                subject_kind = "market-basket"

        if not document_kind and evidence_kind != "unknown":
            document_kind = evidence_kind
        source.evidence_kind = evidence_kind
        source.subject_kind = subject_kind
        source.document_kind = document_kind
        source.entity_binding = entity_binding
        return source

    @classmethod
    def _matches_intent_spec(cls, source: ResearchSource, query: str) -> bool:
        spec = cls._intent_spec(query)
        if spec.mode == "general":
            return True
        if source.provider == "gemini-flash":
            combined = cls._strip_dom_noise_tokens(
                f"{source.title} {source.abstract}"
            ).lower()
            if not combined:
                return False
            if spec.required_context_terms and not any(
                term in combined for term in spec.required_context_terms
            ):
                if spec.mode in {"market", "public-company-market"}:
                    if cls._market_signal_strength(combined) < 2 and (
                        cls._objective_alignment_score(combined, query) < 0.2
                    ):
                        return False
                else:
                    return False
            return True
        cls._assign_source_semantics(source, query)
        if source.evidence_kind in spec.rejected_evidence_kinds:
            return False
        if (
            spec.accepted_evidence_kinds
            and source.evidence_kind not in spec.accepted_evidence_kinds
        ):
            return False
        if (
            spec.accepted_subject_kinds
            and source.subject_kind not in spec.accepted_subject_kinds
        ):
            return False
        combined = cls._strip_dom_noise_tokens(
            f"{source.title} {source.abstract}"
        ).lower()
        has_required_terms = any(
            term in combined for term in spec.required_context_terms
        )
        if spec.required_context_terms and not has_required_terms:
            # Market evidence pages can still be valid even when they don't
            # contain the literal objective wording (e.g. no exact phrase
            # "publicly traded" but clearly analyst/earnings/valuation data).
            if spec.mode in {"market", "public-company-market"}:
                if source.evidence_kind not in {
                    "filing",
                    "earnings-news",
                    "analyst-coverage",
                    "market-analysis",
                    "market-data",
                    "company-profile",
                    "current-news",
                }:
                    return False
                if cls._market_signal_strength(combined) < 2:
                    return False
            else:
                return False
        if (
            spec.mode == "public-company-market"
            and source.evidence_kind == "market-analysis"
            and not source.entity_binding
            and not cls._has_actionable_market_signal(combined)
        ):
            return False
        if spec.require_entity_binding and not source.entity_binding:
            return False
        if spec.require_actionable_signal and not cls._has_actionable_market_signal(
            combined
        ):
            return False
        return True

    @classmethod
    def _text_matches_intent_spec(cls, text: str, query: str) -> bool:
        spec = cls._intent_spec(query)
        if spec.mode == "general":
            return True
        if spec.mode == "software-agent":
            lower = cls._strip_dom_noise_tokens(text).lower()
            return bool(lower) and (
                not spec.required_context_terms
                or any(term in lower for term in spec.required_context_terms)
            )
        probe = ResearchSource(
            provider="intent-probe",
            title=(text or "")[:200],
            url="",
            abstract=(text or "")[:4000],
        )
        return cls._matches_intent_spec(probe, query)

    @classmethod
    def _query_variant_matches_intent(cls, variant: str, query: str) -> bool:
        spec = cls._intent_spec(query)
        if spec.mode == "general":
            return True
        if cls._normalize_title(variant) == cls._normalize_title(query):
            return True
        if spec.mode == "public-company-market":
            original_entity_binding = cls._extract_company_binding(query)
            variant_entity_binding = cls._extract_company_binding(variant)
            if not original_entity_binding and variant_entity_binding:
                return False
        lower = (variant or "").lower()
        if spec.required_context_terms and not any(
            term in lower for term in spec.required_context_terms
        ):
            if spec.mode in {"market", "public-company-market"}:
                entity_terms = cls._entity_terms_from_query(query)
                company_binding = cls._extract_company_binding(query).lower()
                has_entity = any(term in lower for term in entity_terms) or (
                    bool(company_binding) and company_binding in lower
                )
                if cls._market_signal_strength(variant) < 2 and not (
                    has_entity and cls._objective_alignment_score(variant, query) >= 0.2
                ):
                    return False
            else:
                return False
        if spec.mode in {"market", "public-company-market"}:
            return True
        if spec.mode == "software-agent":
            # Query variants for software/agent discovery are text-only probes;
            # requiring full source-semantic classification is over-strict here.
            return True
        return cls._text_matches_intent_spec(variant, query)

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
    def _merge_source_records(
        cls, existing: ResearchSource, source: ResearchSource
    ) -> None:
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

        if cls._abstract_quality(source.abstract) > cls._abstract_quality(
            existing.abstract
        ):
            existing.abstract = source.abstract
        existing.citation_count = max(existing.citation_count, source.citation_count)
        existing.score = max(existing.score, source.score)
        if {existing.provider, source.provider} == {"seed-url", "web-search"}:
            existing.provider = "web-search"

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
        cls._assign_source_semantics(source, query)
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
            if (
                cls._looks_like_market_query(query)
                and source.subject_kind != "market-basket"
                and not cls._has_market_identifiers(f"{source.title} {source.abstract}")
            ):
                # Only hard-zero sources that are truly off-domain (cooking,
                # sports, medicine — not financial content at all).  The
                # missing-market-identifiers penalty in _source_credibility
                # already down-weights sources without tickers; doubling up
                # with a 0.30 threshold eliminated all screener query results.
                if cls._objective_alignment_score(combined, query) < 0.12:
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
        if not cls._matches_intent_spec(source, query):
            source.quality_flags = [
                *(source.quality_flags or []),
                "intent-mismatch",
                "off-topic",
            ]
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
        if source.evidence_kind in {
            "filing",
            "earnings-news",
            "analyst-coverage",
            "market-data",
        }:
            relevance = max(relevance, 0.4)
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

        cls._assign_source_semantics(source, query)

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

        evidence_credibility = {
            "filing": 0.25,
            "analyst-coverage": 0.12,
            "earnings-news": 0.1,
            "market-data": 0.08,
            "repository": 0.08,
            "developer-docs": 0.06,
        }
        credibility += evidence_credibility.get(source.evidence_kind, 0.0)
        if source.evidence_kind in {"app-listing", "map-story", "forum", "social"}:
            penalty += 10.0
            flags.append("intent-mismatch")

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

            if (
                source.provider == "web-search"
                and source.subject_kind != "market-basket"
                and not cls._has_market_identifiers(f"{source.title} {source.abstract}")
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
            key for source in capped for key in cls._source_identity_keys(source)
        }
        capped_urls = {source.url for source in capped if source.url}
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
            if (
                cls._objective_alignment_score(
                    f"{source.title} {source.abstract}",
                    query,
                )
                < 0.12
            ):
                continue
            if source.url and source.url in capped_urls:
                continue
            source_keys = cls._source_identity_keys(source)
            capped.append(source)
            if source.url:
                capped_urls.add(source.url)
            for key in source_keys:
                capped_identity_keys.add(key)
            preserved_browser_sources += 1
        capped = [
            source
            for source in capped
            if "off-topic" not in (source.quality_flags or [])
            and (
                float(source.score or 0.0) <= 0.0
                or (
                    source.provider == "pc-browser-research"
                    and any(
                        flag in (source.quality_flags or [])
                        for flag in (
                            "browser-terminal-verified",
                            "browser-navigation-seed",
                            "browser-judged-source",
                            "browser-fetched-seed",
                        )
                    )
                    and cls._objective_alignment_score(
                        f"{source.title} {source.abstract}",
                        query,
                    )
                    >= 0.12
                )
                or cls._source_is_on_topic(source, query)
            )
        ]
        if not capped:
            capped = [
                source
                for source in ranked[: max(max_sources * 2, max_sources)]
                if "off-topic" not in (source.quality_flags or [])
            ]
            if not capped:
                return []

        is_market_query = cls._looks_like_market_query(query)
        if max_sources >= 80:
            domain_cap = max(8, int(max_sources * 0.12))
        elif max_sources >= 24:
            domain_cap = max(5, int(max_sources * 0.2))
        else:
            domain_cap = max(2, int(max_sources * 0.5))
        weak_domain_cap = max(2, domain_cap // 2) if is_market_query else domain_cap

        def _domain_of(source: ResearchSource) -> str:
            return urllib.parse.urlparse(source.url or "").netloc.lower().lstrip("www.")

        domain_counts: dict[str, int] = {}
        weak_domain_counts: dict[str, int] = {}

        def _can_add(source: ResearchSource, *, allow_override: bool = False) -> bool:
            domain = _domain_of(source)
            if not domain:
                return True
            if not allow_override and domain_counts.get(domain, 0) >= domain_cap:
                return False
            if (
                not allow_override
                and is_market_query
                and source.evidence_grade == "weak"
                and weak_domain_counts.get(domain, 0) >= weak_domain_cap
            ):
                return False
            return True

        def _track_add(source: ResearchSource) -> None:
            domain = _domain_of(source)
            if not domain:
                return
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if source.evidence_grade == "weak":
                weak_domain_counts[domain] = weak_domain_counts.get(domain, 0) + 1

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
                next_index = None
                for candidate_index, candidate in enumerate(bucket):
                    if _can_add(candidate):
                        next_index = candidate_index
                        break
                if next_index is None:
                    continue
                chosen = bucket.pop(next_index)
                selected.append(chosen)
                _track_add(chosen)
                represented.add(provider)
                progressed = True
                if len(selected) >= max_sources:
                    break
            if not progressed:
                break

        def append_preferred(
            provider: str,
            predicate: Any | None = None,
            *,
            allow_override: bool = True,
        ) -> bool:
            provider_sources = by_provider.get(provider) or []
            for index, source in enumerate(provider_sources):
                if predicate is not None and not predicate(source):
                    continue
                if not _can_add(source, allow_override=allow_override):
                    continue
                selected.append(source)
                _track_add(source)
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
                and cls._objective_alignment_score(
                    f"{source.title} {source.abstract}",
                    query,
                )
                >= 0.15
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
            if not _can_add(source):
                continue
            if "off-topic" in (source.quality_flags or []):
                continue
            if not cls._source_is_on_topic(source, query):
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
            _track_add(source)
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
                    cls._finding_confidence_rank(str(finding.get("confidence") or ""))
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

    @staticmethod
    def _generic_market_comparison_terms() -> set[str]:
        return {
            "among",
            "best",
            "highest",
            "lowest",
            "most",
            "next",
            "month",
            "months",
            "over",
            "potential",
            "probability",
            "probability-adjusted",
            "adjusted",
            "top",
            "upside",
            "downside",
            "year",
            "years",
        }

    @staticmethod
    def _anchor_present(term: str, lower_text: str, words: set[str]) -> bool:
        normalized = term.strip().lower()
        if not normalized:
            return False
        if " " in normalized:
            return normalized in lower_text
        return normalized in words

    @classmethod
    def _objective_anchor_terms(cls, query: str) -> set[str]:
        stopwords = cls._generic_query_terms()
        anchors = {
            token
            for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", query.lower())
            if token not in stopwords
        }
        spec = cls._intent_spec(query)
        if spec.mode in {"market", "public-company-market"}:
            anchors -= cls._generic_market_comparison_terms()
            atomic_market_terms = {
                "10-k",
                "10-q",
                "8-k",
                "analyst",
                "earnings",
                "equities",
                "equity",
                "filing",
                "guidance",
                "revenue",
                "share",
                "shares",
                "stock",
                "stocks",
                "ticker",
                "valuation",
            }
            for phrase in spec.required_context_terms:
                normalized = phrase.strip().lower()
                if not normalized:
                    continue
                anchors.add(normalized)
                for token in re.findall(
                    r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b",
                    normalized,
                ):
                    if (
                        token in atomic_market_terms
                        and token not in stopwords
                        and token not in cls._generic_market_comparison_terms()
                    ):
                        anchors.add(token)
            if (
                spec.mode == "public-company-market"
                and not cls._extract_company_binding(query)
            ):
                anchors.update(
                    {
                        "public company",
                        "public companies",
                        "publicly traded",
                        "listed company",
                        "analyst",
                        "earnings",
                        "guidance",
                        "valuation",
                        "price target",
                        "filing",
                    }
                )
        return {term for term in anchors if term and term not in stopwords}

    @classmethod
    def _objective_alignment_score(cls, text: str, query: str) -> float:
        anchors = cls._objective_anchor_terms(query)
        entity_terms = {term.lower() for term in cls._entity_terms_from_query(query)}
        lower_text = text.lower()
        words = {token for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", lower_text)}
        if not words and not lower_text.strip():
            return 0.0
        overlap = sum(
            1 for term in anchors if cls._anchor_present(term, lower_text, words)
        )
        overlap += sum(
            1 for term in entity_terms if cls._anchor_present(term, lower_text, words)
        )
        denominator = len(anchors) + len(entity_terms)
        if denominator <= 0:
            return 0.0
        # Reward overlap but stay conservative unless there are multiple matches.
        return min(overlap / max(min(denominator, 4), 1), 1.0)
