from __future__ import annotations

from collections.abc import Callable


def provider_order(
    search_query: str = "",
    allowed_providers: set[str] | None = None,
    *,
    looks_like_software_agent_query: Callable[[str], bool],
    looks_like_market_query: Callable[[str], bool],
    looks_like_academic_query: Callable[[str], bool],
    looks_like_current_evidence_query: Callable[[str], bool],
) -> tuple[str, ...]:
    allowed = set(allowed_providers or set())
    scholarly = {"openalex", "semantic-scholar", "crossref"}
    market = {
        "sec-edgar",
        "financial-portals",
        "earnings-data",
        "insider-transactions",
        "short-interest",
        "macrotrends",
        "stockanalysis",
        "fed-macro",
        "seeking-alpha",
        "reddit-finance",
    }
    base_order = (
        "web-search",
        "bing-search",
        "google-news-rss",
        "crossref",
        "openalex",
        "semantic-scholar",
        "github-repositories",
        "sec-edgar",
        "financial-portals",
        "earnings-data",
        "insider-transactions",
        "short-interest",
        "macrotrends",
        "stockanalysis",
        "fed-macro",
        "seeking-alpha",
        "reddit-finance",
    )

    if looks_like_software_agent_query(search_query):
        preferred = (
            "github-repositories",
            "web-search",
            "bing-search",
            "google-news-rss",
            "crossref",
            "openalex",
            "semantic-scholar",
        )
    elif looks_like_market_query(search_query):
        preferred = (
            "sec-edgar",
            "earnings-data",
            "insider-transactions",
            "short-interest",
            "fed-macro",
            "google-news-rss",
            "web-search",
            "bing-search",
            "financial-portals",
            "macrotrends",
            "stockanalysis",
            "crossref",
            "seeking-alpha",
            "reddit-finance",
        )
    elif looks_like_academic_query(search_query) or (
        allowed & scholarly and not allowed & market
    ):
        preferred = (
            "openalex",
            "semantic-scholar",
            "crossref",
            "web-search",
            "bing-search",
            "google-news-rss",
        )
    elif looks_like_current_evidence_query(search_query):
        preferred = (
            "web-search",
            "bing-search",
            "google-news-rss",
            "crossref",
            "openalex",
            "semantic-scholar",
            "github-repositories",
        )
    else:
        preferred = base_order

    ordered: list[str] = []
    seen: set[str] = set()
    for provider in (*preferred, *base_order):
        if provider in seen:
            continue
        if allowed and provider not in allowed:
            continue
        seen.add(provider)
        ordered.append(provider)
    return tuple(ordered)
