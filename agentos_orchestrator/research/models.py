from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def extract_ticker_candidates(text: str) -> list[str]:
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
        "ASAP",
        "BUY",
        "RIGHT",
        "BULLISH",
        "USA",
        "US",
        "GDP",
        "CEO",
        "CFO",
        "EPS",
        "YOY",
        "QOQ",
        "NYSE",
        "NASDAQ",
        "SPY",
        "QQQ",
    }
    raw = text or ""
    tokens: list[str] = []
    for pattern in (
        r"\$([A-Z]{1,5})\b",
        r"\b(?:NYSE|NASDAQ|AMEX|TSX|LSE)\s*[:\-]\s*([A-Z]{1,5})\b",
    ):
        tokens.extend(re.findall(pattern, raw))

    for match in re.finditer(r"\b([A-Z]{1,5})\b", raw):
        token = match.group(1)
        left = max(match.start() - 28, 0)
        right = min(match.end() + 28, len(raw))
        window = raw[left:right].lower()
        if any(
            marker in window
            for marker in (
                "ticker",
                "stock",
                "shares",
                "equity",
                "earnings",
                "price target",
                "nasdaq",
                "nyse",
            )
        ):
            tokens.append(token)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        token = token.strip().upper()
        if len(token) < 2:
            continue
        if token in blocked:
            continue
        if token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result[:8]


def sanitize_evidence_claim_text(title: str, abstract: str, url: str) -> str:
    raw = (abstract or "").strip()
    lower = raw.lower()
    if (
        not raw
        or lower.startswith("generic web result")
        or "snippet unavailable" in lower
    ):
        tickers = extract_ticker_candidates(f"{title} {abstract}")
        if tickers:
            return f"Ticker candidates mentioned by source: {', '.join(tickers)}."
        return (title or url)[:240]

    cleaned = re.sub(r"\s+", " ", raw)
    cleaned = re.sub(
        r"window\.initialI18nStore\s*=\s*\{.*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"app\.account\.recovery\.[^\s]+", "", cleaned, flags=re.I)
    sentence = re.split(r"[.!?]", cleaned, maxsplit=1)[0].strip()
    if sentence and len(sentence) >= 30:
        cleaned = sentence
    tickers = extract_ticker_candidates(f"{title} {cleaned}")
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
            "claim": sanitize_evidence_claim_text(self.title, self.abstract, self.url),
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
