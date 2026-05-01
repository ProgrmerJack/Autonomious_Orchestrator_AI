"""Self-documentation loop for unfamiliar applications.

When the agent sees an unknown UI, it should not click blindly. This module
generates a targeted documentation query, fetches official docs/tutorials via
injectable providers, caches the results, and returns concise context for the
frontier planner to cross-reference against Set-of-Mark IDs.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


class DocumentationSearchProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return candidate documentation URLs."""


class DocumentationFetcher(Protocol):
    def fetch(self, url: str, timeout_seconds: int = 15) -> str:
        """Return raw HTML or text for a URL."""


class UrlLibFetcher:
    def fetch(self, url: str, timeout_seconds: int = 15) -> str:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgentOS-SelfDocumentation/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return resp.read().decode("utf-8", errors="replace")


@dataclass(slots=True)
class DocumentationSource:
    url: str
    title: str = ""
    excerpt: str = ""
    official_score: float = 0.0


@dataclass(slots=True)
class DocumentationBundle:
    query: str
    sources: list[DocumentationSource] = field(default_factory=list)
    cache_hit: bool = False

    @property
    def context(self) -> str:
        chunks: list[str] = []
        for index, source in enumerate(self.sources, start=1):
            chunks.append(
                f"[{index}] {source.title or source.url}\n"
                f"URL: {source.url}\n"
                f"Official score: {source.official_score:.2f}\n"
                f"Excerpt: {source.excerpt}"
            )
        return "\n\n".join(chunks)


class SelfDocumentationLoop:
    """Prepare official documentation context before acting in unknown UIs."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        search_provider: DocumentationSearchProvider | None = None,
        fetcher: DocumentationFetcher | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.search_provider = search_provider
        self.fetcher = fetcher or UrlLibFetcher()
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else self.workspace_root / ".agentos" / "docs_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_context(
        self,
        objective: str,
        app_hint: str = "unknown application",
        candidate_urls: list[str] | None = None,
        max_sources: int = 3,
    ) -> DocumentationBundle:
        query = self.generate_query(objective, app_hint)
        cache_path = self._cache_path(query)
        if cache_path.exists():
            return self._load_cache(cache_path)

        urls = list(candidate_urls or [])
        if not urls and self.search_provider is not None:
            urls = self.search_provider.search(query, limit=max_sources * 2)
        urls = self._rank_urls(urls, app_hint)[:max_sources]

        sources: list[DocumentationSource] = []
        for url in urls:
            try:
                raw = self.fetcher.fetch(url)
            except Exception:
                continue
            title, text = self._html_to_text(raw)
            sources.append(
                DocumentationSource(
                    url=url,
                    title=title,
                    excerpt=text[:1800],
                    official_score=self._official_score(url, app_hint),
                )
            )

        bundle = DocumentationBundle(query=query, sources=sources, cache_hit=False)
        self._write_cache(cache_path, bundle)
        return bundle

    @staticmethod
    def generate_query(objective: str, app_hint: str = "unknown application") -> str:
        app = app_hint.strip() or "unknown application"
        objective_clean = re.sub(r"\s+", " ", objective.strip())
        return f"official {app} documentation tutorial {objective_clean}".strip()

    def _rank_urls(self, urls: list[str], app_hint: str) -> list[str]:
        deduped = list(dict.fromkeys(urls))
        scored = [(url, self._official_score(url, app_hint)) for url in deduped]
        official = [(url, score) for url, score in scored if score > 0.0]
        ranked = official or scored
        return [
            url
            for url, _score in sorted(ranked, key=lambda item: item[1], reverse=True)
        ]

    @staticmethod
    def _official_score(url: str, app_hint: str) -> float:
        lower = url.lower()
        app_tokens = [
            t for t in re.split(r"[^a-z0-9]+", app_hint.lower()) if len(t) > 2
        ]
        score = 0.0
        if any(token in lower for token in app_tokens):
            score += 0.25
        if any(
            part in lower
            for part in ("docs", "documentation", "help", "support", "learn")
        ):
            score += 0.35
        if any(part in lower for part in ("official", "developer", "manual", "guide")):
            score += 0.15
        if any(
            domain in lower
            for domain in ("youtube.com", "reddit.com", "x.com", "twitter.com")
        ):
            score -= 0.4
        if lower.startswith("https://"):
            score += 0.05
        return max(0.0, min(1.0, score))

    @staticmethod
    def _html_to_text(raw: str) -> tuple[str, str]:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
        title = html.unescape(title_match.group(1).strip()) if title_match else ""
        text = re.sub(r"<script\b.*?</script>", " ", raw, flags=re.I | re.S)
        text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return title, text

    def _cache_path(self, query: str) -> Path:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _write_cache(path: Path, bundle: DocumentationBundle) -> None:
        payload = {
            "query": bundle.query,
            "sources": [asdict(source) for source in bundle.sources],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _load_cache(path: Path) -> DocumentationBundle:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DocumentationBundle(
            query=payload.get("query", ""),
            sources=[
                DocumentationSource(**source) for source in payload.get("sources", [])
            ],
            cache_hit=True,
        )
