from __future__ import annotations

import json
import html
import os
import re
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
    contradiction_risk: float = 0.0
    evidence_grade: str = "ungraded"

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
            "contradiction_risk": self.contradiction_risk,
        }


@dataclass(slots=True)
class ResearchBrief:
    objective: str
    query: str
    summary: str
    sources: list[ResearchSource]
    artifacts: list[str]
    confidence: float

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

    def run(self, objective: str, run_id: str) -> ResearchBrief:
        self.provider_diagnostics = []
        depth, cleaned_objective = self._split_depth(objective)
        settings = self._settings_for_depth(depth)
        query = self._query_from_objective(cleaned_objective)
        gathered: list[ResearchSource] = []
        query_variants = self._query_variants(query, settings.depth)[
            : settings.max_query_variants
        ]
        for search_query in query_variants:
            gathered.extend(
                self._search_openalex(search_query, settings.per_provider)
            )
            gathered.extend(
                self._search_semantic_scholar(
                    search_query,
                    settings.per_provider,
                )
            )
            if self._looks_like_software_agent_query(search_query):
                gathered.extend(
                    self._search_github_repositories(
                        search_query,
                        min(settings.per_provider, 5),
                    )
                )
        if self._looks_like_software_agent_query(query):
            gathered.extend(self._software_reference_sources(query))
        gathered.extend(self._search_gemini_observation(query, settings.depth))
        sources = self._rank_sources(self._dedupe_sources(gathered), query)
        selected = sources[: settings.max_sources]
        summary = self._summarize(cleaned_objective, selected, settings.depth)
        artifacts = self._write_artifacts(
            run_id,
            cleaned_objective,
            query,
            selected,
            summary,
            settings,
            query_variants,
        )
        confidence = self._confidence(selected)
        return ResearchBrief(
            objective=cleaned_objective,
            query=query,
            summary=summary,
            sources=selected,
            artifacts=artifacts,
            confidence=confidence,
        )

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
        payload = self._get_json(
            f"https://api.github.com/search/repositories?{params}"
        )
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
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
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
    ) -> list[str]:
        artifact_dir = self.workspace_root / "runs" / run_id / "research"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sources_path = artifact_dir / "sources.json"
        brief_path = artifact_dir / "brief.md"
        digest_path = artifact_dir / "digest.json"
        plan_path = artifact_dir / "research_plan.json"
        claim_trace_path = artifact_dir / "claim_trace.json"
        diagnostics_path = artifact_dir / "provider_diagnostics.json"

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
                    "query": query,
                    "query_variants": query_variants,
                    "max_sources": settings.max_sources,
                    "per_provider": settings.per_provider,
                    "token_strategy": (
                        "structured scholarly APIs, software repository "
                        "search, optional model observations, exact "
                        "dedupe, relevance ranking, compressed digest "
                        "artifacts"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        claim_trace_path.write_text(
            json.dumps(
                self._claim_trace(objective, summary, sources),
                indent=2,
            ),
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
            str(claim_trace_path.relative_to(self.workspace_root)),
            str(diagnostics_path.relative_to(self.workspace_root)),
        ]

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
    ) -> str:
        if not sources:
            return (
                "Live research did not return sources from configured public "
                f"providers for: {objective}. Check network policy, API "
                "availability, or attach MCP research servers."
            )
        abstract_text = " ".join(source.abstract for source in sources)
        keywords = self._keywords(abstract_text)
        top_titles = "; ".join(source.title for source in sources[:3])
        theme_text = ", ".join(keywords[:8]) or "source relevance"
        return (
            f"Collected {len(sources)} evidence-backed sources in {depth} "
            "mode for: "
            f"{objective}. "
            f"The strongest starting points are: {top_titles}. "
            f"Recurring themes across abstracts include {theme_text}."
        )

    @staticmethod
    def _query_from_objective(objective: str) -> str:
        cleaned = re.sub(r"\s+", " ", objective).strip()
        prefixes = (
            "Find authoritative sources, prior systems, and gaps for:",
            "Extract implementation constraints, security boundaries,",
        )
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        return cleaned[:240].strip() or objective[:240]

    @staticmethod
    def _split_depth(objective: str) -> tuple[str, str]:
        match = re.search(r"\[(quick|standard|multi-hour)\]\s*", objective)
        if match is None:
            return "standard", objective.strip()
        cleaned = f"{objective[:match.start()]}{objective[match.end():]}"
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
                max_sources=24,
                per_provider=10,
                max_query_variants=8,
            )
        return ResearchSettings(
            depth="standard",
            max_sources=10,
            per_provider=6,
            max_query_variants=5,
        )

    @classmethod
    def _query_variants(cls, query: str, depth: str = "standard") -> list[str]:
        variants = [query]
        lower = query.lower()
        if depth in {"standard", "multi-hour"}:
            variants.extend(
                [
                    f"{query} review",
                    f"{query} methods evaluation",
                ]
            )
        if depth == "multi-hour":
            variants.extend(
                [
                    f"{query} systematic review",
                    f"{query} state of the art",
                    f"{query} benchmarks datasets",
                    f"{query} limitations safety risks",
                    f"{query} implementation framework",
                ]
            )
        if any(term in lower for term in ("desktop", "computer", "gui")):
            variants.append(f"{query} GUI agent computer use")
        if any(term in lower for term in ("accessibility", "vision")):
            variants.append(
                "GUI agent computer control accessibility tree vision"
            )
        if cls._looks_like_software_agent_query(query):
            variants.append(f"{query} github repository agent framework")
        deduped: list[str] = []
        for variant in variants:
            normalized = cls._normalize_title(variant)
            if normalized and normalized not in {
                cls._normalize_title(item) for item in deduped
            }:
                deduped.append(variant[:240])
        return deduped[:8]

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
        by_title: dict[str, ResearchSource] = {}
        for source in sources:
            key = DeepResearchEngine._normalize_title(source.title)
            existing = by_title.get(key)
            if existing is None or source.score > existing.score:
                by_title[key] = source
        return list(by_title.values())

    @classmethod
    def _rank_sources(
        cls,
        sources: list[ResearchSource],
        query: str,
    ) -> list[ResearchSource]:
        query_terms = set(cls._keywords(query))
        if not query_terms:
            query_terms = set(re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", query))
        generic_terms = {
            "agent",
            "agents",
            "model",
            "models",
            "system",
            "systems",
            "research",
            "using",
        }
        distinctive_terms = query_terms - generic_terms
        ranked: list[ResearchSource] = []
        for source in sources:
            if source.provider == "gemini-flash":
                source.relevance = max(source.relevance, 0.65)
                source.recency = cls._recency_score(source.year)
                source.citation_strength = 0.0
                source.contradiction_risk = cls._contradiction_risk(
                    source.abstract
                )
                source.evidence_grade = cls._evidence_grade(source)
                source.score = max(source.score, 62.0)
                source.score += source.recency * 4.0
                source.score -= source.contradiction_risk * 4.0
                ranked.append(source)
                continue
            haystack = f"{source.title} {source.abstract}".lower()
            matched_terms = {term for term in query_terms if term in haystack}
            matched = len(matched_terms)
            relevance = matched / max(len(query_terms), 1)
            if relevance <= 0:
                source.score = 0.0
                continue
            if len(query_terms) >= 3 and matched < 2:
                source.score = 0.0
                continue
            has_distinctive_match = bool(matched_terms & distinctive_terms)
            if not has_distinctive_match and matched < 2:
                source.score = 0.0
                continue
            source.score = relevance * 100.0
            source.relevance = relevance
            source.recency = cls._recency_score(source.year)
            source.citation_strength = min(source.citation_count, 250) / 250
            source.contradiction_risk = cls._contradiction_risk(
                source.abstract
            )
            source.score += source.recency * 8.0
            source.score += source.citation_strength * 6.0
            source.score -= source.contradiction_risk * 6.0
            if source.provider == "semantic-scholar":
                source.score += 5.0
            source.evidence_grade = cls._evidence_grade(source)
            ranked.append(source)
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _quality_summary(sources: list[ResearchSource]) -> str:
        if not sources:
            return "No evidence was available to grade."
        grades = Counter(source.evidence_grade for source in sources)
        strongest = ", ".join(
            f"{grade}: {count}" for grade, count in sorted(grades.items())
        )
        average_relevance = (
            sum(source.relevance for source in sources) / len(sources)
        )
        risk = max(
            (source.contradiction_risk for source in sources),
            default=0.0,
        )
        return (
            f"Evidence grades: {strongest}. Average relevance is "
            f"{average_relevance:.2f}; maximum contradiction risk is "
            f"{risk:.2f}."
        )

    @staticmethod
    def _claim_trace(
        objective: str,
        summary: str,
        sources: list[ResearchSource],
    ) -> dict[str, Any]:
        return {
            "objective": objective,
            "claims": [
                {
                    "claim_id": f"claim_{index}",
                    "claim": claim,
                    "supporting_sources": [
                        {
                            "title": source.title,
                            "url": source.url,
                            "provider": source.provider,
                            "evidence_grade": source.evidence_grade,
                        }
                        for source in sources[:5]
                    ],
                }
                for index, claim in enumerate(_sentences(summary), start=1)
            ],
            "source_count": len(sources),
            "minimum_grade": min(
                (source.evidence_grade for source in sources),
                default="ungraded",
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
            "however",
            "limitation",
            "limitations",
        )
        matches = sum(1 for marker in markers if marker in lower)
        return min(matches / 4, 1.0)

    @staticmethod
    def _evidence_grade(source: ResearchSource) -> str:
        if source.provider == "gemini-flash":
            return "tool-observation"
        if source.relevance >= 0.7 and source.citation_strength >= 0.2:
            return "strong"
        if source.relevance >= 0.45:
            return "moderate"
        return "weak"

    @staticmethod
    def _looks_like_software_agent_query(query: str) -> bool:
        lower = query.lower()
        markers = (
            "agentos",
            "browser research",
            "computer use",
            "desktop agent",
            "github",
            "local pc",
            "openclaw",
            "opencode",
            "openhands",
            "orchestrator",
            "pc agent",
            "software agent",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    def _software_reference_sources(query: str) -> list[ResearchSource]:
        year = datetime.now(UTC).year
        encoded_query = urllib.parse.quote_plus(query)
        return [
            ResearchSource(
                provider="software-reference",
                title=f"GitHub repository search for {query[:80]}",
                url=(
                    "https://github.com/search?type=repositories&q="
                    f"{encoded_query}"
                ),
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
            ResearchSource(
                provider="software-reference",
                title="K-Dense Web product page",
                url="https://www.k-dense.ai/",
                year=year,
                authors=["K-Dense"],
                abstract=(
                    "K-Dense is a web AI co-scientist product surface. "
                    "AgentOS comparisons should test whether local PC "
                    "control, browser workflows, and tool breadth add value."
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
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text)
        if item.strip()
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
