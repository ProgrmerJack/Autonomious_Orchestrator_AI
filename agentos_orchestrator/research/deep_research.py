from __future__ import annotations

import json
import html
import ipaddress
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import settings_policy as _settings_policy
from .artifacts import ResearchArtifactsMixin
from .crawl_state import ResearchCrawlStateMixin
from .provider_search import ProviderSearchMixin
from .retrieval import ResearchRetrievalMixin
from .planning_synthesis import ResearchPlanningSynthesisMixin
from .source_scoring import ResearchSourceScoringMixin
from .models import (
    ResearchBrief,
    ResearchSettings,
    ResearchSource,
)


class DeepResearchEngine(
    ResearchCrawlStateMixin,
    ProviderSearchMixin,
    ResearchArtifactsMixin,
    ResearchSourceScoringMixin,
    ResearchPlanningSynthesisMixin,
    ResearchRetrievalMixin,
):
    """MCP-friendly live research fallback using public scholarly APIs."""

    _provider_parallelism_context = threading.local()

    # Class-level cache: populated on first call to _get_sec_tickers().
    # Maps uppercase ticker → {cik_str, ticker, title} for all ~15k public companies.
    _sec_tickers_cache: "dict[str, dict] | None" = None

    def __init__(
        self,
        workspace_root: str | Path = ".",
        limit_per_provider: int = 6,
        timeout_seconds: int = 20,
        research_state_path: str | Path | None = None,
        crawl_broker_url: str | None = None,
        crawl_broker_token: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.limit_per_provider = limit_per_provider
        self.timeout_seconds = timeout_seconds
        self._research_state_path_override = (
            Path(research_state_path) if research_state_path is not None else None
        )
        self._crawl_broker_url_override = crawl_broker_url
        self._crawl_broker_token_override = crawl_broker_token
        self.provider_diagnostics: list[dict[str, Any]] = []
        self._durable_note_urls: set[str] = set()
        self._durable_note_passes: set[int] = set()
        self._dotenv_loaded = False
        self._semantic_gate_cache: dict[str, bool] = {}
        self._active_run_id = ""
        self._active_objective = ""
        self._research_state_ready = False
        self._crawl_worker_auto_started = False

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
        self._active_run_id = run_id
        self._load_env_from_dotenv()
        self._ensure_research_state_store()
        self._record_provider_preflight()
        depth, cleaned_objective = self._split_depth(objective)
        research_objective = self._clean_objective(cleaned_objective)
        self._active_objective = research_objective
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
        settings = self._settings_for_general_complex_objective(
            settings,
            research_objective,
        )
        if pc_context is None:
            pc_context = self._auto_pc_context_for_run(
                research_objective,
                query,
                settings.depth,
                run_id,
            )
        pc_context_info = self._pc_context_summary(pc_context)
        plan = self._build_research_plan(
            research_objective,
            query,
            settings.depth,
            pc_context_info,
        )
        persistent_query_hints = self._persistent_evidence_query_hints(
            query,
            research_objective,
            limit=32,
        )
        if persistent_query_hints:
            plan["query_plan"] = self._sanitize_query_variants(
                persistent_query_hints + list(plan.get("query_plan") or []),
                query,
            )
        pc_query_seeds = self._pc_query_seeds(pc_context, query)
        if pc_query_seeds:
            plan["query_plan"] = self._sanitize_query_variants(
                pc_query_seeds + list(plan.get("query_plan") or []),
                query,
            )
        persistent_seed_urls = self._persistent_seed_urls(
            query,
            research_objective,
            limit=32,
        )
        seed_urls = self._source_seed_urls(
            research_objective,
            planning_context,
            pc_context,
        )
        if persistent_seed_urls:
            seed_urls = self._persistent_unique_urls(persistent_seed_urls + seed_urls)
        if seed_urls:
            plan["source_seeds"] = seed_urls
            self._enqueue_url_batch(
                seed_urls,
                query,
                run_id,
                source_url="seed-plan",
                priority=12.0,
            )
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
            pc_context=pc_context,
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
        synthesis_packet = self._build_synthesis_packet(
            research_objective,
            query,
            selected,
            settings.depth,
            plan,
            durable_notes,
            synthesis_mode,
        )
        summary = self._summarize(
            research_objective,
            selected,
            settings.depth,
            plan,
            query,
            durable_notes,
            synthesis_mode,
            synthesis_packet=synthesis_packet,
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
            synthesis_packet,
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

    def _research_state_path(self) -> Path:
        override = self._research_state_path_override
        if override is None:
            env_override = os.environ.get(
                "AGENTOS_RESEARCH_STATE_DB",
                "",
            ).strip()
            if env_override:
                override = Path(env_override)
        if override is not None:
            path = Path(override)
        else:
            path = self.workspace_root / ".agentos" / "research_state.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _connect_research_state(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._research_state_path())
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _crawl_broker_base_url(self) -> str:
        if os.environ.get("AGENTOS_DISABLE_CRAWL_BROKER") == "1":
            return ""
        if self._crawl_broker_url_override is not None:
            return str(self._crawl_broker_url_override or "").strip().rstrip("/")
        return os.environ.get("AGENTOS_CRAWL_BROKER_URL", "").strip().rstrip("/")

    def _crawl_broker_token(self) -> str:
        if self._crawl_broker_token_override is not None:
            return str(self._crawl_broker_token_override or "")
        return os.environ.get("AGENTOS_CRAWL_BROKER_TOKEN", "")

    def _crawl_broker_enabled(self) -> bool:
        return bool(self._crawl_broker_base_url())

    def crawl_broker_status(self) -> dict[str, Any]:
        return self._crawl_broker_request("/status")

    def crawl_broker_metrics(self) -> dict[str, Any]:
        return self._crawl_broker_request("/metrics")

    def crawl_broker_queue_inspect(
        self,
        *,
        limit: int = 24,
        statuses: list[str] | None = None,
        domain: str = "",
        worker_id: str = "",
        js_required: bool | None = None,
        shard_index: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"limit": int(limit)}
        normalized_statuses = [
            str(status).strip() for status in (statuses or []) if str(status).strip()
        ]
        if normalized_statuses:
            payload["statuses"] = normalized_statuses
        if str(domain or "").strip():
            payload["domain"] = str(domain).strip()
        if str(worker_id or "").strip():
            payload["worker_id"] = str(worker_id).strip()
        if js_required is not None:
            payload["js_required"] = bool(js_required)
        if shard_index is not None:
            payload["shard_index"] = int(shard_index)
        return self._crawl_broker_request(
            "/queue/inspect",
            payload,
            timeout_seconds=max(10.0, self.timeout_seconds),
        )

    def _crawl_broker_request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        base_url = self._crawl_broker_base_url()
        if not base_url:
            raise RuntimeError("crawl broker is not configured")
        method = "GET" if payload is None else "POST"
        headers = {
            "User-Agent": "AgentOS/1.0",
            "Accept": "application/json",
        }
        request_data: bytes | None = None
        if payload is not None:
            request_data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        token = self._crawl_broker_token().strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=request_data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds or self.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"crawl broker request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"crawl broker unavailable: {exc}") from exc
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("crawl broker returned a non-object JSON payload")
        return parsed

    @classmethod
    def _source_seed_urls(
        cls,
        objective: str,
        planning_context: dict[str, Any] | None,
        pc_context: dict[str, Any] | None,
    ) -> list[str]:
        candidates: list[str] = []
        if pc_context:
            candidates.extend(cls._pc_context_priority_urls(pc_context))
        candidates.extend(cls._software_agent_diagnostic_seed_urls(objective))
        candidates.extend(cls._collect_urls(objective))
        if planning_context:
            candidates.extend(cls._collect_urls(planning_context))

        deduped: list[str] = []
        seen: set[str] = set()
        for url in candidates:
            cleaned = url.rstrip(").,;]}>\"'")
            if not cls._is_safe_public_url(cleaned):
                continue
            if cls._is_low_signal_seed_url(cleaned):
                continue
            if cls._is_search_result_url(cleaned):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped[:48]

    def _auto_pc_context_for_run(
        self,
        objective: str,
        query: str,
        depth: str,
        run_id: str = "",
    ) -> dict[str, Any] | None:
        if os.environ.get("AGENTOS_DISABLE_AUTONOMOUS_BROWSER_CONTEXT") == "1":
            return None
        if not self._should_auto_pc_context_for_run(objective, query, depth):
            return None

        direct_urls = self._source_seed_urls(objective, None, None)[:8]
        if not direct_urls:
            return None

        rendered_pages = self._headless_browser_pool_fetch(
            direct_urls,
            max_chars=12_000,
            timeout_ms=16_000,
        )
        judged_results: list[dict[str, Any]] = []
        discovered_domains: list[str] = []
        kept_urls: list[str] = []
        search_queries = self._software_agent_diagnostic_queries(objective) or [query]

        for url in direct_urls:
            rendered = self._strip_dom_noise_tokens(str(rendered_pages.get(url) or ""))
            used_browser = bool(rendered)
            content = rendered or self._fetch_page_text(url, max_bytes=12_000)
            content = self._strip_dom_noise_tokens(content)
            quality_score = round(self._text_signal_score(content), 3)
            if quality_score < 0.08:
                continue

            title = self._label_from_url(url)
            source = ResearchSource(
                provider="pc-browser-research",
                title=title,
                url=url,
                authors=[self._source_domain(url)],
                abstract=content[:1800],
                score=24.0 if used_browser else 18.0,
                evidence_grade="tool-observation" if used_browser else "ungraded",
            )
            claim = self._compressed_claim(source, query)
            judged_results.append(
                {
                    "query": search_queries[0],
                    "title": title,
                    "url": url,
                    "domain": self._source_domain(url),
                    "abstract": content[:400],
                    "page_excerpt": content[:800],
                    "evidence_claims": [claim] if claim else [],
                    "content_quality": {
                        "quality_score": quality_score,
                        "used_browser": used_browser,
                    },
                    "judgment": (
                        "standalone browser context seeded from authoritative sources"
                    ),
                }
            )
            kept_urls.append(url)
            domain = self._source_domain(url)
            if domain and domain not in discovered_domains:
                discovered_domains.append(domain)

        if not kept_urls:
            return None

        self._record_provider_diagnostic(
            "standalone-browser-context",
            "ok",
            (
                f"seeded {len(kept_urls)} autonomous browser-reviewed urls"
                + (f" for {run_id}" if run_id else "")
            ),
        )
        return {
            "snapshot_path": "__autonomous_browser__",
            "pc_findings": {
                "search_queries": search_queries[:12],
                "judged_results": judged_results,
                "direct_urls": kept_urls,
                "discovered_domains": discovered_domains,
                "candidate_urls": kept_urls,
                "search_result_count": len(kept_urls),
                "frontier": {
                    "mode": "standalone-autonomous",
                    "deep_reads": len(kept_urls),
                    "candidate_urls": len(kept_urls),
                },
                "terminal_verifications": [],
            },
        }

    @classmethod
    def _should_auto_pc_context_for_run(
        cls,
        objective: str,
        query: str,
        depth: str,
    ) -> bool:
        combined = f"{objective} {query}".strip()
        if cls._looks_like_software_agent_diagnostic_objective(combined):
            return True
        if depth != "multi-hour":
            return False
        if not cls._looks_like_current_evidence_query(combined):
            return False
        return bool(
            re.search(
                r"\b(browser|sandbox|pc control|computer use|desktop|local pc)\b",
                combined.lower(),
            )
        )

    @classmethod
    def _pc_context_priority_urls(
        cls,
        pc_context: dict[str, Any],
    ) -> list[str]:
        pc_findings = pc_context.get("pc_findings") or {}
        if not isinstance(pc_findings, dict):
            return []
        candidates: list[str] = []
        candidates.extend(str(url) for url in (pc_findings.get("direct_urls") or []))
        for result in pc_findings.get("judged_results") or []:
            if not isinstance(result, dict):
                continue
            candidates.append(str(result.get("url") or ""))
        candidates.extend(str(url) for url in (pc_findings.get("candidate_urls") or []))
        deduped: list[str] = []
        seen: set[str] = set()
        for url in candidates:
            cleaned = url.strip().rstrip(").,;]}>\"'")
            if not cleaned or cleaned in seen:
                continue
            if not cls._is_safe_public_url(cleaned):
                continue
            host = urllib.parse.urlsplit(cleaned).netloc.lower().lstrip("www.")
            if host in {
                "finance.yahoo.com",
                "marketwatch.com",
                "cnbc.com",
                "news.google.com",
            }:
                continue
            if cls._is_search_result_url(cleaned):
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

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

    @staticmethod
    def _is_low_signal_seed_url(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        if not path:
            return False
        if re.search(
            r"\.(?:css|js|mjs|png|jpe?g|gif|svg|ico|webmanifest|woff2?|ttf|map)$",
            path,
        ):
            return True
        return any(
            token in path
            for token in (
                "favicon",
                "apple-touch-icon",
                "safari-pinned-tab",
                "site.webmanifest",
            )
        )

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
        if self._is_low_signal_seed_url(url):
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

    @staticmethod
    def _verified_terminal_claims(
        pc_findings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        verified: list[dict[str, Any]] = []
        for item in (pc_findings.get("terminal_verifications") or [])[:24]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            expression = str(item.get("expression") or "").strip()
            status = str(item.get("status") or "").strip().lower()
            try:
                exit_code = int(item.get("exit_code") or 0)
            except (TypeError, ValueError):
                exit_code = 1
            if not claim:
                continue
            verified.append(
                {
                    "claim": claim[:400],
                    "expression": expression[:120],
                    "status": status,
                    "exit_code": exit_code,
                    "verified": exit_code == 0
                    and status in {"ok", "completed", "success", "process-executed"},
                }
            )
        return verified

    @classmethod
    def _verification_matches_browser_result(
        cls,
        verification: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        claim = str(verification.get("claim") or "").strip().lower()
        if not claim:
            return False
        result_text = " ".join(
            [
                str(result.get("title") or ""),
                str(result.get("page_excerpt") or result.get("abstract") or ""),
                " ".join(
                    str(claim_text or "")
                    for claim_text in (result.get("evidence_claims") or [])[:4]
                ),
            ]
        ).lower()
        if not result_text.strip():
            return False
        if claim in result_text:
            return True
        keywords = [token for token in cls._keywords(claim) if len(token) >= 4][:5]
        if not keywords:
            return False
        overlap = sum(1 for token in keywords if token in result_text)
        return overlap >= min(2, len(keywords))

    def _pc_finding_seed_sources(
        self,
        pc_context: dict[str, Any] | None,
    ) -> list[ResearchSource]:
        if not pc_context:
            return []
        pc_findings = pc_context.get("pc_findings") or {}
        judged_results = pc_findings.get("judged_results") or []
        candidate_urls = pc_findings.get("candidate_urls") or []
        frontier = pc_findings.get("frontier") or {}
        terminal_verifications = self._verified_terminal_claims(pc_findings)
        sources: list[ResearchSource] = []
        seen_urls: set[str] = set()
        frontier_mode = str(frontier.get("mode") or "").strip().lower()
        judged_limit = 180 if frontier_mode == "expansive" else 40
        supplemental_limit = 120 if frontier_mode == "expansive" else 24

        for result in judged_results[:judged_limit]:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            if not self._is_safe_public_url(url):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(result.get("title") or self._label_from_url(url)).strip()
            domain = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
            excerpt = str(
                result.get("page_excerpt") or result.get("abstract") or ""
            ).strip()
            judgment = str(result.get("judgment") or "").strip()
            evidence_claims = [
                str(claim).strip()
                for claim in (result.get("evidence_claims") or [])
                if str(claim).strip()
            ]
            quality = result.get("content_quality") or {}
            try:
                quality_score = float(quality.get("quality_score") or 0.0)
            except (TypeError, ValueError):
                quality_score = 0.0
            quality_flags = ["browser-judged-source"]
            matched_verifications = [
                item
                for item in terminal_verifications
                if item.get("verified")
                and self._verification_matches_browser_result(item, result)
            ]
            if matched_verifications:
                quality_flags.append("browser-terminal-verified")
                quality_score = max(quality_score, 0.85)

            abstract_parts: list[str] = []
            if judgment:
                abstract_parts.append(judgment)
            if evidence_claims:
                abstract_parts.append("Claims: " + "; ".join(evidence_claims[:3]))
            if matched_verifications:
                formatted = []
                for item in matched_verifications[:2]:
                    claim_text = str(item.get("claim") or "").strip()
                    expression = str(item.get("expression") or "").strip()
                    if expression:
                        formatted.append(f"{claim_text} [terminal:{expression}]")
                    else:
                        formatted.append(claim_text)
                abstract_parts.append("Terminal verification: " + "; ".join(formatted))
            if excerpt:
                abstract_parts.append(excerpt[:1400])
            if not abstract_parts:
                abstract_parts.append(
                    "Browser-judged source captured during sandbox research."
                )

            sources.append(
                ResearchSource(
                    provider="pc-browser-research",
                    title=title[:160],
                    url=url,
                    authors=[domain] if domain else [],
                    abstract=" ".join(abstract_parts)[:3000],
                    score=32.0 + min(max(quality_score, 0.0), 1.0) * 30.0,
                    evidence_grade="tool-observation",
                    quality_flags=quality_flags,
                )
            )

        checkpoint_url_leads = self._pc_checkpoint_url_leads(pc_findings)
        graph_url_leads = self._pc_frontier_graph_top_urls(pc_findings)
        direct_url_count = len(pc_findings.get("direct_urls") or [])
        candidate_url_count = len(candidate_urls)
        checkpoint_url_count = len(checkpoint_url_leads)
        supplemental_urls = (
            list(pc_findings.get("direct_urls") or [])
            + list(candidate_urls)
            + checkpoint_url_leads
            + graph_url_leads
        )
        for index, url in enumerate(supplemental_urls[:supplemental_limit]):
            clean_url = str(url or "").strip()
            if not self._is_safe_public_url(clean_url):
                continue
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            if index < direct_url_count:
                quality_flag = "browser-discovered-url"
                is_candidate = False
            elif index < direct_url_count + candidate_url_count:
                quality_flag = "browser-frontier-candidate"
                is_candidate = True
            elif index < direct_url_count + candidate_url_count + checkpoint_url_count:
                quality_flag = "browser-checkpoint-url-lead"
                is_candidate = True
            else:
                quality_flag = "browser-frontier-graph-lead"
                is_candidate = True
            fetched = self._seed_source(clean_url)
            if fetched is None:
                if frontier_mode != "expansive":
                    continue
                fetched = ResearchSource(
                    provider="pc-browser-research",
                    title=self._label_from_url(clean_url)[:160],
                    url=clean_url,
                    authors=[self._source_domain(clean_url)],
                    abstract=(
                        "Browser frontier URL retained as an unverified lead "
                        "pending content fetch."
                    ),
                    score=0.1,
                    evidence_grade="weak",
                    quality_flags=[quality_flag, "unfetched-browser-lead"],
                )
            elif fetched.abstract.startswith("Seeded external source collected"):
                if frontier_mode != "expansive":
                    continue
                fetched.provider = "pc-browser-research"
                fetched.score = 0.1
                fetched.evidence_grade = "weak"
                fetched.quality_flags = [
                    quality_flag,
                    "unfetched-browser-lead",
                ]
            else:
                fetched.provider = "pc-browser-research"
                fetched.score = max(
                    float(fetched.score or 0.0),
                    18.0 if is_candidate else 24.0,
                )
                fetched.evidence_grade = "ungraded"
                fetched.quality_flags = [quality_flag, "browser-fetched-seed"]
            sources.append(fetched)

        if sources:
            self._record_provider_diagnostic(
                "pc-browser-research",
                "ok",
                f"seeded {len(sources)} browser-judged sources",
            )
        return sources

    @staticmethod
    def _pc_frontier_checkpoints(pc_findings: dict[str, Any]) -> list[dict[str, Any]]:
        checkpoints = pc_findings.get("frontier_checkpoints") or []
        if not isinstance(checkpoints, list):
            return []
        return [item for item in checkpoints if isinstance(item, dict)]

    @staticmethod
    def _pc_unique_strings(items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _pc_checkpoint_url_leads(self, pc_findings: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for checkpoint in self._pc_frontier_checkpoints(pc_findings)[:48]:
            for url in checkpoint.get("url_leads") or []:
                clean_url = str(url or "").strip()
                if clean_url:
                    urls.append(clean_url)
        return self._pc_unique_strings(urls)[:96]

    def _pc_frontier_graph_top_urls(self, pc_findings: dict[str, Any]) -> list[str]:
        frontier_graph = pc_findings.get("frontier_graph") or {}
        summary = frontier_graph.get("summary") or {}
        urls: list[str] = []
        for url in summary.get("top_urls") or []:
            clean_url = str(url or "").strip()
            if clean_url:
                urls.append(clean_url)
        return self._pc_unique_strings(urls)[:48]

    def _pc_query_seeds(
        self,
        pc_context: dict[str, Any] | None,
        query: str,
    ) -> list[str]:
        if not pc_context:
            return []
        pc_findings = pc_context.get("pc_findings") or {}
        candidates: list[str] = []
        for item in (pc_findings.get("search_queries") or [])[:64]:
            text = str(item or "").strip()
            if text:
                candidates.append(text[:240])
        for verification in self._verified_terminal_claims(pc_findings)[:32]:
            if not verification.get("verified"):
                continue
            claim_text = str(verification.get("claim") or "").strip()
            if not claim_text:
                continue
            keywords = self._query_core_terms(claim_text)
            if keywords and len(keywords.split()) >= 3:
                candidates.append(keywords[:240])
            else:
                candidates.append(claim_text[:240])
        for result in (pc_findings.get("judged_results") or [])[:64]:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "").strip()
            if title:
                candidates.append(title[:240])
            for claim in (result.get("evidence_claims") or [])[:2]:
                claim_text = str(claim or "").strip()
                if not claim_text:
                    continue
                keywords = self._query_core_terms(f"{title} {claim_text}".strip())
                if keywords and len(keywords.split()) >= 3:
                    candidates.append(keywords[:240])
        for checkpoint in self._pc_frontier_checkpoints(pc_findings)[:48]:
            for follow_up in checkpoint.get("follow_up_queries") or []:
                text = str(follow_up or "").strip()
                if text:
                    candidates.append(text[:240])
            for contradiction in checkpoint.get("contradictions") or []:
                text = str(contradiction or "").strip()
                if not text:
                    continue
                keywords = self._query_core_terms(f"{query} {text}".strip())
                if keywords and len(keywords.split()) >= 3:
                    candidates.append(keywords[:240])
            for evidence_gap in checkpoint.get("missing_evidence") or []:
                text = str(evidence_gap or "").strip()
                if not text:
                    continue
                keywords = self._query_core_terms(f"{query} {text}".strip())
                if keywords and len(keywords.split()) >= 3:
                    candidates.append(keywords[:240])
            for domain in checkpoint.get("domain_leads") or []:
                domain_text = str(domain or "").strip()
                if not domain_text:
                    continue
                candidates.append(f"{query[:180]} site:{domain_text}"[:240])
        return self._sanitize_query_variants(candidates, query)[:160]

    @staticmethod
    def _source_domain(url: str) -> str:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return host

    @classmethod
    def _unique_source_url_count(cls, sources: list[ResearchSource]) -> int:
        seen: set[str] = set()
        for source in sources:
            url = str(source.url or "").strip()
            if not url:
                continue
            if not cls._is_safe_public_url(url):
                continue
            seen.add(url)
        return len(seen)

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
            # 120 passes guarantees 1000+ real URL fetches through I/O:
            # each pass runs parallel queries + enrichment + citation chasing.
            return 120
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
    def _effective_novelty_threshold(
        depth: str,
        targets: dict[str, Any],
    ) -> float:
        try:
            configured = float(targets.get("min_novelty_rate") or 0.0)
        except (TypeError, ValueError):
            configured = 0.0
        if depth == "multi-hour":
            return max(configured, 0.03)
        if depth == "standard":
            return max(configured, 0.0)
        return max(configured, 0.0)

    @staticmethod
    def _pass_variants(
        variants: list[str],
        pass_index: int,
        limit: int,
    ) -> list[str]:
        if not variants:
            return []
        if len(variants) <= limit:
            stride = max(1, limit // 2)
        else:
            overlap = max(1, limit // 4)
            stride = max(1, limit - overlap)
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
            text = cls._normalize_research_plan_query(str(variant or ""), query)
            if not text:
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
            1 for source in selected if cls._source_is_on_topic(source, query)
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

    @classmethod
    def _source_is_on_topic(cls, source: ResearchSource, query: str) -> bool:
        if not query:
            return True
        if "off-topic" in (source.quality_flags or []):
            return False
        if source.score <= 0.0 and source.provider != "gemini-flash":
            return False
        text = cls._strip_dom_noise_tokens(f"{source.title} {source.abstract}")
        if not text:
            return False
        alignment = cls._objective_alignment_score(text, query)
        entity_hits = cls._entity_hit_count(text, query)
        deterministic = cls._passes_deterministic_semantic_gate(text, query)
        if cls._looks_like_market_query(query):
            market_signal = cls._has_market_signal(text)
            actionable_market_signal = cls._has_actionable_market_signal(text)
            if not market_signal and entity_hits == 0:
                return False
            if entity_hits > 0:
                return deterministic or alignment >= 0.2
            if cls._looks_like_public_security_query(query):
                if not actionable_market_signal and alignment < 0.45:
                    return False
                if not actionable_market_signal and any(
                    flag in (source.quality_flags or [])
                    for flag in (
                        "promo-market-listicle",
                        "low-signal-market-host",
                    )
                ):
                    return False
            return (deterministic and actionable_market_signal) or alignment >= 0.35
        if source.provider == "gemini-flash":
            return alignment >= 0.25 or entity_hits > 0 or source.relevance >= 0.55
        return deterministic or alignment >= 0.35 or entity_hits > 0

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
        generic_terms = self._generic_query_terms()
        query_anchor_terms = self._objective_anchor_terms(query)
        for source in selected[:10]:
            text = f"{source.title} {source.abstract}"
            for token in self._keywords(text):
                if len(token) < 4:
                    continue
                if token in generic_terms:
                    continue
                if query_anchor_terms and token not in query_anchor_terms:
                    if self._objective_alignment_score(token, query) < 0.25:
                        continue
                if token in evidence_terms:
                    continue
                evidence_terms.append(token)
                if len(evidence_terms) >= 12:
                    break
            if len(evidence_terms) >= 12:
                break

        math_mode = self._looks_like_math_query(query)
        axes = self._fallback_research_axes(query, depth)
        if math_mode:
            axes = list(
                dict.fromkeys([*map(str, self._math_focus_terms(query)), *axes])
            )

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
            candidate = self._trim_query_variant_text(variant)
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

    async def _search_openalex_async(
        self,
        query: str,
        limit: int | None = None,
        client: Any | None = None,
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
        payload = await self._get_json_async(
            f"https://api.openalex.org/works?{params}",
            client=client,
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
        # GENERALITY GUARD: an LLM "tool observation" is a parametric-memory
        # restatement, not a primary source.  Admitting it as a ResearchSource
        # with a synthetic URL pollutes evidence with hallucinated content and
        # causes the deep-research output to look like the LLM wrote it from
        # memory instead of from real fetched pages.  Claude/Gemini/GPT deep
        # research all use the LLM as a reasoner *over* fetched pages, never
        # as a fake source.  We mirror that contract: the Gemini call is
        # available for synthesis via ``_call_ai_text`` but must not produce
        # ResearchSource records.  We keep the diagnostic so operators know
        # the provider was deliberately skipped.
        self._record_provider_diagnostic(
            "gemini-flash",
            "disabled",
            (
                "gemini-flash is no longer admitted as an evidence provider; "
                "LLM output is reserved for synthesis/critique over real "
                "fetched pages, never as a primary source."
            ),
        )
        return []
        # Legacy implementation retained below for reference only.  It is
        # unreachable; remove only after dependent tests have migrated.
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

        For market queries, thinks like a Wall Street analyst: what data rooms
        to check, what filings to pull, what catalysts to track, what comps
        to use, what the bear/bull thesis looks like.

        For all other topics, thinks like a rigorous scientist.

        Returns a dict with:
          causal_connections  – list of "A affects B because ..." strings
          authoritative_domains – specific domains/URLs worth targeting
          reasoning_queries – search queries derived from causal thinking
          subquestions – specific sub-questions to answer
        """
        is_market = self._looks_like_market_query(objective)
        if is_market:
            system = (
                "You are a senior equity research analyst at a top-tier investment bank. "
                "Given a research objective, think through the COMPLETE research process "
                "a Wall Street analyst would follow to build a publishable research note:\n\n"
                "THINK ABOUT:\n"
                "- What company(ies) or securities are the primary subjects?\n"
                "- What SEC filings, earnings transcripts, or regulatory documents "
                "would you pull first? (10-K, 10-Q, 8-K, proxy, prospectus)\n"
                "- What authoritative domains have the best primary data? "
                "(sec.gov, company IR pages, Fed data, BLS, BEA, IMF, industry databases)\n"
                "- What are the 5-8 most important sub-questions to answer for "
                "a complete investment thesis? (valuation, catalysts, competition, "
                "management, risks, macro backdrop, technicals)\n"
                "- What 8-12 specific search queries would uncover the strongest "
                "evidence? Include queries for: SEC filings, earnings guidance, "
                "analyst price targets, short interest, insider transactions, "
                "comparable companies, macro factors, and bear case arguments\n"
                "- What causal chains are at play? (e.g., 'Fed rate cuts increase "
                "P/E expansion because cost of equity falls')\n\n"
                "Respond ONLY with valid JSON."
            )
        else:
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
            '  "authoritative_domains": ["sec.gov", "domain.org/path", ...],\n'
            '  "reasoning_queries": ["specific query derived from causal thinking", ...],\n'
            '  "subquestions": ["precise sub-question to answer", ...]\n'
            "}\n"
            "reasoning_queries must be 5-12 concrete search phrases (not generic "
            "variations). authoritative_domains should be real domains likely to "
            "have primary-source evidence — for market queries, include SEC EDGAR, "
            "company investor-relations pages, and authoritative financial data sites."
        )
        raw = self._call_ai_text(system, user)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                return {
                    "causal_connections": list(parsed.get("causal_connections") or [])[
                        :10
                    ],
                    "authoritative_domains": list(
                        parsed.get("authoritative_domains") or []
                    )[:15],
                    "reasoning_queries": list(parsed.get("reasoning_queries") or [])[
                        :12
                    ],
                    "subquestions": list(parsed.get("subquestions") or [])[:10],
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
        for src in selected[:20]:
            year = f" ({src.year})" if src.year else ""
            snippet = (src.abstract or src.title)[:150].replace("\n", " ")
            source_lines.append(f"- [{src.provider}] {src.title}{year}: {snippet}")
        source_summary = "\n".join(source_lines)
        domain_hint = ", ".join(existing_domains[:20]) if existing_domains else "none"
        coverage_hint = coverage or {}

        is_market = self._looks_like_market_query(objective)
        if is_market:
            system = (
                "You are a Managing Director at a top-tier investment bank doing "
                "buy-side fundamental research. You have just completed a retrieval "
                "pass and need to identify critical gaps in your investment thesis.\n\n"
                "Think like a sell-side analyst building a research note:\n"
                "- What financial metrics are still missing? (DCF inputs, margin trends, "
                "FCF yield, EV/EBITDA, net debt, ROIC, capital allocation)\n"
                "- What qualitative factors haven't been sourced yet? (competitive moat, "
                "management quality, regulatory risk, TAM expansion, customer concentration)\n"
                "- What macro factors are unaddressed? (rate sensitivity, FX exposure, "
                "commodity costs, geopolitical risk)\n"
                "- What's the bear case that hasn't been stress-tested yet?\n"
                "- What catalyst events are upcoming and not yet sourced?\n"
                "- What data from SEC EDGAR, earnings transcripts, or insider filings "
                "is still missing?\n"
                "Generate highly specific search queries a real analyst would run — "
                "not generic stock terms. Respond ONLY with valid JSON."
            )
        else:
            system = (
                "You are a research gap analyst. Given what a research engine has "
                "found so far, reason about what important angles are missing, what "
                "causal connections implied by the findings should be followed up, "
                "whether the evidence is current enough, and what specific search "
                "queries would best fill the gaps. Respond ONLY with valid JSON."
            )

        if is_market:
            query_count_hint = "8-12"
        else:
            query_count_hint = "6-8"

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
            f"follow_up_queries must be {query_count_hint} targeted search phrases "
            "derived from your gap analysis — not generic expansions of the core query. "
            "Think: what causal factor is implied but not yet sourced? What "
            "recent event could change the picture? Which authority has not yet "
            "been consulted? For finance: what SEC filing, earnings metric, or "
            "analyst estimate is still unverified?"
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
                    candidate = self._trim_query_variant_text(q)
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
                    # For market queries, use a more permissive alignment threshold
                    # because analyst sub-queries ("NVDA gross margin trend Q1 2026")
                    # may not share all objective terms but are highly relevant.
                    min_overlap = 1 if is_market else 2
                    min_align = 0.25 if is_market else 0.4
                    if (
                        len(obj_terms) >= 4
                        and overlap < min_overlap
                        and self._objective_alignment_score(candidate, objective)
                        < min_align
                    ):
                        continue
                    align_floor = 0.2 if is_market else 0.35
                    if (
                        self._objective_alignment_score(candidate, objective)
                        < align_floor
                    ):
                        continue
                    filtered.append(candidate)
                # Return up to 12 queries for market research, 8 for others.
                return filtered[:12] if is_market else filtered[:8]
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

        axes = self._fallback_research_axes(objective, "standard")
        queries: list[str] = []
        for anchor in anchors[:3]:
            queries.append(anchor)
            for axis in axes[:5]:
                queries.append(f"{anchor} {axis}")

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
        for domain in authoritative_domains[:15]:
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

        For market queries, generates Wall Street analyst perspectives:
        fundamentals, valuation comps, catalyst tracking, bear thesis,
        institutional positioning, macro factors, technical setup, regulatory risk.
        """
        n = 8 if depth == "multi-hour" else (5 if depth == "standard" else 3)
        is_market = DeepResearchEngine._looks_like_market_query(objective)
        if is_market:
            system = (
                "You are a Wall Street equity research analyst. Given a research query "
                "about stocks/markets, generate the specific research perspectives an "
                "analyst would use to build a complete investment thesis. Each perspective "
                "must target a distinct type of evidence that institutional investors care about. "
                "Respond with a valid JSON array only — no prose, no markdown fences."
            )
            user = (
                f"Query: {query}\n"
                f"Objective: {objective}\n\n"
                f"Generate {n} Wall Street research perspectives as a JSON array. "
                "Each element must have: "
                '"name" (short lowercase-hyphenated slug like "earnings-growth", "comps-valuation"), '
                '"goal" (one sentence describing what investment evidence this perspective collects), '
                '"keywords" (list of 4-6 specific financial search keywords — include '
                "company names, tickers, metric names), "
                '"queries" (list of 2-3 specific search phrases for SEC EDGAR, '
                "earnings calls, analyst reports, financial databases — NOT generic placeholders).\n"
                "Cover angles like: fundamentals, valuation vs comps, upcoming catalysts, "
                "bear thesis/short interest, institutional positioning, macro backdrop, "
                "management credibility, competitive moat, regulatory risk."
            )
        else:
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
        except Exception as _persp_exc:
            import warnings

            warnings.warn(
                f"AI research-perspective generation failed ({_persp_exc!r}); "
                "falling back to generic perspectives.",
                RuntimeWarning,
                stacklevel=2,
            )
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
        if DeepResearchEngine._looks_like_math_query(query):
            axis_specs = [
                (
                    "established-results",
                    "Collect known theorems and formal results on the objective.",
                    ["known theorem", "established result", "formal statement"],
                ),
                (
                    "proof-barriers",
                    "Identify proof barriers, failed approaches, and missing lemmas.",
                    ["theorem barrier", "proof obstruction", "missing lemma"],
                ),
                (
                    "computational-verification",
                    "Check finite verification evidence and computational limits.",
                    ["finite verification", "computational evidence", "verified range"],
                ),
                (
                    "definitions",
                    "Pin down definitions, reductions, and equivalent formulations.",
                    ["definition", "equivalent formulation", "reduction"],
                ),
                (
                    "counterexamples",
                    "Search for counterexamples, edge cases, and negative results.",
                    ["counterexample", "edge case", "negative result"],
                ),
            ]
        elif DeepResearchEngine._looks_like_current_evidence_query(query):
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

    # ------------------------------------------------------------------
    # Content enrichment and citation chasing
    # ------------------------------------------------------------------

    # Domains that require JavaScript rendering to return real content.
    # requests/urllib only returns a blank/paywalled shell for these.

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

    @staticmethod
    def _settings_for_depth(depth: str) -> ResearchSettings:
        return _settings_policy.settings_for_depth(depth)

    @staticmethod
    def _settings_for_current_web(settings: ResearchSettings) -> ResearchSettings:
        return _settings_policy.settings_for_current_web(settings)

    @classmethod
    def _settings_for_general_complex_objective(
        cls,
        settings: ResearchSettings,
        objective: str,
    ) -> ResearchSettings:
        return _settings_policy.settings_for_general_complex_objective(
            settings,
            objective,
            looks_like_academic_query=cls._looks_like_academic_query,
            looks_like_software_agent_query=cls._looks_like_software_agent_query,
            looks_like_comprehensive_research=cls._looks_like_comprehensive_research,
        )

    def _query_parallel_worker_count(
        self,
        depth: str,
        query_count: int,
        provider_count: int,
    ) -> int:
        return _settings_policy.query_parallel_worker_count(
            depth,
            query_count,
            provider_count,
            current_web_mode=self._looks_like_current_evidence_query(
                self._active_objective
            ),
        )

    def _provider_parallel_worker_count(self, provider_count: int) -> int:
        depth = self.research_depth_for_objective(self._active_objective)
        return _settings_policy.provider_parallel_worker_count(
            provider_count,
            depth=depth,
            current_web_mode=self._looks_like_current_evidence_query(
                self._active_objective
            ),
        )

    @staticmethod
    def _current_web_targets(depth: str) -> dict[str, int | float]:
        return _settings_policy.current_web_targets(depth)

    @classmethod
    def _current_web_target_overrides(
        cls,
        targets: dict[str, Any],
        depth: str,
    ) -> dict[str, Any]:
        del cls
        return _settings_policy.current_web_target_overrides(targets, depth)

    # Maximum proportion of final selected sources from any single provider.
    # Maximum proportion of final selected sources from any single provider.

    # Scoring weights for scholarly sources (openalex/semantic-scholar/crossref).
    # Named so they can be understood and adjusted without hunting for magic numbers.

    @classmethod
    def _entity_hit_count(cls, text: str, query: str) -> int:
        lower_text = text.lower()
        return sum(
            1 for term in cls._entity_terms_from_query(query) if term in lower_text
        )

    @staticmethod
    def _has_market_signal(text: str) -> bool:
        lower = text.lower()
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
            "cash flow",
            "free cash flow",
            "sec filing",
            "10-k",
            "10-q",
        }
        return DeepResearchEngine._has_market_identifiers(text) or any(
            token in lower for token in market_vocab
        )

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
        if query:
            meaningful_words = [
                word
                for word in words
                if word not in cls._generic_query_terms() and word not in noise_tokens
            ]
            broad_domain_terms = {
                "stock",
                "stocks",
                "market",
                "markets",
                "equity",
                "equities",
                "science",
                "scientific",
                "software",
                "system",
                "systems",
                "policy",
                "policies",
            }
            has_entity = any(
                term in lower for term in cls._entity_terms_from_query(query)
            )
            if len(meaningful_words) < 2 and not has_entity:
                return True
            if (
                len(meaningful_words) == 1
                and meaningful_words[0] in broad_domain_terms
                and not has_entity
            ):
                return True
        if (
            query
            and (
                cls._objective_anchor_terms(query)
                or cls._entity_terms_from_query(query)
            )
            and cls._objective_alignment_score(lower, query) < 0.2
        ):
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
        for word in words:
            if cls._looks_like_gibberish_query_token(word, query_tokens):
                return True
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
    def _looks_like_gibberish_query_token(
        word: str,
        query_tokens: set[str] | None = None,
    ) -> bool:
        token = str(word or "").lower().strip("-")
        if len(token) < 6:
            return False
        if query_tokens and token in query_tokens:
            return False
        if not re.fullmatch(r"[a-z-]+", token):
            return False

        parts = [part for part in token.split("-") if part]
        if len(parts) > 1:
            return any(
                DeepResearchEngine._looks_like_gibberish_query_token(
                    part,
                    query_tokens,
                )
                for part in parts
            )

        vowels = sum(1 for ch in token if ch in "aeiou")
        vowel_ratio = vowels / max(len(token), 1)
        rare_letters = set("qxzjkvwy")
        rare_ratio = sum(1 for ch in token if ch in rare_letters) / len(token)

        if vowels == 0:
            return True
        if (
            vowels <= 1
            and re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", token)
            and (vowel_ratio < 0.25 or rare_ratio >= 0.3)
        ):
            return True
        return rare_ratio >= 0.4 and vowel_ratio < 0.35

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

    @classmethod
    def _looks_like_software_agent_diagnostic_objective(cls, query: str) -> bool:
        if not cls._looks_like_software_agent_query(query):
            return False
        lower = query.lower()
        return bool(
            re.search(
                (
                    r"\b(fix|issue|issues|bug|bugs|failure|failures|gap|gaps|"
                    r"shallow|template|comparable|comparison|benchmark|"
                    r"claude|gpt|gemini|browser|sandbox|pc control|computer use|"
                    r"retrieval|breadth|coverage|10k|1000|ranking|synthesis|"
                    r"underperform|why)\b"
                ),
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
