from __future__ import annotations

import json
import hashlib
import html
import ipaddress
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import combinations
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


class _HeadlessBrowserWorkerPool:
    def __init__(
        self,
        worker_count: int,
        bundle_factory: Any,
        render_with_context: Any,
        bundle_cleanup: Any,
    ) -> None:
        self.worker_count = max(1, worker_count)
        self._bundle_factory = bundle_factory
        self._render_with_context = render_with_context
        self._bundle_cleanup = bundle_cleanup
        self._local = threading.local()
        self._lock = threading.Lock()
        self._bundles: list[dict[str, Any]] = []

    def _bundle(self) -> dict[str, Any] | None:
        bundle = getattr(self._local, "bundle", None)
        if bundle is not None:
            return bundle
        bundle = self._bundle_factory()
        if not bundle:
            return None
        self._local.bundle = bundle
        with self._lock:
            self._bundles.append(bundle)
        return bundle

    def _render_one(self, url: str, max_chars: int, timeout_ms: int) -> str:
        bundle = self._bundle()
        if not bundle:
            return ""
        return self._render_with_context(
            bundle["context"],
            url,
            max_chars,
            timeout_ms,
        )

    def render_many(
        self,
        urls: list[str],
        max_chars: int,
        timeout_ms: int,
    ) -> dict[str, str]:
        if not urls:
            return {}
        results: dict[str, str] = {}
        try:
            with ThreadPoolExecutor(
                max_workers=max(1, min(self.worker_count, len(urls)))
            ) as executor:
                futures = {
                    executor.submit(
                        self._render_one,
                        url,
                        max_chars,
                        timeout_ms,
                    ): url
                    for url in urls
                }
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        text = future.result()
                    except Exception:
                        text = ""
                    if text:
                        results[url] = text
        finally:
            self.close()
        return results

    def close(self) -> None:
        bundles = list(self._bundles)
        self._bundles.clear()
        for bundle in bundles:
            try:
                self._bundle_cleanup(bundle)
            except Exception:
                continue


class DeepResearchEngine:
    """MCP-friendly live research fallback using public scholarly APIs."""

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

    @staticmethod
    def _deserialize_research_sources(payload: Any) -> list[ResearchSource]:
        if not isinstance(payload, list):
            return []
        allowed = set(ResearchSource.__dataclass_fields__)
        sources: list[ResearchSource] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            normalized = {key: item.get(key) for key in allowed if key in item}
            try:
                sources.append(ResearchSource(**normalized))
            except TypeError:
                continue
        return sources

    def _ensure_research_state_store(self) -> None:
        if self._research_state_ready:
            return
        if self._crawl_broker_enabled():
            self._research_state_ready = True
            return
        with closing(self._connect_research_state()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS crawl_queue (
                        url TEXT PRIMARY KEY,
                        domain TEXT NOT NULL,
                        priority REAL NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'queued',
                        source_query TEXT NOT NULL DEFAULT '',
                        source_url TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL DEFAULT '',
                        js_required INTEGER NOT NULL DEFAULT 0,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_claimed_by TEXT NOT NULL DEFAULT '',
                        last_claimed_at TEXT NOT NULL DEFAULT '',
                        last_error TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_crawl_queue_status_priority
                    ON crawl_queue(status, priority DESC, updated_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS crawl_observations (
                        url TEXT PRIMARY KEY,
                        domain TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        excerpt TEXT NOT NULL DEFAULT '',
                        source_query TEXT NOT NULL DEFAULT '',
                        query_hints_json TEXT NOT NULL DEFAULT '[]',
                        outbound_urls_json TEXT NOT NULL DEFAULT '[]',
                        content_hash TEXT NOT NULL DEFAULT '',
                        signal_score REAL NOT NULL DEFAULT 0,
                        used_browser INTEGER NOT NULL DEFAULT 0,
                        worker_id TEXT NOT NULL DEFAULT '',
                        fetched_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_crawl_observations_domain_signal
                    ON crawl_observations(domain, signal_score DESC, updated_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_claims (
                        claim_key TEXT PRIMARY KEY,
                        claim_text TEXT NOT NULL,
                        support_count INTEGER NOT NULL DEFAULT 0,
                        contradiction_count INTEGER NOT NULL DEFAULT 0,
                        confidence_rank INTEGER NOT NULL DEFAULT 0,
                        source_urls_json TEXT NOT NULL DEFAULT '[]',
                        providers_json TEXT NOT NULL DEFAULT '[]',
                        perspectives_json TEXT NOT NULL DEFAULT '[]',
                        last_seen_run TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_domains (
                        domain TEXT PRIMARY KEY,
                        score REAL NOT NULL DEFAULT 0,
                        observation_count INTEGER NOT NULL DEFAULT 0,
                        urls_json TEXT NOT NULL DEFAULT '[]',
                        claim_keys_json TEXT NOT NULL DEFAULT '[]',
                        contradiction_keys_json TEXT NOT NULL DEFAULT '[]',
                        last_seen_run TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_contradictions (
                        contradiction_key TEXT PRIMARY KEY,
                        contradiction_text TEXT NOT NULL,
                        count INTEGER NOT NULL DEFAULT 0,
                        claim_keys_json TEXT NOT NULL DEFAULT '[]',
                        source_urls_json TEXT NOT NULL DEFAULT '[]',
                        last_seen_run TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL
                    )
                    """
                )
        self._research_state_ready = True

    @staticmethod
    def _load_json_text_list(raw: Any) -> list[str]:
        if not raw:
            return []
        parsed = raw
        if not isinstance(parsed, list):
            try:
                parsed = json.loads(str(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                return []
        if not isinstance(parsed, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _merge_text_lists(*groups: list[str], limit: int = 64) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                result.append(text)
                if len(result) >= limit:
                    return result
        return result

    def _persistent_unique_urls(self, urls: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for url in urls:
            clean = str(url or "").strip()
            if not clean or clean in seen:
                continue
            if not self._is_safe_public_url(clean):
                continue
            if self._is_low_signal_seed_url(clean):
                continue
            seen.add(clean)
            result.append(clean)
        return result

    def _persistent_match_score(
        self,
        text: str,
        query: str,
        objective: str,
    ) -> float:
        alignment = max(
            self._objective_alignment_score(text, query),
            self._objective_alignment_score(text, objective),
        )
        entity_hits = max(
            self._entity_hit_count(text, query),
            self._entity_hit_count(text, objective),
        )
        if entity_hits > 0:
            alignment += 0.15
        return min(alignment, 1.0)

    def _enqueue_url_batch(
        self,
        urls: list[str],
        source_query: str,
        run_id: str,
        source_url: str = "",
        priority: float = 0.0,
    ) -> None:
        safe_urls = self._persistent_unique_urls(urls)
        if not safe_urls:
            return
        if self._crawl_broker_enabled():
            self._crawl_broker_request(
                "/queue/enqueue",
                {
                    "urls": safe_urls,
                    "source_query": str(source_query or "")[:320],
                    "run_id": run_id,
                    "source_url": str(source_url or "")[:320],
                    "priority": float(priority),
                },
                timeout_seconds=max(10.0, self.timeout_seconds),
            )
            self._maybe_start_detached_crawl_workers()
            return
        self._ensure_research_state_store()
        now = datetime.now(UTC).isoformat()
        with closing(self._connect_research_state()) as connection:
            with connection:
                for url in safe_urls:
                    connection.execute(
                        """
                        INSERT INTO crawl_queue(
                            url,
                            domain,
                            priority,
                            status,
                            source_query,
                            source_url,
                            run_id,
                            js_required,
                            attempts,
                            created_at,
                            updated_at
                        )
                        VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, 0, ?, ?)
                        ON CONFLICT(url) DO UPDATE SET
                            priority = CASE
                                WHEN excluded.priority > crawl_queue.priority
                                THEN excluded.priority
                                ELSE crawl_queue.priority
                            END,
                            source_query = CASE
                                WHEN length(crawl_queue.source_query) = 0
                                THEN excluded.source_query
                                ELSE crawl_queue.source_query
                            END,
                            source_url = CASE
                                WHEN length(crawl_queue.source_url) = 0
                                THEN excluded.source_url
                                ELSE crawl_queue.source_url
                            END,
                            run_id = excluded.run_id,
                            status = CASE
                                WHEN crawl_queue.status IN ('failed', 'stale')
                                THEN 'queued'
                                ELSE crawl_queue.status
                            END,
                            js_required = CASE
                                WHEN excluded.js_required > crawl_queue.js_required
                                THEN excluded.js_required
                                ELSE crawl_queue.js_required
                            END,
                            updated_at = excluded.updated_at
                        """,
                        (
                            url,
                            self._source_domain(url),
                            float(priority),
                            str(source_query or "")[:320],
                            str(source_url or "")[:320],
                            run_id,
                            1 if self._needs_browser(url) else 0,
                            now,
                            now,
                        ),
                    )
        self._maybe_start_detached_crawl_workers()

    def _auto_start_crawl_workers_enabled(self) -> bool:
        if os.environ.get("AGENTOS_CRAWL_WORKER_PROCESS") == "1":
            return False
        if os.environ.get("AGENTOS_DISABLE_AUTO_CRAWL_WORKERS") == "1":
            return False
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        return True

    def _detached_crawl_worker_count(self) -> int:
        configured = os.environ.get("AGENTOS_CRAWL_WORKER_COUNT", "").strip()
        if configured:
            try:
                return max(0, min(int(configured), 8))
            except ValueError:
                return 0
        cpu_count = os.cpu_count() or 2
        return max(1, min(4, cpu_count // 4 or 1))

    @staticmethod
    def _detached_crawl_batch_size() -> int:
        configured = os.environ.get("AGENTOS_CRAWL_WORKER_BATCH_SIZE", "").strip()
        if configured:
            try:
                return max(1, min(int(configured), 24))
            except ValueError:
                return 6
        return 6

    @staticmethod
    def _detached_crawl_poll_interval() -> float:
        configured = os.environ.get(
            "AGENTOS_CRAWL_WORKER_POLL_INTERVAL",
            "",
        ).strip()
        if configured:
            try:
                return max(2.0, min(float(configured), 300.0))
            except ValueError:
                return 15.0
        return 15.0

    @staticmethod
    def _detached_crawl_claim_ttl() -> int:
        configured = os.environ.get("AGENTOS_CRAWL_WORKER_CLAIM_TTL", "").strip()
        if configured:
            try:
                return max(60, min(int(configured), 7200))
            except ValueError:
                return 900
        return 900

    def _maybe_start_detached_crawl_workers(self) -> None:
        if self._crawl_worker_auto_started:
            return
        if not self._auto_start_crawl_workers_enabled():
            return
        worker_count = self._detached_crawl_worker_count()
        if worker_count <= 0:
            return
        try:
            from agentos_orchestrator.product import (
                CrawlWorkerManager,
                CrawlWorkerServiceManager,
            )

            service_manager = CrawlWorkerServiceManager(
                self.workspace_root,
                python_executable=sys.executable,
            )
            service_status = service_manager.status()
            if service_status.installed:
                if service_status.status != "running":
                    service_manager.start(task_name=service_status.task_name)
                self._crawl_worker_auto_started = True
                return

            manager = CrawlWorkerManager(
                self.workspace_root,
                python_executable=sys.executable,
            )
            status = manager.status()
            if status.status != "running":
                manager.start(
                    worker_count=worker_count,
                    queue_db_path=self._research_state_path(),
                    broker_url=self._crawl_broker_base_url() or None,
                    broker_token=self._crawl_broker_token() or None,
                    poll_interval_seconds=self._detached_crawl_poll_interval(),
                    batch_size=self._detached_crawl_batch_size(),
                    claim_ttl_seconds=self._detached_crawl_claim_ttl(),
                )
            self._crawl_worker_auto_started = True
        except Exception:
            return

    def _requeue_stale_crawl_claims(
        self,
        stale_after_seconds: int = 900,
    ) -> int:
        if self._crawl_broker_enabled():
            response = self._crawl_broker_request(
                "/queue/requeue-stale",
                {"stale_after_seconds": int(stale_after_seconds)},
            )
            return int(response.get("reclaimed_count") or 0)
        self._ensure_research_state_store()
        cutoff = datetime.now(UTC) - timedelta(seconds=max(stale_after_seconds, 60))
        with closing(self._connect_research_state()) as connection:
            with connection:
                result = connection.execute(
                    """
                    UPDATE crawl_queue
                    SET status = 'queued',
                        last_error = 'claim-expired',
                        updated_at = ?
                    WHERE status = 'claimed'
                      AND last_claimed_at != ''
                      AND last_claimed_at < ?
                    """,
                    (
                        datetime.now(UTC).isoformat(),
                        cutoff.isoformat(),
                    ),
                )
        return int(result.rowcount or 0)

    def _peek_crawl_queue_rows(
        self,
        limit: int,
        *,
        statuses: tuple[str, ...] = ("queued",),
        exclude_urls: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        self._ensure_research_state_store()
        normalized_statuses = [
            str(status).strip() for status in statuses if str(status).strip()
        ]
        if not normalized_statuses:
            return []
        excluded_urls = self._persistent_unique_urls(exclude_urls or [])
        status_placeholders = ", ".join("?" for _ in normalized_statuses)
        query = f"""
            SELECT url, domain, priority, status, source_query, source_url,
                   run_id, js_required, attempts, updated_at
            FROM crawl_queue
            WHERE status IN ({status_placeholders})
        """
        params: list[Any] = list(normalized_statuses)
        if excluded_urls:
            exclude_placeholders = ", ".join("?" for _ in excluded_urls)
            query += f" AND url NOT IN ({exclude_placeholders})"
            params.extend(excluded_urls)
        query += " ORDER BY priority DESC, js_required DESC, updated_at DESC LIMIT ?"
        params.append(max(limit, 1))
        with closing(self._connect_research_state()) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def _claim_specific_crawl_urls(
        self,
        urls: list[str],
        worker_id: str,
    ) -> list[dict[str, Any]]:
        safe_urls = self._persistent_unique_urls(urls)
        if not safe_urls:
            return []
        self._ensure_research_state_store()
        claimed: list[dict[str, Any]] = []
        placeholders = ", ".join("?" for _ in safe_urls)
        with closing(self._connect_research_state()) as connection:
            rows = connection.execute(
                f"""
                SELECT url, domain, priority, status, source_query, source_url,
                       run_id, js_required, attempts, updated_at
                FROM crawl_queue
                WHERE status = 'queued' AND url IN ({placeholders})
                """,
                tuple(safe_urls),
            ).fetchall()
            row_map = {str(row["url"] or ""): dict(row) for row in rows}
            now = datetime.now(UTC).isoformat()
            with connection:
                for url in safe_urls:
                    row = row_map.get(url)
                    if row is None:
                        continue
                    updated = connection.execute(
                        """
                        UPDATE crawl_queue
                        SET status = 'claimed',
                            last_claimed_by = ?,
                            last_claimed_at = ?,
                            attempts = attempts + 1,
                            updated_at = ?
                        WHERE url = ? AND status = 'queued'
                        """,
                        (
                            worker_id,
                            now,
                            now,
                            url,
                        ),
                    ).rowcount
                    if not updated:
                        continue
                    claimed.append(row)
        return claimed

    def _claim_crawl_queue_batch(
        self,
        limit: int,
        worker_id: str,
        *,
        allow_js_required: bool = True,
        prefer_js_required: bool = False,
        max_claims_per_domain: int = 2,
        default_domain_cooldown_seconds: float = 0.0,
        js_domain_cooldown_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        if self._crawl_broker_enabled():
            response = self._crawl_broker_request(
                "/queue/claim-batch",
                {
                    "limit": int(limit),
                    "worker_id": str(worker_id or "crawl-worker"),
                    "allow_js_required": bool(allow_js_required),
                    "prefer_js_required": bool(prefer_js_required),
                    "max_claims_per_domain": max(1, int(max_claims_per_domain)),
                    "default_domain_cooldown_seconds": float(
                        default_domain_cooldown_seconds
                    ),
                    "js_domain_cooldown_seconds": float(js_domain_cooldown_seconds),
                },
            )
            claimed = response.get("claimed") or []
            return [dict(item) for item in claimed if isinstance(item, dict)]
        del default_domain_cooldown_seconds, js_domain_cooldown_seconds
        preview_rows = self._peek_crawl_queue_rows(max(limit * 6, limit))
        domain_counts: dict[str, int] = {}
        selected_urls: list[str] = []

        def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
            priority = float(row.get("priority") or 0.0)
            js_required = int(row.get("js_required") or 0)
            updated_at = str(row.get("updated_at") or "")
            if prefer_js_required:
                return (js_required, priority, updated_at)
            return (priority, js_required, updated_at)

        for row in sorted(preview_rows, key=_sort_key, reverse=True):
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            js_required = bool(int(row.get("js_required") or 0))
            if js_required and not allow_js_required:
                continue
            domain = str(row.get("domain") or "").strip() or self._source_domain(url)
            if domain and domain_counts.get(domain, 0) >= max(
                1, int(max_claims_per_domain)
            ):
                continue
            selected_urls.append(url)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(selected_urls) >= limit:
                break
        return self._claim_specific_crawl_urls(selected_urls, worker_id)

    def _claim_persistent_crawl_sources(
        self,
        query: str,
        objective: str,
        limit: int,
        exclude_urls: list[str] | None = None,
    ) -> list[ResearchSource]:
        if limit <= 0:
            return []
        excluded_urls = self._persistent_unique_urls(exclude_urls or [])
        if self._crawl_broker_enabled():
            response = self._crawl_broker_request(
                "/queue/claim-sources",
                {
                    "query": query,
                    "objective": objective,
                    "limit": int(limit),
                    "exclude_urls": excluded_urls,
                },
                timeout_seconds=max(10.0, self.timeout_seconds),
            )
            return self._deserialize_research_sources(response.get("sources") or [])
        self._ensure_research_state_store()
        excluded = set(excluded_urls)
        worker_seed = (
            f"{os.getpid()}:{threading.get_ident()}:{time.monotonic_ns()}:{query[:64]}"
        )
        worker_id = hashlib.sha1(worker_seed.encode("utf-8")).hexdigest()[:12]
        rows_to_claim: list[tuple[sqlite3.Row, float]] = []
        with closing(self._connect_research_state()) as connection:
            rows = connection.execute(
                """
                SELECT url, domain, priority, source_query, source_url,
                       js_required, attempts
                FROM crawl_queue
                WHERE status = 'queued'
                ORDER BY priority DESC, js_required DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 24),),
            ).fetchall()
            now = datetime.now(UTC).isoformat()
            with connection:
                for row in rows:
                    if str(row["url"] or "").strip() in excluded:
                        continue
                    label = " ".join(
                        [
                            str(row["url"] or ""),
                            str(row["source_query"] or ""),
                            str(row["source_url"] or ""),
                        ]
                    )
                    match_score = self._persistent_match_score(
                        label,
                        query,
                        objective,
                    )
                    if match_score < 0.2 and str(row["source_query"] or "").strip():
                        continue
                    updated = connection.execute(
                        """
                        UPDATE crawl_queue
                        SET status = 'claimed',
                            last_claimed_by = ?,
                            last_claimed_at = ?,
                            attempts = attempts + 1,
                            updated_at = ?
                        WHERE url = ? AND status = 'queued'
                        """,
                        (
                            worker_id,
                            now,
                            now,
                            str(row["url"] or ""),
                        ),
                    ).rowcount
                    if not updated:
                        continue
                    rows_to_claim.append((row, match_score))
                    if len(rows_to_claim) >= limit:
                        break
        claimed_sources: list[ResearchSource] = []
        for row, match_score in rows_to_claim:
            url = str(row["url"] or "").strip()
            if not url:
                continue
            domain = str(row["domain"] or "").strip()
            source_query = str(row["source_query"] or "").strip()
            quality_flags = ["persistent-crawl-queue"]
            if int(row["js_required"] or 0) == 1:
                quality_flags.append("js-render-required")
            abstract = (
                source_query
                or str(row["source_url"] or "").strip()
                or self._label_from_url(url)
            )
            claimed_sources.append(
                ResearchSource(
                    provider="persistent-crawl-queue",
                    title=self._label_from_url(url),
                    url=url,
                    authors=[domain] if domain else [],
                    abstract=f"Persistent crawl candidate: {abstract}"[:320],
                    citation_count=0,
                    score=max(
                        float(row["priority"] or 0.0),
                        4.0 + (match_score * 20.0),
                    ),
                    quality_flags=quality_flags,
                )
            )
        return claimed_sources

    def _update_crawl_queue_status(
        self,
        url: str,
        status: str,
        error: str = "",
    ) -> None:
        clean_url = str(url or "").strip()
        if not clean_url:
            return
        if self._crawl_broker_enabled():
            self._crawl_broker_request(
                "/queue/update-status",
                {
                    "url": clean_url,
                    "status": status,
                    "error": str(error or "")[:320],
                },
            )
            return
        self._ensure_research_state_store()
        with closing(self._connect_research_state()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE crawl_queue
                    SET status = ?,
                        last_error = ?,
                        updated_at = ?
                    WHERE url = ?
                    """,
                    (
                        status,
                        str(error or "")[:320],
                        datetime.now(UTC).isoformat(),
                        clean_url,
                    ),
                )

    def _fetch_source_content(
        self,
        source: ResearchSource,
        query: str,
        browser_prefetch: dict[str, str] | None = None,
    ) -> tuple[str, str, str, bool]:
        if not self._is_safe_public_url(source.url):
            return "", "", "skipped", False
        prefetched = browser_prefetch or {}
        used_browser = False
        raw_html = ""
        if self._needs_browser(source.url):
            browser_text = prefetched.get(source.url)
            if not browser_text:
                browser_text = self._get_text_browser(
                    source.url,
                    max_chars=80_000,
                )
            if browser_text and self._text_signal_score(browser_text) >= 0.1:
                raw_html = browser_text
                content = browser_text
                used_browser = True
            else:
                raw_html = self._get_text(
                    source.url,
                    accept="text/html,application/xhtml+xml,*/*",
                    max_bytes=60_000,
                    timeout_seconds=10,
                )
                content = self._get_text_stitched(
                    source.url,
                    accept="text/html,application/xhtml+xml,*/*",
                    page_bytes=60_000,
                    max_pages=4,
                    overlap_bytes=1_600,
                    query=query,
                )
                if not content and raw_html:
                    content = self._html_to_text(raw_html)
        else:
            raw_html = self._get_text(
                source.url,
                accept="text/html,application/xhtml+xml,*/*",
                max_bytes=60_000,
                timeout_seconds=10,
            )
            content = self._get_text_stitched(
                source.url,
                accept="text/html,application/xhtml+xml,*/*",
                page_bytes=60_000,
                max_pages=4,
                overlap_bytes=1_600,
                query=query,
            )
            if not raw_html and not content:
                return "", "", "no-content", False
            signal_before = self._text_signal_score(
                self._html_to_text(raw_html) if raw_html else content
            )
            overlay_markers = self._overlay_marker_count(raw_html) if raw_html else 0
            if signal_before < 0.16 and overlay_markers > 0:
                resolved_html, status = self._interrupt_resolve_overlays(raw_html)
                if status != "resolved":
                    if "unreachable-paywalled" not in source.quality_flags:
                        source.quality_flags.append("unreachable-paywalled")
                    return "", raw_html, "unreachable-paywalled", False
                raw_html = resolved_html
            if self._should_retry_with_browser(
                source.url,
                raw_html or content,
                query,
            ):
                browser_text = prefetched.get(source.url)
                if not browser_text:
                    browser_text = self._get_text_browser(
                        source.url,
                        max_chars=80_000,
                    )
                if browser_text:
                    raw_html = browser_text
                    used_browser = True
                    content = browser_text
            if not content and raw_html:
                content = self._html_to_text(raw_html)
        if not content:
            return "", raw_html, "no-content", used_browser
        return content, raw_html, "processed", used_browser

    def _record_crawl_observation(
        self,
        source: ResearchSource,
        content: str,
        source_query: str,
        query_hints: list[str],
        outbound_urls: list[str],
        worker_id: str,
        used_browser: bool,
    ) -> None:
        clean_url = str(source.url or "").strip()
        if not clean_url:
            return
        if self._crawl_broker_enabled():
            self._crawl_broker_request(
                "/observations/record",
                {
                    "source": asdict(source),
                    "content": content,
                    "source_query": str(source_query or "")[:320],
                    "query_hints": list(query_hints or []),
                    "outbound_urls": list(outbound_urls or []),
                    "worker_id": worker_id[:120],
                    "used_browser": bool(used_browser),
                },
                timeout_seconds=max(10.0, self.timeout_seconds),
            )
            return
        self._ensure_research_state_store()
        excerpt = re.sub(r"\s+", " ", content).strip()[:1600]
        if not excerpt:
            return
        title = str(source.title or self._label_from_url(clean_url)).strip()[:200]
        domain = self._source_domain(clean_url)
        now = datetime.now(UTC).isoformat()
        signal_score = self._text_signal_score(excerpt)
        content_hash = hashlib.sha1(excerpt.encode("utf-8")).hexdigest()
        merged_outbound = self._persistent_unique_urls(outbound_urls)[:32]
        with closing(self._connect_research_state()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO crawl_observations(
                        url,
                        domain,
                        title,
                        excerpt,
                        source_query,
                        query_hints_json,
                        outbound_urls_json,
                        content_hash,
                        signal_score,
                        used_browser,
                        worker_id,
                        fetched_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        domain = excluded.domain,
                        title = excluded.title,
                        excerpt = excluded.excerpt,
                        source_query = excluded.source_query,
                        query_hints_json = excluded.query_hints_json,
                        outbound_urls_json = excluded.outbound_urls_json,
                        content_hash = excluded.content_hash,
                        signal_score = excluded.signal_score,
                        used_browser = excluded.used_browser,
                        worker_id = excluded.worker_id,
                        fetched_at = excluded.fetched_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        clean_url,
                        domain,
                        title,
                        excerpt,
                        str(source_query or "")[:320],
                        json.dumps(query_hints[:16]),
                        json.dumps(merged_outbound),
                        content_hash,
                        float(signal_score),
                        1 if used_browser else 0,
                        worker_id[:120],
                        now,
                        now,
                    ),
                )
                if domain:
                    existing_domain = connection.execute(
                        """
                        SELECT score, observation_count, urls_json,
                               claim_keys_json, contradiction_keys_json
                        FROM evidence_domains
                        WHERE domain = ?
                        """,
                        (domain,),
                    ).fetchone()
                    merged_urls = self._merge_text_lists(
                        self._load_json_text_list(existing_domain["urls_json"])
                        if existing_domain
                        else [],
                        [clean_url],
                        limit=64,
                    )
                    connection.execute(
                        """
                        INSERT INTO evidence_domains(
                            domain,
                            score,
                            observation_count,
                            urls_json,
                            claim_keys_json,
                            contradiction_keys_json,
                            last_seen_run,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(domain) DO UPDATE SET
                            score = excluded.score,
                            observation_count = excluded.observation_count,
                            urls_json = excluded.urls_json,
                            claim_keys_json = excluded.claim_keys_json,
                            contradiction_keys_json = excluded.contradiction_keys_json,
                            last_seen_run = excluded.last_seen_run,
                            updated_at = excluded.updated_at
                        """,
                        (
                            domain,
                            float(signal_score * 10.0)
                            + float(existing_domain["score"] or 0.0)
                            if existing_domain
                            else float(signal_score * 10.0),
                            1 + int(existing_domain["observation_count"] or 0)
                            if existing_domain
                            else 1,
                            json.dumps(merged_urls),
                            json.dumps(
                                self._load_json_text_list(
                                    existing_domain["claim_keys_json"]
                                )
                                if existing_domain
                                else []
                            ),
                            json.dumps(
                                self._load_json_text_list(
                                    existing_domain["contradiction_keys_json"]
                                )
                                if existing_domain
                                else []
                            ),
                            self._active_run_id or worker_id[:120],
                            now,
                        ),
                    )

    def _persistent_evidence_query_hints(
        self,
        query: str,
        objective: str,
        limit: int = 16,
    ) -> list[str]:
        if self._crawl_broker_enabled():
            response = self._crawl_broker_request(
                "/evidence/query-hints",
                {
                    "query": query,
                    "objective": objective,
                    "limit": int(limit),
                },
            )
            return [
                str(item).strip()
                for item in (response.get("query_hints") or [])
                if str(item).strip()
            ][:limit]
        self._ensure_research_state_store()
        candidates: list[str] = []
        with closing(self._connect_research_state()) as connection:
            claim_rows = connection.execute(
                """
                SELECT claim_text, contradiction_count, confidence_rank
                FROM evidence_claims
                ORDER BY support_count DESC, confidence_rank DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 48),),
            ).fetchall()
            contradiction_rows = connection.execute(
                """
                SELECT contradiction_text
                FROM evidence_contradictions
                ORDER BY count DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 4, 24),),
            ).fetchall()
            domain_rows = connection.execute(
                """
                SELECT domain
                FROM evidence_domains
                ORDER BY score DESC, observation_count DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 3, 18),),
            ).fetchall()
            observation_rows = connection.execute(
                """
                SELECT excerpt, source_query, query_hints_json
                FROM crawl_observations
                ORDER BY signal_score DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 48),),
            ).fetchall()

        for row in claim_rows:
            claim_text = str(row["claim_text"] or "").strip()
            if not claim_text:
                continue
            match_score = self._persistent_match_score(claim_text, query, objective)
            if match_score < 0.28:
                continue
            if int(row["contradiction_count"] or 0) > 0:
                candidate = self._query_core_terms(
                    f"{objective} {claim_text} independent verification"
                )
            else:
                candidate = self._query_core_terms(f"{query} {claim_text}")
            if candidate:
                candidates.append(candidate)

        for row in contradiction_rows:
            contradiction_text = str(row["contradiction_text"] or "").strip()
            if not contradiction_text:
                continue
            match_score = self._persistent_match_score(
                contradiction_text,
                query,
                objective,
            )
            if match_score < 0.24:
                continue
            candidate = self._query_core_terms(
                f"{objective} {contradiction_text} counterevidence"
            )
            if candidate:
                candidates.append(candidate)

        for row in domain_rows[: max(4, limit // 2)]:
            domain = str(row["domain"] or "").strip()
            if not domain:
                continue
            candidate = self._query_core_terms(f"site:{domain} {query}")
            if candidate:
                candidates.append(candidate)

        for row in observation_rows:
            excerpt = str(row["excerpt"] or "").strip()
            source_query = str(row["source_query"] or "").strip()
            if self._persistent_match_score(excerpt, query, objective) < 0.22:
                continue
            candidates.extend(self._load_json_text_list(row["query_hints_json"]))
            if source_query:
                candidate = self._query_core_terms(f"{query} {source_query}")
                if candidate:
                    candidates.append(candidate)

        return self._sanitize_query_variants(candidates, query)[:limit]

    def _persistent_seed_urls(
        self,
        query: str,
        objective: str,
        limit: int = 16,
    ) -> list[str]:
        if self._crawl_broker_enabled():
            response = self._crawl_broker_request(
                "/evidence/seed-urls",
                {
                    "query": query,
                    "objective": objective,
                    "limit": int(limit),
                },
            )
            return self._persistent_unique_urls(
                [
                    str(item).strip()
                    for item in (response.get("seed_urls") or [])
                    if str(item).strip()
                ]
            )[:limit]
        self._ensure_research_state_store()
        urls: list[str] = []
        with closing(self._connect_research_state()) as connection:
            claim_rows = connection.execute(
                """
                SELECT claim_text, source_urls_json
                FROM evidence_claims
                ORDER BY support_count DESC, confidence_rank DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 48),),
            ).fetchall()
            observation_rows = connection.execute(
                """
                SELECT url, excerpt, outbound_urls_json
                FROM crawl_observations
                ORDER BY signal_score DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 48),),
            ).fetchall()
            queue_rows = connection.execute(
                """
                SELECT url, source_query
                FROM crawl_queue
                WHERE status IN ('processed', 'queued', 'claimed')
                ORDER BY priority DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 48),),
            ).fetchall()

        for row in claim_rows:
            claim_text = str(row["claim_text"] or "").strip()
            if self._persistent_match_score(claim_text, query, objective) < 0.28:
                continue
            urls.extend(self._load_json_text_list(row["source_urls_json"]))
            if len(urls) >= limit * 3:
                break

        if len(urls) < limit:
            for row in observation_rows:
                excerpt = str(row["excerpt"] or "").strip()
                label = f"{row['url']} {excerpt[:280]}".strip()
                if self._persistent_match_score(label, query, objective) < 0.22:
                    continue
                urls.append(str(row["url"] or "").strip())
                urls.extend(self._load_json_text_list(row["outbound_urls_json"]))
                if len(urls) >= limit * 3:
                    break

        if len(urls) < limit:
            for row in queue_rows:
                source_query = str(row["source_query"] or "").strip()
                label = f"{row['url']} {source_query}".strip()
                if self._persistent_match_score(label, query, objective) < 0.22:
                    continue
                urls.append(str(row["url"] or "").strip())
                if len(urls) >= limit * 3:
                    break

        return self._persistent_unique_urls(urls)[:limit]

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
        if pc_context:
            candidates.extend(cls._collect_urls(pc_context))

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
        coverage_targets = self._coverage_targets(targets)
        max_passes = int(
            targets.get("max_retrieval_passes")
            or self._default_max_passes(settings.depth)
        )
        # No runtime-based pass floor — passes are information-driven.
        max_passes = max(max_passes, min_depth_passes)
        max_passes = max(1, min(max_passes, 240))
        effective_novelty_threshold = self._effective_novelty_threshold(
            settings.depth,
            targets,
        )
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
            # Scale parallel search workers like Claude/Gemini deep research:
            # multi-hour = 20 concurrent query threads to maximize throughput.
            parallel_workers = (
                max(1, min(20, len(pass_variants)))
                if settings.depth == "multi-hour"
                else max(1, min(4, len(pass_variants)))
            )
            if parallel_workers > 1 and len(pass_variants) > 1:

                def _search_one(sq: str) -> list[ResearchSource]:
                    return self._search_query_across_providers(
                        sq, allowed_providers, settings.per_provider
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
            depth_met = (pass_index + 1) >= min_depth_passes
            novelty_threshold = effective_novelty_threshold
            low_novelty_pass = (
                coverage["novelty_rate"] < novelty_threshold and pass_index > 0
            )
            if low_novelty_pass:
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

    def _get_sec_tickers(self) -> "dict[str, dict]":
        """Return {TICKER: {cik_str, ticker, title}} from SEC company_tickers.json.

        Cached at class level after first fetch — the file is ~1.5 MB and lists
        every ~15 000 public company registered with the SEC.  Used to resolve
        a ticker symbol (e.g. "NVDA") to the SEC's numeric CIK so we can call
        data.sec.gov/submissions and data.sec.gov/api/xbrl/companyfacts.
        """
        if DeepResearchEngine._sec_tickers_cache is not None:
            return DeepResearchEngine._sec_tickers_cache
        payload = self._get_sec_json("https://www.sec.gov/files/company_tickers.json")
        if not payload:
            return {}
        result: dict[str, dict] = {}
        for item in payload.values():
            ticker = str(item.get("ticker") or "").upper().strip()
            if ticker:
                result[ticker] = item
        DeepResearchEngine._sec_tickers_cache = result
        return result

    def _fetch_yf_chart_abstract(self, symbol: str, name: str) -> str:
        """Fallback: basic price + 52-week range from Yahoo Finance v8 chart API.

        The v8 chart endpoint works without authentication.  Returns a minimal
        but real abstract (price, 52W range, volume, exchange) for tickers where
        yfinance is unavailable.  Returns empty string on failure.
        """
        payload = self._get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&range=1d"
        )
        meta = ((payload.get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        if not price:
            return ""
        high_52w = meta.get("fiftyTwoWeekHigh")
        low_52w = meta.get("fiftyTwoWeekLow")
        volume = meta.get("regularMarketVolume")
        exchange = meta.get("fullExchangeName") or meta.get("exchangeName") or ""
        long_name = meta.get("longName") or meta.get("shortName") or name

        parts: list[str] = [f"{long_name} ({symbol})"]
        if exchange:
            parts.append(exchange)
        parts.append(f"Price: ${price:.2f}")
        if high_52w and low_52w:
            parts.append(f"52W Range: ${low_52w:.2f}–${high_52w:.2f}")
        if volume:
            parts.append(f"Volume: {volume:,}")
        return " | ".join(parts)

    def _get_sec_json(self, url: str) -> dict[str, Any]:
        """Fetch JSON from SEC APIs (data.sec.gov, efts.sec.gov).

        SEC policy requires a descriptive User-Agent with contact info.
        Returns {} on any error — callers must handle gracefully.
        """
        try:
            return json.loads(self._get_sec_text(url, accept="application/json"))
        except json.JSONDecodeError:
            return {}

    def _get_sec_text(
        self,
        url: str,
        accept: str = "application/json",
    ) -> str:
        """Fetch SEC text content with retry/backoff for transient throttling."""
        sec_ua = os.environ.get(
            "SEC_USER_AGENT",
            "agentos-orchestrator/0.1 research-bot (set SEC_USER_AGENT)",
        )
        headers = {
            "Accept": accept,
            "User-Agent": sec_ua,
        }
        retry_delays = (0.0, 1.0, 2.5)

        try:
            import requests as _requests  # type: ignore[import-not-found]
        except ImportError:
            _requests = None

        if _requests is not None:
            for delay in retry_delays:
                if delay > 0:
                    time.sleep(delay)
                try:
                    response = _requests.get(
                        url,
                        headers=headers,
                        timeout=self.timeout_seconds,
                        allow_redirects=True,
                    )
                except Exception:
                    continue
                if response.status_code == 200:
                    return response.text
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    continue
                break

        request = urllib.request.Request(url, headers=headers)
        for delay in retry_delays:
            if delay > 0:
                time.sleep(delay)
            try:
                with urllib.request.urlopen(  # noqa: S310 - policy-gated URLs
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    return response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                if exc.code in {403, 429, 500, 502, 503, 504}:
                    continue
                break
            except (OSError, urllib.error.URLError):
                continue
        return ""

    def _fetch_yf_fundamentals_abstract(self, symbol: str, name: str) -> str:
        """Fetch REAL fundamental data using yfinance (handles Yahoo Finance auth).

        yfinance handles the Yahoo Finance crumb/cookie authentication that
        blocks raw urllib calls.  Returns a pipe-delimited string of live
        metrics a Wall Street analyst actually uses: P/E, market cap, analyst
        consensus and price targets, growth rates, ROE, FCF, short interest.

        Falls back to the v8 chart API (price + 52W range) if yfinance is
        not installed.  Returns empty string only if ALL data paths fail,
        so callers skip the source rather than emit a placeholder.
        """
        try:
            import yfinance as yf  # optional; falls back gracefully
        except ImportError:
            return self._fetch_yf_chart_abstract(symbol, name)

        try:
            info: dict[str, Any] = yf.Ticker(symbol).info or {}
        except Exception:
            return self._fetch_yf_chart_abstract(symbol, name)

        if not info:
            return self._fetch_yf_chart_abstract(symbol, name)

        long_name = str(info.get("longName") or info.get("shortName") or name)
        curr_price = info.get("currentPrice") or info.get("regularMarketPrice")
        mkt_cap_raw = info.get("marketCap")
        mkt_cap = f"${mkt_cap_raw / 1e9:.1f}B" if mkt_cap_raw else ""
        sector = str(info.get("sector") or "")
        industry = str(info.get("industry") or "")

        trailing_pe = info.get("trailingPE")
        fwd_pe = info.get("forwardPE")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        beta = info.get("beta")
        short_float_raw = info.get("shortPercentOfFloat")
        wk52_chg_raw = info.get("52WeekChange") or info.get("fiftyTwoWeekChangePercent")
        target_price = info.get("targetMeanPrice")
        n_analysts = info.get("numberOfAnalystOpinions")
        rec = str(info.get("recommendationKey") or "").upper().replace("_", " ")
        rev_growth_raw = info.get("revenueGrowth")
        earn_growth_raw = info.get("earningsGrowth")
        roe_raw = info.get("returnOnEquity")
        total_rev = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon")
        free_cf = info.get("freeCashflow")
        fwd_eps = info.get("forwardEps")
        trailing_eps = info.get("trailingEps")

        parts: list[str] = [f"{long_name} ({symbol})"]
        if sector:
            parts.append(f"{sector} — {industry}" if industry else sector)
        if curr_price:
            parts.append(f"Price: ${curr_price:.2f}")
        if mkt_cap:
            parts.append(f"Mkt Cap: {mkt_cap}")
        if trailing_pe:
            parts.append(f"Trailing P/E: {trailing_pe:.1f}x")
        if fwd_pe:
            parts.append(f"Fwd P/E: {fwd_pe:.1f}x")
        if ev_ebitda:
            parts.append(f"EV/EBITDA: {ev_ebitda:.1f}x")
        if pb:
            parts.append(f"P/B: {pb:.2f}x")
        if beta:
            parts.append(f"Beta: {beta:.2f}")
        if short_float_raw:
            parts.append(f"Short Float: {short_float_raw * 100:.1f}%")
        if wk52_chg_raw:
            parts.append(f"52W Chg: {wk52_chg_raw * 100:+.1f}%")
        if target_price and curr_price and curr_price > 0:
            upside_pct = (target_price / curr_price - 1) * 100
            analyst_note = (
                f"({upside_pct:+.1f}% upside, {int(n_analysts)} analysts)"
                if n_analysts
                else f"({upside_pct:+.1f}% upside)"
            )
            parts.append(f"Target: ${target_price:.2f} {analyst_note}")
        if rec:
            parts.append(f"Rating: {rec}")
        if total_rev:
            parts.append(f"Revenue: ${total_rev / 1e9:.1f}B")
        if rev_growth_raw:
            parts.append(f"Rev Growth: {rev_growth_raw * 100:+.1f}%")
        if net_income:
            parts.append(f"Net Income: ${net_income / 1e9:.1f}B")
        if earn_growth_raw:
            parts.append(f"EPS Growth: {earn_growth_raw * 100:+.1f}%")
        if roe_raw:
            parts.append(f"ROE: {roe_raw * 100:+.1f}%")
        if free_cf:
            parts.append(f"FCF: ${free_cf / 1e9:.1f}B")
        if fwd_eps:
            parts.append(f"Fwd EPS: ${fwd_eps:.2f}")
        if trailing_eps:
            parts.append(f"TTM EPS: ${trailing_eps:.2f}")

        result = " | ".join(p for p in parts if p)
        return result if len(parts) > 2 else self._fetch_yf_chart_abstract(symbol, name)

    def _fetch_sec_company_facts(self, ticker: str) -> "ResearchSource | None":
        """Fetch real audited financial data from SEC XBRL API.

        Pipeline:
        1. company_tickers.json → resolve ticker to CIK (cached class-level)
        2. data.sec.gov/submissions/CIK{n}.json → company name, SIC, exchange,
           most recent 10-K and 10-Q filing dates
        3. data.sec.gov/api/xbrl/companyfacts/CIK{n}.json → audited GAAP data:
           revenue, net income, diluted EPS from actual 10-K filings

        Returns a high-confidence (score=72) ResearchSource with real SEC data,
        or None if the ticker is not found or data is insufficient.
        """
        import time

        # Step 1: ticker → CIK via the cached SEC company tickers mapping
        tickers_map = self._get_sec_tickers()
        entry = tickers_map.get(ticker.upper().strip())
        if not entry:
            return None
        cik_int = entry.get("cik_str") or entry.get("cik")
        if not cik_int:
            return None
        cik_padded = str(int(cik_int)).zfill(10)
        company_title = str(entry.get("title") or ticker)

        # Step 2: Company metadata from data.sec.gov/submissions
        submissions = self._get_sec_json(
            f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        )
        if not submissions:
            return None

        company_name = str(submissions.get("name") or company_title)
        sic_desc = str(submissions.get("sicDescription") or "")
        exchanges = submissions.get("exchanges") or []
        exchange = ", ".join(str(e) for e in exchanges) if exchanges else ""

        recent = (submissions.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []

        most_recent_10k_date = ""
        most_recent_10q_date = ""
        for i, form in enumerate(forms):
            if form == "10-K" and not most_recent_10k_date and i < len(dates):
                most_recent_10k_date = dates[i]
            if form == "10-Q" and not most_recent_10q_date and i < len(dates):
                most_recent_10q_date = dates[i]
            if most_recent_10k_date and most_recent_10q_date:
                break

        # Step 3: XBRL financials — audited GAAP data from actual SEC filings.
        # SEC policy: max 10 req/sec.  A brief pause keeps us well under limit.
        time.sleep(0.15)
        facts_payload = self._get_sec_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        )
        gaap = (facts_payload.get("facts") or {}).get("us-gaap") or {}

        def _latest_annual(concept_key: str) -> "tuple[float, str] | None":
            entries = (gaap.get(concept_key) or {}).get("units", {}).get("USD") or []
            annual = sorted(
                [
                    e
                    for e in entries
                    if e.get("form") == "10-K" and e.get("val") is not None
                ],
                key=lambda e: e.get("end", ""),
                reverse=True,
            )
            if annual:
                return float(annual[0]["val"]), str(annual[0].get("end", ""))[:7]
            return None

        def _yoy_growth(concept_key: str) -> "float | None":
            entries = (gaap.get(concept_key) or {}).get("units", {}).get("USD") or []
            annual = sorted(
                [
                    e
                    for e in entries
                    if e.get("form") == "10-K" and e.get("val") is not None
                ],
                key=lambda e: e.get("end", ""),
                reverse=True,
            )
            if len(annual) >= 2 and annual[1].get("val") and annual[1]["val"] != 0:
                return (
                    (float(annual[0]["val"]) - float(annual[1]["val"]))
                    / abs(float(annual[1]["val"]))
                    * 100
                )
            return None

        # Revenue: try common GAAP concepts in priority order
        rev_key = next(
            (
                k
                for k in (
                    "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet",
                    "SalesRevenueGoodsNet",
                    "RevenueFromContractWithCustomerIncludingAssessedTax",
                )
                if k in gaap
            ),
            None,
        )

        abstract_parts: list[str] = [f"{company_name} (SEC CIK: {str(int(cik_int))})"]
        if sic_desc:
            abstract_parts.append(f"Industry: {sic_desc}")
        if exchange:
            abstract_parts.append(f"Exchange: {exchange}")
        if most_recent_10k_date:
            abstract_parts.append(f"Latest 10-K: {most_recent_10k_date}")
        if most_recent_10q_date:
            abstract_parts.append(f"Latest 10-Q: {most_recent_10q_date}")

        has_financials = False
        if rev_key:
            rev_result = _latest_annual(rev_key)
            if rev_result:
                rev_val, rev_period = rev_result
                abstract_parts.append(f"Revenue ({rev_period}): ${rev_val / 1e9:.2f}B")
                rev_growth = _yoy_growth(rev_key)
                if rev_growth is not None:
                    abstract_parts.append(f"Revenue YoY: {rev_growth:+.1f}%")
                has_financials = True

        if "NetIncomeLoss" in gaap:
            ni_result = _latest_annual("NetIncomeLoss")
            if ni_result:
                ni_val, ni_period = ni_result
                abstract_parts.append(f"Net Income ({ni_period}): ${ni_val / 1e9:.2f}B")
                ni_growth = _yoy_growth("NetIncomeLoss")
                if ni_growth is not None:
                    abstract_parts.append(f"Net Income YoY: {ni_growth:+.1f}%")
                has_financials = True

        if "EarningsPerShareDiluted" in gaap:
            eps_result = _latest_annual("EarningsPerShareDiluted")
            if eps_result:
                eps_val, eps_period = eps_result
                abstract_parts.append(f"Diluted EPS ({eps_period}): ${eps_val:.2f}")

        if len(abstract_parts) < 3 and not has_financials:
            return None

        cik_int_str = str(int(cik_int))
        return ResearchSource(
            provider="sec-edgar",
            title=f"{company_name} — SEC XBRL Financial Facts",
            url=(
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik_int_str}"
                f"&type=10-K&dateb=&owner=include&count=10"
            ),
            year=datetime.now(UTC).year,
            abstract=" | ".join(p for p in abstract_parts if p),
            score=72.0,  # Audited primary-source data — highest confidence
        )

    def _search_financial_portals(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Real financial research using Yahoo Finance JSON APIs.

        A Wall Street analyst doesn't care about domain names — they care about
        the DATA.  This method fetches actual financial metrics (P/E, market
        cap, analyst targets, EPS growth, revenue growth, short interest, ROE)
        directly from Yahoo Finance's JSON APIs and puts that real data into
        every source's abstract.

        NO template URL generation. Every source either has real content
        (from a live API call) or is omitted entirely.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        # ── 1. Yahoo Finance search API — real JSON response ──────────────────
        yf_params = urllib.parse.urlencode(
            {
                "q": query,
                "quotesCount": min(target_limit, 12),
                "newsCount": min(target_limit, 12),
                "enableNavLinks": "true",
                "enableEnhancedTrivialQuery": "true",
            }
        )
        yf_payload = self._get_json(
            f"https://query1.finance.yahoo.com/v1/finance/search?{yf_params}"
        )

        # For each equity/ETF result, fetch REAL fundamental data
        fetched_symbols: set[str] = set()
        for item in (yf_payload.get("quotes") or [])[:target_limit]:
            symbol = str(item.get("symbol") or "").strip()
            name = str(item.get("longname") or item.get("shortname") or symbol)
            quote_type = str(item.get("quoteType") or "").upper()
            if not symbol or quote_type not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}:
                continue
            fetched_symbols.add(symbol)

            # Attempt to fetch real fundamental data — no fake placeholders
            abstract = self._fetch_yf_fundamentals_abstract(symbol, name)
            if not abstract:
                # API unavailable for this symbol — skip rather than fake it
                continue
            sources.append(
                ResearchSource(
                    provider="financial-portals",
                    title=f"{name} ({symbol}) — Live Fundamentals",
                    url=f"https://finance.yahoo.com/quote/{symbol}",
                    year=datetime.now(UTC).year,
                    abstract=abstract,
                    score=65.0,
                )
            )

        query_entity_terms = {
            token.lower() for token in re.findall(r"\b[A-Z][A-Za-z]{3,}\b", query)
        }
        query_entity_terms.update(
            token.lower() for token in _extract_ticker_candidates(query)
        )

        # Real news articles from Yahoo Finance (actual URLs, actual titles)
        for item in (yf_payload.get("news") or [])[: min(target_limit, 12)]:
            url = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            publisher = str(item.get("publisher") or "Yahoo Finance").strip()
            pub_time = item.get("providerPublishTime")
            if not url or not title or not url.startswith("http"):
                continue
            if not self._is_safe_public_url(url):
                continue
            title_text = f"{title} {publisher}".lower()
            mentions_entity = any(
                term in title_text or term in url.lower() for term in query_entity_terms
            )
            aligned_news = self._objective_alignment_score(title_text, query) >= 0.35
            if query_entity_terms and not (mentions_entity or aligned_news):
                continue
            year = datetime.now(UTC).year
            if pub_time:
                try:
                    year = datetime.fromtimestamp(int(pub_time), UTC).year
                except (ValueError, OSError):
                    pass
            sources.append(
                ResearchSource(
                    provider="financial-portals",
                    title=title,
                    url=url,
                    year=year,
                    abstract=f"{publisher}: {title}",
                    score=25.0,
                )
            )

        # ── 2. Explicit ticker extraction + company-name search ──────────────
        # Two strategies to resolve companies from the query:
        # (a) Short all-caps tickers via _extract_ticker_candidates
        # (b) Capitalized company-name tokens (e.g. NVIDIA, Apple, Microsoft)
        #     searched individually on YF v1 since the full query returns no quotes.
        tickers = _extract_ticker_candidates(query)
        candidate_names: list[str] = []
        _common_words = {
            "AND",
            "THE",
            "FOR",
            "WITH",
            "FROM",
            "THAT",
            "THIS",
            "STOCK",
            "STOCKS",
            "SHARE",
            "SHARES",
            "MARKET",
            "PRICE",
            "EARNINGS",
            "REVENUE",
            "GROWTH",
            "ANALYSIS",
            "REPORT",
            "QUARTER",
            "ANNUAL",
            "COMPANY",
            "CORP",
            "INC",
            "LLC",
            "LTD",
            "ABOUT",
            "WILL",
            "HAVE",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            # All-caps 2-10 chars that aren't blocked words (company names)
            if re.match(r"^[A-Z]{2,10}$", t) and t not in _common_words:
                candidate_names.append(t)
            # Title-case 5+ chars (Apple, Google, Microsoft, Amazon …)
            elif re.match(r"^[A-Z][a-z]{4,}$", t):
                candidate_names.append(t)

        for name_cand in candidate_names[:5]:
            if name_cand.upper() in fetched_symbols:
                continue
            yf_p = urllib.parse.urlencode(
                {"q": name_cand, "quotesCount": 3, "newsCount": 0}
            )
            yf_data = self._get_json(
                f"https://query1.finance.yahoo.com/v1/finance/search?{yf_p}"
            )
            for item in (yf_data.get("quotes") or [])[:2]:
                sym = str(item.get("symbol") or "").strip()
                nm = str(item.get("longname") or item.get("shortname") or sym)
                qt = str(item.get("quoteType") or "").upper()
                if not sym or qt not in {"EQUITY", "ETF", "MUTUALFUND"}:
                    continue
                if sym in fetched_symbols:
                    continue
                fetched_symbols.add(sym)
                abstract = self._fetch_yf_fundamentals_abstract(sym, nm)
                if not abstract:
                    continue
                sources.append(
                    ResearchSource(
                        provider="financial-portals",
                        title=f"{nm} ({sym}) — Live Fundamentals",
                        url=f"https://finance.yahoo.com/quote/{sym}",
                        year=datetime.now(UTC).year,
                        abstract=abstract,
                        score=65.0,
                    )
                )

        # Direct ticker lookup (short 1-5 char tickers explicitly in query)
        for ticker in tickers[:6]:
            ticker_upper = ticker.upper().strip()
            if ticker_upper in fetched_symbols:
                continue
            abstract = self._fetch_yf_fundamentals_abstract(ticker_upper, ticker_upper)
            if not abstract:
                continue
            fetched_symbols.add(ticker_upper)
            sources.append(
                ResearchSource(
                    provider="financial-portals",
                    title=f"{ticker_upper} — Yahoo Finance Fundamentals",
                    url=f"https://finance.yahoo.com/quote/{ticker_upper}",
                    year=datetime.now(UTC).year,
                    abstract=abstract,
                    score=60.0,
                )
            )

        self._record_provider_diagnostic(
            "financial-portals",
            "ok" if sources else "empty",
            f"returned {len(sources)} sources (real API data) for: {query[:80]}",
        )
        sources.sort(
            key=lambda source: (
                source.score,
                "Live Fundamentals" in source.title,
            ),
            reverse=True,
        )
        return sources[: target_limit * 3]

    def _search_sec_edgar(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Search SEC EDGAR for real company filings using browse-edgar Atom XML.

        The EFTS full-text search (efts.sec.gov) is blocked by Akamai CDN for
        programmatic access.  This method uses the officially-supported
        browse-edgar endpoint which returns Atom XML and is not CDN-protected:

        1. For tickers found in the query: resolve CIK via company_tickers.json
           then call browse-edgar for the exact company's filing history.
        2. For company-name searches: use browse-edgar's company search mode.
        3. For named tickers: call _fetch_sec_company_facts for full XBRL data
           (audited revenue, net income, EPS) — the primary-source numbers a
           sell-side analyst builds their model from.
        """
        import xml.etree.ElementTree as ET

        target_limit = max(1, int(limit or self.limit_per_provider))
        sources: list[ResearchSource] = []
        ATOM_NS = "http://www.w3.org/2005/Atom"
        sec_ua = os.environ.get(
            "SEC_USER_AGENT",
            "agentos-orchestrator/0.1 research-bot (public-research-use)",
        )

        def _fetch_browse_edgar_atom(url: str) -> "list[ResearchSource]":
            """Fetch browse-edgar Atom XML and parse into ResearchSource objects."""
            body = self._get_sec_text(
                url,
                accept="application/atom+xml,text/xml,*/*",
            )
            if not body:
                return []
            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                return []

            atom_tag = f"{{{ATOM_NS}}}"
            # Company name from optional <company-info> element
            ci = root.find("company-info") or root.find(f"{atom_tag}company-info")
            company_name = ""
            if ci is not None:
                company_name = (
                    ci.findtext("conformed-name")
                    or ci.findtext(f"{atom_tag}conformed-name")
                    or ""
                )

            parsed: list[ResearchSource] = []
            for entry_el in list(root.iter(f"{atom_tag}entry"))[:12]:
                title = entry_el.findtext(f"{atom_tag}title") or ""
                link_el = entry_el.find(f"{atom_tag}link")
                href = link_el.get("href", "") if link_el is not None else ""
                updated = entry_el.findtext(f"{atom_tag}updated") or ""
                cat_el = entry_el.find(f"{atom_tag}category")
                form_type = (
                    cat_el.get("term", "Filing") if cat_el is not None else "Filing"
                )
                if not href or not href.startswith("http"):
                    continue
                year = int(updated[:4]) if len(updated) >= 4 else datetime.now(UTC).year
                label = company_name or form_type
                parsed.append(
                    ResearchSource(
                        provider="sec-edgar",
                        title=f"{label} — {form_type} ({updated[:10]})",
                        url=href,
                        year=year,
                        abstract=(
                            f"SEC {form_type} filing"
                            + (f" by {company_name}" if company_name else "")
                            + f". Filed: {updated[:10]}."
                            + (f" {title[:200]}" if title else "")
                        ),
                        score=52.0,
                    )
                )
            return parsed

        tickers = _extract_ticker_candidates(query)

        # ── 1. Per-ticker: direct CIK lookup → browse-edgar filing list + XBRL
        seen_ciks: set[str] = set()
        for ticker in tickers[:4]:
            ticker_up = ticker.upper().strip()

            # High-value XBRL data (audited P&L from actual 10-K filings)
            sec_source = self._fetch_sec_company_facts(ticker_up)
            if sec_source is not None:
                sources.append(sec_source)

            # Filing list via browse-edgar CIK lookup
            tickers_map = self._get_sec_tickers()
            entry = tickers_map.get(ticker_up)
            if entry:
                cik_int = entry.get("cik_str") or entry.get("cik")
                if cik_int:
                    cik_str = str(int(cik_int))
                    if cik_str not in seen_ciks:
                        seen_ciks.add(cik_str)
                        atom_url = (
                            f"https://www.sec.gov/cgi-bin/browse-edgar"
                            f"?action=getcompany&CIK={cik_str}"
                            f"&type=10-K,10-Q,8-K&dateb=&owner=include"
                            f"&count={target_limit}&output=atom"
                        )
                        sources.extend(_fetch_browse_edgar_atom(atom_url))

        # ── 2. Company-name search on browse-edgar (catches non-ticker queries)
        # Browse-edgar company search needs a focused company name, NOT a full
        # analytical query.  Extract the first recognizable company name token.
        if not tickers or len(sources) < target_limit:
            # Extract the best company-name term from the query
            _stop = {
                "AND",
                "THE",
                "FOR",
                "WITH",
                "FROM",
                "THAT",
                "THIS",
                "ABOUT",
                "WILL",
                "HAVE",
                "STOCK",
                "STOCKS",
                "SHARES",
                "MARKET",
                "PRICE",
                "EARNINGS",
                "REVENUE",
                "GROWTH",
                "ANALYSIS",
                "REPORT",
                "QUARTER",
                "ANNUAL",
                "QUARTERLY",
                "COMPANY",
                "CORP",
                "INC",
            }
            company_search_term = query[:60]
            for token in query.split():
                t = token.strip(".,!?;:")
                # Prefer long all-caps tokens (NVIDIA, AMAZON, MICROSOFT …)
                if re.match(r"^[A-Z]{3,}$", t) and t not in _stop:
                    company_search_term = t
                    break
                # Or title-case words of 5+ chars
                if re.match(r"^[A-Z][a-z]{4,}$", t):
                    company_search_term = t
                    break
            search_url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?company={urllib.parse.quote_plus(company_search_term)}"
                f"&CIK=&type=10-K&dateb=&owner=include"
                f"&count={target_limit}&search_text=&action=getcompany&output=atom"
            )
            sources.extend(_fetch_browse_edgar_atom(search_url))

        self._record_provider_diagnostic(
            "sec-edgar",
            "ok" if sources else "empty",
            f"returned {len(sources)} SEC sources for: {query[:80]}",
        )
        return sources

    # ──────────────────────────────────────────────────────────────────────────
    # WALL STREET RESEARCH PROVIDERS
    # Real data sources a sell-side analyst would use to build a research note.
    # Every method here fetches actual numbers from live APIs or public pages —
    # no templates, no placeholder URLs.
    # ──────────────────────────────────────────────────────────────────────────

    def _search_bing_results(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Bing web search — second engine alongside DuckDuckGo.

        Bing covers many financial/news pages that DuckDuckGo doesn't surface,
        especially WSJ, Bloomberg, Reuters, and MarketWatch articles.
        Uses HTML scraping of Bing's public search page (no API key needed).
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        seen_urls: set[str] = set()

        for page in range(min(3, (target_limit + 9) // 10)):
            first_param = page * 10
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "first": str(first_param) if first_param > 0 else "1",
                    "setlang": "en-US",
                    "cc": "US",
                }
            )
            raw = self._get_text(
                f"https://www.bing.com/search?{params}",
                accept="text/html,application/xhtml+xml",
                max_bytes=150_000,
                timeout_seconds=8,
            )
            if not raw:
                break

            # Bing results: <li class="b_algo"> ... <h2><a href="...">title</a></h2>
            for match in re.finditer(
                r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                url = match.group(1).strip()
                title = self._html_to_text(match.group(2)).strip()
                if not url or not title:
                    continue
                if not self._is_safe_public_url(url):
                    continue
                if "bing.com" in url or "microsoft.com" in url:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                # Extract snippet from following text
                tail = raw[match.end() : match.end() + 2000]
                snippet_m = re.search(
                    r'<p[^>]*class="[^"]*b_algoSlug[^"]*"[^>]*>(.*?)</p>',
                    tail,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                snippet = self._html_to_text(snippet_m.group(1)) if snippet_m else ""
                host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
                score = max(target_limit - len(sources), 1)
                sources.append(
                    ResearchSource(
                        provider="bing-search",
                        title=title[:160],
                        url=url,
                        authors=[host] if host else [],
                        abstract=(snippet or f"Bing result: {title}")[:800],
                        score=float(score),
                    )
                )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "bing-search",
            "ok" if sources else "empty",
            f"returned {len(sources)} results",
        )
        return sources

    def _search_google_news_rss(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Google News RSS — real-time financial news with actual article links.

        Google News RSS is public and unauthenticated. Returns genuine news
        articles from WSJ, Reuters, Bloomberg, CNBC, FT, MarketWatch, etc.
        — the same sources a Bloomberg terminal's news feed would show.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        # Encode query for Google News RSS
        params = urllib.parse.urlencode(
            {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        rss_url = f"https://news.google.com/rss/search?{params}"
        raw = self._get_text(
            rss_url,
            accept="application/rss+xml,text/xml,*/*",
            max_bytes=200_000,
            timeout_seconds=8,
        )
        if not raw:
            self._record_provider_diagnostic(
                "google-news-rss", "empty", "no RSS response"
            )
            return []

        # Parse RSS <item> elements
        seen: set[str] = set()
        for item_match in re.finditer(
            r"<item>(.*?)</item>", raw, re.DOTALL | re.IGNORECASE
        ):
            item = item_match.group(1)
            title_m = re.search(
                r"<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>",
                item,
                re.DOTALL,
            )
            link_m = re.search(
                r"<link>(.*?)</link>|<guid[^>]*>(https?://[^<]+)</guid>",
                item,
                re.DOTALL,
            )
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item)
            source_m = re.search(r"<source[^>]*>(.*?)</source>", item, re.DOTALL)
            title = (
                (title_m.group(1) or title_m.group(2) or "").strip() if title_m else ""
            )
            url = (link_m.group(1) or link_m.group(2) or "").strip() if link_m else ""
            # Google News wraps links in its redirect; try to decode
            if "news.google.com" in url:
                # The real URL is after a redirect; use as-is (enrichment will follow)
                pass
            publisher = (
                self._html_to_text(source_m.group(1)).strip()
                if source_m
                else "Google News"
            )
            pub_date = pub_m.group(1).strip() if pub_m else ""
            year = datetime.now(UTC).year
            if pub_date:
                yr_m = re.search(r"\b(20\d\d)\b", pub_date)
                if yr_m:
                    year = int(yr_m.group(1))
            if not title or not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            sources.append(
                ResearchSource(
                    provider="google-news-rss",
                    title=title[:160],
                    url=url,
                    year=year,
                    authors=[publisher] if publisher else [],
                    abstract=f"{publisher}: {title}",
                    score=30.0,
                )
            )
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "google-news-rss",
            "ok" if sources else "empty",
            f"returned {len(sources)} news articles",
        )
        return sources

    def _search_macrotrends(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Macrotrends.net — historical financial metrics without API key.

        Macrotrends is the go-to source for 10–20-year historical charts of
        P/E ratio, revenue, net income, EPS, gross margin, EBITDA, dividend
        yield, and 50+ other metrics. A real analyst ALWAYS pulls Macrotrends
        to understand long-run valuation context and trend direction.

        Extracts company name tokens from the query, maps to Macrotrends
        ticker/slug URLs, and fetches the actual page content.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        # Resolve company name → ticker via yfinance if available
        candidate_tickers: list[tuple[str, str]] = []  # (ticker, name)
        try:
            import yfinance as yf  # type: ignore[import-not-found]

            _candidates = _extract_ticker_candidates(query)
            _stop = {
                "AND",
                "THE",
                "FOR",
                "WITH",
                "FROM",
                "THAT",
                "THIS",
                "STOCK",
                "EARNINGS",
                "REVENUE",
                "GROWTH",
                "ANALYSIS",
            }
            for token in query.split():
                t = token.strip(".,!?;:")
                if re.match(r"^[A-Z]{2,10}$", t) and t not in _stop:
                    if t not in _candidates:
                        _candidates.append(t)
                elif re.match(r"^[A-Z][a-z]{4,}$", t):
                    # Title-case: search YF to get ticker
                    yf_p = urllib.parse.urlencode(
                        {"q": t, "quotesCount": 1, "newsCount": 0}
                    )
                    yf_d = self._get_json(
                        f"https://query1.finance.yahoo.com/v1/finance/search?{yf_p}"
                    )
                    for item in (yf_d.get("quotes") or [])[:1]:
                        sym = str(item.get("symbol") or "").strip()
                        if sym and str(item.get("quoteType") or "").upper() == "EQUITY":
                            _candidates.append(sym)
            for ticker in _candidates[:4]:
                try:
                    info = yf.Ticker(ticker).info or {}
                    name = str(info.get("longName") or info.get("shortName") or ticker)
                    candidate_tickers.append((ticker, name))
                except Exception:
                    candidate_tickers.append((ticker, ticker))
        except ImportError:
            _tickers = _extract_ticker_candidates(query)
            candidate_tickers = [(t, t) for t in _tickers[:4]]

        # Macrotrends metric pages — these have actual historical series
        metrics = [
            ("revenue", "Revenue"),
            ("net-income", "Net Income"),
            ("eps-earnings-per-share-diluted", "Diluted EPS"),
            ("pe-ratio", "P/E Ratio"),
            ("price-to-book-ratio", "P/B Ratio"),
            ("return-on-equity", "ROE"),
            ("free-cash-flow", "Free Cash Flow"),
            ("gross-profit-margin", "Gross Margin"),
        ]

        for ticker, name in candidate_tickers[:3]:
            ticker_lc = ticker.lower()
            name_slug = re.sub(r"[^a-z0-9]+", "-", name.lower())[:30].strip("-")
            for metric_slug, metric_label in metrics[:4]:  # top 4 metrics per company
                url = f"https://www.macrotrends.net/stocks/charts/{ticker_lc}/{name_slug}/{metric_slug}"
                raw = self._get_text(
                    url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
                )
                if not raw:
                    continue
                # Extract JSON data embedded in the page
                # Macrotrends embeds data as: var originalData = [...];
                data_m = re.search(
                    r"var\s+originalData\s*=\s*(\[.*?\]);", raw, re.DOTALL
                )
                if data_m:
                    try:
                        data_rows = json.loads(data_m.group(1))
                        # Each row: {"date": "2024-01-01", "v1": 123456789}
                        rows = [
                            (r.get("date", "")[:7], r.get("v1") or r.get("v2"))
                            for r in data_rows
                            if r.get("date")
                        ]
                        rows = [(d, v) for d, v in rows if v is not None]
                        if rows:
                            recent = rows[-8:]  # last 8 quarters/years
                            series_str = "; ".join(
                                f"{d}: ${v / 1e9:.2f}B"
                                if abs(float(v)) >= 1e9
                                else f"{d}: {float(v):.2f}"
                                if metric_slug.endswith("ratio")
                                or metric_slug.startswith("pe")
                                or "margin" in metric_slug
                                else f"{d}: ${v / 1e6:.0f}M"
                                for d, v in recent
                            )
                            abstract = f"{name} ({ticker}) {metric_label} — Historical Data: {series_str}"
                            sources.append(
                                ResearchSource(
                                    provider="macrotrends",
                                    title=f"{name} ({ticker}) — {metric_label} Historical",
                                    url=url,
                                    year=datetime.now(UTC).year,
                                    abstract=abstract[:2000],
                                    score=55.0,
                                )
                            )
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif raw:
                    # No embedded data — still useful as a page to enrich
                    text = self._html_to_text(raw)[:800]
                    if text and len(text) > 100 and name.lower() in text.lower():
                        sources.append(
                            ResearchSource(
                                provider="macrotrends",
                                title=f"{name} ({ticker}) — {metric_label}",
                                url=url,
                                year=datetime.now(UTC).year,
                                abstract=text[:800],
                                score=40.0,
                            )
                        )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "macrotrends",
            "ok" if sources else "empty",
            f"returned {len(sources)} historical metric sources",
        )
        return sources

    def _search_stockanalysis(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """StockAnalysis.com — free earnings, revenue, balance sheet data.

        StockAnalysis provides clean tables of quarterly/annual financials,
        earnings estimates, institutional ownership, and more. It renders
        as static HTML for many sections, making it scrapable without a
        JavaScript engine. A key data cross-check tool for any analyst.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        seen_urls: set[str] = set()

        # Extract tickers
        tickers: list[str] = []
        _stop = {
            "AND",
            "THE",
            "FOR",
            "WITH",
            "STOCK",
            "EARNINGS",
            "REVENUE",
            "GROWTH",
            "ANALYSIS",
            "MARKET",
            "SHARES",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,10}$", t) and t not in _stop:
                tickers.append(t)
            elif re.match(r"^[A-Z][a-z]{4,}$", t):
                # Title-case name: look up ticker
                yf_p = urllib.parse.urlencode(
                    {"q": t, "quotesCount": 1, "newsCount": 0}
                )
                yf_d = self._get_json(
                    f"https://query1.finance.yahoo.com/v1/finance/search?{yf_p}"
                )
                for item in (yf_d.get("quotes") or [])[:1]:
                    sym = str(item.get("symbol") or "").strip()
                    if sym:
                        tickers.append(sym)

        tickers = list(dict.fromkeys(tickers))[:4]

        sections = [
            ("financials", "Annual Financials"),
            ("financials/quarterly", "Quarterly Financials"),
            ("forecast", "Earnings Estimates"),
        ]

        for ticker in tickers[:3]:
            ticker_lc = ticker.lower()
            for section_path, section_label in sections[:2]:
                url = f"https://stockanalysis.com/stocks/{ticker_lc}/{section_path}/"
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                raw = self._get_text(
                    url, accept="text/html,*/*", max_bytes=100_000, timeout_seconds=8
                )
                if not raw:
                    continue
                text = self._html_to_text(raw)
                if len(text) < 200:
                    continue
                # Extract the financial table portion
                # Look for revenue/earnings patterns in extracted text
                numbers = re.findall(r"\b\d{1,3}(?:\.\d+)?[BMK%]\b", text)
                if not numbers:
                    continue
                # Build a compact summary of financial data found
                first_2000 = re.sub(r"\s+", " ", text)[:2000]
                sources.append(
                    ResearchSource(
                        provider="stockanalysis",
                        title=f"{ticker.upper()} — {section_label} (StockAnalysis)",
                        url=url,
                        year=datetime.now(UTC).year,
                        abstract=first_2000,
                        score=58.0,
                    )
                )
                if len(sources) >= target_limit:
                    break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "stockanalysis",
            "ok" if sources else "empty",
            f"returned {len(sources)} financial table sources",
        )
        return sources

    def _search_insider_transactions(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """SEC Form 4 insider transactions — real smart-money signals.

        When executives and directors buy or sell their own stock, that's
        material non-public-activity-adjacent signal. SEC Form 4s are filed
        within 2 business days of every insider transaction and are public
        record. This is THE primary source for insider-trading analysis on
        Wall Street.

        Uses OpenInsider.com (aggregates SEC Form 4 filings, freely accessible)
        and the SEC EDGAR full-text browse endpoint for Form 4 filings.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        # Extract tickers from query
        tickers: list[str] = []
        _stop = {
            "AND",
            "THE",
            "FOR",
            "INSIDER",
            "TRANSACTION",
            "STOCK",
            "TRADING",
            "BUYING",
            "SELLING",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,6}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:4]

        for ticker in tickers[:3]:
            # OpenInsider — aggregated Form 4 data in human-readable table
            oi_url = f"https://openinsider.com/search?q={urllib.parse.quote(ticker)}"
            raw = self._get_text(
                oi_url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
            )
            if raw:
                text = self._html_to_text(raw)
                # Look for buy/sell transactions
                buy_m = re.findall(
                    r"(?:purchase|buy|bought)[^.]*\$[\d,]+", text.lower()
                )
                sell_m = re.findall(r"(?:sale|sell|sold)[^.]*\$[\d,]+", text.lower())
                if text and len(text) > 200:
                    sources.append(
                        ResearchSource(
                            provider="insider-transactions",
                            title=f"{ticker.upper()} — SEC Form 4 Insider Transactions",
                            url=oi_url,
                            year=datetime.now(UTC).year,
                            abstract=re.sub(r"\s+", " ", text)[:1500],
                            score=62.0,
                        )
                    )

            # Also pull directly from SEC EDGAR Form 4 browse
            tickers_map = self._get_sec_tickers()
            entry = tickers_map.get(ticker.upper())
            if entry:
                cik_int = entry.get("cik_str") or entry.get("cik")
                if cik_int:
                    cik_str = str(int(cik_int))
                    form4_url = (
                        f"https://www.sec.gov/cgi-bin/browse-edgar"
                        f"?action=getcompany&CIK={cik_str}"
                        f"&type=4&dateb=&owner=include&count={target_limit}"
                        f"&search_text=&output=atom"
                    )
                    form4_sources = self._fetch_browse_edgar_atom_generic(
                        form4_url, "4", entry.get("title", ticker)
                    )
                    sources.extend(form4_sources[:3])

            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "insider-transactions",
            "ok" if sources else "empty",
            f"returned {len(sources)} insider transaction sources",
        )
        return sources

    def _fetch_browse_edgar_atom_generic(
        self,
        url: str,
        form_type: str,
        company_name: str,
    ) -> list[ResearchSource]:
        """Reusable browse-edgar Atom XML fetcher for any form type."""
        import xml.etree.ElementTree as ET

        ATOM_NS = "http://www.w3.org/2005/Atom"
        body = self._get_sec_text(
            url,
            accept="application/atom+xml,text/xml,*/*",
        )
        if not body:
            return []
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return []
        sources: list[ResearchSource] = []
        for entry_el in list(root.iter(f"{{{ATOM_NS}}}entry"))[:8]:
            link_el = entry_el.find(f"{{{ATOM_NS}}}link")
            href = link_el.get("href", "") if link_el is not None else ""
            updated = entry_el.findtext(f"{{{ATOM_NS}}}updated") or ""
            if not href or not href.startswith("http"):
                continue
            year = int(updated[:4]) if len(updated) >= 4 else datetime.now(UTC).year
            sources.append(
                ResearchSource(
                    provider="sec-edgar",
                    title=f"{company_name} — Form {form_type} ({updated[:10]})",
                    url=href,
                    year=year,
                    abstract=f"SEC Form {form_type} filing by {company_name}. Filed: {updated[:10]}.",
                    score=48.0,
                )
            )
        return sources

    def _search_short_interest(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Short interest data from FINRA and public financial sites.

        Short interest is one of the most important signals for identifying:
        1. Potential short squeezes (high short interest + rising price)
        2. Market consensus that a stock is overvalued
        3. Hidden risk in positions

        FINRA publishes bi-monthly short interest data for all exchange-listed
        securities and it's freely accessible. We also check finviz and
        Yahoo Finance (already have via yfinance short float).
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        tickers: list[str] = []
        _stop = {"AND", "THE", "FOR", "SHORT", "INTEREST", "SQUEEZE", "FLOAT"}
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,6}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:4]

        for ticker in tickers[:3]:
            # Finviz has real-time short float and short ratio in their screener
            finviz_url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
            raw = self._get_text(
                finviz_url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
            )
            if raw:
                text = self._html_to_text(raw)
                # Look for key Finviz table values
                short_float_m = re.search(
                    r"Short Float[^\d]*([\d.]+%)", raw, re.IGNORECASE
                )
                short_ratio_m = re.search(
                    r"Short Ratio[^\d]*([\d.]+)", raw, re.IGNORECASE
                )
                short_int_m = re.search(
                    r"Short Interest[^\d]*([\d,.]+[MK]?)", raw, re.IGNORECASE
                )
                shares_outstanding_m = re.search(
                    r"Shs Outstand[^\d]*([\d.]+[MBK]?)", raw, re.IGNORECASE
                )
                float_m = re.search(
                    r"(?:^|\s)Float[^\d]*([\d.]+[MBK]?)", raw, re.IGNORECASE
                )
                inst_own_m = re.search(r"Inst Own[^\d]*([\d.]+%)", raw, re.IGNORECASE)
                insider_own_m = re.search(
                    r"Insider Own[^\d]*([\d.]+%)", raw, re.IGNORECASE
                )
                perf_ytd_m = re.search(r"Perf YTD[^\s]*([-+\d.]+%)", raw, re.IGNORECASE)

                parts: list[str] = [f"{ticker.upper()} (Finviz)"]
                if short_float_m:
                    parts.append(f"Short Float: {short_float_m.group(1)}")
                if short_ratio_m:
                    parts.append(f"Short Ratio: {short_ratio_m.group(1)} days")
                if short_int_m:
                    parts.append(f"Short Interest: {short_int_m.group(1)}")
                if float_m:
                    parts.append(f"Float: {float_m.group(1)}")
                if inst_own_m:
                    parts.append(f"Inst Own: {inst_own_m.group(1)}")
                if insider_own_m:
                    parts.append(f"Insider Own: {insider_own_m.group(1)}")
                if perf_ytd_m:
                    parts.append(f"Perf YTD: {perf_ytd_m.group(1)}")

                if len(parts) > 2:
                    sources.append(
                        ResearchSource(
                            provider="short-interest",
                            title=f"{ticker.upper()} — Short Interest & Ownership (Finviz)",
                            url=finviz_url,
                            year=datetime.now(UTC).year,
                            abstract=" | ".join(parts),
                            score=60.0,
                        )
                    )
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "short-interest",
            "ok" if sources else "empty",
            f"returned {len(sources)} short interest sources",
        )
        return sources

    def _search_earnings_data(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Earnings estimates and calendar from Yahoo Finance earnings API.

        Earnings surprises (beat vs miss) and forward guidance revisions are
        the single most important short-term catalyst for stock prices. This
        method fetches the earnings calendar and analyst EPS estimates — the
        same data a sell-side analyst builds their quarterly model around.
        Uses yfinance for earnings history and the YF earnings calendar API.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        tickers: list[str] = []
        _stop = {
            "AND",
            "THE",
            "FOR",
            "EARNINGS",
            "CALENDAR",
            "ESTIMATE",
            "EPS",
            "GUIDANCE",
            "BEAT",
            "MISS",
        }
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,10}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:4]

        try:
            import yfinance as yf  # type: ignore[import-not-found]

            for ticker in tickers[:4]:
                try:
                    tk = yf.Ticker(ticker.upper())
                    # Earnings history
                    history = tk.earnings_history
                    if history is not None and not history.empty:
                        rows = history.tail(8)
                        lines: list[str] = []
                        for idx, row in rows.iterrows():
                            eps_est = row.get("epsEstimate")
                            eps_act = row.get("epsActual")
                            surprise = row.get("epsDifference")
                            surprise_pct = row.get("surprisePercent")
                            if eps_act is not None:
                                period = str(idx)[:10] if idx else "?"
                                line = f"{period}: EPS ${eps_act:.2f}"
                                if eps_est is not None:
                                    line += f" (est ${eps_est:.2f}"
                                if surprise_pct is not None:
                                    line += f", {surprise_pct:+.1f}% surprise)"
                                elif eps_est is not None:
                                    line += ")"
                                lines.append(line)
                        if lines:
                            sources.append(
                                ResearchSource(
                                    provider="earnings-data",
                                    title=f"{ticker.upper()} — EPS History & Surprise Track",
                                    url=f"https://finance.yahoo.com/quote/{ticker.upper()}/earnings",
                                    year=datetime.now(UTC).year,
                                    abstract=f"{ticker.upper()} Earnings History: "
                                    + "; ".join(lines),
                                    score=68.0,
                                )
                            )
                    # Next earnings date estimate
                    cal = tk.calendar
                    if cal is not None and not cal.empty:
                        next_date = cal.get("Earnings Date")
                        if next_date is not None:
                            dates = (
                                list(next_date)
                                if hasattr(next_date, "__iter__")
                                else [next_date]
                            )
                            date_str = ", ".join(str(d)[:10] for d in dates[:2])
                            eps_est_low = cal.get("EPS Estimate")
                            rev_est = cal.get("Revenue Estimate")
                            cal_parts = [f"{ticker.upper()} Next Earnings: {date_str}"]
                            if eps_est_low is not None:
                                try:
                                    cal_parts.append(
                                        f"EPS Est: ${float(eps_est_low.iloc[0]):.2f}"
                                    )
                                except Exception:
                                    pass
                            if rev_est is not None:
                                try:
                                    rv = float(rev_est.iloc[0])
                                    cal_parts.append(f"Rev Est: ${rv / 1e9:.2f}B")
                                except Exception:
                                    pass
                            sources.append(
                                ResearchSource(
                                    provider="earnings-data",
                                    title=f"{ticker.upper()} — Next Earnings Date & Estimates",
                                    url=f"https://finance.yahoo.com/quote/{ticker.upper()}/earnings",
                                    year=datetime.now(UTC).year,
                                    abstract=" | ".join(cal_parts),
                                    score=65.0,
                                )
                            )
                except Exception:
                    pass
                if len(sources) >= target_limit:
                    break
        except ImportError:
            pass

        self._record_provider_diagnostic(
            "earnings-data",
            "ok" if sources else "empty",
            f"returned {len(sources)} earnings sources",
        )
        return sources

    def _search_fed_macro_data(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Federal Reserve (FRED) and BEA/BLS macro data — public APIs.

        Macro context is essential for equity valuation. Interest rates,
        inflation, GDP growth, unemployment, and yield curve shape all affect
        DCF discount rates and earnings multiples. This method fetches key
        macro indicators from FRED (Federal Reserve Economic Data) which
        provides a public JSON API for hundreds of economic series.

        A real macro/equity analyst checks these before forming a view on
        valuation multiples and risk premium.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        lower_q = query.lower()

        # FRED series IDs for macro indicators a WS analyst would check
        series_map = {
            "DFF": ("Fed Funds Rate (Effective)", "interest rate macro"),
            "GS10": ("10-Year Treasury Yield", "interest rate yield curve"),
            "GS2": ("2-Year Treasury Yield", "yield curve short rate"),
            "T10Y2Y": (
                "10Y-2Y Yield Spread (Inversion Signal)",
                "yield curve inversion",
            ),
            "CPIAUCSL": ("CPI Inflation (All Items)", "inflation price level"),
            "PCE": ("Personal Consumption Expenditures", "consumer spending gdp"),
            "GDPC1": ("Real GDP (Annualized)", "economic growth gdp"),
            "UNRATE": ("Unemployment Rate", "labor market jobs"),
            "VIXCLS": ("CBOE Volatility Index (VIX)", "market volatility fear"),
            "SP500": ("S&P 500 Index Level", "stock market equity"),
            "DEXUSEU": ("USD/EUR Exchange Rate", "dollar currency forex"),
            "DCOILWTICO": ("WTI Crude Oil Price", "oil energy commodity"),
        }

        # Determine which series are relevant to this query
        relevant_ids: list[str] = []
        for series_id, (label, keywords) in series_map.items():
            for kw in keywords.split():
                if kw in lower_q:
                    relevant_ids.append(series_id)
                    break

        # If no specific match, include core macro indicators for any WS query
        if not relevant_ids and self._looks_like_market_query(query):
            relevant_ids = ["DFF", "GS10", "T10Y2Y", "CPIAUCSL", "GDPC1", "VIXCLS"]

        if not relevant_ids:
            self._record_provider_diagnostic(
                "fed-macro", "skipped", "query not macro-relevant"
            )
            return []

        # FRED API — returns JSON, completely free and public
        fred_key = (
            os.environ.get("FRED_API_KEY") or ""
        )  # optional key for higher rate limits
        for series_id in relevant_ids[: min(target_limit, 6)]:
            label, _ = series_map[series_id]
            params = {
                "series_id": series_id,
                "file_type": "json",
                "limit": "8",
                "sort_order": "desc",
            }
            if fred_key:
                params["api_key"] = fred_key
            fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            # Use the public CSV endpoint (no API key needed)
            csv_raw = self._get_text(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&vintage_date=",
                accept="text/csv,text/plain,*/*",
                max_bytes=20_000,
                timeout_seconds=6,
            )
            if not csv_raw:
                continue
            lines = [
                line.strip()
                for line in csv_raw.splitlines()
                if line.strip() and not line.startswith("DATE")
            ]
            # Last 8 data points
            recent_lines = [line for line in lines if "." in line][-8:]
            if not recent_lines:
                continue
            data_points: list[str] = []
            for line in recent_lines:
                parts = line.split(",")
                if (
                    len(parts) >= 2
                    and parts[1].replace(".", "").replace("-", "").isdigit()
                ):
                    data_points.append(f"{parts[0]}: {parts[1]}")
            if not data_points:
                continue
            abstract = f"{label}: " + "; ".join(data_points[-4:])
            sources.append(
                ResearchSource(
                    provider="fed-macro",
                    title=f"FRED: {label} ({series_id})",
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    year=datetime.now(UTC).year,
                    abstract=abstract,
                    score=50.0,
                )
            )
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "fed-macro",
            "ok" if sources else "empty",
            f"returned {len(sources)} macro indicator sources",
        )
        return sources

    def _search_seeking_alpha_news(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Seeking Alpha public earnings articles and analysis.

        Seeking Alpha is the largest independent equity research platform.
        Their public pages (no paywall for news/earnings articles) contain:
        - Earnings call transcripts summaries
        - Analyst ratings changes
        - Company news articles
        These are the kind of qualitative research inputs a WS analyst reads
        to supplement the quantitative data.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))

        tickers: list[str] = []
        _stop = {"AND", "THE", "FOR", "ANALYSIS", "STOCK", "MARKET"}
        for token in query.split():
            t = token.strip(".,!?;:")
            if re.match(r"^[A-Z]{1,6}$", t) and t not in _stop:
                tickers.append(t)
        tickers = _extract_ticker_candidates(query) + tickers
        tickers = list(dict.fromkeys(tickers))[:3]

        for ticker in tickers[:3]:
            # Seeking Alpha news page (public)
            sa_url = f"https://seekingalpha.com/symbol/{ticker.upper()}/news"
            raw = self._get_text(
                sa_url, accept="text/html,*/*", max_bytes=80_000, timeout_seconds=8
            )
            if raw:
                # Extract article titles and links
                article_links = re.findall(
                    r'href="(/article/[^"]+)"[^>]*>([^<]{20,200})', raw, re.IGNORECASE
                )
                for path, title_raw in article_links[:target_limit]:
                    title = self._html_to_text(title_raw).strip()
                    if not title:
                        continue
                    full_url = f"https://seekingalpha.com{path}"
                    sources.append(
                        ResearchSource(
                            provider="seeking-alpha",
                            title=title[:160],
                            url=full_url,
                            year=datetime.now(UTC).year,
                            abstract=f"Seeking Alpha analysis: {title}",
                            score=35.0,
                        )
                    )
                    if len(sources) >= target_limit:
                        break
            if len(sources) >= target_limit:
                break

        self._record_provider_diagnostic(
            "seeking-alpha",
            "ok" if sources else "empty",
            f"returned {len(sources)} SA articles",
        )
        return sources

    def _search_reddit_finance(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[ResearchSource]:
        """Search Reddit finance communities for stock discussion and sentiment.

        Reddit's r/investing, r/stocks, r/wallstreetbets, and r/SecurityAnalysis
        provide real-time retail sentiment, DD (due diligence) posts, and often
        surface information before it reaches mainstream media.

        Uses Reddit's free JSON API — no authentication required for read-only.
        """
        sources: list[ResearchSource] = []
        target_limit = max(1, int(limit or self.limit_per_provider))
        try:
            encoded = urllib.parse.quote_plus(query[:120])
            # Search across the 4 most relevant finance subreddits.
            subreddits = "investing+stocks+wallstreetbets+SecurityAnalysis"
            url = (
                f"https://www.reddit.com/r/{subreddits}/search.json"
                f"?q={encoded}&sort=top&t=month&limit={min(target_limit * 3, 25)}"
                f"&restrict_sr=1"
            )
            raw = self._get_text(
                url,
                accept="application/json",
                max_bytes=120_000,
                timeout_seconds=8,
                extra_headers={"Accept": "application/json"},
            )
            if not raw:
                raise ValueError("empty response")
            data = json.loads(raw)
            posts = (data.get("data") or {}).get("children") or []
            for post in posts:
                post_data = post.get("data") or {}
                title = (post_data.get("title") or "").strip()
                selftext = (post_data.get("selftext") or "").strip()[:500]
                permalink = post_data.get("permalink") or ""
                score_val = int(post_data.get("score") or 0)
                subreddit = post_data.get("subreddit") or "reddit"
                num_comments = int(post_data.get("num_comments") or 0)
                if not title or not permalink:
                    continue
                full_url = f"https://www.reddit.com{permalink}"
                abstract = selftext if selftext else f"Reddit discussion: {title}"
                # Higher upvote/comment count = more market signal.
                source_score = min(
                    45.0, 15.0 + (score_val / 500.0) + (num_comments / 20.0)
                )
                sources.append(
                    ResearchSource(
                        provider="reddit-finance",
                        title=title[:180],
                        url=full_url,
                        year=datetime.now(UTC).year,
                        authors=[f"r/{subreddit}"],
                        abstract=abstract,
                        score=source_score,
                        quality_flags=["social-media-signal"],
                    )
                )
                if len(sources) >= target_limit:
                    break
        except Exception as exc:
            self._record_provider_diagnostic("reddit-finance", "error", str(exc)[:120])
        self._record_provider_diagnostic(
            "reddit-finance",
            "ok" if sources else "empty",
            f"returned {len(sources)} Reddit posts",
        )
        return sources

    def _fetch_urls_async_httpx(
        self,
        urls: list[str],
        max_bytes: int = 60_000,
        timeout_seconds: float = 10.0,
    ) -> dict[str, str]:
        """Fetch multiple URLs in parallel using httpx async — dramatically faster
        than sequential requests for bulk URL fetching (e.g., chain expansion).

        Returns a dict mapping url → plain text content.
        Used by the chain expansion mechanism to process 100+ URLs at once.
        """
        import asyncio

        results: dict[str, str] = {}
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError:
            # Fall back to threaded requests if httpx is unavailable.
            def _fetch_one(url: str) -> tuple[str, str]:
                text = self._get_text(
                    url, max_bytes=max_bytes, timeout_seconds=int(timeout_seconds)
                )
                return url, text

            with ThreadPoolExecutor(max_workers=min(20, len(urls))) as pool:
                for url, text in pool.map(_fetch_one, urls):
                    if text:
                        results[url] = text
            return results

        async def _fetch_all(url_list: list[str]) -> None:
            ua = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            limits = httpx.Limits(max_connections=30, max_keepalive_connections=10)
            async with httpx.AsyncClient(
                headers=headers,
                timeout=timeout_seconds,
                limits=limits,
                follow_redirects=True,
                verify=False,  # noqa: S501 — bulk fetching, no cert pinning needed
            ) as client:
                tasks = [client.get(u) for u in url_list]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for url, resp in zip(url_list, responses):
                    if isinstance(resp, Exception):
                        continue
                    try:
                        ct = resp.headers.get("content-type", "").lower()
                        if ct and not any(
                            m in ct for m in ("text/", "html", "json", "xml")
                        ):
                            continue
                        text = resp.text[:max_bytes]
                        if text:
                            results[url] = text
                    except Exception:
                        pass

        try:
            asyncio.run(_fetch_all(urls))
        except Exception:
            pass
        return results

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
            "sec-edgar",
            "financial-portals",
            "earnings-data",
            "insider-transactions",
            "short-interest",
            "macrotrends",
            "stockanalysis",
            "fed-macro",
            "google-news-rss",
            "seeking-alpha",
            "reddit-finance",
            "bing-search",
            "openalex",
            "semantic-scholar",
            "crossref",
            "web-search",
            "github-repositories",
        )

    def _provider_searchers(self) -> dict[str, Any]:
        return {
            "sec-edgar": self._search_sec_edgar,
            "financial-portals": self._search_financial_portals,
            "earnings-data": self._search_earnings_data,
            "insider-transactions": self._search_insider_transactions,
            "short-interest": self._search_short_interest,
            "macrotrends": self._search_macrotrends,
            "stockanalysis": self._search_stockanalysis,
            "fed-macro": self._search_fed_macro_data,
            "google-news-rss": self._search_google_news_rss,
            "seeking-alpha": self._search_seeking_alpha_news,
            "reddit-finance": self._search_reddit_finance,
            "bing-search": self._search_bing_results,
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
            return {"web-search", "bing-search", "google-news-rss", "gemini-flash"}

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
                # Reddit finance adds crowd sentiment and early signal.
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
                    "seeking-alpha",
                    "reddit-finance",
                    "gemini-flash",
                }
            # Include financial-portals for price/product data, and crossref for
            # reports/whitepapers that may index financial analyses.
            return {
                "web-search",
                "bing-search",
                "financial-portals",
                "sec-edgar",
                "google-news-rss",
                "gemini-flash",
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
                    "seeking-alpha",
                    "reddit-finance",
                }
            )
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

        # Run enrichment in parallel — same as Claude/Gemini deep research.
        if not sources:
            return []
        # Scale enrichment workers — I/O-bound fetches benefit from many
        # concurrent threads. 30 workers handles Playwright + requests concurrently.
        with ThreadPoolExecutor(max_workers=max(1, min(30, len(sources)))) as executor:
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

        # Try requests first (handles gzip, cookies, TLS better than urllib —
        # critical for financial sites that use Cloudflare or CDN protection).
        try:
            import requests as _requests  # type: ignore[import-not-found]

            resp = _requests.get(
                url,
                headers=headers,
                timeout=effective_timeout,
                stream=True,
                allow_redirects=True,
                verify=True,
            )
            ct = resp.headers.get("Content-Type", "").lower()
            if ct and not any(m in ct for m in ("text/", "html", "xml", "json")):
                return ""
            raw = b""
            for chunk in resp.iter_content(chunk_size=8192):
                raw += chunk
                if len(raw) >= max_bytes:
                    break
            return raw[:max_bytes].decode("utf-8", errors="replace")
        except Exception:
            pass

        # Fallback to urllib when requests is unavailable or fails.
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(  # noqa: S310 - policy-gated URLs
                request,
                timeout=effective_timeout,
            ) as response:
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if content_type and not any(
                    marker in content_type
                    for marker in ("text/", "html", "xml", "json")
                ):
                    return ""
                return response.read(max_bytes).decode("utf-8", errors="replace")
        except (Exception,):
            return ""

    # Domains that require JavaScript rendering to return real content.
    # requests/urllib only returns a blank/paywalled shell for these.
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
        if self._looks_like_current_evidence_query(self._active_objective):
            return max(2, min(6, url_count))
        return max(1, min(4, url_count))

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

    @staticmethod
    def _has_actionable_market_signal(text: str) -> bool:
        lower = (text or "").lower()
        if DeepResearchEngine._has_market_identifiers(text):
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

    # Maps non-standard .env key names to the canonical env var names used by the code.
    _ENV_KEY_ALIASES: dict[str, str] = {
        "gemini-api": "GEMINI_API_KEY",
        "gemini_api": "GEMINI_API_KEY",
        "google-api": "GOOGLE_API_KEY",
        "google_api": "GOOGLE_API_KEY",
        "gemini-api-key": "GEMINI_API_KEY",
        "google-api-key": "GOOGLE_API_KEY",
    }

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
                if not line or line.startswith("#"):
                    continue
                # Support both "KEY=VALUE" and "KEY: VALUE" separators.
                if "=" in line:
                    sep = "="
                elif ":" in line:
                    sep = ":"
                else:
                    continue
                key, value = line.split(sep, 1)
                env_key = key.strip().rstrip(":")
                if not env_key:
                    continue
                # Resolve non-standard key aliases (e.g. "Gemini-API" -> "GEMINI_API_KEY").
                canonical = self._ENV_KEY_ALIASES.get(env_key.lower())
                if canonical:
                    env_key = canonical
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
        synthesis_packet: dict[str, Any],
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
        synthesis_packet_path = artifact_dir / "synthesis_packet.json"
        durable_notes_path = self._durable_report_path(run_id_for_durable_notes)
        findings = self._finding_ledger(query, sources, plan)
        claim_trace_payload = self._claim_trace(objective, summary, findings)
        retrieval_payload = {
            "coverage": retrieval["coverage"],
            "passes": retrieval["passes"],
            "stop_reason": retrieval["stop_reason"],
            "query_variants": retrieval["query_variants"],
        }
        benchmark_adapters = self._benchmark_adapters(sources)
        evidence_graph_payload = self._evidence_graph(
            objective,
            sources,
            retrieval,
            pc_context_info,
            findings,
        )
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
            json.dumps(claim_trace_payload, indent=2),
            encoding="utf-8",
        )
        evidence_graph_path.write_text(
            json.dumps(evidence_graph_payload, indent=2),
            encoding="utf-8",
        )
        benchmark_adapters_path.write_text(
            json.dumps(benchmark_adapters, indent=2),
            encoding="utf-8",
        )
        synthesis_packet_path.write_text(
            json.dumps(synthesis_packet, indent=2),
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            json.dumps(self.provider_diagnostics, indent=2),
            encoding="utf-8",
        )
        self._update_persistent_evidence_index(
            run_id,
            objective,
            query,
            sources,
            claim_trace_payload,
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
            str(synthesis_packet_path.relative_to(self.workspace_root)),
            str(diagnostics_path.relative_to(self.workspace_root)),
        ]
        artifacts.extend(
            self._write_persistent_state_snapshots(run_id, query, objective)
        )
        if durable_notes_path is not None and durable_notes_path.exists():
            artifacts.append(str(durable_notes_path.relative_to(self.workspace_root)))
        return artifacts

    def _update_persistent_evidence_index(
        self,
        run_id: str,
        objective: str,
        query: str,
        sources: list[ResearchSource],
        claim_trace: dict[str, Any],
    ) -> None:
        del sources
        claims = list(claim_trace.get("claims") or [])
        if not claims:
            return
        if self._crawl_broker_enabled():
            self._crawl_broker_request(
                "/evidence/update-index",
                {
                    "run_id": run_id,
                    "objective": objective,
                    "query": query,
                    "claim_trace": claim_trace,
                },
                timeout_seconds=max(15.0, self.timeout_seconds),
            )
            return
        self._ensure_research_state_store()
        now = datetime.now(UTC).isoformat()
        try:
            with closing(self._connect_research_state()) as connection:
                with connection:
                    for claim in claims:
                        claim_text = str(claim.get("claim") or "").strip()
                        claim_key = self._normalize_title(claim_text)
                        if not claim_key:
                            continue
                        support_count = int(claim.get("support_count") or 0)
                        contradiction_count = int(claim.get("contradiction_count") or 0)
                        confidence_rank = self._finding_confidence_rank(
                            str(claim.get("confidence") or "")
                        )
                        supporting_sources = list(claim.get("supporting_sources") or [])
                        source_urls = self._persistent_unique_urls(
                            [
                                str(source.get("url") or "")
                                for source in supporting_sources
                                if isinstance(source, dict)
                            ]
                        )
                        providers = self._merge_text_lists(
                            [
                                str(source.get("provider") or "").strip()
                                for source in supporting_sources
                                if isinstance(source, dict)
                            ],
                            limit=24,
                        )
                        perspectives = self._merge_text_lists(
                            [str(claim.get("perspective") or "").strip()],
                            limit=12,
                        )
                        existing_claim = connection.execute(
                            """
                            SELECT support_count, contradiction_count,
                                   confidence_rank, source_urls_json,
                                   providers_json, perspectives_json
                            FROM evidence_claims
                            WHERE claim_key = ?
                            """,
                            (claim_key,),
                        ).fetchone()
                        merged_source_urls = self._merge_text_lists(
                            self._load_json_text_list(
                                existing_claim["source_urls_json"]
                            )
                            if existing_claim
                            else [],
                            source_urls,
                            limit=64,
                        )
                        merged_providers = self._merge_text_lists(
                            self._load_json_text_list(existing_claim["providers_json"])
                            if existing_claim
                            else [],
                            providers,
                            limit=24,
                        )
                        merged_perspectives = self._merge_text_lists(
                            self._load_json_text_list(
                                existing_claim["perspectives_json"]
                            )
                            if existing_claim
                            else [],
                            perspectives,
                            limit=16,
                        )
                        connection.execute(
                            """
                            INSERT INTO evidence_claims(
                                claim_key,
                                claim_text,
                                support_count,
                                contradiction_count,
                                confidence_rank,
                                source_urls_json,
                                providers_json,
                                perspectives_json,
                                last_seen_run,
                                updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(claim_key) DO UPDATE SET
                                claim_text = excluded.claim_text,
                                support_count = excluded.support_count,
                                contradiction_count = excluded.contradiction_count,
                                confidence_rank = excluded.confidence_rank,
                                source_urls_json = excluded.source_urls_json,
                                providers_json = excluded.providers_json,
                                perspectives_json = excluded.perspectives_json,
                                last_seen_run = excluded.last_seen_run,
                                updated_at = excluded.updated_at
                            """,
                            (
                                claim_key,
                                claim_text[:500],
                                support_count
                                + int(existing_claim["support_count"] or 0)
                                if existing_claim
                                else support_count,
                                contradiction_count
                                + int(existing_claim["contradiction_count"] or 0)
                                if existing_claim
                                else contradiction_count,
                                max(
                                    confidence_rank,
                                    int(existing_claim["confidence_rank"] or 0)
                                    if existing_claim
                                    else 0,
                                ),
                                json.dumps(merged_source_urls),
                                json.dumps(merged_providers),
                                json.dumps(merged_perspectives),
                                run_id,
                                now,
                            ),
                        )

                        contradiction_keys: list[str] = []
                        if contradiction_count > 0:
                            contradiction_key = claim_key
                            contradiction_keys.append(contradiction_key)
                            existing_contradiction = connection.execute(
                                """
                                SELECT count, claim_keys_json, source_urls_json
                                FROM evidence_contradictions
                                WHERE contradiction_key = ?
                                """,
                                (contradiction_key,),
                            ).fetchone()
                            merged_claim_keys = self._merge_text_lists(
                                self._load_json_text_list(
                                    existing_contradiction["claim_keys_json"]
                                )
                                if existing_contradiction
                                else [],
                                [claim_key],
                                limit=32,
                            )
                            merged_contradiction_urls = self._merge_text_lists(
                                self._load_json_text_list(
                                    existing_contradiction["source_urls_json"]
                                )
                                if existing_contradiction
                                else [],
                                merged_source_urls,
                                limit=64,
                            )
                            connection.execute(
                                """
                                INSERT INTO evidence_contradictions(
                                    contradiction_key,
                                    contradiction_text,
                                    count,
                                    claim_keys_json,
                                    source_urls_json,
                                    last_seen_run,
                                    updated_at
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(contradiction_key) DO UPDATE SET
                                    contradiction_text = excluded.contradiction_text,
                                    count = excluded.count,
                                    claim_keys_json = excluded.claim_keys_json,
                                    source_urls_json = excluded.source_urls_json,
                                    last_seen_run = excluded.last_seen_run,
                                    updated_at = excluded.updated_at
                                """,
                                (
                                    contradiction_key,
                                    claim_text[:500],
                                    contradiction_count
                                    + int(existing_contradiction["count"] or 0)
                                    if existing_contradiction
                                    else contradiction_count,
                                    json.dumps(merged_claim_keys),
                                    json.dumps(merged_contradiction_urls),
                                    run_id,
                                    now,
                                ),
                            )

                        for url in merged_source_urls:
                            domain = self._source_domain(url)
                            if not domain:
                                continue
                            existing_domain = connection.execute(
                                """
                                SELECT score, observation_count, urls_json,
                                       claim_keys_json, contradiction_keys_json
                                FROM evidence_domains
                                WHERE domain = ?
                                """,
                                (domain,),
                            ).fetchone()
                            merged_domain_urls = self._merge_text_lists(
                                self._load_json_text_list(existing_domain["urls_json"])
                                if existing_domain
                                else [],
                                [url],
                                limit=64,
                            )
                            merged_domain_claims = self._merge_text_lists(
                                self._load_json_text_list(
                                    existing_domain["claim_keys_json"]
                                )
                                if existing_domain
                                else [],
                                [claim_key],
                                limit=48,
                            )
                            merged_domain_contradictions = self._merge_text_lists(
                                self._load_json_text_list(
                                    existing_domain["contradiction_keys_json"]
                                )
                                if existing_domain
                                else [],
                                contradiction_keys,
                                limit=32,
                            )
                            connection.execute(
                                """
                                INSERT INTO evidence_domains(
                                    domain,
                                    score,
                                    observation_count,
                                    urls_json,
                                    claim_keys_json,
                                    contradiction_keys_json,
                                    last_seen_run,
                                    updated_at
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(domain) DO UPDATE SET
                                    score = excluded.score,
                                    observation_count = excluded.observation_count,
                                    urls_json = excluded.urls_json,
                                    claim_keys_json = excluded.claim_keys_json,
                                    contradiction_keys_json = excluded.contradiction_keys_json,
                                    last_seen_run = excluded.last_seen_run,
                                    updated_at = excluded.updated_at
                                """,
                                (
                                    domain,
                                    float(support_count + confidence_rank)
                                    + float(existing_domain["score"] or 0.0)
                                    if existing_domain
                                    else float(support_count + confidence_rank),
                                    1 + int(existing_domain["observation_count"] or 0)
                                    if existing_domain
                                    else 1,
                                    json.dumps(merged_domain_urls),
                                    json.dumps(merged_domain_claims),
                                    json.dumps(merged_domain_contradictions),
                                    run_id,
                                    now,
                                ),
                            )
        except sqlite3.Error:
            return

    def _persistent_evidence_snapshot(
        self,
        query: str,
        objective: str,
        limit: int = 12,
    ) -> dict[str, Any]:
        if self._crawl_broker_enabled():
            return self._crawl_broker_request(
                "/evidence/snapshot",
                {
                    "query": query,
                    "objective": objective,
                    "limit": int(limit),
                },
                timeout_seconds=max(10.0, self.timeout_seconds),
            )
        self._ensure_research_state_store()
        with closing(self._connect_research_state()) as connection:
            claim_rows = connection.execute(
                """
                SELECT claim_key, claim_text, support_count,
                       contradiction_count, confidence_rank,
                       source_urls_json, perspectives_json
                FROM evidence_claims
                ORDER BY support_count DESC, confidence_rank DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 6, 48),),
            ).fetchall()
            contradiction_rows = connection.execute(
                """
                SELECT contradiction_key, contradiction_text, count,
                       claim_keys_json, source_urls_json
                FROM evidence_contradictions
                ORDER BY count DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 4, 24),),
            ).fetchall()
            domain_rows = connection.execute(
                """
                SELECT domain, score, observation_count, urls_json,
                       claim_keys_json, contradiction_keys_json
                FROM evidence_domains
                ORDER BY score DESC, observation_count DESC, updated_at DESC
                LIMIT ?
                """,
                (max(limit * 4, 24),),
            ).fetchall()

        claims: list[dict[str, Any]] = []
        for row in claim_rows:
            claim_text = str(row["claim_text"] or "").strip()
            if self._persistent_match_score(claim_text, query, objective) < 0.24:
                continue
            claims.append(
                {
                    "claim_key": str(row["claim_key"] or ""),
                    "claim": claim_text,
                    "support_count": int(row["support_count"] or 0),
                    "contradiction_count": int(row["contradiction_count"] or 0),
                    "confidence_rank": int(row["confidence_rank"] or 0),
                    "source_urls": self._load_json_text_list(row["source_urls_json"])[
                        :8
                    ],
                    "perspectives": self._load_json_text_list(row["perspectives_json"])[
                        :6
                    ],
                }
            )
            if len(claims) >= limit:
                break

        contradictions: list[dict[str, Any]] = []
        for row in contradiction_rows:
            contradiction_text = str(row["contradiction_text"] or "").strip()
            if (
                self._persistent_match_score(
                    contradiction_text,
                    query,
                    objective,
                )
                < 0.22
            ):
                continue
            contradictions.append(
                {
                    "contradiction_key": str(row["contradiction_key"] or ""),
                    "text": contradiction_text,
                    "count": int(row["count"] or 0),
                    "claim_keys": self._load_json_text_list(row["claim_keys_json"])[:8],
                    "source_urls": self._load_json_text_list(row["source_urls_json"])[
                        :8
                    ],
                }
            )
            if len(contradictions) >= limit:
                break

        domains: list[dict[str, Any]] = []
        for row in domain_rows[:limit]:
            domains.append(
                {
                    "domain": str(row["domain"] or ""),
                    "score": float(row["score"] or 0.0),
                    "observation_count": int(row["observation_count"] or 0),
                    "urls": self._load_json_text_list(row["urls_json"])[:8],
                    "claim_keys": self._load_json_text_list(row["claim_keys_json"])[:8],
                    "contradiction_keys": self._load_json_text_list(
                        row["contradiction_keys_json"]
                    )[:8],
                }
            )

        return {
            "query": query,
            "objective": objective,
            "seed_urls": self._persistent_seed_urls(query, objective, limit=limit),
            "claims": claims,
            "contradictions": contradictions,
            "domains": domains,
        }

    def _persistent_crawl_queue_snapshot(self, limit: int = 24) -> dict[str, Any]:
        if self._crawl_broker_enabled():
            return self._crawl_broker_request(
                "/queue/snapshot",
                {"limit": int(limit)},
                timeout_seconds=max(10.0, self.timeout_seconds),
            )
        self._ensure_research_state_store()
        with closing(self._connect_research_state()) as connection:
            rows = connection.execute(
                """
                SELECT url, domain, priority, status, source_query,
                       source_url, js_required, attempts
                FROM crawl_queue
                ORDER BY priority DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {
            "queued": [
                {
                    "url": str(row["url"] or ""),
                    "domain": str(row["domain"] or ""),
                    "priority": float(row["priority"] or 0.0),
                    "status": str(row["status"] or ""),
                    "source_query": str(row["source_query"] or ""),
                    "source_url": str(row["source_url"] or ""),
                    "js_required": bool(int(row["js_required"] or 0)),
                    "attempts": int(row["attempts"] or 0),
                }
                for row in rows
            ]
        }

    def _write_persistent_state_snapshots(
        self,
        run_id: str,
        query: str,
        objective: str,
    ) -> list[str]:
        if not run_id:
            return []
        artifact_dir = self.workspace_root / "runs" / run_id / "research"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        evidence_snapshot_path = artifact_dir / "evidence_index_snapshot.json"
        crawl_snapshot_path = artifact_dir / "crawl_queue_snapshot.json"
        try:
            evidence_snapshot_path.write_text(
                json.dumps(
                    self._persistent_evidence_snapshot(query, objective),
                    indent=2,
                ),
                encoding="utf-8",
            )
            crawl_snapshot_path.write_text(
                json.dumps(self._persistent_crawl_queue_snapshot(), indent=2),
                encoding="utf-8",
            )
        except (OSError, sqlite3.Error):
            return []
        return [
            str(evidence_snapshot_path.relative_to(self.workspace_root)),
            str(crawl_snapshot_path.relative_to(self.workspace_root)),
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
        sanitized = _sanitize_evidence_claim_text(
            source.title,
            source.abstract,
            source.url,
        )
        sentence = _sentences(sanitized)
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
        return _sanitize_evidence_claim_text(source.title, "", source.url)[:220]

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

    @classmethod
    def _synthesis_source_limit(
        cls,
        depth: str,
        source_count: int,
        synthesis_mode: str,
        objective: str = "",
    ) -> int:
        if source_count <= 0:
            return 0
        if synthesis_mode == "durable-notes-only":
            return min(source_count, 24)
        limits = {
            "quick": 12,
            "standard": 24,
            "multi-hour": 32,
        }
        if (
            depth == "multi-hour"
            and objective
            and (
                cls._looks_like_software_agent_query(objective)
                or cls._looks_like_current_evidence_query(objective)
                or cls._looks_like_comprehensive_research(objective.lower())
            )
        ):
            return max(1, min(source_count, 96))
        return max(1, min(source_count, limits.get(depth, 24)))

    def _build_synthesis_packet(
        self,
        objective: str,
        query: str,
        sources: list[ResearchSource],
        depth: str,
        plan: dict[str, Any] | None,
        durable_notes: str,
        synthesis_mode: str,
    ) -> dict[str, Any]:
        source_limit = self._synthesis_source_limit(
            depth,
            len(sources),
            synthesis_mode,
            objective,
        )
        synthesis_sources = sources[:source_limit]
        findings = self._finding_ledger(query or objective, synthesis_sources, plan)
        perspective_coverage = self._perspective_coverage(
            sources,
            (plan or {}).get("perspectives") or [],
        )
        provider_counts: dict[str, int] = {}
        for source in sources:
            provider_counts[source.provider] = (
                provider_counts.get(source.provider, 0) + 1
            )
        return {
            "objective": objective,
            "query": query,
            "depth": depth,
            "synthesis_mode": synthesis_mode,
            "total_ranked_sources": len(sources),
            "synthesis_source_count": len(synthesis_sources),
            "durable_notes_available": bool(durable_notes.strip()),
            "provider_counts": provider_counts,
            "perspective_coverage": perspective_coverage,
            "findings": findings,
            "market_signals": (
                self._market_signal_snapshot(synthesis_sources)
                if self._looks_like_market_query(query or objective)
                else []
            ),
            "top_sources": [
                {
                    "provider": source.provider,
                    "title": source.title,
                    "url": source.url,
                    "year": source.year,
                    "evidence_grade": source.evidence_grade,
                    "score": round(source.score, 3),
                    "abstract": (source.abstract or source.title)[:240],
                }
                for source in synthesis_sources
            ],
        }

    def _summarize(
        self,
        objective: str,
        sources: list[ResearchSource],
        depth: str = "standard",
        plan: dict[str, Any] | None = None,
        query: str = "",
        durable_notes: str = "",
        synthesis_mode: str = "hybrid",
        synthesis_packet: dict[str, Any] | None = None,
    ) -> str:
        if not sources:
            return (
                "Live research did not return sources from configured public "
                f"providers for: {objective}. Check network policy, API "
                "availability, or attach MCP research servers."
            )
        packet = synthesis_packet or self._build_synthesis_packet(
            objective,
            query,
            sources,
            depth,
            plan,
            durable_notes,
            synthesis_mode,
        )
        synthesis_source_count = int(packet.get("synthesis_source_count") or 0)
        synthesis_sources = (
            sources[:synthesis_source_count] if synthesis_source_count > 0 else sources
        )
        if synthesis_mode == "durable-notes-only" and durable_notes.strip():
            ai_synthesis = self._ai_durable_notes_synthesis(
                objective,
                synthesis_sources,
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
        findings = list(packet.get("findings") or [])
        perspective_coverage = dict(packet.get("perspective_coverage") or {})
        subquestion_count = len((plan or {}).get("subquestions", []))

        # ------------------------------------------------------------------
        # SCIENTIST SYNTHESIS: ask AI to reason about what the evidence
        # actually says — forming conclusions, noting contradictions, and
        # flagging gaps — rather than generating a boilerplate summary.
        # ------------------------------------------------------------------
        ai_synthesis = self._ai_scientist_synthesis(
            objective,
            synthesis_sources,
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
        market_signal_lines = list(packet.get("market_signals") or [])
        market_snapshot_text = (
            "Market signal snapshot: " + "; ".join(market_signal_lines[:5]) + ". "
            if market_signal_lines
            else ""
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
            f"{market_snapshot_text}"
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
        """Synthesize evidence like a Wall Street analyst (for market queries)
        or a senior research scientist (for all other topics).

        Wall Street mode produces: investment thesis, bull/bear/base case with
        probability weights, named tickers, catalysts with dates, risk factors
        ranked by severity, comp valuation, and a conviction call.

        Science mode produces: evidence weighing, contradiction analysis,
        credibility grading, gap identification, and confidence calibration.

        Returns an empty string when AI is unavailable.
        """
        if not sources:
            return ""
        market_mode = self._looks_like_market_query(objective)

        # Build compact evidence digest.
        source_lines: list[str]
        if synthesis_mode == "durable-notes-only":
            source_lines = self._minimal_source_metadata_lines(sources)
        else:
            source_lines = []
            for src in sources:
                year = f" ({src.year})" if src.year else ""
                grade = src.evidence_grade
                snippet = (src.abstract or src.title)[:200].replace("\n", " ")
                source_lines.append(
                    f"[{src.provider}/{grade}] {src.title}{year}: {snippet}"
                )

        finding_lines: list[str] = []
        for f in findings[:10]:
            finding_lines.append(
                f"  {f['perspective']} ({f['confidence']}, "
                f"{f['support_count']} sources, "
                f"{f['contradiction_count']} contradictions): {f['finding']}"
            )
        missing = ", ".join(perspective_coverage.get("missing") or []) or "none"
        market_signal_lines = (
            self._market_signal_snapshot(sources) if market_mode else []
        )
        subquestions = (
            "\n".join(f"  - {sq}" for sq in (plan or {}).get("subquestions", [])[:10])
            or "  (none recorded)"
        )

        if market_mode:
            system = (
                "You are a Managing Director at a top-tier Wall Street investment bank "
                "(think Goldman Sachs, Morgan Stanley, JPMorgan). You are writing a "
                "research note that will be distributed to institutional investors. "
                "This is NOT a summary — it is a rigorous investment analysis.\n\n"
                "YOUR ANALYSIS MUST INCLUDE ALL OF THE FOLLOWING:\n\n"
                "1. EXECUTIVE SUMMARY & RECOMMENDATION (1 paragraph)\n"
                "   - Clear Buy / Overweight / Hold / Underweight / Sell call\n"
                "   - Conviction level: High / Medium / Low (with justification)\n"
                "   - 12-month price target with upside/downside % vs current price\n"
                "   - One-sentence thesis statement\n\n"
                "2. INVESTMENT THESIS — BULL CASE (probability weight: X%)\n"
                "   - Specific catalysts that would drive the thesis\n"
                "   - Valuation in upside scenario (P/E, EV/EBITDA, or P/S)\n"
                "   - Key data points supporting this case with sources cited\n\n"
                "3. INVESTMENT THESIS — BASE CASE (probability weight: X%)\n"
                "   - Expected trajectory with specific metrics (revenue growth, margin)\n"
                "   - Fair value under consensus assumptions\n"
                "   - Upcoming catalysts and their expected impact\n\n"
                "4. INVESTMENT THESIS — BEAR CASE (probability weight: X%)\n"
                "   - Specific risks that would derail the thesis\n"
                "   - Downside scenario valuation\n"
                "   - Counterarguments and short thesis points\n\n"
                "5. CATALYST CALENDAR\n"
                "   - List specific upcoming events with dates/quarters "
                "(earnings, product launches, regulatory decisions, management events)\n"
                "   - Rate each catalyst: positive / negative / binary\n\n"
                "6. COMPARABLE COMPANY ANALYSIS\n"
                "   - Name at least 3-5 comparable companies WITH TICKERS\n"
                "   - Compare key multiples (P/E, EV/EBITDA, P/S, EV/Revenue)\n"
                "   - State whether target is cheap, fairly valued, or expensive vs comps\n\n"
                "7. KEY RISKS (ranked by severity)\n"
                "   - At least 5 specific risks with quantified potential impact\n"
                "   - Include: competitive risk, regulatory risk, macro risk, "
                "execution risk, balance sheet risk\n\n"
                "8. EVIDENCE QUALITY ASSESSMENT\n"
                "   - Which findings are supported by primary sources "
                "(SEC filings, earnings transcripts, official data)\n"
                "   - Which are based on secondary sources (analyst reports, news)\n"
                "   - Key uncertainties that require more diligence\n\n"
                "CRITICAL RULES:\n"
                "- ALWAYS name specific companies with their ticker symbols in parentheses\n"
                "- ALWAYS use specific numbers (%, $, multiples) not vague language\n"
                "- NEVER say 'the company' without naming it\n"
                "- NEVER use phrases like 'it is clear that' or 'obviously'\n"
                "- If evidence is thin on a section, say so explicitly — do not fabricate\n"
                "- Bull/bear/base probabilities must sum to 100%"
            )
            user = (
                f"Research objective: {objective}\n\n"
                f"Subquestions investigated:\n{subquestions}\n\n"
                f"Evidence collected ({len(sources)} sources across "
                f"{len({s.provider for s in sources})} providers):\n"
                + "\n".join(source_lines)
                + (
                    "\n\nMarket signal candidates identified in evidence:\n"
                    + "\n".join(market_signal_lines)
                    if market_signal_lines
                    else ""
                )
                + (
                    "\n\nDurable distilled report notes (deep research accumulation):\n"
                    + durable_notes[:16000]
                    if durable_notes
                    else ""
                )
                + f"\n\nPer-perspective findings:\n"
                + "\n".join(finding_lines or ["  (none yet)"])
                + f"\n\nUncovered perspectives: {missing}\n\n"
                + "Write the complete Wall Street research note as specified above. "
                + "Be forensically specific — institutional investors will scrutinize "
                + "every number and claim. Cite the evidence type for each major assertion."
            )
        else:
            system = (
                "You are a senior research scientist writing an evidence synthesis "
                "for a peer-reviewed audience. Your task is NOT to summarize — "
                "it is to ANALYZE with the rigor of a Nature or Science Methods paper.\n\n"
                "YOUR ANALYSIS MUST:\n"
                "1. FORM EXPLICIT CONCLUSIONS with stated confidence levels "
                "(high/moderate/low/speculative) and the minimum evidence "
                "threshold that would change each conclusion.\n"
                "2. WEIGH CONTRADICTIONS forensically: for every conflicting claim, "
                "identify whether the cause is methodological, scope-related, "
                "recency bias, or sampling artifact — and state which side the "
                "weight of evidence favors and why.\n"
                "3. GRADE SOURCE CREDIBILITY for every major claim: "
                "peer-reviewed > pre-registered > government data > industry reports "
                "> preprints > blogs > uncited claims. Flag over-reliance on "
                "low-credibility sources explicitly.\n"
                "4. MAP CAUSAL MECHANISMS: don't just state what was found — "
                "explain the mechanism. What drives the effect? What are the "
                "confounders? What are the effect sizes?\n"
                "5. IDENTIFY CRITICAL GAPS: name exactly what evidence is missing, "
                "why it matters, and what kind of study would fill it.\n"
                "6. ASSESS REPLICATION STATUS: has the finding been independently "
                "replicated? Are there failed replications? What is the p-curve?\n"
                "7. PRACTICAL IMPLICATIONS: what do the findings mean for real-world "
                "application? What are the scope conditions and boundary cases?\n\n"
                "Be specific, technical, and never use filler language. "
                "A reader should be able to cite your analysis in a paper."
            )
            user = (
                f"Research objective: {objective}\n\n"
                f"Subquestions investigated:\n{subquestions}\n\n"
                f"Evidence found ({len(sources)} sources):\n"
                + "\n".join(source_lines)
                + (
                    "\n\nDurable distilled report notes:\n" + durable_notes[:16000]
                    if durable_notes
                    else ""
                )
                + f"\n\nPer-perspective findings:\n"
                + "\n".join(finding_lines or ["  (none yet)"])
                + f"\n\nUncovered perspectives: {missing}\n\n"
                + "Synthesize this evidence as a senior research scientist would. "
                + "Be substantive, specific, and technically rigorous. "
                + "A reader must be able to make informed decisions or design "
                + "follow-up studies based on your synthesis."
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
        software_agent_mode = self._looks_like_software_agent_diagnostic_objective(
            f"{objective} {query}"
        )
        diagnostic_queries = self._software_agent_diagnostic_queries(objective)
        diagnostic_perspectives = self._software_agent_diagnostic_perspectives(
            objective
        )
        diagnostic_subquestions = self._software_agent_diagnostic_subquestions(
            objective
        )
        diagnostic_authorities = self._software_agent_diagnostic_seed_urls(
            objective
        )

        # ------------------------------------------------------------------
        # ADAPTIVE PLANNING: derive perspectives, comparative axes, and
        # evidence requirements from AI reasoning about THIS specific
        # objective, not from domain-type templates.
        # ------------------------------------------------------------------
        perspectives = (
            diagnostic_perspectives
            if diagnostic_perspectives
            else self._research_perspectives(query, objective, depth)
        )

        ai_axes = self._ai_research_axes(objective, query, depth)
        comparative_axes = ai_axes.get("comparative_axes") or []
        evidence_requirements = ai_axes.get("evidence_requirements") or []
        if software_agent_mode:
            comparative_axes = list(comparative_axes) + [
                "planner/query distillation quality",
                "browser and sandbox tool routing",
                "retrieval breadth and crawl scaling",
                "evidence ranking and synthesis fidelity",
                "benchmark performance against leading research agents",
            ]
            evidence_requirements = list(evidence_requirements) + [
                "code-path evidence for planning, routing, retrieval, and synthesis",
                "browser or sandbox traces showing active tool usage",
                "crawl breadth metrics, queue state, or frontier expansion evidence",
                "authoritative benchmark or product documentation for competitor comparison",
            ]

        # AI-derived subquestions from strategy call above.
        ai_subquestions = list(ai_strategy.get("subquestions") or [])
        if diagnostic_subquestions:
            ai_subquestions = diagnostic_subquestions + ai_subquestions

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
        plan_queries = list(diagnostic_queries)
        plan_queries.extend(self._entity_queries(query, objective))

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

        merged_domains: list[str] = []
        seen_domains: set[str] = set()
        for domain in diagnostic_authorities + list(
            ai_strategy.get("authoritative_domains") or []
        ):
            text = str(domain or "").strip()
            if not text:
                continue
            normalized = text.lower().rstrip("/")
            if normalized in seen_domains:
                continue
            seen_domains.add(normalized)
            merged_domains.append(text)

        return {
            "core_question": objective[:300],
            "subquestions": subquestions,
            "comparative_axes": comparative_axes,
            "evidence_requirements": evidence_requirements,
            "perspectives": perspectives,
            "query_plan": deduped_queries,
            # AI-reasoned authoritative domains are stored so that
            # _iterative_retrieval can seed the source list with them.
            "ai_authoritative_domains": merged_domains,
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
        combined = " ".join(part for part in (query, objective) if part).strip()
        lower = combined.lower()
        entities = sorted(
            DeepResearchEngine._entity_terms_from_query(combined),
            key=lambda item: (
                lower.find(item.lower())
                if lower.find(item.lower()) >= 0
                else len(lower) + len(item),
                len(item),
            ),
        )
        generic_terms = DeepResearchEngine._generic_query_terms()
        entity_tokens = {
            token
            for entity in entities
            for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", entity.lower())
        }

        anchor_tokens: list[str] = []
        seen_anchor_tokens: set[str] = set()
        for token in re.findall(r"\b[a-z][a-z0-9-]{3,}\b", lower):
            if (
                token in generic_terms
                or token in entity_tokens
                or token in seen_anchor_tokens
            ):
                continue
            seen_anchor_tokens.add(token)
            anchor_tokens.append(token)
            if len(anchor_tokens) >= 8:
                break
        for token in DeepResearchEngine._keywords(combined):
            if (
                token in generic_terms
                or token in entity_tokens
                or token in seen_anchor_tokens
            ):
                continue
            seen_anchor_tokens.add(token)
            anchor_tokens.append(token)
            if len(anchor_tokens) >= 8:
                break

        focus_phrases: list[str] = []
        if DeepResearchEngine._looks_like_current_evidence_query(combined):
            focus_phrases.extend(["latest evidence", "current analysis"])
        if "risk" in lower or "uncertainty" in lower:
            focus_phrases.append("risk analysis")
        if "compare" in lower or len(entities) > 1:
            focus_phrases.append("comparison")

        focused: list[str] = []
        if entities:
            for entity in entities[:4]:
                focused.append(entity)
                for token in anchor_tokens[:4]:
                    focused.append(f"{entity} {token}")
                for phrase in focus_phrases[:3]:
                    if phrase not in entity.lower():
                        focused.append(f"{entity} {phrase}")
            for left, right in combinations(entities[:4], 2):
                focused.append(f"{left} {right} comparison")
        elif DeepResearchEngine._looks_like_current_evidence_query(combined):
            core = DeepResearchEngine._query_core_terms(combined)
            if core:
                focused.append(core)
                for phrase in focus_phrases[:2]:
                    focused.append(f"{core} {phrase}")

        deduped: list[str] = []
        seen_queries: set[str] = set()
        for candidate in focused:
            text = candidate[:120].strip()
            if not text:
                continue
            if DeepResearchEngine._is_low_signal_query_variant(text, combined):
                continue
            if DeepResearchEngine._is_noisy_query_variant(text, combined):
                continue
            normalized = DeepResearchEngine._normalize_title(text)
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            deduped.append(text)
        return deduped[:18]

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
        diagnostic_queries = DeepResearchEngine._software_agent_diagnostic_queries(
            cleaned
        )
        if diagnostic_queries:
            return diagnostic_queries[0]
        distilled = DeepResearchEngine._query_core_terms(cleaned)
        return distilled[:240].strip() or cleaned[:240].strip() or objective[:240]

    @classmethod
    def _software_agent_diagnostic_queries(cls, objective: str) -> list[str]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        lower = re.sub(r"\s+", " ", objective).strip().lower()
        anchors = ["deep research agent"]
        if "agentos" in lower:
            anchors.insert(0, "agentos deep research")
        elif "orchestrator" in lower:
            anchors.insert(0, "research orchestrator")

        aspects: list[str] = []
        if re.search(
            r"\b(browser|sandbox|pc control|desktop|computer use|local pc|web browsing)\b",
            lower,
        ):
            aspects.append("browser sandbox pc control routing")
        if re.search(
            r"\b(website|websites|url|urls|crawl|crawler|retrieval|breadth|coverage|10k|1000)\b",
            lower,
        ):
            aspects.append("retrieval breadth crawl scaling")
        if re.search(
            r"\b(template|general|generic|useful data|ranking|evidence|synthesis|analyst|scientist)\b",
            lower,
        ):
            aspects.append("evidence quality ranking synthesis")
        if re.search(
            r"\b(compare|comparison|comparable|claude|gpt|gemini|openhands|openclaw)\b",
            lower,
        ):
            aspects.append("benchmark comparison")
        if re.search(
            r"\b(fix|issue|issues|failure|failures|gap|gaps|bug|bugs|why|underperform|shallow)\b",
            lower,
        ):
            aspects.append("failure modes architecture gaps")
        if not aspects:
            aspects = [
                "failure modes architecture gaps",
                "retrieval breadth crawl scaling",
                "evidence quality ranking synthesis",
            ]

        queries: list[str] = []
        comparator_terms = [
            token
            for token in ("claude", "gpt", "gemini", "openhands", "openclaw")
            if token in lower
        ]
        for anchor in anchors:
            queries.append(anchor)
            for aspect in aspects:
                queries.append(f"{anchor} {aspect}")
            for comparator in comparator_terms:
                queries.append(f"{anchor} {comparator} comparison")
        if "mcp" in lower:
            queries.append("model context protocol research agent tool routing")
        if "browser" in lower or "sandbox" in lower:
            queries.append("computer use browser automation research agent architecture")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            text = str(query or "").strip()[:240]
            normalized = cls._normalize_title(text)
            if not text or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(text)
        return deduped[:14]

    @classmethod
    def _software_agent_diagnostic_seed_urls(cls, objective: str) -> list[str]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        lower = objective.lower()
        candidates = ["https://github.com"]
        if "mcp" in lower:
            candidates.append("https://modelcontextprotocol.io")
        if "browser" in lower or "sandbox" in lower:
            candidates.append("https://playwright.dev")
        if "claude" in lower:
            candidates.append("https://docs.anthropic.com")
        if "gpt" in lower or "openai" in lower:
            candidates.append("https://platform.openai.com/docs")
        if "gemini" in lower or "google" in lower:
            candidates.append("https://ai.google.dev")

        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped[:8]

    @classmethod
    def _software_agent_diagnostic_perspectives(
        cls,
        objective: str,
    ) -> list[dict[str, Any]]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        lower = objective.lower()
        perspectives = [
            {
                "name": "architecture",
                "goal": "Audit orchestration, planning, and execution seams.",
                "keywords": [
                    "architecture",
                    "planner",
                    "orchestrator",
                    "routing",
                    "runtime",
                ],
                "queries": [
                    "deep research agent failure modes architecture gaps",
                    "deep research agent planner routing diagnostics",
                ],
            },
            {
                "name": "browser-tooling",
                "goal": "Verify browser, sandbox, and pc-control tool routing.",
                "keywords": [
                    "browser",
                    "sandbox",
                    "pc control",
                    "computer use",
                    "tool routing",
                ],
                "queries": [
                    "deep research agent browser sandbox pc control routing",
                    "computer use browser automation research agent architecture",
                ],
            },
            {
                "name": "retrieval-breadth",
                "goal": "Measure crawl scaling, frontier expansion, and source breadth.",
                "keywords": [
                    "retrieval",
                    "crawl",
                    "breadth",
                    "coverage",
                    "frontier",
                ],
                "queries": [
                    "deep research agent retrieval breadth crawl scaling",
                    "deep research agent frontier expansion coverage",
                ],
            },
            {
                "name": "evidence-quality",
                "goal": "Assess ranking, evidence quality, and synthesis fidelity.",
                "keywords": [
                    "evidence quality",
                    "ranking",
                    "synthesis",
                    "grounding",
                    "useful data",
                ],
                "queries": [
                    "deep research agent evidence quality ranking synthesis",
                    "deep research agent grounding useful data diagnostics",
                ],
            },
        ]
        if re.search(
            r"\b(compare|comparison|comparable|claude|gpt|gemini|openhands|openclaw)\b",
            lower,
        ):
            perspectives.append(
                {
                    "name": "benchmarks",
                    "goal": "Compare observed behavior against leading research agents.",
                    "keywords": [
                        "benchmark",
                        "comparison",
                        "claude",
                        "gpt",
                        "gemini",
                    ],
                    "queries": [
                        "deep research agent benchmark comparison",
                        "deep research agent claude gpt gemini comparison",
                    ],
                }
            )
        return perspectives[:5]

    @classmethod
    def _software_agent_diagnostic_subquestions(cls, objective: str) -> list[str]:
        if not cls._looks_like_software_agent_diagnostic_objective(objective):
            return []
        questions = [
            "Which planning or query-distillation steps are causing low-signal retrieval?",
            "Which browser, sandbox, or pc-control routing decisions are limiting active research?",
            "Where do retrieval, ranking, or synthesis stages discard high-signal evidence?",
            "Which benchmark or competitor capabilities reveal the largest architectural gaps?",
        ]
        if "10k" in objective.lower() or "1000" in objective.lower():
            questions.append(
                "Which crawl, frontier, or queue limits prevent broad multi-thousand URL coverage?"
            )
        return questions[:5]

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
                    resolved = depth
                    break
            else:
                resolved = ""
            if resolved:
                lower_objective = objective.lower()
                if resolved == "quick" and cls._looks_like_market_query(
                    lower_objective
                ):
                    if any(
                        cue in lower_objective
                        for cue in (
                            "valuation",
                            "undervalued",
                            "wall street",
                            "ticker",
                            "catalyst",
                            "risk",
                        )
                    ):
                        return "multi-hour"
                    return "standard"
                return resolved
        except Exception:
            pass

        # Minimal heuristic fallback if AI is unavailable.
        lower = objective.lower()
        if cls._looks_like_market_query(lower):
            if any(
                cue in lower
                for cue in (
                    "valuation",
                    "undervalued",
                    "wall street",
                    "ticker",
                    "catalyst",
                    "risk",
                    "current evidence",
                )
            ):
                return "multi-hour"
            return "standard"
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
            # Wall-Street / Claude-style deep research: cast a wide net.
            # High frontier budget for institutional-grade deep research.
            return ResearchSettings(
                depth="multi-hour",
                max_sources=1200,
                per_provider=120,
                max_query_variants=90,
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
            # For current-web multi-hour (market research, breaking news analysis),
            # target 1000+ URL fetches like a Claude/Gemini deep research session.
            return ResearchSettings(
                depth=settings.depth,
                max_sources=max(settings.max_sources, 1200),
                per_provider=max(settings.per_provider, 120),
                max_query_variants=max(settings.max_query_variants, 90),
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

    @classmethod
    def _settings_for_general_complex_objective(
        cls,
        settings: ResearchSettings,
        objective: str,
    ) -> ResearchSettings:
        if settings.depth == "multi-hour":
            return settings
        if cls._looks_like_academic_query(objective):
            return settings
        lower = objective.lower()
        if not (
            cls._looks_like_software_agent_query(objective)
            or cls._looks_like_comprehensive_research(lower)
        ):
            return settings
        if settings.depth == "standard":
            return ResearchSettings(
                depth=settings.depth,
                max_sources=max(settings.max_sources, 72),
                per_provider=max(settings.per_provider, 24),
                max_query_variants=max(settings.max_query_variants, 24),
            )
        return ResearchSettings(
            depth=settings.depth,
            max_sources=max(settings.max_sources, 18),
            per_provider=max(settings.per_provider, 8),
            max_query_variants=max(settings.max_query_variants, 6),
        )

    @staticmethod
    def _current_web_targets(depth: str) -> dict[str, int | float]:
        if depth == "multi-hour":
            # 120 passes * parallel queries * enrichment = 1000+ real URL fetches.
            return {
                "max_retrieval_passes": 120,
                "depth_pass_floor": 20,
                "max_low_novelty_streak": 12,
                "min_unique_urls": 1000,
                "min_perspective_count": 6,
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
        if depth == "multi-hour":
            source_floor = 12
        merged["min_source_count"] = min(
            int(merged.get("min_source_count") or source_floor),
            source_floor,
        )
        merged["min_source_count"] = max(int(merged["min_source_count"]), 1)
        # Current-web and market mode rely heavily on live web signals where
        # evidence grades can remain ungraded despite being useful. Enforce
        # source quality via on-topic/weak-ratio gates, not strong-grade counts.
        merged["min_strong_or_moderate"] = 0
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
        for keyword in sorted(cls._objective_anchor_terms(query)):
            if len(keyword) < 4:
                continue
            if keyword in anchors:
                continue
            anchors.append(keyword)
            if len(anchors) >= 8:
                break
        for keyword in cls._keywords(query):
            if len(keyword) < 4:
                continue
            if keyword in cls._generic_query_terms():
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
        """Generate search queries using AI.

        For market/current-evidence queries, behaves like a Wall Street analyst
        decomposing an investment thesis into 40-60 targeted searches:
        SEC filings, earnings, comps, catalysts, macro factors, short interest,
        analyst ratings, regulatory filings, technical setups, etc.

        For all other topics, generates academically rigorous sub-questions
        covering methodology, prior work, limitations, and counterevidence.

        Returns an empty list when AI is unavailable so callers fall back to
        the heuristic variant generator.
        """
        if depth == "quick":
            return []
        is_market = cls._looks_like_market_query(query)
        is_current = cls._looks_like_current_evidence_query(query)
        n_queries = 40 if depth == "multi-hour" else 12
        if is_market and (is_current or depth == "multi-hour"):
            system = (
                "You are a senior equity analyst at a top-tier Wall Street firm. "
                "You have been given a research objective and must decompose it into "
                "the exact search queries you would run to build a complete, "
                "publication-quality investment thesis. Think like a real analyst:\n\n"
                "WHAT WALL STREET ACTUALLY RESEARCHES:\n"
                "- Company fundamentals: earnings growth, revenue beats/misses, "
                "margin trajectory, free cash flow, debt/equity, ROE, ROIC\n"
                "- Valuation: P/E, P/S, EV/EBITDA vs sector comps, DCF assumptions, "
                "implied upside to consensus price targets\n"
                "- Catalysts: upcoming earnings dates, product launches, FDA decisions, "
                "contract wins, regulatory approvals, activist events, M&A rumors\n"
                "- Institutional positioning: 13-F filings, insider buying/selling, "
                "short interest %, days-to-cover, options flow\n"
                "- Macro factors: sector rotation, rate sensitivity, FX exposure, "
                "commodity inputs, tariff risk, geopolitical headwinds\n"
                "- Competition: market share trends, moat analysis, competitive threats, "
                "pricing power, customer retention\n"
                "- Bear case: what could go wrong, short thesis, regulatory risk, "
                "management execution risk, leverage concerns\n"
                "- Industry data: TAM growth, channel checks, supply chain dynamics, "
                "inventory levels, demand signals\n"
                "- Technical setup: 52-week range, RSI, moving averages, relative "
                "strength vs index, support/resistance levels\n\n"
                f"Generate exactly {n_queries} specific, targeted search queries "
                "that together would give a complete picture for an investment decision. "
                "Each query must be concrete — include company names, ticker symbols, "
                "specific metrics, and time horizons. Respond ONLY with a JSON array "
                "of strings. No prose, no explanations."
            )
        else:
            system = (
                "You are a senior research scientist. Decompose the research objective "
                "into targeted search queries that together would give a complete, "
                "systematic understanding of the topic. Think like a rigorous scientist:\n\n"
                "WHAT RIGOROUS RESEARCH COVERS:\n"
                "- Primary evidence and mechanism: what actually causes the effect\n"
                "- Methodology: how was it measured, what instruments, what controls\n"
                "- Replication: have independent groups confirmed this\n"
                "- Counterevidence: what contradicts the main finding\n"
                "- Limitations: scope conditions, confounders, publication bias\n"
                "- Recency: what has changed since the original work\n"
                "- Applications: how is it used in practice, what are edge cases\n"
                "- Experts: who are the leading voices, what are their positions\n\n"
                f"Generate exactly {n_queries} specific, targeted search queries. "
                "Each must be concrete and grounded in the actual objective terms. "
                "Respond ONLY with a JSON array of strings. No prose."
            )
        user = f"Research objective: {query}\nDepth: {depth}"
        try:
            raw = cls()._call_ai_text(system, user)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list) and len(parsed) >= 3:
                    result: list[str] = []
                    for item in parsed:
                        candidate = str(item or "").strip()[:240]
                        if candidate and len(candidate) >= 6:
                            result.append(candidate)
                    if len(result) >= 3:
                        return result[:n_queries]
        except Exception:
            pass
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

        generic_noise_terms = {
            "generate",
            "analysis",
            "style",
            "explicit",
            "symbols",
            "confidence",
            "levels",
            "using",
            "evidence",
            "report",
            "please",
            "could",
            "would",
            "wall",
            "street",
            "based",
            "on",
            "and",
            "for",
            "from",
            "into",
            "with",
            "the",
            "a",
            "an",
            "to",
            "of",
        }
        normalized_tokens = [
            token
            for token in re.findall(r"\b[0-9a-z]+(?:-[0-9a-z]+)*\b", cleaned.lower())
            if token not in generic_noise_terms
        ]
        if DeepResearchEngine._looks_like_math_query(query):
            focus_tokens: list[str] = []
            for focus in DeepResearchEngine._math_focus_terms(query):
                for token in re.findall(
                    r"\b[0-9a-z]+(?:-[0-9a-z]+)*\b",
                    str(focus).lower(),
                ):
                    if token in generic_noise_terms or token in focus_tokens:
                        continue
                    focus_tokens.append(token)
            if focus_tokens:
                normalized_tokens = focus_tokens + [
                    token for token in normalized_tokens if token not in focus_tokens
                ]
        if len(normalized_tokens) >= 4:
            cleaned = " ".join(normalized_tokens[:18])

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

    @staticmethod
    def _select_balanced_top(
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
            for key in DeepResearchEngine._source_identity_keys(source)
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
            if not DeepResearchEngine._source_is_on_topic(source, query):
                continue
            source_keys = DeepResearchEngine._source_identity_keys(source)
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
                or DeepResearchEngine._source_is_on_topic(source, query)
            )
        ]
        if not capped:
            return []
        by_provider: dict[str, list[ResearchSource]] = {}
        for source in capped:
            by_provider.setdefault(source.provider, []).append(source)
        if DeepResearchEngine._looks_like_market_query(query):
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
        if DeepResearchEngine._looks_like_software_agent_query(query):
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
                and DeepResearchEngine._source_is_on_topic(source, query)
            ),
        )

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
        raw_query = query or ""
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
        generic = DeepResearchEngine._generic_query_terms()
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
        return {
            "about",
            "across",
            "after",
            "also",
            "all",
            "analyst",
            "analysis",
            "analyze",
            "and",
            "available",
            "best",
            "brief",
            "candidate",
            "candidates",
            "chief",
            "companies",
            "company",
            "current",
            "data",
            "deep",
            "deepest",
            "direct",
            "evidence",
            "expert",
            "current",
            "for",
            "from",
            "gathering",
            "general",
            "highest",
            "latest",
            "live",
            "major",
            "near-term",
            "now",
            "opportunities",
            "opportunity",
            "potential",
            "primary",
            "proper",
            "public",
            "produce",
            "report",
            "reports",
            "research",
            "risk",
            "risks",
            "rigorous",
            "review",
            "scenario",
            "scenarios",
            "signal",
            "signals",
            "soar",
            "source",
            "sources",
            "specialised",
            "specialized",
            "that",
            "the",
            "their",
            "these",
            "this",
            "through",
            "timeline",
            "today",
            "tool",
            "tools",
            "uncertainties",
            "uncertainty",
            "using",
            "web",
            "website",
            "websites",
            "what",
            "with",
        }

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
