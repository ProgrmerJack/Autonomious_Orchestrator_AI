from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import ResearchSource


class ResearchCrawlStateMixin:
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

    def _queued_crawl_backlog_count(self) -> int | None:
        if self._crawl_broker_enabled():
            return None
        self._ensure_research_state_store()
        with closing(self._connect_research_state()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM crawl_queue WHERE status = 'queued'"
            ).fetchone()
        if not row:
            return 0
        return max(0, int(row[0] or 0))

    @staticmethod
    def _detached_crawl_auto_start_backlog_threshold() -> int:
        configured = os.environ.get("AGENTOS_CRAWL_AUTO_START_BACKLOG", "").strip()
        if configured:
            try:
                return max(0, min(int(configured), 512))
            except ValueError:
                return 24
        return 24

    def _detached_crawl_worker_count(self, queued_count: int | None = None) -> int:
        configured = os.environ.get("AGENTOS_CRAWL_WORKER_COUNT", "").strip()
        if configured:
            try:
                return max(0, min(int(configured), 16))
            except ValueError:
                return 0
        cpu_count = os.cpu_count() or 2
        if queued_count is None:
            return max(1, min(2, cpu_count // 6 or 1))
        if queued_count < 96:
            return 1
        if queued_count < 224:
            return max(1, min(2, cpu_count // 6 or 1))
        if queued_count < 448:
            return max(2, min(3, cpu_count // 4 or 1))
        return max(2, min(4, cpu_count // 3 or 1))

    @staticmethod
    def _detached_crawl_batch_size(queued_count: int | None = None) -> int:
        configured = os.environ.get("AGENTOS_CRAWL_WORKER_BATCH_SIZE", "").strip()
        if configured:
            try:
                return max(1, min(int(configured), 48))
            except ValueError:
                return 6
        if queued_count is None or queued_count < 96:
            return 6
        if queued_count < 224:
            return 8
        if queued_count < 448:
            return 12
        return 16

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
        queued_count = self._queued_crawl_backlog_count()
        if (
            queued_count is not None
            and queued_count < self._detached_crawl_auto_start_backlog_threshold()
        ):
            return
        worker_count = self._detached_crawl_worker_count(queued_count)
        if worker_count <= 0:
            return
        batch_size = self._detached_crawl_batch_size(queued_count)
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
                    service_manager.start(
                        task_name=service_status.task_name,
                        worker_count=worker_count,
                        batch_size=batch_size,
                        poll_interval_seconds=self._detached_crawl_poll_interval(),
                    )
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
                    batch_size=batch_size,
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

