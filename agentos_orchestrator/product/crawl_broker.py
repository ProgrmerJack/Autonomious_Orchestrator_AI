from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentos_orchestrator.research import DeepResearchEngine, ResearchSource


@dataclass(slots=True)
class CrawlBrokerRecord:
    status: str
    url: str
    host: str
    port: int
    workspace_root: str
    queue_db_path: str
    auth_enabled: bool
    backend: str = "sqlite-sharded"
    shard_count: int = 1
    detail: str = ""


class _ShardedCrawlBrokerState:
    def __init__(
        self,
        workspace_root: Path,
        queue_db_path: Path,
        shard_count: int,
    ) -> None:
        self.workspace_root = workspace_root
        self.queue_db_path = queue_db_path
        self.shard_count = max(1, int(shard_count))
        self.backend = "sqlite-sharded" if self.shard_count > 1 else "sqlite"
        self._claim_lock = threading.Lock()
        self._domain_lease_until: dict[str, float] = {}
        self._shard_paths = [
            self._path_for_shard(index) for index in range(self.shard_count)
        ]
        self.engines = [self._new_engine(path) for path in self._shard_paths]
        self.primary_engine = self.engines[0]

    def _new_engine(self, research_state_path: Path) -> DeepResearchEngine:
        engine = DeepResearchEngine(
            workspace_root=self.workspace_root,
            research_state_path=research_state_path,
            crawl_broker_url="",
            crawl_broker_token="",
        )
        engine._crawl_worker_auto_started = True
        return engine

    def _path_for_shard(self, index: int) -> Path:
        if self.shard_count <= 1 or index == 0:
            return self.queue_db_path
        suffix = self.queue_db_path.suffix or ".sqlite3"
        stem = (
            self.queue_db_path.name[: -len(suffix)]
            if self.queue_db_path.suffix
            else self.queue_db_path.name
        )
        return self.queue_db_path.with_name(f"{stem}.shard{index:02d}{suffix}")

    @staticmethod
    def _unique_texts(items: list[str], limit: int | None = None) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
            if limit is not None and len(ordered) >= limit:
                break
        return ordered

    def _shard_index_for_key(self, key: str) -> int:
        normalized = str(key or "").strip()
        if self.shard_count <= 1 or not normalized:
            return 0
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % self.shard_count

    def _engine_for_url(self, url: str) -> tuple[int, DeepResearchEngine]:
        shard_index = self._shard_index_for_key(url)
        return shard_index, self.engines[shard_index]

    def enqueue_url_batch(
        self,
        urls: list[str],
        source_query: str,
        run_id: str,
        *,
        source_url: str,
        priority: float,
    ) -> int:
        safe_urls = self.primary_engine._persistent_unique_urls(urls)
        if not safe_urls:
            return 0
        grouped: dict[int, list[str]] = {}
        for url in safe_urls:
            shard_index, _ = self._engine_for_url(url)
            grouped.setdefault(shard_index, []).append(url)
        for shard_index, shard_urls in grouped.items():
            self.engines[shard_index]._enqueue_url_batch(
                shard_urls,
                source_query,
                run_id,
                source_url=source_url,
                priority=priority,
            )
        return len(safe_urls)

    def requeue_stale_crawl_claims(self, stale_after_seconds: int) -> int:
        return sum(
            engine._requeue_stale_crawl_claims(stale_after_seconds)
            for engine in self.engines
        )

    def _prune_domain_leases(self, now: float) -> None:
        expired = [
            domain
            for domain, lease_until in self._domain_lease_until.items()
            if lease_until <= now
        ]
        for domain in expired:
            self._domain_lease_until.pop(domain, None)

    @staticmethod
    def _max_timestamp(*values: str) -> str:
        timestamps = [
            str(value or "").strip() for value in values if str(value or "").strip()
        ]
        if not timestamps:
            return ""
        return max(timestamps)

    @staticmethod
    def _lease_payload(
        domain: str,
        lease_until: float,
        now: float,
    ) -> dict[str, Any]:
        remaining_seconds = max(0.0, float(lease_until) - now)
        return {
            "domain": domain,
            "lease_until": datetime.fromtimestamp(
                float(lease_until),
                UTC,
            ).isoformat(),
            "seconds_remaining": round(remaining_seconds, 3),
        }

    def _active_domain_leases(
        self,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        lease_now = float(now if now is not None else time.time())
        self._prune_domain_leases(lease_now)
        leases = [
            self._lease_payload(domain, lease_until, lease_now)
            for domain, lease_until in sorted(self._domain_lease_until.items())
            if float(lease_until) > lease_now
        ]
        leases.sort(
            key=lambda item: (
                float(item.get("seconds_remaining") or 0.0),
                str(item.get("domain") or ""),
            ),
            reverse=True,
        )
        return leases

    def _shard_metrics(self, shard_index: int) -> dict[str, Any]:
        engine = self.engines[shard_index]
        engine._ensure_research_state_store()
        status_counts: dict[str, int] = {}
        js_status_counts: dict[str, int] = {}
        worker_stats: dict[str, dict[str, Any]] = {}
        with closing(engine._connect_research_state()) as connection:
            queue_summary = connection.execute(
                """
                SELECT status,
                       COUNT(*) AS row_count,
                       COALESCE(SUM(js_required), 0) AS js_required_count
                FROM crawl_queue
                GROUP BY status
                """
            ).fetchall()
            for row in queue_summary:
                status = str(row["status"] or "").strip() or "unknown"
                status_counts[status] = int(row["row_count"] or 0)
                js_status_counts[status] = int(row["js_required_count"] or 0)

            totals_row = connection.execute(
                """
                SELECT COUNT(*) AS queue_total,
                       COALESCE(SUM(js_required), 0) AS js_total,
                       COUNT(DISTINCT domain) AS domain_total
                FROM crawl_queue
                """
            ).fetchone()
            observation_row = connection.execute(
                """
                SELECT COUNT(*) AS observation_total,
                       COALESCE(SUM(used_browser), 0) AS browser_total
                FROM crawl_observations
                """
            ).fetchone()
            active_claim_rows = connection.execute(
                """
                SELECT last_claimed_by AS worker_id,
                       COUNT(*) AS active_claims,
                       COALESCE(SUM(js_required), 0) AS active_js_claims,
                       MAX(last_claimed_at) AS last_claimed_at
                FROM crawl_queue
                WHERE status = 'claimed' AND last_claimed_by != ''
                GROUP BY last_claimed_by
                ORDER BY active_claims DESC,
                         active_js_claims DESC,
                         worker_id ASC
                """
            ).fetchall()
            for row in active_claim_rows:
                worker_id = str(row["worker_id"] or "").strip()
                if not worker_id:
                    continue
                worker_stats[worker_id] = {
                    "worker_id": worker_id,
                    "active_claims": int(row["active_claims"] or 0),
                    "active_js_claims": int(row["active_js_claims"] or 0),
                    "observation_count": 0,
                    "browser_observations": 0,
                    "last_claimed_at": str(row["last_claimed_at"] or ""),
                    "last_observed_at": "",
                }

            observation_rows = connection.execute(
                """
                SELECT worker_id,
                       COUNT(*) AS observation_count,
                       COALESCE(SUM(used_browser), 0) AS browser_observations,
                       MAX(updated_at) AS last_observed_at
                FROM crawl_observations
                WHERE worker_id != ''
                GROUP BY worker_id
                ORDER BY observation_count DESC,
                         browser_observations DESC,
                         worker_id ASC
                """
            ).fetchall()
            for row in observation_rows:
                worker_id = str(row["worker_id"] or "").strip()
                if not worker_id:
                    continue
                entry = worker_stats.setdefault(
                    worker_id,
                    {
                        "worker_id": worker_id,
                        "active_claims": 0,
                        "active_js_claims": 0,
                        "observation_count": 0,
                        "browser_observations": 0,
                        "last_claimed_at": "",
                        "last_observed_at": "",
                    },
                )
                entry["observation_count"] = int(row["observation_count"] or 0)
                entry["browser_observations"] = int(row["browser_observations"] or 0)
                entry["last_observed_at"] = str(row["last_observed_at"] or "")

        worker_items = sorted(
            worker_stats.values(),
            key=lambda item: (
                int(item.get("active_js_claims") or 0),
                int(item.get("active_claims") or 0),
                int(item.get("browser_observations") or 0),
                int(item.get("observation_count") or 0),
                str(item.get("worker_id") or ""),
            ),
            reverse=True,
        )
        return {
            "shard_index": shard_index,
            "queue_db_path": str(self._shard_paths[shard_index]),
            "queue_total": (int(totals_row["queue_total"] or 0) if totals_row else 0),
            "domain_total": (int(totals_row["domain_total"] or 0) if totals_row else 0),
            "status_counts": status_counts,
            "js_required_counts": {
                "total": int(totals_row["js_total"] or 0) if totals_row else 0,
                **js_status_counts,
            },
            "active_claims": int(status_counts.get("claimed", 0)),
            "observation_total": int(observation_row["observation_total"] or 0)
            if observation_row
            else 0,
            "browser_observation_total": int(observation_row["browser_total"] or 0)
            if observation_row
            else 0,
            "workers": worker_items,
        }

    def broker_metrics(self) -> dict[str, Any]:
        now = time.time()
        active_domain_leases = self._active_domain_leases(now)
        shard_metrics = [
            self._shard_metrics(index) for index in range(self.shard_count)
        ]
        status_counts: dict[str, int] = {}
        js_required_counts: dict[str, int] = {"total": 0}
        worker_utilization: dict[str, dict[str, Any]] = {}
        queue_total = 0
        domain_total = 0
        observation_total = 0
        browser_observation_total = 0
        for shard in shard_metrics:
            queue_total += int(shard.get("queue_total") or 0)
            domain_total += int(shard.get("domain_total") or 0)
            observation_total += int(shard.get("observation_total") or 0)
            browser_observation_total += int(
                shard.get("browser_observation_total") or 0
            )
            for status, count in dict(shard.get("status_counts") or {}).items():
                status_counts[status] = status_counts.get(status, 0) + int(count or 0)
            for status, count in dict(shard.get("js_required_counts") or {}).items():
                js_required_counts[status] = js_required_counts.get(
                    status,
                    0,
                ) + int(count or 0)
            for worker in list(shard.get("workers") or []):
                if not isinstance(worker, dict):
                    continue
                worker_id = str(worker.get("worker_id") or "").strip()
                if not worker_id:
                    continue
                entry = worker_utilization.setdefault(
                    worker_id,
                    {
                        "worker_id": worker_id,
                        "active_claims": 0,
                        "active_js_claims": 0,
                        "observation_count": 0,
                        "browser_observations": 0,
                        "last_claimed_at": "",
                        "last_observed_at": "",
                        "shards": [],
                    },
                )
                entry["active_claims"] += int(worker.get("active_claims") or 0)
                entry["active_js_claims"] += int(worker.get("active_js_claims") or 0)
                entry["observation_count"] += int(worker.get("observation_count") or 0)
                entry["browser_observations"] += int(
                    worker.get("browser_observations") or 0
                )
                entry["last_claimed_at"] = self._max_timestamp(
                    str(entry.get("last_claimed_at") or ""),
                    str(worker.get("last_claimed_at") or ""),
                )
                entry["last_observed_at"] = self._max_timestamp(
                    str(entry.get("last_observed_at") or ""),
                    str(worker.get("last_observed_at") or ""),
                )
                shard_list = entry.setdefault("shards", [])
                shard_value = int(shard.get("shard_index") or 0)
                if shard_value not in shard_list:
                    shard_list.append(shard_value)

        worker_items = sorted(
            worker_utilization.values(),
            key=lambda item: (
                int(item.get("active_js_claims") or 0),
                int(item.get("active_claims") or 0),
                int(item.get("browser_observations") or 0),
                int(item.get("observation_count") or 0),
                str(item.get("worker_id") or ""),
            ),
            reverse=True,
        )
        return {
            "backend": self.backend,
            "shard_count": self.shard_count,
            "generated_at": datetime.now(UTC).isoformat(),
            "queue": {
                "total": queue_total,
                "domain_total": domain_total,
                "status_counts": status_counts,
                "js_required_counts": js_required_counts,
            },
            "observations": {
                "total": observation_total,
                "browser_total": browser_observation_total,
            },
            "domain_leases": {
                "active_count": len(active_domain_leases),
                "items": active_domain_leases,
            },
            "worker_utilization": {
                "active_worker_count": len(worker_items),
                "items": worker_items,
            },
            "shards": shard_metrics,
        }

    def queue_inspect(
        self,
        *,
        limit: int = 24,
        statuses: list[str] | None = None,
        domain: str = "",
        worker_id: str = "",
        js_required: bool | None = None,
        shard_index: int | None = None,
    ) -> dict[str, Any]:
        requested_limit = max(1, int(limit))
        normalized_statuses = [
            str(status or "").strip()
            for status in (statuses or [])
            if str(status or "").strip()
        ]
        normalized_domain = str(domain or "").strip().lower()
        normalized_worker_id = str(worker_id or "").strip()
        shard_indexes: tuple[int, ...] = tuple(range(self.shard_count))
        if shard_index is not None:
            shard_value = max(0, min(int(shard_index), self.shard_count - 1))
            shard_indexes = (shard_value,)

        items: list[dict[str, Any]] = []
        total_matches = 0
        per_shard_matches: dict[int, int] = {}
        for current_shard in shard_indexes:
            engine = self.engines[current_shard]
            engine._ensure_research_state_store()
            where_clauses: list[str] = []
            params: list[Any] = []
            if normalized_statuses:
                placeholders = ", ".join("?" for _ in normalized_statuses)
                where_clauses.append(f"status IN ({placeholders})")
                params.extend(normalized_statuses)
            if normalized_domain:
                where_clauses.append("LOWER(domain) = ?")
                params.append(normalized_domain)
            if normalized_worker_id:
                where_clauses.append("last_claimed_by = ?")
                params.append(normalized_worker_id)
            if js_required is not None:
                where_clauses.append("js_required = ?")
                params.append(1 if js_required else 0)
            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            with closing(engine._connect_research_state()) as connection:
                match_row = connection.execute(
                    (f"SELECT COUNT(*) AS row_count FROM crawl_queue {where_sql}"),
                    tuple(params),
                ).fetchone()
                shard_match_total = int(match_row["row_count"] or 0) if match_row else 0
                total_matches += shard_match_total
                per_shard_matches[current_shard] = shard_match_total
                query_params = list(params)
                query_params.append(max(requested_limit, 1))
                rows = connection.execute(
                    f"""
                    SELECT url,
                           domain,
                           priority,
                           status,
                           source_query,
                           source_url,
                           run_id,
                           js_required,
                           attempts,
                           last_claimed_by,
                           last_claimed_at,
                           last_error,
                           created_at,
                           updated_at
                    FROM crawl_queue
                    {where_sql}
                    ORDER BY priority DESC, js_required DESC, updated_at DESC
                    LIMIT ?
                    """,
                    tuple(query_params),
                ).fetchall()
            for row in rows:
                items.append(
                    {
                        "shard_index": current_shard,
                        "url": str(row["url"] or ""),
                        "domain": str(row["domain"] or ""),
                        "priority": float(row["priority"] or 0.0),
                        "status": str(row["status"] or ""),
                        "source_query": str(row["source_query"] or ""),
                        "source_url": str(row["source_url"] or ""),
                        "run_id": str(row["run_id"] or ""),
                        "js_required": bool(int(row["js_required"] or 0)),
                        "attempts": int(row["attempts"] or 0),
                        "last_claimed_by": str(row["last_claimed_by"] or ""),
                        "last_claimed_at": str(row["last_claimed_at"] or ""),
                        "last_error": str(row["last_error"] or ""),
                        "created_at": str(row["created_at"] or ""),
                        "updated_at": str(row["updated_at"] or ""),
                    }
                )

        items.sort(
            key=lambda item: (
                float(item.get("priority") or 0.0),
                int(bool(item.get("js_required"))),
                str(item.get("updated_at") or ""),
            ),
            reverse=True,
        )
        return {
            "filters": {
                "statuses": normalized_statuses,
                "domain": normalized_domain,
                "worker_id": normalized_worker_id,
                "js_required": js_required,
                "shard_index": shard_index,
            },
            "total_matches": total_matches,
            "returned": min(len(items), requested_limit),
            "per_shard_matches": per_shard_matches,
            "items": items[:requested_limit],
        }

    def _claim_rows_by_shard(
        self,
        rows: list[dict[str, Any]],
        worker_id: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[int, list[str]] = {}
        ordered_urls: list[str] = []
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            ordered_urls.append(url)
            grouped.setdefault(int(row.get("_shard_index") or 0), []).append(url)
        claimed_by_url: dict[str, dict[str, Any]] = {}
        for shard_index, shard_urls in grouped.items():
            for claimed in self.engines[shard_index]._claim_specific_crawl_urls(
                shard_urls,
                worker_id,
            ):
                claimed_by_url[str(claimed.get("url") or "").strip()] = dict(claimed)
        return [claimed_by_url[url] for url in ordered_urls if url in claimed_by_url]

    def claim_crawl_queue_batch(
        self,
        limit: int,
        worker_id: str,
        *,
        allow_js_required: bool,
        prefer_js_required: bool,
        max_claims_per_domain: int,
        default_domain_cooldown_seconds: float,
        js_domain_cooldown_seconds: float,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._claim_lock:
            now = time.time()
            self._prune_domain_leases(now)
            preview_limit = max(limit * 6, 24)
            candidates: list[dict[str, Any]] = []
            for shard_index, engine in enumerate(self.engines):
                for row in engine._peek_crawl_queue_rows(preview_limit):
                    candidate = dict(row)
                    candidate["_shard_index"] = shard_index
                    candidates.append(candidate)

            def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
                priority = float(row.get("priority") or 0.0)
                js_required = int(row.get("js_required") or 0)
                updated_at = str(row.get("updated_at") or "")
                if prefer_js_required:
                    return (js_required, priority, updated_at)
                return (priority, js_required, updated_at)

            selected: list[dict[str, Any]] = []
            domain_counts: dict[str, int] = {}
            for row in sorted(candidates, key=_sort_key, reverse=True):
                url = str(row.get("url") or "").strip()
                if not url:
                    continue
                js_required = bool(int(row.get("js_required") or 0))
                if js_required and not allow_js_required:
                    continue
                domain = str(
                    row.get("domain") or ""
                ).strip() or self.primary_engine._source_domain(url)
                if domain and self._domain_lease_until.get(domain, 0.0) > now:
                    continue
                if domain and domain_counts.get(domain, 0) >= max(
                    1, int(max_claims_per_domain)
                ):
                    continue
                selected.append(row)
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if len(selected) >= limit:
                    break
            claimed = self._claim_rows_by_shard(selected, worker_id)
            for row in claimed:
                domain = str(
                    row.get("domain") or ""
                ).strip() or self.primary_engine._source_domain(
                    str(row.get("url") or "")
                )
                if not domain:
                    continue
                lease_seconds = (
                    float(js_domain_cooldown_seconds)
                    if bool(int(row.get("js_required") or 0))
                    else float(default_domain_cooldown_seconds)
                )
                if lease_seconds > 0:
                    self._domain_lease_until[domain] = now + lease_seconds
            return claimed

    def claim_persistent_crawl_sources(
        self,
        query: str,
        objective: str,
        limit: int,
        exclude_urls: list[str] | None = None,
    ) -> list[ResearchSource]:
        if limit <= 0:
            return []
        excluded = set(self.primary_engine._persistent_unique_urls(exclude_urls or []))
        worker_seed = f"broker:{query[:64]}:{time.monotonic_ns()}"
        worker_id = hashlib.sha1(worker_seed.encode("utf-8")).hexdigest()[:12]
        with self._claim_lock:
            preview_limit = max(limit * 8, 24)
            candidates: list[dict[str, Any]] = []
            for shard_index, engine in enumerate(self.engines):
                preview_rows = engine._peek_crawl_queue_rows(
                    preview_limit,
                    statuses=("queued",),
                    exclude_urls=list(excluded),
                )
                for row in preview_rows:
                    url = str(row.get("url") or "").strip()
                    if not url or url in excluded:
                        continue
                    label = " ".join(
                        [
                            url,
                            str(row.get("source_query") or ""),
                            str(row.get("source_url") or ""),
                        ]
                    )
                    match_score = engine._persistent_match_score(
                        label, query, objective
                    )
                    if match_score < 0.2 and str(row.get("source_query") or "").strip():
                        continue
                    candidate = dict(row)
                    candidate["_shard_index"] = shard_index
                    candidate["_match_score"] = match_score
                    candidates.append(candidate)

            def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
                score = max(
                    float(row.get("priority") or 0.0),
                    4.0 + (float(row.get("_match_score") or 0.0) * 20.0),
                )
                return (
                    score,
                    int(row.get("js_required") or 0),
                    str(row.get("updated_at") or ""),
                )

            selected: list[dict[str, Any]] = []
            seen_urls: set[str] = set(excluded)
            for row in sorted(candidates, key=_sort_key, reverse=True):
                url = str(row.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                selected.append(row)
                if len(selected) >= limit:
                    break
            claimed = self._claim_rows_by_shard(selected, worker_id)
        selected_by_url = {str(row.get("url") or "").strip(): row for row in selected}
        sources: list[ResearchSource] = []
        for row in claimed:
            url = str(row.get("url") or "").strip()
            selected_row = selected_by_url.get(url, row)
            domain = str(row.get("domain") or "").strip()
            source_query = str(row.get("source_query") or "").strip()
            quality_flags = ["persistent-crawl-queue"]
            if int(row.get("js_required") or 0) == 1:
                quality_flags.append("js-render-required")
            abstract = (
                source_query
                or str(row.get("source_url") or "").strip()
                or self.primary_engine._label_from_url(url)
            )
            match_score = float(selected_row.get("_match_score") or 0.0)
            sources.append(
                ResearchSource(
                    provider="persistent-crawl-queue",
                    title=self.primary_engine._label_from_url(url),
                    url=url,
                    authors=[domain] if domain else [],
                    abstract=f"Persistent crawl candidate: {abstract}"[:320],
                    citation_count=0,
                    score=max(
                        float(row.get("priority") or 0.0), 4.0 + (match_score * 20.0)
                    ),
                    quality_flags=quality_flags,
                )
            )
        return sources

    def update_crawl_queue_status(self, url: str, status: str, error: str = "") -> None:
        _, engine = self._engine_for_url(url)
        engine._update_crawl_queue_status(url, status, error)

    def persistent_crawl_queue_snapshot(self, limit: int = 24) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for engine in self.engines:
            rows.extend(
                list(
                    engine._persistent_crawl_queue_snapshot(limit=limit).get("queued")
                    or []
                )
            )
        rows.sort(
            key=lambda item: (
                float(item.get("priority") or 0.0),
                int(bool(item.get("js_required"))),
                str(item.get("status") or ""),
            ),
            reverse=True,
        )
        return {"queued": rows[:limit]}

    def record_crawl_observation(
        self,
        source: ResearchSource,
        content: str,
        source_query: str,
        query_hints: list[str],
        outbound_urls: list[str],
        worker_id: str,
        used_browser: bool,
    ) -> None:
        _, engine = self._engine_for_url(source.url)
        engine._record_crawl_observation(
            source,
            content,
            source_query,
            query_hints,
            outbound_urls,
            worker_id,
            used_browser,
        )

    def persistent_evidence_query_hints(
        self,
        query: str,
        objective: str,
        limit: int,
    ) -> list[str]:
        hints: list[str] = []
        for engine in self.engines:
            hints.extend(
                engine._persistent_evidence_query_hints(query, objective, limit=limit)
            )
        return self._unique_texts(hints, limit=max(limit, 0))

    def persistent_seed_urls(
        self,
        query: str,
        objective: str,
        limit: int,
    ) -> list[str]:
        seed_urls: list[str] = []
        for engine in self.engines:
            seed_urls.extend(
                engine._persistent_seed_urls(query, objective, limit=limit)
            )
        return self._unique_texts(seed_urls, limit=max(limit, 0))

    def update_persistent_evidence_index(
        self,
        run_id: str,
        objective: str,
        query: str,
        claim_trace: dict[str, Any],
    ) -> None:
        claims = [
            claim
            for claim in list(claim_trace.get("claims") or [])
            if isinstance(claim, dict)
        ]
        if not claims:
            return
        grouped: dict[int, list[dict[str, Any]]] = {}
        for claim in claims:
            claim_key = self.primary_engine._normalize_title(
                str(claim.get("claim") or "")
            )
            if not claim_key:
                continue
            shard_index = self._shard_index_for_key(claim_key)
            grouped.setdefault(shard_index, []).append(claim)
        for shard_index, shard_claims in grouped.items():
            shard_trace = dict(claim_trace)
            shard_trace["claims"] = shard_claims
            self.engines[shard_index]._update_persistent_evidence_index(
                run_id,
                objective,
                query,
                [],
                shard_trace,
            )

    def persistent_evidence_snapshot(
        self,
        query: str,
        objective: str,
        limit: int,
    ) -> dict[str, Any]:
        claims: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []
        domains: list[str] = []
        for engine in self.engines:
            snapshot = engine._persistent_evidence_snapshot(
                query, objective, limit=limit
            )
            claims.extend(
                dict(item)
                for item in list(snapshot.get("claims") or [])
                if isinstance(item, dict)
            )
            contradictions.extend(
                dict(item)
                for item in list(snapshot.get("contradictions") or [])
                if isinstance(item, dict)
            )
            domains.extend(str(item) for item in list(snapshot.get("domains") or []))
        claims.sort(
            key=lambda item: (
                int(item.get("support_count") or 0),
                -int(item.get("contradiction_count") or 0),
                int(item.get("confidence_rank") or 0),
            ),
            reverse=True,
        )
        contradictions.sort(
            key=lambda item: (
                int(item.get("count") or 0),
                len(list(item.get("claim_keys") or [])),
            ),
            reverse=True,
        )
        return {
            "query": query,
            "objective": objective,
            "seed_urls": self.persistent_seed_urls(query, objective, limit=limit),
            "claims": claims[:limit],
            "contradictions": contradictions[:limit],
            "domains": self._unique_texts(domains, limit=max(limit, 12)),
        }


class CrawlBrokerServer:
    def __init__(
        self,
        workspace_root: str | Path = ".",
        host: str = "127.0.0.1",
        port: int = 0,
        queue_db_path: str | Path | None = None,
        auth_token: str = "",
        shard_count: int = 4,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.host = str(host).strip() or "127.0.0.1"
        self.port = max(0, int(port))
        if queue_db_path is None:
            self.queue_db_path = (
                self.workspace_root / ".agentos" / "research_state.sqlite3"
            )
        else:
            self.queue_db_path = Path(queue_db_path).resolve()
        self.auth_token = str(auth_token or "")
        self.store = _ShardedCrawlBrokerState(
            self.workspace_root,
            self.queue_db_path,
            shard_count=shard_count,
        )
        self.engine = self.store.primary_engine
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> CrawlBrokerRecord:
        if self._server is not None:
            return self.status(detail="already-running")
        server = ThreadingHTTPServer((self.host, self.port), self._handler_type())
        server.daemon_threads = True
        server.crawl_broker = self
        self._server = server
        self.host = str(server.server_address[0])
        self.port = int(server.server_address[1])
        return self.status(detail="running")

    def start_in_thread(self) -> CrawlBrokerRecord:
        record = self.start()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self.serve_forever,
                name="agentos-crawl-broker",
                daemon=True,
            )
            self._thread.start()
        return record

    def serve_forever(self) -> None:
        if self._server is None:
            self.start()
        assert self._server is not None
        self._server.serve_forever()

    def shutdown(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def status(self, detail: str = "") -> CrawlBrokerRecord:
        running = self._server is not None
        return CrawlBrokerRecord(
            status="running" if running else "stopped",
            url=self.url,
            host=self.host,
            port=self.port,
            workspace_root=str(self.workspace_root),
            queue_db_path=str(self.queue_db_path),
            auth_enabled=bool(self.auth_token),
            backend=self.store.backend,
            shard_count=self.store.shard_count,
            detail=detail,
        )

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        broker = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AgentOSCrawlBroker/1.0"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                del format, args

            def _send_json(
                self,
                payload: dict[str, Any],
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json_body(self) -> dict[str, Any]:
                raw_length = self.headers.get("Content-Length", "0").strip() or "0"
                try:
                    length = max(0, int(raw_length))
                except ValueError:
                    raise ValueError("invalid content length")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                if not raw.strip():
                    return {}
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("request payload must be a JSON object")
                return payload

            def _authorized(self) -> bool:
                if not broker.auth_token:
                    return True
                header = str(self.headers.get("Authorization") or "").strip()
                if header == f"Bearer {broker.auth_token}":
                    return True
                self._send_json(
                    {"error": "unauthorized"},
                    status=HTTPStatus.UNAUTHORIZED,
                )
                return False

            def do_GET(self) -> None:  # noqa: N802
                if not self._authorized():
                    return
                if self.path == "/status":
                    self._send_json(asdict(broker.status(detail="running")))
                    return
                if self.path == "/metrics":
                    self._send_json(broker.store.broker_metrics())
                    return
                self._send_json(
                    {"error": f"unknown path: {self.path}"},
                    status=HTTPStatus.NOT_FOUND,
                )

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    return
                try:
                    payload = self._json_body()
                    response = broker._dispatch(self.path, payload)
                except ValueError as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                except Exception as exc:  # pragma: no cover - defensive server path
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return
                self._send_json(response)

        return Handler

    def _dispatch(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/queue/enqueue":
            urls = [str(url).strip() for url in (payload.get("urls") or [])]
            queued = self.store.enqueue_url_batch(
                urls,
                str(payload.get("source_query") or ""),
                str(payload.get("run_id") or ""),
                source_url=str(payload.get("source_url") or ""),
                priority=float(payload.get("priority") or 0.0),
            )
            return {"queued_count": queued}
        if path == "/queue/requeue-stale":
            reclaimed = self.store.requeue_stale_crawl_claims(
                int(payload.get("stale_after_seconds") or 900)
            )
            return {"reclaimed_count": reclaimed}
        if path == "/queue/claim-batch":
            claimed = self.store.claim_crawl_queue_batch(
                int(payload.get("limit") or 0),
                str(payload.get("worker_id") or "crawl-worker"),
                allow_js_required=bool(payload.get("allow_js_required", True)),
                prefer_js_required=bool(payload.get("prefer_js_required", False)),
                max_claims_per_domain=max(
                    1, int(payload.get("max_claims_per_domain") or 2)
                ),
                default_domain_cooldown_seconds=float(
                    payload.get("default_domain_cooldown_seconds") or 0.0
                ),
                js_domain_cooldown_seconds=float(
                    payload.get("js_domain_cooldown_seconds") or 0.0
                ),
            )
            return {"claimed": claimed}
        if path == "/queue/claim-sources":
            sources = self.store.claim_persistent_crawl_sources(
                str(payload.get("query") or ""),
                str(payload.get("objective") or ""),
                int(payload.get("limit") or 0),
                exclude_urls=[
                    str(url).strip() for url in (payload.get("exclude_urls") or [])
                ],
            )
            return {"sources": [asdict(source) for source in sources]}
        if path == "/queue/update-status":
            self.store.update_crawl_queue_status(
                str(payload.get("url") or ""),
                str(payload.get("status") or ""),
                str(payload.get("error") or ""),
            )
            return {"ok": True}
        if path == "/queue/snapshot":
            return self.store.persistent_crawl_queue_snapshot(
                limit=int(payload.get("limit") or 24)
            )
        if path == "/queue/inspect":
            js_filter = payload.get("js_required")
            normalized_js_filter: bool | None
            if js_filter is None:
                normalized_js_filter = None
            else:
                normalized_js_filter = bool(js_filter)
            raw_shard_index = payload.get("shard_index")
            raw_limit = payload.get("limit", 24)
            normalized_shard_index = None
            if raw_shard_index not in (None, ""):
                normalized_shard_index = int(raw_shard_index)
            normalized_limit = 24 if raw_limit in (None, "") else int(raw_limit)
            return self.store.queue_inspect(
                limit=normalized_limit,
                statuses=[
                    str(status).strip()
                    for status in (payload.get("statuses") or [])
                    if str(status).strip()
                ],
                domain=str(payload.get("domain") or ""),
                worker_id=str(payload.get("worker_id") or ""),
                js_required=normalized_js_filter,
                shard_index=normalized_shard_index,
            )
        if path == "/observations/record":
            source = self._source_from_payload(payload.get("source") or {})
            self.store.record_crawl_observation(
                source,
                str(payload.get("content") or ""),
                str(payload.get("source_query") or ""),
                [
                    str(item).strip()
                    for item in (payload.get("query_hints") or [])
                    if str(item).strip()
                ],
                [
                    str(item).strip()
                    for item in (payload.get("outbound_urls") or [])
                    if str(item).strip()
                ],
                str(payload.get("worker_id") or "crawl-worker"),
                bool(payload.get("used_browser")),
            )
            return {"ok": True}
        if path == "/evidence/query-hints":
            return {
                "query_hints": self.store.persistent_evidence_query_hints(
                    str(payload.get("query") or ""),
                    str(payload.get("objective") or ""),
                    limit=int(payload.get("limit") or 16),
                )
            }
        if path == "/evidence/seed-urls":
            return {
                "seed_urls": self.store.persistent_seed_urls(
                    str(payload.get("query") or ""),
                    str(payload.get("objective") or ""),
                    limit=int(payload.get("limit") or 16),
                )
            }
        if path == "/evidence/update-index":
            self.store.update_persistent_evidence_index(
                str(payload.get("run_id") or ""),
                str(payload.get("objective") or ""),
                str(payload.get("query") or ""),
                dict(payload.get("claim_trace") or {}),
            )
            return {"ok": True}
        if path == "/evidence/snapshot":
            return self.store.persistent_evidence_snapshot(
                str(payload.get("query") or ""),
                str(payload.get("objective") or ""),
                limit=int(payload.get("limit") or 12),
            )
        raise ValueError(f"unknown path: {path}")

    @staticmethod
    def _source_from_payload(payload: dict[str, Any]) -> ResearchSource:
        if not isinstance(payload, dict):
            raise ValueError("source payload must be an object")
        fields = ResearchSource.__dataclass_fields__
        normalized = {key: payload.get(key) for key in fields if key in payload}
        return ResearchSource(**normalized)
