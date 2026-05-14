from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .models import (
    ResearchBrief,
    ResearchSettings,
    ResearchSource,
    sanitize_evidence_claim_text as _sanitize_evidence_claim_text,
)


_PROVIDER_DIAGNOSTIC_RAM_CAP = max(
    32,
    int(os.environ.get("AGENTOS_PROVIDER_DIAGNOSTIC_RAM_CAP", "512") or 512),
)


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


class ResearchArtifactsMixin:
    def _record_provider_diagnostic(
        self,
        provider: str,
        status: str,
        detail: str = "",
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        compact_detail = detail[:500]
        if self.provider_diagnostics:
            last = self.provider_diagnostics[-1]
            if (
                last.get("provider") == provider
                and last.get("status") == status
                and last.get("detail") == compact_detail
            ):
                last["repeat_count"] = int(last.get("repeat_count") or 1) + 1
                last["created_at"] = created_at
                self._flush_live_provider_diagnostics()
                return
        self.provider_diagnostics.append(
            {
                "provider": provider,
                "status": status,
                "detail": compact_detail,
                "created_at": created_at,
            }
        )
        overflow = getattr(self, "_provider_diagnostic_overflow", 0)
        if len(self.provider_diagnostics) > _PROVIDER_DIAGNOSTIC_RAM_CAP:
            overflow += len(self.provider_diagnostics) - _PROVIDER_DIAGNOSTIC_RAM_CAP
            self.provider_diagnostics = self.provider_diagnostics[
                -_PROVIDER_DIAGNOSTIC_RAM_CAP:
            ]
            self._provider_diagnostic_overflow = overflow
        self._flush_live_provider_diagnostics()

    def _flush_live_provider_diagnostics(self) -> None:
        if not self._active_run_id:
            return
        artifact_dir = self.workspace_root / "runs" / self._active_run_id / "research"
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload = list(self.provider_diagnostics)
            overflow = int(getattr(self, "_provider_diagnostic_overflow", 0) or 0)
            if overflow > 0:
                payload = [
                    {
                        "provider": "provider-diagnostics",
                        "status": "compacted",
                        "detail": (
                            "Older provider diagnostics were compacted from in-memory "
                            f"retention to prevent runaway RAM growth. Dropped entries: {overflow}."
                        ),
                        "overflow_count": overflow,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    *payload,
                ]
            (artifact_dir / "provider_diagnostics.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

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
                "generative synthesis key configured"
                if gemini_present
                else "generative synthesis key not found in env/.env; AI synthesis and critique may be skipped"
            ),
        )

    @staticmethod
    def _frontier_shard_index(key: str, shard_count: int) -> int:
        normalized = str(key or "").strip()
        if shard_count <= 1 or not normalized:
            return 0
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % max(1, int(shard_count))

    @staticmethod
    def _frontier_unique_values(items: list[str], limit: int = 8) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
            if len(values) >= limit:
                break
        return values

    @staticmethod
    def _frontier_backlog_status(
        queue_total: int,
        active_claims: int,
    ) -> str:
        if queue_total >= 512 or active_claims >= 16:
            return "high"
        if queue_total >= 128 or active_claims >= 4:
            return "medium"
        return "low"

    def _detached_frontier_metrics_snapshot(self) -> dict[str, Any]:
        if self._crawl_broker_enabled():
            try:
                metrics = self.crawl_broker_metrics()
            except RuntimeError:
                return {}
            return metrics if isinstance(metrics, dict) else {}

        self._ensure_research_state_store()
        status_counts: dict[str, int] = {}
        js_status_counts: dict[str, int] = {}
        worker_stats: dict[str, dict[str, Any]] = {}
        with closing(self._connect_research_state()) as connection:
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
        shard_entry = {
            "shard_index": 0,
            "queue_db_path": str(self._research_state_path()),
            "queue_total": int(totals_row["queue_total"] or 0) if totals_row else 0,
            "domain_total": int(totals_row["domain_total"] or 0) if totals_row else 0,
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
        return {
            "backend": "sqlite",
            "shard_count": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "queue": {
                "total": shard_entry["queue_total"],
                "domain_total": shard_entry["domain_total"],
                "status_counts": status_counts,
                "js_required_counts": shard_entry["js_required_counts"],
            },
            "observations": {
                "total": shard_entry["observation_total"],
                "browser_total": shard_entry["browser_observation_total"],
            },
            "domain_leases": {
                "active_count": 0,
                "items": [],
            },
            "worker_utilization": {
                "active_worker_count": len(worker_items),
                "items": worker_items,
            },
            "shards": [shard_entry],
        }

    def _detached_frontier_queue_inspect(
        self,
        *,
        limit: int = 24,
        statuses: list[str] | None = None,
        shard_index: int | None = None,
    ) -> dict[str, Any]:
        if self._crawl_broker_enabled():
            try:
                return self.crawl_broker_queue_inspect(
                    limit=limit,
                    statuses=statuses,
                    shard_index=shard_index,
                )
            except RuntimeError:
                return {"items": [], "total_matches": 0, "per_shard_matches": {}}

        del shard_index
        self._ensure_research_state_store()
        normalized_statuses = [
            str(status).strip() for status in (statuses or []) if str(status).strip()
        ]
        if not normalized_statuses:
            normalized_statuses = ["queued", "claimed", "processed"]
        status_placeholders = ", ".join("?" for _ in normalized_statuses)
        where_sql = f"WHERE status IN ({status_placeholders})"
        params: list[Any] = list(normalized_statuses)
        with closing(self._connect_research_state()) as connection:
            match_row = connection.execute(
                f"SELECT COUNT(*) AS row_count FROM crawl_queue {where_sql}",
                tuple(params),
            ).fetchone()
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
                (*params, max(int(limit), 1)),
            ).fetchall()
        items = [
            {
                "shard_index": 0,
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
            for row in rows
        ]
        total_matches = int(match_row["row_count"] or 0) if match_row else 0
        return {
            "limit": max(int(limit), 1),
            "statuses": normalized_statuses,
            "total_matches": total_matches,
            "per_shard_matches": {0: total_matches},
            "items": items,
        }

    @staticmethod
    def _frontier_top_domains(
        queue_items: list[dict[str, Any]], limit: int = 4
    ) -> list[str]:
        counts: dict[str, int] = {}
        for item in queue_items:
            domain = str(item.get("domain") or "").strip()
            if not domain:
                continue
            counts[domain] = counts.get(domain, 0) + 1
        ranked = sorted(
            counts.items(),
            key=lambda item: (int(item[1]), item[0]),
            reverse=True,
        )
        return [domain for domain, _ in ranked[:limit]]

    def _detached_frontier_schedule(
        self,
        objective: str,
        query: str,
        query_variants: list[str],
        plan: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        metrics = self._detached_frontier_metrics_snapshot()
        if not metrics:
            return {}, [], ""

        shard_metrics = [
            dict(item)
            for item in (metrics.get("shards") or [])
            if isinstance(item, dict)
        ]
        shard_count = max(
            1,
            int(metrics.get("shard_count") or len(shard_metrics) or 1),
        )
        variants = self._frontier_unique_values(list(query_variants or []), limit=64)
        source_seeds = self._frontier_unique_values(
            [str(item) for item in (plan.get("source_seeds") or [])],
            limit=64,
        )
        shard_summaries: list[dict[str, Any]] = []
        for index in range(shard_count):
            shard_metric = next(
                (
                    item
                    for item in shard_metrics
                    if int(item.get("shard_index") or 0) == index
                ),
                {"shard_index": index},
            )
            inspected = self._detached_frontier_queue_inspect(
                limit=16,
                statuses=["queued", "claimed", "processed"],
                shard_index=index,
            )
            queue_items = [
                dict(item)
                for item in (inspected.get("items") or [])
                if isinstance(item, dict)
            ]
            assigned_queries = [
                item
                for item in variants
                if self._frontier_shard_index(item, shard_count) == index
            ]
            assigned_seed_urls = [
                item
                for item in source_seeds
                if self._frontier_shard_index(item, shard_count) == index
            ]
            queued_urls = self._frontier_unique_values(
                [
                    str(item.get("url") or "")
                    for item in queue_items
                    if str(item.get("status") or "") in {"queued", "claimed"}
                ],
                limit=6,
            )
            processed_urls = self._frontier_unique_values(
                [
                    str(item.get("url") or "")
                    for item in queue_items
                    if str(item.get("status") or "") == "processed"
                ],
                limit=6,
            )
            pending_queries = self._frontier_unique_values(
                [
                    str(item.get("source_query") or "")
                    for item in queue_items
                    if str(item.get("status") or "") in {"queued", "claimed"}
                ],
                limit=6,
            )
            summary = {
                "shard_index": index,
                "queue_total": int(shard_metric.get("queue_total") or 0),
                "domain_total": int(shard_metric.get("domain_total") or 0),
                "status_counts": dict(shard_metric.get("status_counts") or {}),
                "active_claims": int(shard_metric.get("active_claims") or 0),
                "observation_total": int(shard_metric.get("observation_total") or 0),
                "browser_observation_total": int(
                    shard_metric.get("browser_observation_total") or 0
                ),
                "workers": [
                    dict(item)
                    for item in list(shard_metric.get("workers") or [])[:3]
                    if isinstance(item, dict)
                ],
                "assigned_query_variants": assigned_queries,
                "assigned_seed_urls": assigned_seed_urls,
                "sample_urls": queued_urls,
                "processed_urls": processed_urls,
                "pending_queries": pending_queries,
                "top_domains": self._frontier_top_domains(queue_items),
                "backlog_status": self._frontier_backlog_status(
                    int(shard_metric.get("queue_total") or 0),
                    int(shard_metric.get("active_claims") or 0),
                ),
            }
            shard_summaries.append(summary)

        schedule = {
            "mode": (
                "detached-sharded-frontier" if shard_count > 1 else "detached-frontier"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "backend": str(metrics.get("backend") or "sqlite"),
            "objective": objective,
            "query": query,
            "stop_reason": str(retrieval.get("stop_reason") or ""),
            "shard_count": shard_count,
            "queue_total": int((metrics.get("queue") or {}).get("total") or 0),
            "observation_total": int(
                (metrics.get("observations") or {}).get("total") or 0
            ),
            "active_worker_count": int(
                (metrics.get("worker_utilization") or {}).get("active_worker_count")
                or 0
            ),
            "query_variant_count": len(variants),
            "source_seed_count": len(source_seeds),
            "shards": [
                {
                    "shard_index": int(item.get("shard_index") or 0),
                    "backlog_status": str(item.get("backlog_status") or "low"),
                    "queue_total": int(item.get("queue_total") or 0),
                    "observation_total": int(item.get("observation_total") or 0),
                    "assigned_query_variants": list(
                        item.get("assigned_query_variants") or []
                    ),
                    "assigned_seed_urls": list(item.get("assigned_seed_urls") or []),
                    "top_domains": list(item.get("top_domains") or []),
                }
                for item in shard_summaries
            ],
        }
        markdown_lines = [
            "# Detached Frontier Shard Summaries",
            "",
            f"Mode: {schedule['mode']}",
            f"Backend: {schedule['backend']}",
            f"Shards: {schedule['shard_count']}",
            f"Queue total: {schedule['queue_total']}",
            f"Observation total: {schedule['observation_total']}",
            f"Active workers: {schedule['active_worker_count']}",
            f"Stop reason: {schedule['stop_reason'] or 'unknown'}",
            "",
        ]
        for item in shard_summaries:
            markdown_lines.extend(
                [
                    f"## Shard {int(item.get('shard_index') or 0)}",
                    "",
                    (
                        "Queue: "
                        f"total={int(item.get('queue_total') or 0)}, "
                        f"active_claims={int(item.get('active_claims') or 0)}, "
                        f"observations={int(item.get('observation_total') or 0)}, "
                        f"backlog={str(item.get('backlog_status') or 'low')}"
                    ),
                    (
                        "Assigned query variants: "
                        + (
                            "; ".join(item.get("assigned_query_variants") or [])
                            or "none"
                        )
                    ),
                    (
                        "Pending queries: "
                        + ("; ".join(item.get("pending_queries") or []) or "none")
                    ),
                    (
                        "Sample queued URLs: "
                        + ("; ".join(item.get("sample_urls") or []) or "none")
                    ),
                    (
                        "Sample processed URLs: "
                        + ("; ".join(item.get("processed_urls") or []) or "none")
                    ),
                    (
                        "Top domains: "
                        + (", ".join(item.get("top_domains") or []) or "none")
                    ),
                    "",
                ]
            )
        return schedule, shard_summaries, "\n".join(markdown_lines).rstrip() + "\n"

    def _checkpoint_research_artifact_dir(self, run_id: str) -> Any:
        artifact_dir = self.workspace_root / "runs" / run_id / "research"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _prepare_detached_merge_payloads(
        self,
        *,
        run_id: str,
        objective: str,
        query: str,
        settings: ResearchSettings,
        query_variants: list[str],
        plan: dict[str, Any],
        pc_context_info: dict[str, Any],
        retrieval: dict[str, Any],
        run_id_for_durable_notes: str,
        synthesis_mode: str,
        sources: list[ResearchSource],
        frontier_schedule_payload: dict[str, Any] | None = None,
        frontier_shard_payload: list[dict[str, Any]] | None = None,
        frontier_shard_markdown: str = "",
        shard_synthesis_packets: list[dict[str, Any]] | None = None,
        merge_coordinator: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        durable_notes = self._load_durable_notes(run_id_for_durable_notes)
        if frontier_schedule_payload is None or frontier_shard_payload is None:
            (
                frontier_schedule_payload,
                frontier_shard_payload,
                frontier_shard_markdown,
            ) = self._detached_frontier_schedule(
                objective,
                query,
                list(retrieval.get("query_variants") or query_variants),
                plan,
                retrieval,
            )
        if shard_synthesis_packets is None:
            shard_synthesis_packets = self._build_shard_synthesis_packets(
                objective,
                query,
                sources,
                settings.depth,
                plan,
                durable_notes,
                synthesis_mode,
                shard_count=int(
                    (frontier_schedule_payload or {}).get("shard_count")
                    or len(frontier_shard_payload or [])
                    or 1
                ),
                frontier_shards=frontier_shard_payload,
            )
        merged_packet, computed_merge = self._merge_shard_synthesis_packets(
            objective,
            query,
            settings.depth,
            plan,
            durable_notes,
            synthesis_mode,
            shard_synthesis_packets,
        )
        if merge_coordinator is None:
            merge_coordinator = computed_merge
        else:
            merge_coordinator = {
                **computed_merge,
                **dict(merge_coordinator),
                "shards": list(
                    dict(merge_coordinator).get("shards")
                    or computed_merge.get("shards")
                    or []
                ),
            }
        stop_reason = str(retrieval.get("stop_reason") or "")
        if stop_reason:
            merged_packet["retrieval_stop_reason"] = stop_reason
        replay_manifest = {
            "kind": "detached-merge-replay-manifest",
            "generated_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "objective": objective,
            "query": query,
            "depth": settings.depth,
            "max_sources": settings.max_sources,
            "per_provider": settings.per_provider,
            "max_query_variants": settings.max_query_variants,
            "query_variants": list(
                query_variants or retrieval.get("query_variants") or []
            ),
            "plan": dict(plan),
            "pc_context_info": dict(pc_context_info),
            "retrieval": {
                "coverage": dict(retrieval.get("coverage") or {}),
                "passes": [
                    dict(item)
                    for item in (retrieval.get("passes") or [])
                    if isinstance(item, dict)
                ],
                "stop_reason": stop_reason,
                "query_variants": list(
                    retrieval.get("query_variants") or query_variants
                ),
            },
            "synthesis_mode": synthesis_mode,
            "detached_frontier": dict(frontier_schedule_payload or {}),
            "frontier_shards": [
                dict(item)
                for item in (frontier_shard_payload or [])
                if isinstance(item, dict)
            ],
            "detached_merge": dict(merge_coordinator or {}),
        }
        return {
            "frontier_schedule_payload": frontier_schedule_payload or {},
            "frontier_shard_payload": frontier_shard_payload or [],
            "frontier_shard_markdown": frontier_shard_markdown,
            "shard_synthesis_packets": shard_synthesis_packets or [],
            "merge_coordinator": merge_coordinator or {},
            "replay_manifest": replay_manifest,
            "merged_packet": merged_packet,
        }

    def _write_periodic_synthesis_checkpoint(
        self,
        *,
        run_id: str,
        objective: str,
        query: str,
        settings: ResearchSettings,
        query_variants: list[str],
        plan: dict[str, Any],
        pc_context_info: dict[str, Any],
        retrieval: dict[str, Any],
        synthesis_mode: str,
        sources: list[ResearchSource],
    ) -> dict[str, Any]:
        if not run_id:
            return {}
        artifact_dir = self._checkpoint_research_artifact_dir(run_id)
        payloads = self._prepare_detached_merge_payloads(
            run_id=run_id,
            objective=objective,
            query=query,
            settings=settings,
            query_variants=query_variants,
            plan=plan,
            pc_context_info=pc_context_info,
            retrieval=retrieval,
            run_id_for_durable_notes=run_id,
            synthesis_mode=synthesis_mode,
            sources=sources,
        )
        frontier_schedule_path = artifact_dir / "frontier_schedule.json"
        frontier_shards_path = artifact_dir / "frontier_shards.json"
        frontier_shards_markdown_path = artifact_dir / "frontier_shards.md"
        synthesis_shards_path = artifact_dir / "synthesis_shards.json"
        synthesis_merge_path = artifact_dir / "synthesis_merge_coordinator.json"
        replay_manifest_path = artifact_dir / "synthesis_replay_manifest.json"
        try:
            frontier_schedule_path.write_text(
                json.dumps(payloads["frontier_schedule_payload"], indent=2),
                encoding="utf-8",
            )
            frontier_shards_path.write_text(
                json.dumps(payloads["frontier_shard_payload"], indent=2),
                encoding="utf-8",
            )
            frontier_shards_markdown_path.write_text(
                str(payloads["frontier_shard_markdown"] or ""),
                encoding="utf-8",
            )
            synthesis_shards_path.write_text(
                json.dumps(payloads["shard_synthesis_packets"], indent=2),
                encoding="utf-8",
            )
            synthesis_merge_path.write_text(
                json.dumps(payloads["merge_coordinator"], indent=2),
                encoding="utf-8",
            )
            replay_manifest_path.write_text(
                json.dumps(payloads["replay_manifest"], indent=2),
                encoding="utf-8",
            )
        except OSError:
            return {}
        return {
            "detached_frontier": {
                "mode": payloads["frontier_schedule_payload"].get("mode"),
                "backend": payloads["frontier_schedule_payload"].get("backend"),
                "shard_count": payloads["frontier_schedule_payload"].get("shard_count"),
                "schedule_path": str(
                    frontier_schedule_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
            },
            "detached_merge": {
                "mode": payloads["merge_coordinator"].get("mode"),
                "shard_count": payloads["merge_coordinator"].get("shard_count"),
                "merge_ready_packet_count": payloads["merge_coordinator"].get(
                    "merge_ready_packet_count"
                ),
                "non_empty_shards": payloads["merge_coordinator"].get(
                    "non_empty_shards"
                ),
                "shard_packet_path": str(
                    synthesis_shards_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "coordinator_path": str(
                    synthesis_merge_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "replay_manifest_path": str(
                    replay_manifest_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
            },
        }

    def replay_detached_merge(self, run_id: str) -> ResearchBrief:
        artifact_dir = self._checkpoint_research_artifact_dir(run_id)
        replay_manifest_path = artifact_dir / "synthesis_replay_manifest.json"
        synthesis_shards_path = artifact_dir / "synthesis_shards.json"
        synthesis_merge_path = artifact_dir / "synthesis_merge_coordinator.json"
        if not replay_manifest_path.exists():
            raise FileNotFoundError(
                f"Detached merge replay manifest not found for run '{run_id}'"
            )
        if not synthesis_shards_path.exists():
            raise FileNotFoundError(
                f"Shard synthesis packets not found for run '{run_id}'"
            )
        manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
        shard_packets = json.loads(synthesis_shards_path.read_text(encoding="utf-8"))
        persisted_merge: dict[str, Any] = {}
        if synthesis_merge_path.exists():
            persisted_merge = json.loads(
                synthesis_merge_path.read_text(encoding="utf-8")
            )
        objective = str(manifest.get("objective") or run_id)
        query = str(manifest.get("query") or objective)
        depth = str(manifest.get("depth") or "standard")
        settings = ResearchSettings(
            depth=depth,
            max_sources=max(
                1,
                int(manifest.get("max_sources") or len(shard_packets) or 1),
            ),
            per_provider=max(
                1, int(manifest.get("per_provider") or self.limit_per_provider)
            ),
            max_query_variants=max(
                1,
                int(
                    manifest.get("max_query_variants")
                    or len(manifest.get("query_variants") or [])
                    or 1
                ),
            ),
        )
        plan = dict(manifest.get("plan") or {})
        retrieval = dict(manifest.get("retrieval") or {})
        query_variants = [
            str(item)
            for item in (
                manifest.get("query_variants") or retrieval.get("query_variants") or []
            )
            if str(item).strip()
        ]
        frontier_schedule_payload = dict(manifest.get("detached_frontier") or {})
        frontier_shard_payload = [
            dict(item)
            for item in (manifest.get("frontier_shards") or [])
            if isinstance(item, dict)
        ]
        durable_notes = self._load_durable_notes(run_id)
        synthesis_mode = str(
            manifest.get("synthesis_mode")
            or self._resolve_final_synthesis_mode(depth, durable_notes)
        )
        synthesis_packet, merge_coordinator = self._merge_shard_synthesis_packets(
            objective,
            query,
            depth,
            plan,
            durable_notes,
            synthesis_mode,
            [dict(item) for item in shard_packets if isinstance(item, dict)],
        )
        if persisted_merge:
            merge_coordinator = {
                **merge_coordinator,
                **persisted_merge,
                "shards": list(
                    persisted_merge.get("shards")
                    or merge_coordinator.get("shards")
                    or []
                ),
            }
        synthesis_packet["merge_coordinator"] = merge_coordinator
        stop_reason = str(
            retrieval.get("stop_reason")
            or persisted_merge.get("stop_reason")
            or "checkpoint-replay"
        )
        synthesis_packet["retrieval_stop_reason"] = stop_reason
        summary_sources = self._synthesis_packet_sources(synthesis_packet)
        summary = self._summarize(
            objective,
            summary_sources,
            depth,
            plan,
            query,
            durable_notes,
            synthesis_mode,
            coverage=dict(retrieval.get("coverage") or {}),
            synthesis_packet=synthesis_packet,
        )
        replay_retrieval = {
            "coverage": dict(retrieval.get("coverage") or {}),
            "passes": [
                dict(item)
                for item in (retrieval.get("passes") or [])
                if isinstance(item, dict)
            ],
            "stop_reason": stop_reason,
            "query_variants": query_variants,
            "replay_mode": "detached-merge-only",
        }
        artifacts = self._write_artifacts(
            run_id,
            objective,
            query,
            summary_sources,
            summary,
            settings,
            query_variants,
            plan,
            dict(manifest.get("pc_context_info") or {}),
            replay_retrieval,
            run_id,
            synthesis_mode,
            synthesis_packet,
            frontier_schedule_payload,
            frontier_shard_payload,
            "",
            [dict(item) for item in shard_packets if isinstance(item, dict)],
            merge_coordinator,
        )
        return ResearchBrief(
            objective=objective,
            query=query,
            summary=summary,
            sources=summary_sources,
            artifacts=artifacts,
            confidence=self._confidence(summary_sources),
            metadata={
                "replay_mode": "detached-merge-only",
                "run_id": run_id,
                "retrieval": replay_retrieval,
            },
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
        frontier_schedule_payload: dict[str, Any] | None = None,
        frontier_shard_payload: list[dict[str, Any]] | None = None,
        frontier_shard_markdown: str = "",
        shard_synthesis_packets: list[dict[str, Any]] | None = None,
        merge_coordinator: dict[str, Any] | None = None,
    ) -> list[str]:
        artifact_dir = self.workspace_root / "runs" / run_id / "research"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sources_path = artifact_dir / "sources.json"
        brief_path = artifact_dir / "brief.md"
        digest_path = artifact_dir / "digest.json"
        plan_path = artifact_dir / "research_plan.json"
        frontier_schedule_path = artifact_dir / "frontier_schedule.json"
        frontier_shards_path = artifact_dir / "frontier_shards.json"
        frontier_shards_markdown_path = artifact_dir / "frontier_shards.md"
        claim_trace_path = artifact_dir / "claim_trace.json"
        findings_path = artifact_dir / "findings.json"
        diagnostics_path = artifact_dir / "provider_diagnostics.json"
        analysis_report_path = artifact_dir / "analysis_report.md"
        paper_report_path = artifact_dir / "paper_report.md"
        retrieval_metrics_path = artifact_dir / "retrieval_metrics.json"
        evidence_graph_path = artifact_dir / "evidence_graph.json"
        benchmark_adapters_path = artifact_dir / "benchmark_adapters.json"
        synthesis_packet_path = artifact_dir / "synthesis_packet.json"
        synthesis_shards_path = artifact_dir / "synthesis_shards.json"
        synthesis_merge_path = artifact_dir / "synthesis_merge_coordinator.json"
        synthesis_replay_manifest_path = artifact_dir / "synthesis_replay_manifest.json"
        durable_notes_path = self._durable_report_path(run_id_for_durable_notes)
        findings = self._finding_ledger(query, sources, plan)
        claim_trace_payload = self._claim_trace(objective, summary, findings)
        retrieval_payload = {
            "coverage": retrieval["coverage"],
            "passes": retrieval["passes"],
            "stop_reason": retrieval["stop_reason"],
            "query_variants": retrieval["query_variants"],
        }
        detached_payloads = self._prepare_detached_merge_payloads(
            run_id=run_id,
            objective=objective,
            query=query,
            settings=settings,
            query_variants=query_variants,
            plan=plan,
            pc_context_info=pc_context_info,
            retrieval=retrieval,
            run_id_for_durable_notes=run_id_for_durable_notes,
            synthesis_mode=synthesis_mode,
            sources=sources,
            frontier_schedule_payload=frontier_schedule_payload,
            frontier_shard_payload=frontier_shard_payload,
            frontier_shard_markdown=frontier_shard_markdown,
            shard_synthesis_packets=shard_synthesis_packets,
            merge_coordinator=merge_coordinator,
        )
        frontier_schedule_payload = dict(
            detached_payloads.get("frontier_schedule_payload") or {}
        )
        frontier_shard_payload = [
            dict(item)
            for item in (detached_payloads.get("frontier_shard_payload") or [])
            if isinstance(item, dict)
        ]
        frontier_shard_markdown = str(
            detached_payloads.get("frontier_shard_markdown") or ""
        )
        shard_synthesis_packets = [
            dict(item)
            for item in (detached_payloads.get("shard_synthesis_packets") or [])
            if isinstance(item, dict)
        ]
        merge_coordinator = dict(detached_payloads.get("merge_coordinator") or {})
        replay_manifest = dict(detached_payloads.get("replay_manifest") or {})
        if frontier_schedule_payload:
            retrieval_payload["detached_frontier"] = frontier_schedule_payload
            retrieval_payload["frontier_shards"] = frontier_shard_payload
        if merge_coordinator:
            retrieval_payload["detached_merge"] = merge_coordinator
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
                "optional model-based synthesis and critique",
                "exact dedupe",
                "plan-first multi-perspective query decomposition",
                "finding support/conflict ledger",
                "relevance ranking",
                "compressed digest artifacts",
            ]
        )
        plan_payload = {
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
        }
        if frontier_schedule_payload:
            plan_payload["detached_frontier"] = {
                "mode": frontier_schedule_payload.get("mode"),
                "backend": frontier_schedule_payload.get("backend"),
                "shard_count": frontier_schedule_payload.get("shard_count"),
                "schedule_path": str(
                    frontier_schedule_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "shard_summary_path": str(
                    frontier_shards_markdown_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
            }
        if merge_coordinator:
            plan_payload["detached_merge"] = {
                "mode": merge_coordinator.get("mode"),
                "shard_count": merge_coordinator.get("shard_count"),
                "non_empty_shards": merge_coordinator.get("non_empty_shards"),
                "shard_packet_path": str(
                    synthesis_shards_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "coordinator_path": str(
                    synthesis_merge_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "replay_manifest_path": str(
                    synthesis_replay_manifest_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
            }
        analysis_retrieval = dict(retrieval)
        if frontier_schedule_payload:
            analysis_retrieval["detached_frontier"] = frontier_schedule_payload
            analysis_retrieval["frontier_shards"] = frontier_shard_payload
        if merge_coordinator:
            analysis_retrieval["detached_merge"] = merge_coordinator

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
            json.dumps(plan_payload, indent=2),
            encoding="utf-8",
        )
        if frontier_schedule_payload:
            frontier_schedule_path.write_text(
                json.dumps(frontier_schedule_payload, indent=2),
                encoding="utf-8",
            )
            frontier_shards_path.write_text(
                json.dumps(frontier_shard_payload, indent=2),
                encoding="utf-8",
            )
            frontier_shards_markdown_path.write_text(
                frontier_shard_markdown,
                encoding="utf-8",
            )
        analysis_report_path.write_text(
            self._analysis_report_markdown(
                objective,
                summary,
                sources,
                plan,
                pc_context_info,
                analysis_retrieval,
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
        if shard_synthesis_packets:
            synthesis_shards_path.write_text(
                json.dumps(shard_synthesis_packets, indent=2),
                encoding="utf-8",
            )
        if merge_coordinator:
            synthesis_merge_path.write_text(
                json.dumps(merge_coordinator, indent=2),
                encoding="utf-8",
            )
        if replay_manifest:
            synthesis_replay_manifest_path.write_text(
                json.dumps(replay_manifest, indent=2),
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
            *(
                [
                    str(frontier_schedule_path.relative_to(self.workspace_root)),
                    str(frontier_shards_path.relative_to(self.workspace_root)),
                    str(frontier_shards_markdown_path.relative_to(self.workspace_root)),
                ]
                if frontier_schedule_payload
                else []
            ),
            str(analysis_report_path.relative_to(self.workspace_root)),
            str(paper_report_path.relative_to(self.workspace_root)),
            str(retrieval_metrics_path.relative_to(self.workspace_root)),
            str(findings_path.relative_to(self.workspace_root)),
            str(claim_trace_path.relative_to(self.workspace_root)),
            str(evidence_graph_path.relative_to(self.workspace_root)),
            str(benchmark_adapters_path.relative_to(self.workspace_root)),
            str(synthesis_packet_path.relative_to(self.workspace_root)),
            *(
                [str(synthesis_shards_path.relative_to(self.workspace_root))]
                if shard_synthesis_packets
                else []
            ),
            *(
                [str(synthesis_merge_path.relative_to(self.workspace_root))]
                if merge_coordinator
                else []
            ),
            *(
                [str(synthesis_replay_manifest_path.relative_to(self.workspace_root))]
                if replay_manifest
                else []
            ),
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
