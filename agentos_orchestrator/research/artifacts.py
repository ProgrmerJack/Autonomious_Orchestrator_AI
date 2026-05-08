from __future__ import annotations

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
    ResearchSettings,
    ResearchSource,
    sanitize_evidence_claim_text as _sanitize_evidence_claim_text,
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
        self.provider_diagnostics.append(
            {
                "provider": provider,
                "status": status,
                "detail": detail[:500],
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self._flush_live_provider_diagnostics()

    def _flush_live_provider_diagnostics(self) -> None:
        if not self._active_run_id:
            return
        artifact_dir = self.workspace_root / "runs" / self._active_run_id / "research"
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "provider_diagnostics.json").write_text(
                json.dumps(self.provider_diagnostics, indent=2),
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
                "optional model-based synthesis and critique",
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
