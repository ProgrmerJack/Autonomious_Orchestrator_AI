from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .deep_research import DeepResearchEngine, ResearchSource


@dataclass(slots=True)
class CrawlWorkerLoopConfig:
    batch_size: int = 6
    poll_interval_seconds: float = 15.0
    claim_ttl_seconds: int = 900
    once: bool = False


@dataclass(slots=True)
class CrawlWorkerBatchResult:
    worker_id: str
    claimed_count: int
    processed_count: int
    failed_count: int
    enqueued_count: int
    observation_count: int
    reclaimed_claims: int
    idle: bool


class ResearchCrawlWorker:
    def __init__(
        self,
        engine: DeepResearchEngine,
        worker_id: str = "crawl-worker",
        config: CrawlWorkerLoopConfig | None = None,
    ) -> None:
        self.engine = engine
        self.worker_id = worker_id
        self.config = config or CrawlWorkerLoopConfig()

    def run_once(self) -> dict[str, Any]:
        self.engine._ensure_research_state_store()
        reclaimed_claims = self.engine._requeue_stale_crawl_claims(
            self.config.claim_ttl_seconds
        )
        claimed_rows = self.engine._claim_crawl_queue_batch(
            self.config.batch_size,
            self.worker_id,
        )
        if not claimed_rows:
            return asdict(
                CrawlWorkerBatchResult(
                    worker_id=self.worker_id,
                    claimed_count=0,
                    processed_count=0,
                    failed_count=0,
                    enqueued_count=0,
                    observation_count=0,
                    reclaimed_claims=reclaimed_claims,
                    idle=True,
                )
            )

        sources: list[ResearchSource] = []
        for row in claimed_rows:
            url = str(row.get("url") or "").strip()
            domain = str(row.get("domain") or "").strip()
            sources.append(
                ResearchSource(
                    provider="crawl-worker",
                    title=self.engine._label_from_url(url),
                    url=url,
                    authors=[domain] if domain else [],
                    abstract=(
                        str(row.get("source_query") or "").strip()
                        or str(row.get("source_url") or "").strip()
                        or self.engine._label_from_url(url)
                    )[:320],
                    score=float(row.get("priority") or 0.0),
                    quality_flags=["crawl-worker"],
                )
            )

        browser_prefetch = self.engine._headless_browser_pool_fetch(
            self.engine._persistent_unique_urls(
                [
                    source.url
                    for source, row in zip(sources, claimed_rows)
                    if int(row.get("js_required") or 0) == 1
                ]
            ),
            max_chars=80_000,
            timeout_ms=18_000,
        )

        processed_count = 0
        failed_count = 0
        enqueued_count = 0
        observation_count = 0
        for source, row in zip(sources, claimed_rows):
            query = (
                str(row.get("source_query") or "").strip()
                or source.title
                or self.engine._label_from_url(source.url)
            )
            run_id = str(row.get("run_id") or "").strip()
            content, raw_html, status, used_browser = self.engine._fetch_source_content(
                source,
                query,
                browser_prefetch,
            )
            if status != "processed":
                self.engine._update_crawl_queue_status(source.url, "failed", status)
                failed_count += 1
                continue
            outbound_candidates = self.engine._extract_outbound_source_candidates(
                raw_html,
                query,
                source.url,
            )
            outbound_urls = [candidate.url for candidate in outbound_candidates]
            if outbound_urls:
                self.engine._enqueue_url_batch(
                    outbound_urls,
                    query,
                    run_id,
                    source_url=source.url,
                    priority=max(float(row.get("priority") or 0.0) + 1.0, 5.0),
                )
                enqueued_count += len(outbound_urls)
            query_hints = self.engine._content_to_new_queries(
                content,
                source.title,
                query,
            )
            self.engine._record_crawl_observation(
                source,
                content,
                query,
                query_hints,
                outbound_urls,
                self.worker_id,
                used_browser,
            )
            self.engine._update_crawl_queue_status(source.url, "processed")
            processed_count += 1
            observation_count += 1

        return asdict(
            CrawlWorkerBatchResult(
                worker_id=self.worker_id,
                claimed_count=len(claimed_rows),
                processed_count=processed_count,
                failed_count=failed_count,
                enqueued_count=enqueued_count,
                observation_count=observation_count,
                reclaimed_claims=reclaimed_claims,
                idle=False,
            )
        )

    def run_forever(self) -> None:
        while True:
            result = self.run_once()
            if self.config.once:
                return
            if result.get("idle"):
                time.sleep(self.config.poll_interval_seconds)
            else:
                time.sleep(min(self.config.poll_interval_seconds, 3.0))
