from __future__ import annotations

import json
import threading
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
    detail: str = ""


class CrawlBrokerServer:
    def __init__(
        self,
        workspace_root: str | Path = ".",
        host: str = "127.0.0.1",
        port: int = 0,
        queue_db_path: str | Path | None = None,
        auth_token: str = "",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.host = str(host).strip() or "127.0.0.1"
        self.port = max(0, int(port))
        if queue_db_path is None:
            self.queue_db_path = self.workspace_root / ".agentos" / "research_state.sqlite3"
        else:
            self.queue_db_path = Path(queue_db_path).resolve()
        self.auth_token = str(auth_token or "")
        self.engine = DeepResearchEngine(
            workspace_root=self.workspace_root,
            research_state_path=self.queue_db_path,
            crawl_broker_url="",
            crawl_broker_token="",
        )
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
            self.engine._enqueue_url_batch(
                urls,
                str(payload.get("source_query") or ""),
                str(payload.get("run_id") or ""),
                source_url=str(payload.get("source_url") or ""),
                priority=float(payload.get("priority") or 0.0),
            )
            return {"queued_count": len(urls)}
        if path == "/queue/requeue-stale":
            reclaimed = self.engine._requeue_stale_crawl_claims(
                int(payload.get("stale_after_seconds") or 900)
            )
            return {"reclaimed_count": reclaimed}
        if path == "/queue/claim-batch":
            claimed = self.engine._claim_crawl_queue_batch(
                int(payload.get("limit") or 0),
                str(payload.get("worker_id") or "crawl-worker"),
            )
            return {"claimed": claimed}
        if path == "/queue/claim-sources":
            sources = self.engine._claim_persistent_crawl_sources(
                str(payload.get("query") or ""),
                str(payload.get("objective") or ""),
                int(payload.get("limit") or 0),
                exclude_urls=[
                    str(url).strip() for url in (payload.get("exclude_urls") or [])
                ],
            )
            return {"sources": [asdict(source) for source in sources]}
        if path == "/queue/update-status":
            self.engine._update_crawl_queue_status(
                str(payload.get("url") or ""),
                str(payload.get("status") or ""),
                str(payload.get("error") or ""),
            )
            return {"ok": True}
        if path == "/queue/snapshot":
            return self.engine._persistent_crawl_queue_snapshot(
                limit=int(payload.get("limit") or 24)
            )
        if path == "/observations/record":
            source = self._source_from_payload(payload.get("source") or {})
            self.engine._record_crawl_observation(
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
                "query_hints": self.engine._persistent_evidence_query_hints(
                    str(payload.get("query") or ""),
                    str(payload.get("objective") or ""),
                    limit=int(payload.get("limit") or 16),
                )
            }
        if path == "/evidence/seed-urls":
            return {
                "seed_urls": self.engine._persistent_seed_urls(
                    str(payload.get("query") or ""),
                    str(payload.get("objective") or ""),
                    limit=int(payload.get("limit") or 16),
                )
            }
        if path == "/evidence/update-index":
            self.engine._update_persistent_evidence_index(
                str(payload.get("run_id") or ""),
                str(payload.get("objective") or ""),
                str(payload.get("query") or ""),
                [],
                dict(payload.get("claim_trace") or {}),
            )
            return {"ok": True}
        if path == "/evidence/snapshot":
            return self.engine._persistent_evidence_snapshot(
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
        normalized = {
            key: payload.get(key)
            for key in fields
            if key in payload
        }
        return ResearchSource(**normalized)