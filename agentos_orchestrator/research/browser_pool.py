from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class HeadlessBrowserWorkerPool:
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
