from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class AgentOSClient:
    """Tiny stdlib REST client for the local AgentOS gateway."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def product_status(self) -> dict[str, Any]:
        return self._request("GET", "/setup/checks")

    def providers(self) -> list[dict[str, Any]]:
        return self._request("GET", "/providers")

    def channels(self) -> list[dict[str, Any]]:
        return self._request("GET", "/channels")

    def benchmarks(self) -> dict[str, Any]:
        return self._request("GET", "/benchmarks")

    def golden_traces(self) -> dict[str, Any]:
        return self._request("GET", "/benchmarks/golden-traces")

    def replay_benchmarks(self, trace_id: str = "") -> dict[str, Any]:
        return self._request(
            "POST",
            "/benchmarks/replay",
            {"trace_id": trace_id},
        )

    def eval_pack(
        self,
        pack: str = "",
        max_tasks: int | None = None,
    ) -> dict[str, Any]:
        query_items: dict[str, str | int] = {}
        if pack:
            query_items["pack"] = pack
        if max_tasks is not None:
            query_items["max_tasks"] = max_tasks
        if not query_items:
            return self._request("GET", "/benchmarks/eval-pack")
        query = urllib.parse.urlencode(query_items)
        return self._request("GET", f"/benchmarks/eval-pack?{query}")

    def replay_debug(
        self,
        run_id: str = "",
        limit: int = 1,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/debug/replay",
            {"run_id": run_id, "limit": limit},
        )

    def live_fire_eval(
        self,
        backend: str = "virtual-desktop-sandbox",
        max_tasks: int | None = None,
        surfaces: list[str] | None = None,
        intents: list[str] | None = None,
        pack: str = "combined",
        run_id: str = "",
        windows_safe_pack: bool = False,
        repeat: int = 1,
        promote_failures: bool = True,
        promote_after: int = 1,
        replay_limit: int = 10,
        training_output: str = "",
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": backend,
            "max_tasks": max_tasks,
            "surfaces": surfaces or [],
            "intents": intents or [],
            "pack": pack,
            "run_id": run_id,
            "windows_safe_pack": windows_safe_pack,
            "repeat": repeat,
            "promote_failures": promote_failures,
            "promote_after": promote_after,
            "replay_limit": replay_limit,
            "training_output": training_output,
        }
        if approval_token:
            payload["approval_token"] = approval_token
        return self._request("POST", "/benchmarks/live-fire-eval", payload)

    def live_fire_review(self, limit: int = 10) -> dict[str, Any]:
        query = urllib.parse.urlencode({"limit": limit})
        return self._request("GET", f"/benchmarks/live-fire-review?{query}")

    def promote_live_fire_failure(
        self,
        run_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/benchmarks/live-fire-review/promote",
            {"run_id": run_id, "task_id": task_id},
        )

    def live_fire_shadow_training(
        self,
        trajectory_paths: list[str] | None = None,
        output_dir: str = "",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/benchmarks/live-fire-shadow-training",
            {
                "trajectory_paths": trajectory_paths or [],
                "output_dir": output_dir,
            },
        )

    def daemon_status(self) -> dict[str, Any]:
        return self._request("GET", "/daemon/status")

    def daemon_stop(self) -> dict[str, Any]:
        return self._request("POST", "/daemon/stop")

    def commands(self) -> list[dict[str, Any]]:
        return self._request("GET", "/commands")

    def start_run(
        self,
        objective: str,
        depth: str = "adaptive",
        background: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/runs",
            {
                "objective": objective,
                "depth": depth,
                "background": background,
            },
        )

    def command(
        self,
        text: str,
        channel: str = "sdk",
        sender_id: str = "sdk",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/channels/command",
            {
                "channel": channel,
                "sender_id": sender_id,
                "text": text,
            },
        )

    def policy_inspect(
        self,
        action_type: str,
        target: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/policy/inspect",
            {"action_type": action_type, "target": target},
        )

    def pc_snapshot(
        self,
        backend: str = "windows-uia",
        limit: int = 120,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode({"backend": backend, "limit": limit})
        return self._request("GET", f"/pc/snapshot?{query}")

    def pc_debug_selector(
        self,
        selector: str,
        backend: str = "windows-uia",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/pc/debug-selector",
            {"backend": backend, "selector": selector},
        )

    def pc_receipts(self) -> list[dict[str, Any]]:
        return self._request("GET", "/pc/receipts")

    def pc_workflow_plan(self, objective: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/pc/workflow/plan",
            {"objective": objective},
        )

    def pc_workflow_execute(
        self,
        objective: str,
        backend: str = "virtual-desktop-sandbox",
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "objective": objective,
            "backend": backend,
        }
        if approval_token:
            payload["approval_token"] = approval_token
        return self._request(
            "POST",
            "/pc/workflow/execute",
            payload,
        )

    def channel_deliveries(self) -> list[dict[str, Any]]:
        return self._request("GET", "/channels/deliveries")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - local gateway client
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(text or exc.reason) from exc
