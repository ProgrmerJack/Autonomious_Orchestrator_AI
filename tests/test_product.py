from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from agentos_orchestrator.product import (
    CommandRegistry,
    CrawlWorkerManager,
    CrawlWorkerRecord,
    CrawlWorkerServiceManager,
    DaemonManager,
    WorkflowCommand,
    collect_product_status,
)
from agentos_orchestrator.sdk import AgentOSClient


class FakeClient(AgentOSClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((method, path, payload))
        return {"ok": True, "path": path, "payload": payload}


class ProductTests(unittest.TestCase):
    def test_product_status_reports_core_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "default": "deny",
                        "allow": {"actions": [], "paths": []},
                        "forbid": {"actions": [], "paths": []},
                        "require_approval": {"actions": ["os.act"]},
                    }
                ),
                encoding="utf-8",
            )

            status = collect_product_status(
                root,
                policy,
                root / ".agentos/state.sqlite3",
                root / ".agentos/memory.sqlite3",
                {"passed": True},
            )

            checks = {check.check_id: check for check in status.checks}
            self.assertEqual(checks["policy-safety"].status, "pass")
            self.assertIn("readiness_score", status.benchmarks)

    def test_provider_status_detects_environment_keys(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            status = collect_product_status(
                Path.cwd(),
                "examples/policies/deep_research.json",
                ".agentos/state.sqlite3",
                ".agentos/memory.sqlite3",
            )

        providers = {provider.provider_id: provider for provider in status.providers}
        self.assertTrue(providers["openai"].configured)

    def test_daemon_manager_reports_stopped_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DaemonManager(temp_dir)
            status = manager.status()

            self.assertEqual(status.status, "stopped")
            self.assertIsNone(status.launcher_pid)

    def test_crawl_worker_manager_reports_stopped_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CrawlWorkerManager(temp_dir)
            status = manager.status()

            self.assertEqual(status.status, "stopped")
            self.assertEqual(status.worker_pids, [])

    def test_crawl_worker_manager_start_records_detached_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_db = Path(temp_dir) / ".agentos/research_state.sqlite3"

            class _FakeProcess:
                def __init__(self, pid: int) -> None:
                    self.pid = pid

            with patch(
                "agentos_orchestrator.product.crawl_worker.subprocess.Popen"
            ) as popen:
                popen.side_effect = [_FakeProcess(111), _FakeProcess(222)]
                manager = CrawlWorkerManager(temp_dir, python_executable="python")
                record = manager.start(
                    worker_count=2,
                    queue_db_path=queue_db,
                    poll_interval_seconds=9.0,
                    batch_size=4,
                    claim_ttl_seconds=300,
                )

            self.assertEqual(record.status, "running")
            self.assertEqual(record.worker_count, 2)
            self.assertEqual(record.worker_pids, [111, 222])
            payload = json.loads(
                (Path(temp_dir) / ".agentos/crawl_worker.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["worker_count"], 2)
            self.assertEqual(payload["worker_pids"], [111, 222])
            self.assertEqual(payload["queue_db_path"], str(queue_db))

    def test_crawl_worker_manager_start_records_broker_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_db = Path(temp_dir) / ".agentos/research_state.sqlite3"

            class _FakeProcess:
                def __init__(self, pid: int) -> None:
                    self.pid = pid

            with patch(
                "agentos_orchestrator.product.crawl_worker.subprocess.Popen"
            ) as popen:
                popen.return_value = _FakeProcess(333)
                manager = CrawlWorkerManager(temp_dir, python_executable="python")
                record = manager.start(
                    worker_count=1,
                    queue_db_path=queue_db,
                    broker_url="http://127.0.0.1:8787",
                    broker_token="secret-token",
                )

            self.assertEqual(record.broker_url, "http://127.0.0.1:8787")
            self.assertTrue(record.broker_token_configured)
            self.assertIn("--broker-url", popen.call_args.args[0])
            self.assertIn("http://127.0.0.1:8787", popen.call_args.args[0])
            self.assertIn("--broker-token", popen.call_args.args[0])
            payload = json.loads(
                (Path(temp_dir) / ".agentos/crawl_worker.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["broker_url"], "http://127.0.0.1:8787")
            self.assertEqual(payload["broker_token"], "secret-token")

    def test_crawl_worker_manager_supervise_once_uses_reconcile_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_db = Path(temp_dir) / ".agentos/research_state.sqlite3"
            manager = CrawlWorkerManager(temp_dir, python_executable="python")
            running = CrawlWorkerRecord(
                status="running",
                worker_pids=[111],
                worker_count=1,
                queue_db_path=str(queue_db),
                log_paths=[str(Path(temp_dir) / ".agentos/logs/crawl-worker-1.log")],
                detail="ok",
            )

            with patch.object(
                manager, "ensure_running", return_value=running
            ) as ensure:
                record = manager.supervise(
                    worker_count=1,
                    queue_db_path=queue_db,
                    poll_interval_seconds=9.0,
                    batch_size=4,
                    claim_ttl_seconds=300,
                    reconcile_interval_seconds=15.0,
                    once=True,
                )

            ensure.assert_called_once_with(
                worker_count=1,
                queue_db_path=queue_db,
                poll_interval_seconds=9.0,
                batch_size=4,
                claim_ttl_seconds=300,
            )
            self.assertEqual(record.status, "running")

    def test_crawl_worker_service_manager_install_writes_task_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_db = Path(temp_dir) / ".agentos/research_state.sqlite3"
            service = CrawlWorkerServiceManager(temp_dir, python_executable="python")
            schtasks_result = subprocess.CompletedProcess(
                args=["schtasks"],
                returncode=0,
                stdout="ok",
                stderr="",
            )
            running = CrawlWorkerRecord(
                status="running",
                worker_pids=[111, 222],
                worker_count=2,
                queue_db_path=str(queue_db),
                log_paths=[
                    str(Path(temp_dir) / ".agentos/logs/crawl-worker-1.log"),
                    str(Path(temp_dir) / ".agentos/logs/crawl-worker-2.log"),
                ],
                detail="ok",
            )

            with (
                patch.object(service, "_is_supported", return_value=True),
                patch.object(
                    service,
                    "_task_exists",
                    return_value=True,
                ),
                patch.object(
                    service,
                    "_run_schtasks",
                    return_value=schtasks_result,
                ) as schtasks,
                patch(
                    "agentos_orchestrator.product.crawl_worker.CrawlWorkerManager.status",
                    return_value=running,
                ),
            ):
                record = service.install(
                    worker_count=2,
                    queue_db_path=queue_db,
                    poll_interval_seconds=9.0,
                    batch_size=4,
                    claim_ttl_seconds=300,
                    reconcile_interval_seconds=45.0,
                    task_name="AgentOS Test Crawl",
                    start_now=True,
                )

            self.assertEqual(record.status, "running")
            self.assertTrue(record.installed)
            self.assertEqual(record.task_name, "AgentOS Test Crawl")
            payload = json.loads(service.config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["worker_count"], 2)
            self.assertEqual(payload["queue_db_path"], str(queue_db))
            task_xml = service.task_xml_path.read_text(encoding="utf-16")
            self.assertIn("crawl-worker supervise", task_xml)
            self.assertIn(str(queue_db), task_xml)
            self.assertEqual(schtasks.call_count, 2)
            self.assertIn("/Create", schtasks.call_args_list[0].args[0])
            self.assertIn("/Run", schtasks.call_args_list[1].args[0])

    def test_crawl_worker_service_install_writes_broker_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_db = Path(temp_dir) / ".agentos/research_state.sqlite3"
            service = CrawlWorkerServiceManager(temp_dir, python_executable="python")
            schtasks_result = subprocess.CompletedProcess(
                args=["schtasks"],
                returncode=0,
                stdout="ok",
                stderr="",
            )

            with (
                patch.object(service, "_is_supported", return_value=True),
                patch.object(service, "_task_exists", return_value=True),
                patch.object(
                    service,
                    "_run_schtasks",
                    return_value=schtasks_result,
                ),
                patch(
                    "agentos_orchestrator.product.crawl_worker.CrawlWorkerManager.status",
                    return_value=CrawlWorkerRecord(
                        status="running",
                        worker_pids=[111],
                        worker_count=1,
                        queue_db_path=str(queue_db),
                        broker_url="http://127.0.0.1:8787",
                        broker_token_configured=True,
                        log_paths=[
                            str(Path(temp_dir) / ".agentos/logs/crawl-worker-1.log")
                        ],
                        detail="ok",
                    ),
                ),
            ):
                service.install(
                    worker_count=1,
                    queue_db_path=queue_db,
                    broker_url="http://127.0.0.1:8787",
                    broker_token="secret-token",
                    task_name="AgentOS Broker Crawl",
                    start_now=False,
                )

            task_xml = service.task_xml_path.read_text(encoding="utf-16")
            self.assertIn("--broker-url", task_xml)
            self.assertIn("http://127.0.0.1:8787", task_xml)
            self.assertIn("--broker-token", task_xml)

    def test_sdk_shapes_run_and_command_requests(self) -> None:
        client = FakeClient()

        run = client.start_run("topic", depth="quick")
        command = client.command("/run topic")

        self.assertEqual(run["path"], "/runs")
        self.assertEqual(
            client.calls[0][2],
            {
                "objective": "topic",
                "depth": "quick",
                "background": True,
            },
        )
        self.assertEqual(command["path"], "/channels/command")

        self.assertEqual(client.daemon_status()["path"], "/daemon/status")
        self.assertEqual(
            client.golden_traces()["path"],
            "/benchmarks/golden-traces",
        )
        self.assertEqual(client.eval_pack()["path"], "/benchmarks/eval-pack")
        self.assertEqual(client.replay_debug()["path"], "/debug/replay")

    def test_sdk_shapes_benchmark_and_pc_requests(self) -> None:
        client = FakeClient()

        self.assertEqual(
            client.live_fire_eval(
                max_tasks=1,
                windows_safe_pack=True,
                repeat=2,
            )["path"],
            "/benchmarks/live-fire-eval",
        )
        payload = cast(dict[str, Any], client.calls[-1][2])
        self.assertTrue(payload["windows_safe_pack"])
        self.assertEqual(payload["repeat"], 2)
        self.assertEqual(
            client.live_fire_review()["path"],
            "/benchmarks/live-fire-review?limit=10",
        )
        self.assertEqual(
            client.promote_live_fire_failure("run", "task")["path"],
            "/benchmarks/live-fire-review/promote",
        )
        self.assertEqual(
            client.live_fire_shadow_training(["trace.jsonl"])["path"],
            "/benchmarks/live-fire-shadow-training",
        )
        self.assertEqual(
            client.pc_debug_selector("name=AgentOS")["path"],
            "/pc/debug-selector",
        )
        self.assertEqual(
            client.pc_workflow_plan("write a report")["path"],
            "/pc/workflow/plan",
        )
        self.assertEqual(
            client.pc_workflow_execute("write a report")["path"],
            "/pc/workflow/execute",
        )

    def test_command_registry_merges_defaults_and_user_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "commands.json"
            registry = CommandRegistry(path)
            saved = registry.save(
                WorkflowCommand(
                    command_id="audit",
                    label="Audit",
                    description="Custom audit",
                )
            )

            self.assertEqual(saved.objective_for("topic"), "topic")
            self.assertIsNotNone(registry.get("quick-research"))
            self.assertIsNotNone(registry.get("research"))
            loaded = registry.get("/audit")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.command_id, "audit")


if __name__ == "__main__":
    unittest.main()
