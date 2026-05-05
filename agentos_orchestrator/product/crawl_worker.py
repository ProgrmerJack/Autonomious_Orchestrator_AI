from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Any

from agentos_orchestrator.core.types import utc_now


@dataclass(slots=True)
class CrawlWorkerRecord:
    status: str
    worker_pids: list[int]
    worker_count: int
    queue_db_path: str
    log_paths: list[str]
    broker_url: str = ""
    broker_token_configured: bool = False
    started_at: str | None = None
    poll_interval_seconds: float = 0.0
    batch_size: int = 0
    claim_ttl_seconds: int = 0
    allow_js_required: bool = True
    prefer_js_required: bool = False
    max_claims_per_domain: int = 0
    default_domain_cooldown_seconds: float = 0.0
    js_domain_cooldown_seconds: float = 0.0
    detail: str = ""


@dataclass(slots=True)
class CrawlWorkerServiceRecord:
    status: str
    task_name: str
    supported: bool
    installed: bool
    backend: str
    workspace_root: str
    config_path: str
    worker_count: int
    queue_db_path: str
    broker_url: str = ""
    broker_token_configured: bool = False
    poll_interval_seconds: float = 0.0
    batch_size: int = 0
    claim_ttl_seconds: int = 0
    allow_js_required: bool = True
    prefer_js_required: bool = False
    max_claims_per_domain: int = 0
    default_domain_cooldown_seconds: float = 0.0
    js_domain_cooldown_seconds: float = 0.0
    reconcile_interval_seconds: float = 0.0
    detail: str = ""


class CrawlWorkerManager:
    def __init__(
        self,
        workspace_root: str | Path = ".",
        python_executable: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.python_executable = python_executable or sys.executable
        self.agent_dir = self.workspace_root / ".agentos"
        self.state_path = self.agent_dir / "crawl_worker.json"
        self.service_state_path = self.agent_dir / "crawl_worker_service.json"
        self.log_dir = self.agent_dir / "logs"

    def status(self) -> CrawlWorkerRecord:
        payload = self._read_state()
        if payload is None:
            return CrawlWorkerRecord(
                status="stopped",
                worker_pids=[],
                worker_count=0,
                queue_db_path=str(
                    self.workspace_root / ".agentos" / "research_state.sqlite3"
                ),
                broker_url="",
                broker_token_configured=False,
                log_paths=[str(self.log_dir / "crawl_worker_1.log")],
            )
        worker_pids = [
            int(value)
            for value in list(payload.get("worker_pids") or [])
            if _int_or_none(value) is not None
        ]
        running_pids = [pid for pid in worker_pids if _process_running(pid)]
        if worker_pids and len(running_pids) == len(worker_pids):
            status = "running"
            detail = "all crawl workers are alive"
        elif running_pids:
            status = "degraded"
            detail = "some crawl workers are alive"
        else:
            status = "stale"
            detail = "state exists only"
        return CrawlWorkerRecord(
            status=status,
            worker_pids=worker_pids,
            worker_count=int(payload.get("worker_count") or len(worker_pids) or 0),
            queue_db_path=str(payload.get("queue_db_path") or ""),
            broker_url=str(payload.get("broker_url") or ""),
            broker_token_configured=bool(payload.get("broker_token") or ""),
            log_paths=[str(path) for path in list(payload.get("log_paths") or [])],
            started_at=payload.get("started_at"),
            poll_interval_seconds=float(payload.get("poll_interval_seconds") or 0.0),
            batch_size=int(payload.get("batch_size") or 0),
            claim_ttl_seconds=int(payload.get("claim_ttl_seconds") or 0),
            allow_js_required=bool(payload.get("allow_js_required", True)),
            prefer_js_required=bool(payload.get("prefer_js_required", False)),
            max_claims_per_domain=int(payload.get("max_claims_per_domain") or 0),
            default_domain_cooldown_seconds=float(
                payload.get("default_domain_cooldown_seconds") or 0.0
            ),
            js_domain_cooldown_seconds=float(
                payload.get("js_domain_cooldown_seconds") or 0.0
            ),
            detail=detail,
        )

    def start(
        self,
        worker_count: int = 1,
        queue_db_path: str | Path | None = None,
        broker_url: str | None = None,
        broker_token: str | None = None,
        poll_interval_seconds: float = 15.0,
        batch_size: int = 6,
        claim_ttl_seconds: int = 900,
        allow_js_required: bool = True,
        prefer_js_required: bool = False,
        max_claims_per_domain: int = 2,
        default_domain_cooldown_seconds: float = 2.0,
        js_domain_cooldown_seconds: float = 8.0,
    ) -> CrawlWorkerRecord:
        desired_count = max(1, min(int(worker_count), 8))
        queue_path = (
            Path(queue_db_path)
            if queue_db_path is not None
            else (self.workspace_root / ".agentos" / "research_state.sqlite3")
        )
        resolved_broker_url = str(broker_url or "").strip()
        resolved_broker_token = str(broker_token or "")
        state_payload = self._read_state() or {}
        current = self.status()
        if (
            current.status == "running"
            and current.worker_count == desired_count
            and Path(current.queue_db_path) == queue_path
            and current.broker_url == resolved_broker_url
            and str(state_payload.get("broker_token") or "") == resolved_broker_token
            and current.allow_js_required == bool(allow_js_required)
            and current.prefer_js_required == bool(prefer_js_required)
            and current.max_claims_per_domain == int(max_claims_per_domain)
            and current.default_domain_cooldown_seconds
            == float(default_domain_cooldown_seconds)
            and current.js_domain_cooldown_seconds == float(js_domain_cooldown_seconds)
        ):
            return current
        if current.status in {"running", "degraded", "stale"}:
            self.stop()
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        worker_pids: list[int] = []
        log_paths: list[str] = []
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        for index in range(desired_count):
            worker_label = f"crawl-worker-{index + 1}"
            log_path = self.log_dir / f"{worker_label}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                self.python_executable,
                "-m",
                "agentos_orchestrator",
                "crawl-worker",
                "run",
                "--workspace-root",
                str(self.workspace_root),
                "--queue-db",
                str(queue_path),
                "--worker-id",
                worker_label,
                "--poll-interval",
                str(poll_interval_seconds),
                "--batch-size",
                str(batch_size),
                "--claim-ttl",
                str(claim_ttl_seconds),
            ]
            if resolved_broker_url:
                command.extend(["--broker-url", resolved_broker_url])
            if resolved_broker_token:
                command.extend(["--broker-token", resolved_broker_token])
            if prefer_js_required:
                command.append("--prefer-js")
            if not allow_js_required:
                command.append("--no-js")
            command.extend(
                [
                    "--max-claims-per-domain",
                    str(max(1, int(max_claims_per_domain))),
                    "--domain-cooldown",
                    str(max(0.0, float(default_domain_cooldown_seconds))),
                    "--js-domain-cooldown",
                    str(max(0.0, float(js_domain_cooldown_seconds))),
                ]
            )
            env = os.environ.copy()
            env["AGENTOS_CRAWL_WORKER_PROCESS"] = "1"
            env["AGENTOS_DISABLE_AUTO_CRAWL_WORKERS"] = "1"
            process = subprocess.Popen(
                command,
                cwd=self.workspace_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                env=env,
            )
            log_handle.close()
            worker_pids.append(process.pid)
            log_paths.append(str(log_path))
        record = CrawlWorkerRecord(
            status="running",
            worker_pids=worker_pids,
            worker_count=desired_count,
            queue_db_path=str(queue_path),
            broker_url=resolved_broker_url,
            broker_token_configured=bool(resolved_broker_token),
            log_paths=log_paths,
            started_at=utc_now(),
            poll_interval_seconds=float(poll_interval_seconds),
            batch_size=int(batch_size),
            claim_ttl_seconds=int(claim_ttl_seconds),
            allow_js_required=bool(allow_js_required),
            prefer_js_required=bool(prefer_js_required),
            max_claims_per_domain=max(1, int(max_claims_per_domain)),
            default_domain_cooldown_seconds=max(
                0.0,
                float(default_domain_cooldown_seconds),
            ),
            js_domain_cooldown_seconds=max(0.0, float(js_domain_cooldown_seconds)),
            detail="detached crawl workers started",
        )
        payload = asdict(record)
        payload["broker_token"] = resolved_broker_token
        self._write_state(payload)
        return record

    def ensure_running(
        self,
        worker_count: int = 1,
        queue_db_path: str | Path | None = None,
        broker_url: str | None = None,
        broker_token: str | None = None,
        poll_interval_seconds: float = 15.0,
        batch_size: int = 6,
        claim_ttl_seconds: int = 900,
        allow_js_required: bool = True,
        prefer_js_required: bool = False,
        max_claims_per_domain: int = 2,
        default_domain_cooldown_seconds: float = 2.0,
        js_domain_cooldown_seconds: float = 8.0,
    ) -> CrawlWorkerRecord:
        desired_count = max(1, min(int(worker_count), 8))
        queue_path = (
            Path(queue_db_path)
            if queue_db_path is not None
            else (self.workspace_root / ".agentos" / "research_state.sqlite3")
        )
        current = self.status()
        if (
            current.status == "running"
            and current.worker_count == desired_count
            and Path(current.queue_db_path) == queue_path
            and current.broker_url == str(broker_url or "").strip()
            and current.allow_js_required == bool(allow_js_required)
            and current.prefer_js_required == bool(prefer_js_required)
            and current.max_claims_per_domain == int(max_claims_per_domain)
            and current.default_domain_cooldown_seconds
            == float(default_domain_cooldown_seconds)
            and current.js_domain_cooldown_seconds == float(js_domain_cooldown_seconds)
        ):
            return current
        return self.start(
            worker_count=desired_count,
            queue_db_path=queue_path,
            broker_url=broker_url,
            broker_token=broker_token,
            poll_interval_seconds=poll_interval_seconds,
            batch_size=batch_size,
            claim_ttl_seconds=claim_ttl_seconds,
            allow_js_required=allow_js_required,
            prefer_js_required=prefer_js_required,
            max_claims_per_domain=max_claims_per_domain,
            default_domain_cooldown_seconds=default_domain_cooldown_seconds,
            js_domain_cooldown_seconds=js_domain_cooldown_seconds,
        )

    def supervise(
        self,
        worker_count: int = 1,
        queue_db_path: str | Path | None = None,
        broker_url: str | None = None,
        broker_token: str | None = None,
        poll_interval_seconds: float = 15.0,
        batch_size: int = 6,
        claim_ttl_seconds: int = 900,
        allow_js_required: bool = True,
        prefer_js_required: bool = False,
        max_claims_per_domain: int = 2,
        default_domain_cooldown_seconds: float = 2.0,
        js_domain_cooldown_seconds: float = 8.0,
        reconcile_interval_seconds: float = 30.0,
        once: bool = False,
    ) -> CrawlWorkerRecord:
        interval = max(5.0, min(float(reconcile_interval_seconds), 300.0))
        while True:
            ensure_kwargs: dict[str, Any] = {
                "worker_count": worker_count,
                "queue_db_path": queue_db_path,
                "poll_interval_seconds": poll_interval_seconds,
                "batch_size": batch_size,
                "claim_ttl_seconds": claim_ttl_seconds,
                "allow_js_required": allow_js_required,
                "prefer_js_required": prefer_js_required,
                "max_claims_per_domain": max_claims_per_domain,
                "default_domain_cooldown_seconds": default_domain_cooldown_seconds,
                "js_domain_cooldown_seconds": js_domain_cooldown_seconds,
            }
            if broker_url:
                ensure_kwargs["broker_url"] = broker_url
            if broker_token:
                ensure_kwargs["broker_token"] = broker_token
            record = self.ensure_running(**ensure_kwargs)
            if once:
                return record
            time.sleep(interval)

    def stop(self) -> CrawlWorkerRecord:
        current = self.status()
        for pid in current.worker_pids:
            _terminate_tree(pid)
        if self.state_path.exists():
            self.state_path.unlink()
        return CrawlWorkerRecord(
            status="stopped",
            worker_pids=[],
            worker_count=0,
            queue_db_path=current.queue_db_path,
            broker_url=current.broker_url,
            broker_token_configured=current.broker_token_configured,
            log_paths=current.log_paths,
            detail="crawl workers stopped",
        )

    def _read_state(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_state(self, payload: dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


class CrawlWorkerServiceManager:
    def __init__(
        self,
        workspace_root: str | Path = ".",
        python_executable: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.python_executable = python_executable or sys.executable
        self.agent_dir = self.workspace_root / ".agentos"
        self.config_path = self.agent_dir / "crawl_worker_service.json"
        self.task_xml_path = self.agent_dir / "crawl_worker_service.xml"

    def default_task_name(self) -> str:
        label = self.workspace_root.name.strip() or "workspace"
        safe = "".join(
            char if char.isalnum() or char in {"-", "_", " ", "."} else "-"
            for char in label
        )
        compact = " ".join(safe.split())[:80].strip(" -") or "workspace"
        return f"AgentOS Crawl Workers - {compact}"

    def status(
        self,
        task_name: str | None = None,
    ) -> CrawlWorkerServiceRecord:
        payload = self._read_config()
        configured_task_name = str(payload.get("task_name") or "").strip()
        resolved_task_name = str(
            task_name or configured_task_name or self.default_task_name()
        ).strip()
        queue_db_path = self._resolve_queue_db_path(
            str(payload.get("queue_db_path") or "")
        )
        broker_url = str(payload.get("broker_url") or "")
        worker_count = max(0, int(payload.get("worker_count") or 0))
        poll_interval_seconds = float(payload.get("poll_interval_seconds") or 0.0)
        batch_size = int(payload.get("batch_size") or 0)
        claim_ttl_seconds = int(payload.get("claim_ttl_seconds") or 0)
        allow_js_required = bool(payload.get("allow_js_required", True))
        prefer_js_required = bool(payload.get("prefer_js_required", False))
        max_claims_per_domain = int(payload.get("max_claims_per_domain") or 0)
        default_domain_cooldown_seconds = float(
            payload.get("default_domain_cooldown_seconds") or 0.0
        )
        js_domain_cooldown_seconds = float(
            payload.get("js_domain_cooldown_seconds") or 0.0
        )
        reconcile_interval_seconds = float(
            payload.get("reconcile_interval_seconds") or 0.0
        )
        supported = self._is_supported()
        installed = supported and self._task_exists(resolved_task_name)
        worker_status = CrawlWorkerManager(
            self.workspace_root,
            python_executable=self.python_executable,
        ).status()
        if not supported:
            status = "unsupported"
            detail = "crawl worker service wrapper is only implemented for Windows"
        elif installed and worker_status.status == "running":
            status = "running"
            detail = "scheduled task installed and supervising live crawl workers"
        elif installed and worker_status.status in {"degraded", "stale"}:
            status = "degraded"
            detail = "scheduled task installed but crawl worker pool is degraded"
        elif installed:
            status = "installed"
            detail = "scheduled task installed and ready to supervise crawl workers"
        else:
            status = "stopped"
            detail = "scheduled task is not installed"
        return CrawlWorkerServiceRecord(
            status=status,
            task_name=resolved_task_name,
            supported=supported,
            installed=installed,
            backend="windows-task-scheduler" if supported else "unsupported",
            workspace_root=str(self.workspace_root),
            config_path=str(self.config_path),
            worker_count=worker_count,
            queue_db_path=str(queue_db_path),
            broker_url=broker_url,
            broker_token_configured=bool(payload.get("broker_token") or ""),
            poll_interval_seconds=poll_interval_seconds,
            batch_size=batch_size,
            claim_ttl_seconds=claim_ttl_seconds,
            allow_js_required=allow_js_required,
            prefer_js_required=prefer_js_required,
            max_claims_per_domain=max_claims_per_domain,
            default_domain_cooldown_seconds=default_domain_cooldown_seconds,
            js_domain_cooldown_seconds=js_domain_cooldown_seconds,
            reconcile_interval_seconds=reconcile_interval_seconds,
            detail=detail,
        )

    def install(
        self,
        worker_count: int = 1,
        queue_db_path: str | Path | None = None,
        broker_url: str | None = None,
        broker_token: str | None = None,
        poll_interval_seconds: float = 15.0,
        batch_size: int = 6,
        claim_ttl_seconds: int = 900,
        allow_js_required: bool = True,
        prefer_js_required: bool = False,
        max_claims_per_domain: int = 2,
        default_domain_cooldown_seconds: float = 2.0,
        js_domain_cooldown_seconds: float = 8.0,
        reconcile_interval_seconds: float = 30.0,
        task_name: str | None = None,
        start_now: bool = True,
    ) -> CrawlWorkerServiceRecord:
        if not self._is_supported():
            return self.status(task_name=task_name)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        resolved_task_name = str(task_name or self.default_task_name()).strip()
        queue_path = self._resolve_queue_db_path(queue_db_path)
        payload = {
            "task_name": resolved_task_name,
            "worker_count": max(1, min(int(worker_count), 8)),
            "queue_db_path": str(queue_path),
            "broker_url": str(broker_url or "").strip(),
            "broker_token": str(broker_token or ""),
            "poll_interval_seconds": float(poll_interval_seconds),
            "batch_size": max(1, min(int(batch_size), 24)),
            "claim_ttl_seconds": max(60, min(int(claim_ttl_seconds), 7_200)),
            "allow_js_required": bool(allow_js_required),
            "prefer_js_required": bool(prefer_js_required),
            "max_claims_per_domain": max(1, int(max_claims_per_domain)),
            "default_domain_cooldown_seconds": max(
                0.0,
                float(default_domain_cooldown_seconds),
            ),
            "js_domain_cooldown_seconds": max(
                0.0,
                float(js_domain_cooldown_seconds),
            ),
            "reconcile_interval_seconds": max(
                5.0,
                min(float(reconcile_interval_seconds), 300.0),
            ),
            "workspace_root": str(self.workspace_root),
        }
        self._write_config(payload)
        self._write_task_xml(payload)
        create_result = self._run_schtasks(
            [
                "/Create",
                "/TN",
                resolved_task_name,
                "/XML",
                str(self.task_xml_path),
                "/F",
            ]
        )
        if create_result.returncode != 0:
            raise RuntimeError(self._command_error("install", create_result))
        if start_now:
            self.start(task_name=resolved_task_name)
        return self.status(task_name=resolved_task_name)

    def start(
        self,
        task_name: str | None = None,
    ) -> CrawlWorkerServiceRecord:
        if not self._is_supported():
            return self.status(task_name=task_name)
        resolved_task_name = self._resolve_task_name(task_name)
        result = self._run_schtasks(
            [
                "/Run",
                "/TN",
                resolved_task_name,
            ]
        )
        if result.returncode != 0:
            output = self._command_error("start", result)
            if "already running" not in output.lower():
                raise RuntimeError(output)
        return self.status(task_name=resolved_task_name)

    def stop(
        self,
        task_name: str | None = None,
    ) -> CrawlWorkerServiceRecord:
        resolved_task_name = self._resolve_task_name(task_name)
        if self._is_supported() and self._task_exists(resolved_task_name):
            result = self._run_schtasks(
                [
                    "/End",
                    "/TN",
                    resolved_task_name,
                ]
            )
            if result.returncode != 0:
                output = self._command_error("stop", result)
                if "there is no running instance" not in output.lower():
                    raise RuntimeError(output)
        CrawlWorkerManager(
            self.workspace_root,
            python_executable=self.python_executable,
        ).stop()
        return self.status(task_name=resolved_task_name)

    def restart(
        self,
        task_name: str | None = None,
    ) -> CrawlWorkerServiceRecord:
        self.stop(task_name=task_name)
        return self.start(task_name=task_name)

    def uninstall(
        self,
        task_name: str | None = None,
    ) -> CrawlWorkerServiceRecord:
        resolved_task_name = self._resolve_task_name(task_name)
        if self._is_supported() and self._task_exists(resolved_task_name):
            self.stop(task_name=resolved_task_name)
            result = self._run_schtasks(
                [
                    "/Delete",
                    "/TN",
                    resolved_task_name,
                    "/F",
                ]
            )
            if result.returncode != 0:
                raise RuntimeError(self._command_error("uninstall", result))
        if self.config_path.exists():
            self.config_path.unlink()
        if self.task_xml_path.exists():
            self.task_xml_path.unlink()
        return self.status(task_name=resolved_task_name)

    def _is_supported(self) -> bool:
        return os.name == "nt"

    def _resolve_task_name(self, task_name: str | None) -> str:
        payload = self._read_config()
        return str(
            task_name or payload.get("task_name") or self.default_task_name()
        ).strip()

    def _resolve_queue_db_path(
        self,
        queue_db_path: str | Path | None,
    ) -> Path:
        if queue_db_path is None:
            configured = ""
        else:
            configured = str(queue_db_path).strip()
        if not configured:
            return self.workspace_root / ".agentos" / "research_state.sqlite3"
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        return candidate

    def _read_config(self) -> dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_config(self, payload: dict[str, Any]) -> None:
        self.config_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _task_exists(self, task_name: str) -> bool:
        result = self._run_schtasks(
            [
                "/Query",
                "/TN",
                task_name,
            ]
        )
        return result.returncode == 0

    def _write_task_xml(self, payload: dict[str, Any]) -> None:
        namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
        ET.register_namespace("", namespace)
        task = ET.Element(f"{{{namespace}}}Task", version="1.3")

        registration = ET.SubElement(task, f"{{{namespace}}}RegistrationInfo")
        ET.SubElement(registration, f"{{{namespace}}}Author").text = "AgentOS"
        ET.SubElement(registration, f"{{{namespace}}}Date").text = utc_now()
        ET.SubElement(registration, f"{{{namespace}}}Description").text = (
            "Supervise AgentOS crawl workers so the research queue resumes "
            "after reboot and task failure."
        )
        ET.SubElement(
            registration, f"{{{namespace}}}URI"
        ).text = f"\\AgentOS\\{payload['task_name']}"

        principals = ET.SubElement(task, f"{{{namespace}}}Principals")
        principal = ET.SubElement(
            principals,
            f"{{{namespace}}}Principal",
            id="Author",
        )
        ET.SubElement(principal, f"{{{namespace}}}UserId").text = "S-1-5-18"
        ET.SubElement(principal, f"{{{namespace}}}LogonType").text = "ServiceAccount"
        ET.SubElement(principal, f"{{{namespace}}}RunLevel").text = "HighestAvailable"

        triggers = ET.SubElement(task, f"{{{namespace}}}Triggers")
        boot_trigger = ET.SubElement(triggers, f"{{{namespace}}}BootTrigger")
        ET.SubElement(boot_trigger, f"{{{namespace}}}Enabled").text = "true"
        logon_trigger = ET.SubElement(triggers, f"{{{namespace}}}LogonTrigger")
        ET.SubElement(logon_trigger, f"{{{namespace}}}Enabled").text = "true"

        settings = ET.SubElement(task, f"{{{namespace}}}Settings")
        ET.SubElement(
            settings,
            f"{{{namespace}}}MultipleInstancesPolicy",
        ).text = "IgnoreNew"
        ET.SubElement(
            settings,
            f"{{{namespace}}}DisallowStartIfOnBatteries",
        ).text = "false"
        ET.SubElement(
            settings,
            f"{{{namespace}}}StopIfGoingOnBatteries",
        ).text = "false"
        ET.SubElement(settings, f"{{{namespace}}}AllowHardTerminate").text = "true"
        ET.SubElement(settings, f"{{{namespace}}}StartWhenAvailable").text = "true"
        ET.SubElement(
            settings,
            f"{{{namespace}}}RunOnlyIfNetworkAvailable",
        ).text = "false"
        idle = ET.SubElement(settings, f"{{{namespace}}}IdleSettings")
        ET.SubElement(idle, f"{{{namespace}}}StopOnIdleEnd").text = "false"
        ET.SubElement(idle, f"{{{namespace}}}RestartOnIdle").text = "false"
        ET.SubElement(settings, f"{{{namespace}}}AllowStartOnDemand").text = "true"
        ET.SubElement(settings, f"{{{namespace}}}Enabled").text = "true"
        ET.SubElement(settings, f"{{{namespace}}}Hidden").text = "false"
        ET.SubElement(settings, f"{{{namespace}}}RunOnlyIfIdle").text = "false"
        ET.SubElement(settings, f"{{{namespace}}}WakeToRun").text = "false"
        ET.SubElement(settings, f"{{{namespace}}}ExecutionTimeLimit").text = "PT0S"
        ET.SubElement(settings, f"{{{namespace}}}Priority").text = "5"
        restart = ET.SubElement(settings, f"{{{namespace}}}RestartOnFailure")
        ET.SubElement(restart, f"{{{namespace}}}Interval").text = "PT1M"
        ET.SubElement(restart, f"{{{namespace}}}Count").text = "255"

        argument_list = [
            "-m",
            "agentos_orchestrator",
            "crawl-worker",
            "supervise",
            "--workspace-root",
            str(self.workspace_root),
            "--queue-db",
            str(payload["queue_db_path"]),
            "--workers",
            str(payload["worker_count"]),
            "--poll-interval",
            str(payload["poll_interval_seconds"]),
            "--batch-size",
            str(payload["batch_size"]),
            "--claim-ttl",
            str(payload["claim_ttl_seconds"]),
            "--supervisor-interval",
            str(payload["reconcile_interval_seconds"]),
            "--max-claims-per-domain",
            str(payload.get("max_claims_per_domain") or 2),
            "--domain-cooldown",
            str(payload.get("default_domain_cooldown_seconds") or 0.0),
            "--js-domain-cooldown",
            str(payload.get("js_domain_cooldown_seconds") or 0.0),
        ]
        broker_url = str(payload.get("broker_url") or "").strip()
        broker_token = str(payload.get("broker_token") or "")
        if broker_url:
            argument_list.extend(["--broker-url", broker_url])
        if broker_token:
            argument_list.extend(["--broker-token", broker_token])
        if bool(payload.get("prefer_js_required", False)):
            argument_list.append("--prefer-js")
        if not bool(payload.get("allow_js_required", True)):
            argument_list.append("--no-js")
        arguments = subprocess.list2cmdline(argument_list)
        actions = ET.SubElement(task, f"{{{namespace}}}Actions", Context="Author")
        exec_action = ET.SubElement(actions, f"{{{namespace}}}Exec")
        ET.SubElement(exec_action, f"{{{namespace}}}Command").text = str(
            Path(self.python_executable).resolve()
        )
        ET.SubElement(exec_action, f"{{{namespace}}}Arguments").text = arguments
        ET.SubElement(exec_action, f"{{{namespace}}}WorkingDirectory").text = str(
            self.workspace_root
        )

        tree = ET.ElementTree(task)
        tree.write(
            self.task_xml_path,
            encoding="utf-16",
            xml_declaration=True,
        )

    @staticmethod
    def _command_error(action: str, result: subprocess.CompletedProcess[str]) -> str:
        detail = " ".join(
            part.strip()
            for part in [str(result.stdout or ""), str(result.stderr or "")]
            if part.strip()
        )
        if detail:
            return f"Failed to {action} crawl worker service: {detail}"
        return f"Failed to {action} crawl worker service"

    @staticmethod
    def _run_schtasks(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["schtasks", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    killpg = getattr(os, "killpg", None)
    try:
        if killpg is not None:
            killpg(pid, signal.SIGTERM)
            return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
