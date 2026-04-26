from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import JsonObject, utc_now


@dataclass(slots=True)
class WorkflowStepRecord:
    run_id: str
    step_id: str
    kind: str
    status: str
    input: JsonObject
    result: JsonObject | None
    attempts: int
    error: str | None
    updated_at: str


@dataclass(slots=True)
class WorkflowRunRecord:
    run_id: str
    objective: str
    tasks: list[JsonObject]
    status: str
    created_at: str
    updated_at: str


class DurableExecutionStore:
    """Replayable workflow state for automatic crash recovery."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_schema(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_runs (
                        run_id TEXT PRIMARY KEY,
                        objective TEXT NOT NULL,
                        tasks_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_steps (
                        run_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        input_json TEXT NOT NULL,
                        result_json TEXT,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(run_id, step_id),
                        FOREIGN KEY(run_id)
                            REFERENCES workflow_runs(run_id)
                            ON DELETE CASCADE
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_transitions (
                        transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

    def save_manifest(
        self,
        run_id: str,
        objective: str,
        tasks: list[JsonObject],
        status: str = "running",
    ) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO workflow_runs(
                        run_id,
                        objective,
                        tasks_json,
                        status,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        objective = excluded.objective,
                        tasks_json = excluded.tasks_json,
                        status = excluded.status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        run_id,
                        objective,
                        json.dumps(tasks, sort_keys=True),
                        status,
                        now,
                        now,
                    ),
                )

    def complete_run(self, run_id: str) -> None:
        self._update_run_status(run_id, "completed")

    def fail_run(self, run_id: str) -> None:
        self._update_run_status(run_id, "failed")

    def load_manifest(self, run_id: str) -> JsonObject | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT objective, tasks_json, status, updated_at
                FROM workflow_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "objective": row[0],
            "tasks": json.loads(row[1]),
            "status": row[2],
            "updated_at": row[3],
        }

    def list_runs(
        self,
        status: str | None = None,
    ) -> list[WorkflowRunRecord]:
        query = (
            "SELECT run_id, objective, tasks_json, status, "
            "created_at, updated_at FROM workflow_runs"
        )
        parameters: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            parameters.append(status)
        query += " ORDER BY updated_at ASC"

        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            WorkflowRunRecord(
                run_id=row[0],
                objective=row[1],
                tasks=json.loads(row[2]),
                status=row[3],
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        ]

    def run_json_step(
        self,
        run_id: str,
        step_id: str,
        kind: str,
        input_payload: JsonObject,
        operation: Callable[[], JsonObject],
    ) -> JsonObject:
        existing = self.get_step(run_id, step_id)
        if existing and existing.status == "completed":
            return existing.result or {}

        self.record_transition(
            run_id,
            step_id,
            kind,
            "running",
            input_payload,
        )
        try:
            result = operation()
        except Exception as exc:
            self.record_transition(
                run_id,
                step_id,
                kind,
                "failed",
                input_payload,
                error=str(exc),
            )
            self.fail_run(run_id)
            raise

        self.record_transition(
            run_id,
            step_id,
            kind,
            "completed",
            input_payload,
            result=result,
        )
        return result

    def record_transition(
        self,
        run_id: str,
        step_id: str,
        kind: str,
        status: str,
        input_payload: JsonObject,
        result: JsonObject | None = None,
        error: str | None = None,
    ) -> WorkflowStepRecord:
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                current = connection.execute(
                    """
                    SELECT attempts FROM workflow_steps
                    WHERE run_id = ? AND step_id = ?
                    """,
                    (run_id, step_id),
                ).fetchone()
                attempts = int(current[0]) if current else 0
                if status == "running":
                    attempts += 1
                connection.execute(
                    """
                    INSERT INTO workflow_steps(
                        run_id,
                        step_id,
                        kind,
                        status,
                        input_json,
                        result_json,
                        attempts,
                        error,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, step_id) DO UPDATE SET
                        kind = excluded.kind,
                        status = excluded.status,
                        input_json = excluded.input_json,
                        result_json = excluded.result_json,
                        attempts = excluded.attempts,
                        error = excluded.error,
                        updated_at = excluded.updated_at
                    """,
                    (
                        run_id,
                        step_id,
                        kind,
                        status,
                        json.dumps(input_payload, sort_keys=True),
                        self._json_or_none(result),
                        attempts,
                        error,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO workflow_transitions(
                        run_id,
                        step_id,
                        status,
                        payload_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        step_id,
                        status,
                        json.dumps(
                            {
                                "input": input_payload,
                                "result": result,
                                "error": error,
                            },
                            sort_keys=True,
                        ),
                        now,
                    ),
                )
        return WorkflowStepRecord(
            run_id=run_id,
            step_id=step_id,
            kind=kind,
            status=status,
            input=input_payload,
            result=result,
            attempts=attempts,
            error=error,
            updated_at=now,
        )

    def get_step(
        self,
        run_id: str,
        step_id: str,
    ) -> WorkflowStepRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    kind,
                    status,
                    input_json,
                    result_json,
                    attempts,
                    error,
                    updated_at
                FROM workflow_steps
                WHERE run_id = ? AND step_id = ?
                """,
                (run_id, step_id),
            ).fetchone()
        if row is None:
            return None
        return WorkflowStepRecord(
            run_id=run_id,
            step_id=step_id,
            kind=row[0],
            status=row[1],
            input=json.loads(row[2]),
            result=self._loads_or_none(row[3]),
            attempts=int(row[4]),
            error=row[5],
            updated_at=row[6],
        )

    def list_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    step_id,
                    kind,
                    status,
                    input_json,
                    result_json,
                    attempts,
                    error,
                    updated_at
                FROM workflow_steps
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            WorkflowStepRecord(
                run_id=run_id,
                step_id=row[0],
                kind=row[1],
                status=row[2],
                input=json.loads(row[3]),
                result=self._loads_or_none(row[4]),
                attempts=int(row[5]),
                error=row[6],
                updated_at=row[7],
            )
            for row in rows
        ]

    def recover_stale_steps(self, run_id: str) -> int:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE workflow_steps
                    SET status = 'recovering', updated_at = ?
                    WHERE run_id = ? AND status = 'running'
                    """,
                    (utc_now(), run_id),
                )
                return cursor.rowcount

    def _update_run_status(self, run_id: str, status: str) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE workflow_runs
                    SET status = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (status, utc_now(), run_id),
                )

    @staticmethod
    def _json_or_none(value: JsonObject | None) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _loads_or_none(value: str | bytes | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return json.loads(value)
