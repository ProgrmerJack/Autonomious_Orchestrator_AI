from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path

from .types import JsonObject, utc_now


@dataclass(slots=True)
class Checkpoint:
    run_id: str
    stage: str
    state: JsonObject
    updated_at: str


class CheckpointStore:
    """Durable run state used for crash recovery and context-limit resume."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_schema(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        run_id TEXT PRIMARY KEY,
                        stage TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )

    def save(self, run_id: str, stage: str, state: JsonObject) -> Checkpoint:
        checkpoint = Checkpoint(
            run_id=run_id,
            stage=stage,
            state=state,
            updated_at=utc_now(),
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO checkpoints(
                        run_id,
                        stage,
                        state_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        stage = excluded.stage,
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        run_id,
                        stage,
                        json.dumps(state, sort_keys=True),
                        checkpoint.updated_at,
                    ),
                )
        return checkpoint

    def load(self, run_id: str) -> Checkpoint | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT run_id, stage, state_json, updated_at
                FROM checkpoints
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            run_id=row[0],
            stage=row[1],
            state=json.loads(row[2]),
            updated_at=row[3],
        )
