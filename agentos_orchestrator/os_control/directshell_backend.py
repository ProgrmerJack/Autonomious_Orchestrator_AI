from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .base import UiAction, UiNode


class DirectShellBackend:
    """SQLite bridge compatible with a DirectShell-style Rust body."""

    name = "directshell"

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def init_schema(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ui_nodes (
                        node_id TEXT PRIMARY KEY,
                        role TEXT NOT NULL,
                        name TEXT NOT NULL,
                        x INTEGER,
                        y INTEGER,
                        width INTEGER,
                        height INTEGER,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        focused INTEGER NOT NULL DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS actions (
                        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action_type TEXT NOT NULL,
                        selector TEXT NOT NULL,
                        value TEXT,
                        metadata_json TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'queued',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def available(self) -> bool:
        return self.db_path.exists()

    def snapshot(self) -> list[UiNode]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    node_id,
                    role,
                    name,
                    x,
                    y,
                    width,
                    height,
                    enabled,
                    focused,
                    metadata_json
                FROM ui_nodes
                ORDER BY updated_at DESC, node_id ASC
                """
            ).fetchall()
        nodes: list[UiNode] = []
        for row in rows:
            bounds = None
            if None not in row[3:7]:
                bounds = (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
            nodes.append(
                UiNode(
                    node_id=row[0],
                    role=row[1],
                    name=row[2],
                    bounds=bounds,
                    enabled=bool(row[7]),
                    focused=bool(row[8]),
                    metadata=json.loads(row[9]),
                )
            )
        return nodes

    def perform(self, action: UiAction) -> str:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO actions(
                        action_type,
                        selector,
                        value,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        action.action_type,
                        action.selector,
                        action.value,
                        json.dumps(action.metadata, sort_keys=True),
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("action insert did not return an id")
                return str(cursor.lastrowid)
