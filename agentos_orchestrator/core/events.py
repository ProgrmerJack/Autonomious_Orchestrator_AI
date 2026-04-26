from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from .types import Event, JsonObject

EventHandler = Callable[[Event], None]


class DurableEventLog:
    """SQLite-backed append-only event log."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
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
                    CREATE TABLE IF NOT EXISTS events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        run_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_run_sequence
                ON events(run_id, sequence)
                """
            )

    def append(self, event: Event) -> Event:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO events(
                        event_id,
                        run_id,
                        type,
                        source,
                        payload_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.run_id,
                        event.type,
                        event.source,
                        json.dumps(event.payload, sort_keys=True),
                        event.created_at,
                    ),
                )
                if cursor.lastrowid is None:
                    message = "event insert did not return a sequence"
                    raise RuntimeError(message)
                event.sequence = int(cursor.lastrowid)
                return event

    def list_events(
        self,
        run_id: str | None = None,
        after_sequence: int = 0,
    ) -> list[Event]:
        query = (
            "SELECT sequence, event_id, run_id, type, source, "
            "payload_json, created_at "
            "FROM events WHERE sequence > ?"
        )
        parameters: list[object] = [after_sequence]
        if run_id is not None:
            query += " AND run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY sequence ASC"

        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()

        events: list[Event] = []
        for row in rows:
            (
                sequence,
                event_id,
                row_run_id,
                event_type,
                source,
                payload_json,
                created_at,
            ) = row
            events.append(
                Event(
                    run_id=row_run_id,
                    type=event_type,
                    source=source,
                    payload=json.loads(payload_json),
                    event_id=event_id,
                    created_at=created_at,
                    sequence=sequence,
                )
            )
        return events


class EventBus:
    """Small synchronous event bus."""

    def __init__(self, log: DurableEventLog) -> None:
        self.log = log
        self._handlers: dict[str, list[EventHandler]] = {"*": []}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(
        self,
        run_id: str,
        event_type: str,
        source: str,
        payload: JsonObject,
    ) -> Event:
        event = self.log.append(
            Event(
                run_id=run_id,
                type=event_type,
                source=source,
                payload=payload,
            )
        )
        handlers = [
            *self._handlers.get(event_type, []),
            *self._handlers.get("*", []),
        ]
        for handler in handlers:
            handler(event)
        return event
