"""Differentiable memory architectures.

Working Memory: A scratchpad for the current sub-task (limited capacity,
fast read/write, forgets quickly).
Episodic Memory: A vector database of previous actions and outcomes,
preventing repeated mistakes and enabling analogical reasoning.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentos_orchestrator.os_control.base import UiAction


@dataclass(slots=True)
class EpisodicEvent:
    """A single event stored in episodic memory."""

    event_id: str
    timestamp: float
    objective: str
    action: UiAction
    observation_summary: str
    outcome: str
    reward: float
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass(slots=True)
class WorkingMemoryItem:
    """A single item in working memory."""

    item_id: str
    content: str
    item_type: str  # "plan", "observation", "hypothesis", "action", "goal"
    priority: float = 0.5
    created_at: float = 0.0
    expires_at: float | None = None


class WorkingMemoryScratchpad:
    """Fast, limited-capacity scratchpad for the current sub-task.

    Items decay over time and are pruned when capacity is exceeded.
    Higher-priority items survive longer.
    """

    def __init__(self, capacity: int = 32, default_ttl_seconds: float = 300.0) -> None:
        self.capacity = capacity
        self.default_ttl = default_ttl_seconds
        self._items: list[WorkingMemoryItem] = []
        self._counter = 0

    def write(
        self,
        content: str,
        item_type: str = "observation",
        priority: float = 0.5,
        ttl_seconds: float | None = None,
    ) -> str:
        """Add an item to working memory. Returns the item ID."""
        self._counter += 1
        now = time.time()
        item_id = f"wm_{self._counter}_{int(now)}"
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        item = WorkingMemoryItem(
            item_id=item_id,
            content=content,
            item_type=item_type,
            priority=priority,
            created_at=now,
            expires_at=now + ttl,
        )
        self._items.append(item)
        self._prune()
        return item_id

    def read(
        self, item_type: str | None = None, min_priority: float = 0.0
    ) -> list[WorkingMemoryItem]:
        """Retrieve items, optionally filtered by type and priority."""
        now = time.time()
        result = []
        for item in self._items:
            if item.expires_at and item.expires_at < now:
                continue
            if item_type and item.item_type != item_type:
                continue
            if item.priority < min_priority:
                continue
            result.append(item)
        return sorted(result, key=lambda i: i.priority, reverse=True)

    def read_recent(self, n: int = 5) -> list[WorkingMemoryItem]:
        """Get the n most recent non-expired items."""
        now = time.time()
        active = [i for i in self._items if not i.expires_at or i.expires_at >= now]
        return sorted(active, key=lambda i: i.created_at, reverse=True)[:n]

    def clear(self, item_type: str | None = None) -> int:
        """Remove items. Returns count removed."""
        before = len(self._items)
        if item_type:
            self._items = [i for i in self._items if i.item_type != item_type]
        else:
            self._items = []
        return before - len(self._items)

    def update_priority(self, item_id: str, new_priority: float) -> bool:
        """Boost or reduce an item's priority."""
        for item in self._items:
            if item.item_id == item_id:
                item.priority = new_priority
                return True
        return False

    def summarize(self) -> str:
        """Return a text summary of current working memory contents."""
        active = self.read()
        lines = [f"Working Memory ({len(active)} items):"]
        for item in active[:10]:
            lines.append(
                f"  [{item.item_type}] (p={item.priority:.2f}) {item.content[:80]}"
            )
        return "\n".join(lines)

    def _prune(self) -> None:
        """Remove expired and low-priority items to stay under capacity."""
        now = time.time()
        # First remove expired
        self._items = [
            i for i in self._items if not i.expires_at or i.expires_at >= now
        ]
        # Then sort by effective score (priority weighted by recency)
        self._items.sort(
            key=lambda i: i.priority / (1 + now - i.created_at),
            reverse=True,
        )
        if len(self._items) > self.capacity:
            self._items = self._items[: self.capacity]


class EpisodicMemoryBank:
    """Persistent vector database of past experiences.

    Stores actions taken, their outcomes, and embeddings for similarity search.
    Prevents repeated mistakes and enables analogical transfer.
    """

    def __init__(self, db_path: str | Path = ".agentos/episodic_memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._counter = 0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    objective TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    selector TEXT NOT NULL,
                    value TEXT,
                    observation_summary TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reward REAL NOT NULL,
                    tags_json TEXT NOT NULL,
                    embedding_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_objective
                ON episodic_events(objective)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_tags
                ON episodic_events(tags_json)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        objective: str,
        action: UiAction,
        observation_summary: str,
        outcome: str,
        reward: float,
        tags: list[str] | None = None,
    ) -> str:
        """Record an event in episodic memory."""
        self._counter += 1
        event_id = f"ep_{self._counter}_{int(time.time() * 1000)}"
        embedding = self._compute_embedding(objective, action, observation_summary)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO episodic_events
                (event_id, timestamp, objective, action_type, selector, value,
                 observation_summary, outcome, reward, tags_json, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    time.time(),
                    objective,
                    action.action_type,
                    action.selector,
                    action.value,
                    observation_summary,
                    outcome,
                    reward,
                    json.dumps(tags or []),
                    json.dumps(embedding),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return event_id

    def retrieve_similar(
        self,
        objective: str,
        action_hint: UiAction | None = None,
        top_k: int = 5,
    ) -> list[EpisodicEvent]:
        """Retrieve past events most similar to the current situation."""
        query_embedding = self._compute_embedding(objective, action_hint)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM episodic_events ORDER BY timestamp DESC LIMIT 500"
            ).fetchall()
        finally:
            conn.close()
        events: list[tuple[float, EpisodicEvent]] = []
        for row in rows:
            event = self._row_to_event(row)
            if event.embedding:
                sim = self._cosine_similarity(query_embedding, event.embedding)
                events.append((sim, event))
        events.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in events[:top_k]]

    def retrieve_by_objective_substring(
        self,
        substring: str,
        top_k: int = 10,
    ) -> list[EpisodicEvent]:
        """Simple substring search for objectives."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM episodic_events WHERE objective LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{substring}%", top_k),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_event(row) for row in rows]

    def get_failure_patterns(
        self, objective: str, top_k: int = 5
    ) -> list[EpisodicEvent]:
        """Retrieve past failures for this objective to avoid repeating them."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM episodic_events
                WHERE objective LIKE ? AND reward < 0
                ORDER BY timestamp DESC LIMIT ?
                """,
                (f"%{objective}%", top_k),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_event(row) for row in rows]

    def get_success_patterns(
        self, objective: str, top_k: int = 5
    ) -> list[EpisodicEvent]:
        """Retrieve past successes for this objective to replicate."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM episodic_events
                WHERE objective LIKE ? AND reward > 0
                ORDER BY timestamp DESC LIMIT ?
                """,
                (f"%{objective}%", top_k),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_event(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        """Return statistics about the memory bank."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM episodic_events").fetchone()[0]
            avg_reward = conn.execute(
                "SELECT AVG(reward) FROM episodic_events"
            ).fetchone()[0]
            failures = conn.execute(
                "SELECT COUNT(*) FROM episodic_events WHERE reward < 0"
            ).fetchone()[0]
        finally:
            conn.close()
        return {
            "total_events": total,
            "average_reward": avg_reward or 0.0,
            "failure_count": failures,
        }

    def close(self) -> None:
        """Close any open database connections."""
        pass  # Connections are already closed after each operation

    @staticmethod
    def _compute_embedding(
        objective: str,
        action: UiAction | None = None,
        observation: str = "",
    ) -> list[float]:
        """Simple bag-of-characters embedding for fast retrieval.

        In production, replace with a sentence-transformer or CLIP embedding.
        """
        text = objective.lower()
        if action:
            text += f" {action.action_type} {action.selector}"
        if observation:
            text += f" {observation.lower()}"
        # Character n-gram frequency vector (256 dims)
        vec = [0.0] * 256
        for ch in text:
            idx = ord(ch) % 256
            vec[idx] += 1.0
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EpisodicEvent:
        return EpisodicEvent(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            objective=row["objective"],
            action=UiAction(
                action_type=row["action_type"],
                selector=row["selector"],
                value=row["value"],
            ),
            observation_summary=row["observation_summary"],
            outcome=row["outcome"],
            reward=row["reward"],
            tags=json.loads(row["tags_json"]),
            embedding=json.loads(row["embedding_json"])
            if row["embedding_json"]
            else None,
        )
