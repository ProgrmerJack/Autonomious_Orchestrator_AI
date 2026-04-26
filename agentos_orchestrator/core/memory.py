from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from contextlib import closing
from pathlib import Path

from .types import JsonObject, new_id, utc_now


@dataclass(slots=True)
class MemoryCandidate:
    run_id: str
    statement: str
    evidence: list[JsonObject]
    confidence: float
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompressionDecision:
    accepted: bool
    statement: str
    reasons: list[str]
    observation_id: str | None = None


class CognitiveCompressor:
    """Gatekeeper that keeps unsupported hypotheses out of durable memory."""

    def __init__(
        self,
        min_confidence: float = 0.7,
        max_statement_chars: int = 1200,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_statement_chars = max_statement_chars

    def evaluate(self, candidate: MemoryCandidate) -> CompressionDecision:
        reasons: list[str] = []
        if candidate.confidence < self.min_confidence:
            reasons.append("confidence below durable-memory threshold")
        if not candidate.evidence:
            reasons.append("no evidence attached")
        lowered = candidate.statement.lower()
        unresolved_markers = ("maybe", "unverified", "hypothesis")
        if any(marker in lowered for marker in unresolved_markers):
            reasons.append("statement is framed as unresolved or unverified")
        poison_markers = (
            "ignore previous instructions",
            "developer message",
            "system prompt",
            "disable security",
        )
        if any(marker in lowered for marker in poison_markers):
            reasons.append("statement resembles prompt-injection content")
        acceptance_reason = "accepted with evidence and sufficient confidence"
        compressed = " ".join(candidate.statement.split())
        compressed = compressed[: self.max_statement_chars]
        return CompressionDecision(
            accepted=not reasons,
            statement=compressed,
            reasons=reasons or [acceptance_reason],
        )


class SemanticMemory:
    """SQLite semantic-memory ledger."""

    def __init__(
        self,
        db_path: str | Path,
        compressor: CognitiveCompressor | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.compressor = compressor or CognitiveCompressor()
        self.fts_enabled = False
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
                    CREATE TABLE IF NOT EXISTS memory_observations (
                        observation_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        candidate_json TEXT NOT NULL,
                        accepted INTEGER NOT NULL,
                        reasons_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        observation_id TEXT,
                        run_id TEXT NOT NULL,
                        statement TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        tags_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                self._init_fts(connection)

    def commit(self, candidate: MemoryCandidate) -> CompressionDecision:
        return self.observe(candidate)

    def observe(self, candidate: MemoryCandidate) -> CompressionDecision:
        observation_id = new_id("obs")
        decision = self.compressor.evaluate(candidate)
        decision.observation_id = observation_id
        self._record_observation(observation_id, candidate, decision)
        if not decision.accepted:
            return decision
        self._commit_accepted(observation_id, candidate, decision)
        return decision

    def _record_observation(
        self,
        observation_id: str,
        candidate: MemoryCandidate,
        decision: CompressionDecision,
    ) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO memory_observations(
                        observation_id,
                        run_id,
                        candidate_json,
                        accepted,
                        reasons_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        candidate.run_id,
                        json.dumps(asdict(candidate), sort_keys=True),
                        1 if decision.accepted else 0,
                        json.dumps(decision.reasons, sort_keys=True),
                        utc_now(),
                    ),
                )

    def _commit_accepted(
        self,
        observation_id: str,
        candidate: MemoryCandidate,
        decision: CompressionDecision,
    ) -> None:
        memory_id = new_id("mem")
        created_at = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO memories(
                        memory_id,
                        observation_id,
                        run_id,
                        statement,
                        evidence_json,
                        confidence,
                        tags_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        observation_id,
                        candidate.run_id,
                        decision.statement,
                        json.dumps(candidate.evidence, sort_keys=True),
                        candidate.confidence,
                        json.dumps(candidate.tags, sort_keys=True),
                        created_at,
                    ),
                )
                if self._fts_available(connection):
                    connection.execute(
                        """
                        INSERT INTO memories_fts(
                            rowid,
                            statement,
                            tags
                        )
                        VALUES (
                            (SELECT rowid FROM memories WHERE memory_id = ?),
                            ?,
                            ?
                        )
                        """,
                        (
                            memory_id,
                            decision.statement,
                            " ".join(candidate.tags),
                        ),
                    )

    def list_for_run(self, run_id: str) -> list[JsonObject]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    memory_id,
                    observation_id,
                    statement,
                    evidence_json,
                    confidence,
                    tags_json,
                    created_at
                FROM memories
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "memory_id": row[0],
                "observation_id": row[1],
                "statement": row[2],
                "evidence": json.loads(row[3]),
                "confidence": row[4],
                "tags": json.loads(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]

    def list_observations(self, run_id: str) -> list[JsonObject]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    observation_id,
                    candidate_json,
                    accepted,
                    reasons_json,
                    created_at
                FROM memory_observations
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "observation_id": row[0],
                "candidate": json.loads(row[1]),
                "accepted": bool(row[2]),
                "reasons": json.loads(row[3]),
                "created_at": row[4],
            }
            for row in rows
        ]

    def search(self, query: str, limit: int = 10) -> list[JsonObject]:
        if not self.fts_enabled:
            return self._like_search(query, limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT m.memory_id, m.statement, m.confidence, m.tags_json
                FROM memories_fts f
                JOIN memories m ON m.rowid = f.rowid
                WHERE memories_fts MATCH ?
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [
            {
                "memory_id": row[0],
                "statement": row[1],
                "confidence": row[2],
                "tags": json.loads(row[3]),
            }
            for row in rows
        ]

    def _like_search(self, query: str, limit: int) -> list[JsonObject]:
        pattern = f"%{query}%"
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, statement, confidence, tags_json
                FROM memories
                WHERE statement LIKE ?
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()
        return [
            {
                "memory_id": row[0],
                "statement": row[1],
                "confidence": row[2],
                "tags": json.loads(row[3]),
            }
            for row in rows
        ]

    def _init_fts(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(statement, tags, content='')
                """
            )
        except sqlite3.OperationalError:
            self.fts_enabled = False
        else:
            self.fts_enabled = True

    def _fts_available(self, connection: sqlite3.Connection) -> bool:
        if not self.fts_enabled:
            return False
        try:
            connection.execute("SELECT 1 FROM memories_fts LIMIT 1")
        except sqlite3.OperationalError:
            self.fts_enabled = False
            return False
        return True
