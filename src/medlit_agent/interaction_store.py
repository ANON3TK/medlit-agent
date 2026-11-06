"""Persistent storage for question and answer interactions."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InteractionRecord:
    interaction_id: str
    operation: str
    question: str
    answer: str
    document_id: str | None
    tool_used: str | None
    created_at: str


class InMemoryInteractionStore:
    def __init__(self) -> None:
        self._interactions: dict[
            str,
            InteractionRecord,
        ] = {}

    def save(
        self,
        interaction: InteractionRecord,
    ) -> None:
        self._interactions[
            interaction.interaction_id
        ] = interaction

    def list_recent(
        self,
        limit: int,
    ) -> list[InteractionRecord]:
        if limit <= 0:
            raise ValueError(
                "limit must be positive."
            )

        ordered = sorted(
            self._interactions.values(),
            key=lambda interaction: interaction.created_at,
            reverse=True,
        )

        return ordered[:limit]

class SQLiteInteractionStore:
    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self._database_path,
        )

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    document_id TEXT,
                    tool_used TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        interaction: InteractionRecord,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    interaction_id,
                    operation,
                    question,
                    answer,
                    document_id,
                    tool_used,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(interaction_id)
                DO UPDATE SET
                    operation = excluded.operation,
                    question = excluded.question,
                    answer = excluded.answer,
                    document_id = excluded.document_id,
                    tool_used = excluded.tool_used,
                    created_at = excluded.created_at
                """,
                (
                    interaction.interaction_id,
                    interaction.operation,
                    interaction.question,
                    interaction.answer,
                    interaction.document_id,
                    interaction.tool_used,
                    interaction.created_at,
                ),
            )

    def list_recent(
        self,
        limit: int,
    ) -> list[InteractionRecord]:
        if limit <= 0:
            raise ValueError(
                "limit must be positive."
            )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    interaction_id,
                    operation,
                    question,
                    answer,
                    document_id,
                    tool_used,
                    created_at
                FROM interactions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            InteractionRecord(
                interaction_id=row[0],
                operation=row[1],
                question=row[2],
                answer=row[3],
                document_id=row[4],
                tool_used=row[5],
                created_at=row[6],
            )
            for row in rows
        ]