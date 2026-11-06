"""SQLite-backed storage for uploaded documents."""

import sqlite3
from pathlib import Path
import json
from medlit_agent.chunker import TextChunk
from medlit_agent.document_store import StoredDocument
from medlit_agent.vector_store import (
    SearchHit,
    VectorRecord,
    search_vectors,
)

class SQLiteDocumentStore:
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
        connection = sqlite3.connect(
            self._database_path,
        )
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    document_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (document_id, position),
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                )
                """
            )

    def save(
        self,
        document: StoredDocument,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename
                )
                VALUES (?, ?)
                ON CONFLICT(document_id)
                DO UPDATE SET
                    filename = excluded.filename
                """,
                (
                    document.document_id,
                    document.filename,
                ),
            )

            connection.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = ?
                """,
                (document.document_id,),
            )

            connection.executemany(
                """
                INSERT INTO document_chunks (
                    document_id,
                    position,
                    page_number,
                    chunk_index,
                    text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document.document_id,
                        position,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.text,
                    )
                    for position, chunk in enumerate(
                        document.chunks
                    )
                ],
            )

    def get(
        self,
        document_id: str,
    ) -> StoredDocument | None:
        with self._connect() as connection:
            document_row = connection.execute(
                """
                SELECT document_id, filename
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if document_row is None:
                return None

            chunk_rows = connection.execute(
                """
                SELECT
                    page_number,
                    chunk_index,
                    text
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()

        chunks = tuple(
            TextChunk(
                page_number=row[0],
                chunk_index=row[1],
                text=row[2],
            )
            for row in chunk_rows
        )

        return StoredDocument(
            document_id=document_row[0],
            filename=document_row[1],
            chunks=chunks,
        )

class SQLiteVectorStore:
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
                CREATE TABLE IF NOT EXISTS vector_records (
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY (
                        document_id,
                        page_number,
                        chunk_index
                    )
                )
                """
            )

    def add(
        self,
        records: list[VectorRecord],
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO vector_records (
                    document_id,
                    page_number,
                    chunk_index,
                    text,
                    vector_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (
                    document_id,
                    page_number,
                    chunk_index
                )
                DO UPDATE SET
                    text = excluded.text,
                    vector_json = excluded.vector_json
                """,
                [
                    (
                        record.document_id,
                        record.page_number,
                        record.chunk_index,
                        record.text,
                        json.dumps(record.vector),
                    )
                    for record in records
                ],
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        with self._connect() as connection:
            if document_id is None:
                rows = connection.execute(
                    """
                    SELECT
                        document_id,
                        page_number,
                        chunk_index,
                        text,
                        vector_json
                    FROM vector_records
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        document_id,
                        page_number,
                        chunk_index,
                        text,
                        vector_json
                    FROM vector_records
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchall()

        records = [
            VectorRecord(
                document_id=row[0],
                page_number=row[1],
                chunk_index=row[2],
                text=row[3],
                vector=tuple(json.loads(row[4])),
            )
            for row in rows
        ]

        return search_vectors(
            records=records,
            query_vector=query_vector,
            top_k=top_k,
        )