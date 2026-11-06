"""Choose storage implementations from application configuration."""

from pathlib import Path

from medlit_agent.document_store import InMemoryDocumentStore
from medlit_agent.sqlite_store import (
    SQLiteDocumentStore,
    SQLiteVectorStore,
)
from medlit_agent.interaction_store import (
    InMemoryInteractionStore,
    SQLiteInteractionStore,
)
from medlit_agent.vector_store import InMemoryVectorStore


DocumentStore = (
    InMemoryDocumentStore
    | SQLiteDocumentStore
)

VectorStore = (
    InMemoryVectorStore
    | SQLiteVectorStore
)
InteractionStore = (
    InMemoryInteractionStore
    | SQLiteInteractionStore
)


def create_stores(
    database_path: str | Path | None,
) -> tuple[DocumentStore, VectorStore]:
    if database_path is None:
        return (
            InMemoryDocumentStore(),
            InMemoryVectorStore(),
        )

    return (
        SQLiteDocumentStore(
            database_path=database_path,
        ),
        SQLiteVectorStore(
            database_path=database_path,
        ),
    )

def create_interaction_store(
    database_path: str | Path | None,
) -> InteractionStore:
    if database_path is None:
        return InMemoryInteractionStore()

    return SQLiteInteractionStore(
        database_path=database_path,
    )