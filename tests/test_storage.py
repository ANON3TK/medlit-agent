from medlit_agent.document_store import InMemoryDocumentStore
from medlit_agent.sqlite_store import (
    SQLiteDocumentStore,
    SQLiteVectorStore,
)
from medlit_agent.storage import (
    create_interaction_store,
    create_stores,
)
from medlit_agent.vector_store import InMemoryVectorStore
from medlit_agent.interaction_store import (
    InMemoryInteractionStore,
    SQLiteInteractionStore,
)


def test_creates_in_memory_stores_without_database_path() -> None:
    document_store, vector_store = create_stores(
        database_path=None,
    )

    assert isinstance(
        document_store,
        InMemoryDocumentStore,
    )
    assert isinstance(
        vector_store,
        InMemoryVectorStore,
    )


def test_creates_sqlite_stores_with_database_path(
    tmp_path,
) -> None:
    database_path = tmp_path / "medlit.db"

    document_store, vector_store = create_stores(
        database_path=database_path,
    )

    assert isinstance(
        document_store,
        SQLiteDocumentStore,
    )
    assert isinstance(
        vector_store,
        SQLiteVectorStore,
    )

def test_creates_in_memory_interaction_store_without_path() -> None:
    interaction_store = create_interaction_store(
        database_path=None,
    )

    assert isinstance(
        interaction_store,
        InMemoryInteractionStore,
    )


def test_creates_sqlite_interaction_store_with_path(
    tmp_path,
) -> None:
    database_path = tmp_path / "medlit.db"

    interaction_store = create_interaction_store(
        database_path=database_path,
    )

    assert isinstance(
        interaction_store,
        SQLiteInteractionStore,
    )