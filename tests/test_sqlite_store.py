from medlit_agent.chunker import TextChunk
from medlit_agent.document_store import StoredDocument
from medlit_agent.sqlite_store import (
    SQLiteDocumentStore,
    SQLiteVectorStore,
)
from medlit_agent.vector_store import VectorRecord

def test_persists_document_across_store_instances(
    tmp_path,
) -> None:
    database_path = tmp_path / "medlit.db"

    first_store = SQLiteDocumentStore(
        database_path=database_path,
    )
    document = StoredDocument(
        document_id="doc-1",
        filename="sample.pdf",
        chunks=(
            TextChunk(
                page_number=1,
                chunk_index=0,
                text="first chunk",
            ),
            TextChunk(
                page_number=2,
                chunk_index=0,
                text="second chunk",
            ),
        ),
    )

    first_store.save(document)

    reopened_store = SQLiteDocumentStore(
        database_path=database_path,
    )
    loaded_document = reopened_store.get("doc-1")

    assert loaded_document == document

def test_persists_vectors_across_store_instances(
    tmp_path,
) -> None:
    database_path = tmp_path / "medlit.db"

    record = VectorRecord(
        document_id="doc-1",
        page_number=1,
        chunk_index=0,
        text="first chunk",
        vector=(1.0, 0.0),
    )

    first_store = SQLiteVectorStore(
        database_path=database_path,
    )
    first_store.add([record])

    reopened_store = SQLiteVectorStore(
        database_path=database_path,
    )
    hits = reopened_store.search(
        query_vector=[1.0, 0.0],
        top_k=1,
        document_id="doc-1",
    )

    assert len(hits) == 1
    assert hits[0].record == record
    assert abs(hits[0].score - 1.0) < 0.000001