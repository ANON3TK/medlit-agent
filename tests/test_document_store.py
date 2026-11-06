"""Tests for in-memory document storage."""

from medlit_agent.chunker import TextChunk
from medlit_agent.document_store import InMemoryDocumentStore, StoredDocument


def test_saves_and_gets_document() -> None:
    store = InMemoryDocumentStore()
    document = StoredDocument(
        document_id="doc-1",
        filename="sample.pdf",
        chunks=(
            TextChunk(page_number=1, chunk_index=0, text="example"),
        ),
    )

    store.save(document)

    assert store.get("doc-1") == document
    assert store.get("missing") is None

