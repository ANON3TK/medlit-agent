"""Tests for vector storage and retrieval."""

from medlit_agent.chunker import TextChunk

import pytest
from medlit_agent.vector_store import (
    VectorRecord,
    build_vector_records,
    search_vectors,
    InMemoryVectorStore,
)

def test_builds_vector_records_in_order() -> None:
    chunks = [
        TextChunk(page_number=2, chunk_index=0, text="first"),
        TextChunk(page_number=2, chunk_index=1, text="second"),
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    records = build_vector_records("doc-1", chunks, vectors)

    assert records == [
        VectorRecord(
            document_id="doc-1",
            page_number=2,
            chunk_index=0,
            text="first",
            vector=(1.0, 0.0),
        ),
        VectorRecord(
            document_id="doc-1",
            page_number=2,
            chunk_index=1,
            text="second",
            vector=(0.0, 1.0),
        ),
    ]


def test_rejects_mismatched_chunk_and_vector_counts() -> None:
    chunks = [
        TextChunk(page_number=1, chunk_index=0, text="first"),
        TextChunk(page_number=1, chunk_index=1, text="second"),
    ]
    vectors = [
        [1.0, 0.0],
    ]

    with pytest.raises(
            ValueError,
            match="Each chunk must have exactly one vector.",
    ):
        build_vector_records("doc-1", chunks, vectors)


def test_returns_most_similar_vector_record() -> None:
    records = [
        VectorRecord(
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            text="horizontal",
            vector=(1.0, 0.0),
        ),
        VectorRecord(
            document_id="doc-1",
            page_number=1,
            chunk_index=1,
            text="vertical",
            vector=(0.0, 1.0),
        ),
    ]

    hits = search_vectors(
        records,
        query_vector=[1.0, 0.0],
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0].record.text == "horizontal"
    assert hits[0].score == 1.0

def test_vector_store_adds_and_searches_by_document() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            VectorRecord(
                document_id="doc-1",
                page_number=1,
                chunk_index=0,
                text="from document one",
                vector=(1.0, 0.0),
            ),
            VectorRecord(
                document_id="doc-2",
                page_number=1,
                chunk_index=0,
                text="from document two",
                vector=(0.0, 1.0),
            ),
        ]
    )

    hits = store.search(
        query_vector=[0.0, 1.0],
        top_k=1,
        document_id="doc-1",
    )

    assert len(hits) == 1
    assert hits[0].record.document_id == "doc-1"