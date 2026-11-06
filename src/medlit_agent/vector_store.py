"""Storage and retrieval models for embedding vectors."""

from dataclasses import dataclass
from medlit_agent.chunker import TextChunk
from math import sqrt

@dataclass(frozen=True)
class VectorRecord:
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    vector: tuple[float, ...]

@dataclass(frozen=True)
class SearchHit:
    record: VectorRecord
    score: float

def build_vector_records(
    document_id: str,
    chunks: list[TextChunk],
    vectors: list[list[float]],
) -> list[VectorRecord]:
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector.")

    records = []

    for chunk, vector in zip(chunks, vectors):
        records.append(
            VectorRecord(
                document_id=document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                vector=tuple(vector),
            )
        )

    return records

def _cosine_similarity(
    left: tuple[float, ...],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension.")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    left_length = sqrt(sum(value * value for value in left))
    right_length = sqrt(sum(value * value for value in right))

    if left_length == 0 or right_length == 0:
        raise ValueError("Vectors must not be zero vectors.")

    return dot_product / (left_length * right_length)


def search_vectors(
    records: list[VectorRecord],
    query_vector: list[float],
    top_k: int,
) -> list[SearchHit]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    hits = [
        SearchHit(
            record=record,
            score=_cosine_similarity(record.vector, query_vector),
        )
        for record in records
    ]

    return sorted(
        hits,
        key=lambda hit: hit.score,
        reverse=True,
    )[:top_k]

class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, records: list[VectorRecord]) -> None:
        self._records.extend(records)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        candidates = self._records

        if document_id is not None:
            candidates = [
                record
                for record in self._records
                if record.document_id == document_id
            ]

        return search_vectors(
            candidates,
            query_vector=query_vector,
            top_k=top_k,
        )