"""In-memory storage for uploaded documents."""

from dataclasses import dataclass

from medlit_agent.chunker import TextChunk


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    filename: str
    chunks: tuple[TextChunk, ...]


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}

    def save(self, document: StoredDocument) -> None:
        self._documents[document.document_id] = document

    def get(self, document_id: str) -> StoredDocument | None:
        return self._documents.get(document_id)
