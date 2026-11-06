"""Text chunking helpers."""

from dataclasses import dataclass
from medlit_agent.pdf_parser import PageText


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    text: str


def chunk_page(
        page: PageText,
        chunk_size: int,
        overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    chunks = []
    start = 0

    while start < len(page.text):
        end = min(start + chunk_size, len(page.text))

        chunks.append(
            TextChunk(
                page_number=page.page_number,
                chunk_index=len(chunks),
                text=page.text[start:end],
            )
        )

        if end == len(page.text):
            break

        start = end - overlap

    return chunks

def chunk_pages(
    pages: list[PageText],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in pages:
        page_chunks = chunk_page(page, chunk_size, overlap)
        chunks.extend(page_chunks)

    return chunks