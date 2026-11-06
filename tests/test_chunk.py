"""Tests for text chunking helpers."""
import pytest
from medlit_agent.chunker import chunk_page,chunk_pages
from medlit_agent.pdf_parser import PageText


def test_chunks_page_with_overlap() -> None:
    page = PageText(page_number=2, text="abcdefghij")

    chunks = chunk_page(page, chunk_size=4, overlap=1)

    assert [
               (chunk.page_number, chunk.chunk_index, chunk.text)
               for chunk in chunks
           ] == [
               (2, 0, "abcd"),
               (2, 1, "defg"),
               (2, 2, "ghij"),
           ]


def test_rejects_overlap_equal_to_chunk_size() -> None:
    page = PageText(page_number=1, text="abcdefghij")

    with pytest.raises(ValueError):
        chunk_page(page, chunk_size=4, overlap=4)


def test_rejects_zero_chunk_size() -> None:
    page = PageText(page_number=1, text="abc")

    with pytest.raises(ValueError):
        chunk_page(page, chunk_size=0, overlap=0)


def test_empty_page_produces_no_chunks() -> None:
    page = PageText(page_number=1, text="")

    chunks = chunk_page(page, chunk_size=4, overlap=1)

    assert chunks == []

def test_chunks_multiple_pages() -> None:
    pages = [
        PageText(page_number=1, text="abcde"),
        PageText(page_number=2, text="vwxyz"),
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert [
        (chunk.page_number, chunk.chunk_index, chunk.text)
        for chunk in chunks
    ] == [
        (1, 0, "abcd"),
        (1, 1, "de"),
        (2, 0, "vwxy"),
        (2, 1, "yz"),
    ]