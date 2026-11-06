"""Tests for reusable PDF parsing helpers."""

from io import BytesIO

from pypdf import PdfWriter

from medlit_agent.pdf_parser import extract_pdf_pages


def make_blank_pdf(page_count: int) -> bytes:
    output = BytesIO()
    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)

    writer.write(output)
    return output.getvalue()


def test_extracts_pages_with_one_based_page_numbers() -> None:
    pages = extract_pdf_pages(make_blank_pdf(page_count=2))

    assert [page.page_number for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["", ""]