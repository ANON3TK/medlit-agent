"""Reusable PDF parsing helpers."""

from dataclasses import dataclass
from io import BytesIO
from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def extract_pdf_pages(content: bytes) -> list[PageText]:
    reader = PdfReader(BytesIO(content))

    return [
        PageText(
            page_number=index,
            text=(page.extract_text() or "").strip(),
        )
        for index, page in enumerate(reader.pages, start=1)
    ]
