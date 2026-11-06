"""Tests for PDF upload validation and inspection."""

from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from uuid import UUID
from medlit_agent.main import app
from medlit_agent import documents
from medlit_agent.chunker import TextChunk
from medlit_agent.interaction_store import InMemoryInteractionStore

import httpx

client = TestClient(app)

class FixedEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]

class FailIfCalledChatModel:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise AssertionError(
            "The chat model must not be called without evidence."
        )
class FailingEmbedder:
    def embed(self, text: str) -> list[float]:
        raise httpx.ConnectError(
            "SiliconFlow is unavailable."
        )

class FailingChatModel:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise httpx.ConnectError(
            "DeepSeek is unavailable."
        )

def make_blank_pdf() -> bytes:
    """Create a small valid one-page PDF entirely in memory."""
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(output)
    return output.getvalue()


def test_inspect_valid_pdf() -> None:
    response = client.post(
        "/documents",
        files={"file": ("sample.pdf", make_blank_pdf(), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    document_id = payload.pop("document_id")

    assert UUID(document_id).version == 4
    assert payload == {
        "filename": "sample.pdf",
        "chunk_count": 0,
        "page_count": 1,
        "text_chars": 0,
        "has_extractable_text": False,
    }
    stored_document = documents.document_store.get(document_id)

    assert stored_document is not None
    assert stored_document.document_id == document_id
    assert stored_document.filename == "sample.pdf"
    assert stored_document.chunks == ()


def test_rejects_non_pdf_upload() -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only PDF files are supported."


def test_rejects_unreadable_pdf() -> None:
    response = client.post(
        "/documents",
        files={"file": ("broken.pdf", b"not really a pdf", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a readable PDF."


def test_rejects_oversized_pdf() -> None:
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/documents",
        files={"file": ("large.pdf", oversized_content, "application/pdf")},
    )

    assert response.status_code == 413


def test_gets_stored_document_by_id() -> None:
    upload_response = client.post(
        "/documents",
        files={"file": ("sample.pdf", make_blank_pdf(), "application/pdf")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "filename": "sample.pdf",
        "chunk_count": 0,
    }


def test_returns_404_for_missing_document() -> None:
    response = client.get("/documents/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_indexes_uploaded_document_chunks(monkeypatch) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )

    response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = response.json()["document_id"]

    hits = documents.vector_store.search(
        query_vector=[7.0, 1.0],
        top_k=1,
        document_id=document_id,
    )

    assert response.status_code == 200
    assert len(hits) == 1
    assert hits[0].record.document_id == document_id
    assert hits[0].record.text == "example"
    assert hits[0].record.vector == (7.0, 1.0)


def test_searches_uploaded_document(monkeypatch) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    search_response = client.post(
        f"/documents/{document_id}/search",
        json={
            "query": "example",
            "top_k": 1,
        },
    )

    assert search_response.status_code == 200

    payload = search_response.json()
    result = payload["results"][0]
    score = result.pop("score")

    assert payload["document_id"] == document_id
    assert result == {
        "page_number": 1,
        "chunk_index": 0,
        "text": "example",
    }
    assert abs(score - 1.0) < 0.000001

def test_excludes_results_below_minimum_score(monkeypatch) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    search_response = client.post(
        f"/documents/{document_id}/search",
        json={
            "query": "x",
            "top_k": 1,
            "min_score": 0.99,
        },
    )

    assert search_response.status_code == 200
    assert search_response.json()["results"] == []

class FixedChatModel:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        assert "example" in user_prompt
        assert "第1页" in user_prompt

        return "测试答案。[来源1，第1页]"

def test_answers_question_with_cited_sources(monkeypatch) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    interaction_store = InMemoryInteractionStore()

    monkeypatch.setattr(
        documents,
        "interaction_store",
        interaction_store,
        raising=False,
    )

    logged_events = []

    def fake_log_event(
        _logger,
        event,
        **fields,
    ) -> None:
        logged_events.append(
            {
                "event": event,
                **fields,
            }
        )

    monkeypatch.setattr(
        documents,
        "log_event",
        fake_log_event,
        raising=False,
    )

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_chat_model",
        lambda: FixedChatModel(),
        raising=False,
    )

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    answer_response = client.post(
        f"/documents/{document_id}/ask",
        json={
            "query": "example",
            "top_k": 1,
            "min_score": 0.3,
        },
    )

    assert answer_response.status_code == 200

    payload = answer_response.json()
    source = payload["sources"][0]
    score = source.pop("score")

    assert payload["document_id"] == document_id
    assert payload["answer"] == "测试答案。[来源1，第1页]"
    assert source == {
        "page_number": 1,
        "chunk_index": 0,
        "text": "example",
    }
    assert abs(score - 1.0) < 0.000001
    interactions = interaction_store.list_recent(limit=1)

    assert len(interactions) == 1

    interaction = interactions[0]

    assert interaction.operation == "document_ask"
    assert interaction.question == "example"
    assert interaction.answer == "测试答案。[来源1，第1页]"
    assert interaction.document_id == document_id
    assert interaction.tool_used is None
    assert interaction.created_at
    assert logged_events == [
        {
            "event": "interaction_saved",
            "interaction_id": (
                interaction.interaction_id
            ),
            "operation": "document_ask",
            "document_id": document_id,
            "tool_used": None,
            "source_count": 1,
        }
    ]

def test_ask_does_not_generate_without_evidence(monkeypatch) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_chat_model",
        lambda: FailIfCalledChatModel(),
    )

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    answer_response = client.post(
        f"/documents/{document_id}/ask",
        json={
            "query": "x",
            "top_k": 1,
            "min_score": 0.99,
        },
    )

    assert answer_response.status_code == 200
    assert answer_response.json() == {
        "document_id": document_id,
        "answer": "未在文档中找到足够相关的证据。",
        "sources": [],
    }

def test_returns_502_when_embedding_service_fails(
    monkeypatch,
) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FailingEmbedder(),
    )

    error_client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = error_client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "External AI service is unavailable."
    }

def test_returns_502_when_chat_service_fails(
    monkeypatch,
) -> None:
    fake_chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_chat_model",
        lambda: FailingChatModel(),
    )

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    error_client = TestClient(
        app,
        raise_server_exceptions=False,
    )
    response = error_client.post(
        f"/documents/{document_id}/ask",
        json={
            "query": "example",
            "top_k": 1,
            "min_score": 0.3,
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "External AI service is unavailable."
    }


def test_evaluates_document_retrieval(
    monkeypatch,
) -> None:
    fake_chunks = [
        TextChunk(
            page_number=2,
            chunk_index=0,
            text="example",
        )
    ]

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: fake_chunks,
    )
    monkeypatch.setattr(
        documents,
        "create_siliconflow_embedder",
        lambda: FixedEmbedder(),
    )

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    evaluation_response = client.post(
        f"/documents/{document_id}/evaluate",
        json={
            "cases": [
                {
                    "question": "example",
                    "expected_page_number": 2,
                },
                {
                    "question": "example",
                    "expected_page_number": 1,
                },
            ],
            "top_k": 1,
        },
    )

    assert evaluation_response.status_code == 200
    assert evaluation_response.json() == {
        "total_cases": 2,
        "hit_count": 1,
        "hit_rate": 0.5,
    }