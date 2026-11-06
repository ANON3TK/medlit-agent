"""HTTP endpoints for inspecting uploaded PDF documents."""
import logging
from medlit_agent.runtime import (
    document_store,
    interaction_store,
    vector_store,
)
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from pypdf.errors import PdfReadError
from medlit_agent.pdf_parser import extract_pdf_pages
from medlit_agent.chunker import chunk_pages
from uuid import uuid4
from medlit_agent.embedder import (
    create_siliconflow_embedder,
    embed_chunks,
)
from medlit_agent.vector_store import (
    SearchHit,
    build_vector_records,
)
from medlit_agent.answering import generate_answer
from medlit_agent.llm import create_siliconflow_chat_model
from medlit_agent.interaction_store import InteractionRecord
from medlit_agent.document_store import StoredDocument
from medlit_agent.logging_utils import log_event
from medlit_agent.evaluation import (
    RetrievalEvaluationQuestion,
    run_retrieval_evaluation,
)




logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
router = APIRouter(prefix="/documents", tags=["documents"])
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200



class DocumentInspection(BaseModel):
    """Metadata extracted from an uploaded PDF without storing the file."""
    chunk_count: int
    filename: str
    page_count: int
    text_chars: int
    has_extractable_text: bool
    document_id: str

class StoredDocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int

class DocumentSearchRequest(BaseModel):
    query: str
    top_k: int = 3
    min_score:float = 0.3


class DocumentSearchResult(BaseModel):
    page_number: int
    chunk_index: int
    text: str
    score: float


class DocumentSearchResponse(BaseModel):
    document_id: str
    results: list[DocumentSearchResult]

class DocumentAnswerResponse(BaseModel):
    document_id: str
    answer: str
    sources: list[DocumentSearchResult]

class DocumentEvaluationCaseRequest(BaseModel):
    question: str
    expected_page_number: int


class DocumentEvaluationRequest(BaseModel):
    cases: list[DocumentEvaluationCaseRequest]
    top_k: int = 3


class DocumentEvaluationResponse(BaseModel):
    total_cases: int
    hit_count: int
    hit_rate: float


@router.post("", response_model=DocumentInspection)
async def inspect_document(file: UploadFile = File(...)) -> DocumentInspection:
    """Validate a PDF upload and return basic extraction information."""
    filename = Path(file.filename or "uploaded.pdf").name
    suffix = Path(filename).suffix.lower()

    if suffix != ".pdf" and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The uploaded file is too large.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    try:
        pages = extract_pdf_pages(content)
    except (PdfReadError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The uploaded file is not a readable PDF.",
        ) from exc

    text_chars = sum(len(page.text) for page in pages)
    chunks = chunk_pages(
        pages,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )
    document_id = str(uuid4())
    records = []

    if chunks:
        embedder = create_siliconflow_embedder()
        vectors = embed_chunks(chunks, embedder)
        records = build_vector_records(
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
        )

    document_store.save(
        StoredDocument(
            document_id=document_id,
            filename=filename,
            chunks=tuple(chunks),
        )
    )
    vector_store.add(records)
    return DocumentInspection(
        document_id=document_id,
        chunk_count=len(chunks),
        filename=filename,
        page_count=len(pages),
        text_chars=text_chars,
        has_extractable_text=text_chars > 0,
    )

@router.get("/{document_id}", response_model=StoredDocumentSummary)
def get_document(document_id: str) -> StoredDocumentSummary:
    document = document_store.get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return StoredDocumentSummary(
        document_id=document.document_id,
        filename=document.filename,
        chunk_count=len(document.chunks),
    )

@router.post(
    "/{document_id}/search",
    response_model=DocumentSearchResponse,
)
def search_document(
    document_id: str,
    request: DocumentSearchRequest,
) -> DocumentSearchResponse:
    document = document_store.get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be positive.",
        )

    embedder = create_siliconflow_embedder()
    query_vector = embedder.embed(request.query)

    hits = vector_store.search(
        query_vector=query_vector,
        top_k=request.top_k,
        document_id=document_id,
    )
    hits = [
        hit
        for hit in hits
        if hit.score >= request.min_score
    ]

    return DocumentSearchResponse(
        document_id=document_id,
        results=[
            DocumentSearchResult(
                page_number=hit.record.page_number,
                chunk_index=hit.record.chunk_index,
                text=hit.record.text,
                score=hit.score,
            )
            for hit in hits
        ],
    )



@router.post(
    "/{document_id}/evaluate",
    response_model=DocumentEvaluationResponse,
)
def evaluate_document(
    document_id: str,
    request: DocumentEvaluationRequest,
) -> DocumentEvaluationResponse:
    document = document_store.get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if not request.cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation cases must not be empty.",
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be positive.",
        )

    embedder = create_siliconflow_embedder()

    def retrieve(question: str) -> list[SearchHit]:
        query_vector = embedder.embed(question)

        return vector_store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            document_id=document_id,
        )

    result = run_retrieval_evaluation(
        questions=[
            RetrievalEvaluationQuestion(
                question=case.question,
                expected_page_number=(
                    case.expected_page_number
                ),
            )
            for case in request.cases
        ],
        retrieve=retrieve,
    )

    return DocumentEvaluationResponse(
        total_cases=result.total_cases,
        hit_count=result.hit_count,
        hit_rate=result.hit_rate,
    )

@router.post(
    "/{document_id}/ask",
    response_model=DocumentAnswerResponse,
)
def answer_document(
    document_id: str,
    request: DocumentSearchRequest,
) -> DocumentAnswerResponse:
    document = document_store.get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be positive.",
        )

    embedder = create_siliconflow_embedder()
    query_vector = embedder.embed(request.query)

    hits = vector_store.search(
        query_vector=query_vector,
        top_k=request.top_k,
        document_id=document_id,
    )

    hits = [
        hit
        for hit in hits
        if hit.score >= request.min_score
    ]

    chat_model = create_siliconflow_chat_model()
    generated = generate_answer(
        question=request.query,
        hits=hits,
        model=chat_model,
    )

    interaction_id = str(uuid4())

    interaction_store.save(
        InteractionRecord(
            interaction_id=interaction_id,
            operation="document_ask",
            question=request.query,
            answer=generated.answer,
            document_id=document_id,
            tool_used=None,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )
    )

    log_event(
        logger,
        "interaction_saved",
        interaction_id=interaction_id,
        operation="document_ask",
        document_id=document_id,
        tool_used=None,
        source_count=len(generated.sources),
    )

    return DocumentAnswerResponse(
        document_id=document_id,
        answer=generated.answer,
        sources=[
            DocumentSearchResult(
                page_number=hit.record.page_number,
                chunk_index=hit.record.chunk_index,
                text=hit.record.text,
                score=hit.score,
            )
            for hit in generated.sources
        ],
    )