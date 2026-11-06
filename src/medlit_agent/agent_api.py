"""HTTP endpoints for the tool-calling agent."""
import logging
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import BaseModel
from medlit_agent.interaction_store import InteractionRecord
from medlit_agent.runtime import interaction_store
from medlit_agent.agent import (
    create_siliconflow_tool_router,
    run_pubmed_agent,
)
from medlit_agent.pubmed import PubMedClient
from medlit_agent.logging_utils import log_event

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


class AgentAskRequest(BaseModel):
    question: str


class PubMedArticleResponse(BaseModel):
    pubmed_id: str
    title: str
    authors: list[str]
    publication_date: str
    url: str


class AgentAskResponse(BaseModel):
    answer: str
    tool_used: str | None
    articles: list[PubMedArticleResponse]


@router.post(
    "/ask",
    response_model=AgentAskResponse,
)
def ask_agent(
    request: AgentAskRequest,
) -> AgentAskResponse:
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    router_model = create_siliconflow_tool_router()
    pubmed = PubMedClient()

    result = run_pubmed_agent(
        question=request.question,
        router=router_model,
        pubmed=pubmed,
    )

    interaction_id = str(uuid4())

    interaction_store.save(
        InteractionRecord(
            interaction_id=interaction_id,
            operation="agent_ask",
            question=request.question,
            answer=result.answer,
            document_id=None,
            tool_used=result.tool_used,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )
    )

    log_event(
        logger,
        "interaction_saved",
        interaction_id=interaction_id,
        operation="agent_ask",
        document_id=None,
        tool_used=result.tool_used,
        article_count=len(result.articles),
    )

    return AgentAskResponse(
        answer=result.answer,
        tool_used=result.tool_used,
        articles=[
            PubMedArticleResponse(
                pubmed_id=article.pubmed_id,
                title=article.title,
                authors=list(article.authors),
                publication_date=(
                    article.publication_date
                ),
                url=article.url,
            )
            for article in result.articles
        ],
    )