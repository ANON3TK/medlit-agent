"""HTTP endpoints for interaction history."""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from medlit_agent import runtime


router = APIRouter(
    prefix="/interactions",
    tags=["interactions"],
)


class InteractionResponse(BaseModel):
    interaction_id: str
    operation: str
    question: str
    answer: str
    document_id: str | None
    tool_used: str | None
    created_at: str


@router.get(
    "",
    response_model=list[InteractionResponse],
)
def list_interactions(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> list[InteractionResponse]:
    interactions = (
        runtime.interaction_store.list_recent(
            limit=limit,
        )
    )

    return [
        InteractionResponse(
            interaction_id=interaction.interaction_id,
            operation=interaction.operation,
            question=interaction.question,
            answer=interaction.answer,
            document_id=interaction.document_id,
            tool_used=interaction.tool_used,
            created_at=interaction.created_at,
        )
        for interaction in interactions
    ]