from fastapi.testclient import TestClient

from medlit_agent import runtime
from medlit_agent.interaction_store import (
    InMemoryInteractionStore,
    InteractionRecord,
)
from medlit_agent.main import app


client = TestClient(app)


def test_lists_recent_interactions(
    monkeypatch,
) -> None:
    interaction_store = InMemoryInteractionStore()

    interaction_store.save(
        InteractionRecord(
            interaction_id="interaction-old",
            operation="document_ask",
            question="旧问题",
            answer="旧回答",
            document_id="doc-1",
            tool_used=None,
            created_at="2026-10-15T08:00:00+00:00",
        )
    )
    interaction_store.save(
        InteractionRecord(
            interaction_id="interaction-new",
            operation="agent_ask",
            question="新问题",
            answer="新回答",
            document_id=None,
            tool_used="search_pubmed",
            created_at="2026-10-16T08:00:00+00:00",
        )
    )

    monkeypatch.setattr(
        runtime,
        "interaction_store",
        interaction_store,
    )

    response = client.get(
        "/interactions",
        params={"limit": 1},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "interaction_id": "interaction-new",
            "operation": "agent_ask",
            "question": "新问题",
            "answer": "新回答",
            "document_id": None,
            "tool_used": "search_pubmed",
            "created_at": (
                "2026-10-16T08:00:00+00:00"
            ),
        }
    ]

def test_rejects_invalid_interaction_limits() -> None:
    zero_response = client.get(
        "/interactions",
        params={"limit": 0},
    )
    too_large_response = client.get(
        "/interactions",
        params={"limit": 101},
    )

    assert zero_response.status_code == 422
    assert too_large_response.status_code == 422