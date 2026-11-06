from fastapi.testclient import TestClient

from medlit_agent import agent_api
from medlit_agent.agent import AgentResult
from medlit_agent.main import app
from medlit_agent.pubmed import PubMedArticle
from medlit_agent.interaction_store import InMemoryInteractionStore

client = TestClient(app)


def test_agent_asks_model_to_search_pubmed(
    monkeypatch,
) -> None:
    article = PubMedArticle(
        pubmed_id="12345678",
        title="Atrial fibrillation study.",
        authors=("Zhang Y",),
        publication_date="2026 Jan",
        url=(
            "https://pubmed.ncbi.nlm.nih.gov/"
            "12345678/"
        ),
    )

    interaction_store = InMemoryInteractionStore()

    monkeypatch.setattr(
        agent_api,
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
        agent_api,
        "log_event",
        fake_log_event,
        raising=False,
    )

    def fake_run_pubmed_agent(
        question,
        router,
        pubmed,
    ) -> AgentResult:
        assert question == "搜索房颤文献"

        return AgentResult(
            answer=(
                "找到一篇房颤相关文献。"
                "[PMID: 12345678]"
            ),
            tool_used="search_pubmed",
            articles=(article,),
        )

    monkeypatch.setattr(
        agent_api,
        "run_pubmed_agent",
        fake_run_pubmed_agent,
    )
    monkeypatch.setattr(
        agent_api,
        "create_siliconflow_tool_router",
        lambda: object(),
    )
    monkeypatch.setattr(
        agent_api,
        "PubMedClient",
        lambda: object(),
    )

    response = client.post(
        "/agent/ask",
        json={
            "question": "搜索房颤文献",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "找到一篇房颤相关文献。"
            "[PMID: 12345678]"
        ),
        "tool_used": "search_pubmed",
        "articles": [
            {
                "pubmed_id": "12345678",
                "title": "Atrial fibrillation study.",
                "authors": ["Zhang Y"],
                "publication_date": "2026 Jan",
                "url": (
                    "https://pubmed.ncbi.nlm.nih.gov/"
                    "12345678/"
                ),
            }
        ],
    }

    interactions = interaction_store.list_recent(limit=1)

    assert len(interactions) == 1

    interaction = interactions[0]

    assert interaction.operation == "agent_ask"
    assert interaction.question == "搜索房颤文献"
    assert interaction.answer == (
        "找到一篇房颤相关文献。"
        "[PMID: 12345678]"
    )
    assert interaction.document_id is None
    assert interaction.tool_used == "search_pubmed"
    assert interaction.created_at

    assert logged_events == [
        {
            "event": "interaction_saved",
            "interaction_id": (
                interaction.interaction_id
            ),
            "operation": "agent_ask",
            "document_id": None,
            "tool_used": "search_pubmed",
            "article_count": 1,
        }
    ]