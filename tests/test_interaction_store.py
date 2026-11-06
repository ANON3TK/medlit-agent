from medlit_agent.interaction_store import (
    InMemoryInteractionStore,
    InteractionRecord,
    SQLiteInteractionStore,
)


def test_persists_interaction_across_store_instances(
    tmp_path,
) -> None:
    database_path = tmp_path / "medlit.db"

    interaction = InteractionRecord(
        interaction_id="interaction-1",
        operation="document_ask",
        question="房颤是什么？",
        answer="房颤是一种心律失常。",
        document_id="doc-1",
        tool_used=None,
        created_at="2026-10-14T10:00:00+00:00",
    )

    first_store = SQLiteInteractionStore(
        database_path=database_path,
    )
    first_store.save(interaction)

    reopened_store = SQLiteInteractionStore(
        database_path=database_path,
    )
    interactions = reopened_store.list_recent(
        limit=10,
    )

    assert interactions == [interaction]

def test_in_memory_store_lists_newest_interaction_first() -> None:
    store = InMemoryInteractionStore()

    older = InteractionRecord(
        interaction_id="interaction-1",
        operation="document_ask",
        question="旧问题",
        answer="旧回答",
        document_id="doc-1",
        tool_used=None,
        created_at="2026-10-14T10:00:00+00:00",
    )
    newer = InteractionRecord(
        interaction_id="interaction-2",
        operation="agent_ask",
        question="新问题",
        answer="新回答",
        document_id=None,
        tool_used="search_pubmed",
        created_at="2026-10-14T11:00:00+00:00",
    )

    store.save(older)
    store.save(newer)

    assert store.list_recent(limit=1) == [newer]