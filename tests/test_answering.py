from medlit_agent.answering import generate_answer
from medlit_agent.vector_store import SearchHit, VectorRecord


class RecordingChatModel:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return "房颤是一种心律失常。[来源1，第2页]"


def test_generates_answer_from_search_hits() -> None:
    hit = SearchHit(
        record=VectorRecord(
            document_id="doc-1",
            page_number=2,
            chunk_index=0,
            text="房颤是一种常见的心律失常。",
            vector=(1.0, 0.0),
        ),
        score=0.9,
    )
    model = RecordingChatModel()

    result = generate_answer(
        question="房颤是什么？",
        hits=[hit],
        model=model,
    )

    assert result.answer == "房颤是一种心律失常。[来源1，第2页]"
    assert result.sources == (hit,)
    assert "房颤是什么？" in model.user_prompt
    assert "房颤是一种常见的心律失常。" in model.user_prompt
    assert "第2页" in model.user_prompt

class FailIfCalledChatModel:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise AssertionError(
            "The chat model must not be called without evidence."
        )


def test_does_not_call_model_without_evidence() -> None:
    result = generate_answer(
        question="文档中没有涉及的问题",
        hits=[],
        model=FailIfCalledChatModel(),
    )

    assert result.answer == "未在文档中找到足够相关的证据。"
    assert result.sources == ()