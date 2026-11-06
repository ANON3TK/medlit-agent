"""Build grounded answers from retrieved document evidence."""

from dataclasses import dataclass

from medlit_agent.llm import ChatModel
from medlit_agent.vector_store import SearchHit


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: tuple[SearchHit, ...]


def generate_answer(
    question: str,
    hits: list[SearchHit],
    model: ChatModel,
) -> GeneratedAnswer:
    if not hits:
        return GeneratedAnswer(
            answer="未在文档中找到足够相关的证据。",
            sources=(),
        )

    evidence_sections = []

    for source_number, hit in enumerate(hits, start=1):
        evidence_sections.append(
            (
                f"[来源{source_number}，第"
                f"{hit.record.page_number}页]\n"
                f"{hit.record.text}"
            )
        )

    evidence_text = "\n\n".join(evidence_sections)

    system_prompt = (
        "你是一个医学文献助手。"
        "只能根据用户提供的证据回答问题，"
        "不能补充证据之外的事实。"
        "回答中的事实必须使用"
        "[来源编号，第几页]标注出处。"
        "如果证据不足，必须明确说明。"
    )

    user_prompt = (
        f"问题：{question}\n\n"
        f"证据：\n{evidence_text}"
    )

    answer = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return GeneratedAnswer(
        answer=answer,
        sources=tuple(hits),
    )