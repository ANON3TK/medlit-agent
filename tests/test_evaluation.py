from medlit_agent.evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationQuestion,
    RetrievalEvaluationResult,
    evaluate_retrieval_cases,
    run_retrieval_evaluation,
)
from medlit_agent.vector_store import (
    SearchHit,
    VectorRecord,
)


def test_calculates_retrieval_hit_rate() -> None:
    cases = [
        RetrievalEvaluationCase(
            question="问题1",
            expected_page_number=1,
            retrieved_page_numbers=(1, 2),
        ),
        RetrievalEvaluationCase(
            question="问题2",
            expected_page_number=3,
            retrieved_page_numbers=(2, 3),
        ),
        RetrievalEvaluationCase(
            question="问题3",
            expected_page_number=5,
            retrieved_page_numbers=(4, 6),
        ),
    ]

    result = evaluate_retrieval_cases(cases)

    assert result == RetrievalEvaluationResult(
        total_cases=3,
        hit_count=2,
        hit_rate=2 / 3,
    )


def test_empty_evaluation_has_zero_hit_rate() -> None:
    result = evaluate_retrieval_cases([])

    assert result == RetrievalEvaluationResult(
        total_cases=0,
        hit_count=0,
        hit_rate=0.0,
    )

def make_search_hit(
    page_number: int,
) -> SearchHit:
    return SearchHit(
        record=VectorRecord(
            document_id="doc-1",
            page_number=page_number,
            chunk_index=0,
            text="测试证据",
            vector=(1.0, 0.0),
        ),
        score=0.9,
    )


def test_runs_retrieval_for_each_question() -> None:
    questions = [
        RetrievalEvaluationQuestion(
            question="问题1",
            expected_page_number=1,
        ),
        RetrievalEvaluationQuestion(
            question="问题2",
            expected_page_number=3,
        ),
    ]

    def retrieve(
        question: str,
    ) -> list[SearchHit]:
        if question == "问题1":
            return [
                make_search_hit(1),
                make_search_hit(2),
            ]

        return [
            make_search_hit(2),
        ]

    result = run_retrieval_evaluation(
        questions=questions,
        retrieve=retrieve,
    )

    assert result == RetrievalEvaluationResult(
        total_cases=2,
        hit_count=1,
        hit_rate=0.5,
    )