"""Metrics for evaluating retrieval quality."""

from dataclasses import dataclass
from collections.abc import Callable

from medlit_agent.vector_store import SearchHit

@dataclass(frozen=True)
class RetrievalEvaluationQuestion:
    question: str
    expected_page_number: int



@dataclass(frozen=True)
class RetrievalEvaluationCase:
    question: str
    expected_page_number: int
    retrieved_page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    total_cases: int
    hit_count: int
    hit_rate: float


def evaluate_retrieval_cases(
    cases: list[RetrievalEvaluationCase],
) -> RetrievalEvaluationResult:
    hit_count = 0

    for case in cases:
        if (
            case.expected_page_number
            in case.retrieved_page_numbers
        ):
            hit_count += 1

    total_cases = len(cases)

    if total_cases == 0:
        hit_rate = 0.0
    else:
        hit_rate = hit_count / total_cases

    return RetrievalEvaluationResult(
        total_cases=total_cases,
        hit_count=hit_count,
        hit_rate=hit_rate,
    )


def run_retrieval_evaluation(
    questions: list[RetrievalEvaluationQuestion],
    retrieve: Callable[
        [str],
        list[SearchHit],
    ],
) -> RetrievalEvaluationResult:
    cases = []

    for question in questions:
        hits = retrieve(question.question)

        retrieved_page_numbers = tuple(
            hit.record.page_number
            for hit in hits
        )

        cases.append(
            RetrievalEvaluationCase(
                question=question.question,
                expected_page_number=(
                    question.expected_page_number
                ),
                retrieved_page_numbers=(
                    retrieved_page_numbers
                ),
            )
        )

    return evaluate_retrieval_cases(cases)