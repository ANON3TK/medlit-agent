import json

import httpx

from medlit_agent.agent import (
    AgentResult,
    SiliconFlowToolRouter,
    ToolCall,
    execute_tool_call,
    run_pubmed_agent,
)
from medlit_agent.pubmed import PubMedArticle


def test_model_requests_pubmed_search() -> None:
    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content)

        assert payload["tool_choice"] == "auto"
        assert (
            payload["tools"][0]["function"]["name"]
            == "search_pubmed"
        )
        assert (
            payload["messages"][0]["content"]
            == "请帮我搜索最新的房颤消融文献。"
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_pubmed",
                                        "arguments": json.dumps(
                                            {
                                                "query": (
                                                    "atrial fibrillation "
                                                    "ablation"
                                                ),
                                                "limit": 3,
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
    )
    router = SiliconFlowToolRouter(
        api_key="test-key",
        client=client,
    )

    tool_call = router.choose_tool(
        question="请帮我搜索最新的房颤消融文献。"
    )

    assert tool_call == ToolCall(
        call_id="call-1",
        name="search_pubmed",
        arguments={
            "query": "atrial fibrillation ablation",
            "limit": 3,
        },
    )

class RecordingPubMedClient:
    def __init__(self) -> None:
        self.query = ""
        self.limit = 0

    def search_articles(
        self,
        query: str,
        limit: int,
    ) -> list[PubMedArticle]:
        self.query = query
        self.limit = limit

        return [
            PubMedArticle(
                pubmed_id="12345678",
                title="Atrial fibrillation study.",
                authors=("Zhang Y",),
                publication_date="2026 Jan",
                url=(
                    "https://pubmed.ncbi.nlm.nih.gov/"
                    "12345678/"
                ),
            )
        ]


def test_executes_pubmed_tool_call() -> None:
    pubmed = RecordingPubMedClient()
    tool_call = ToolCall(
        call_id="call-1",
        name="search_pubmed",
        arguments={
            "query": "atrial fibrillation",
            "limit": 3,
        },
    )

    articles = execute_tool_call(
        tool_call=tool_call,
        pubmed=pubmed,
    )

    assert pubmed.query == "atrial fibrillation"
    assert pubmed.limit == 3
    assert articles == [
        PubMedArticle(
            pubmed_id="12345678",
            title="Atrial fibrillation study.",
            authors=("Zhang Y",),
            publication_date="2026 Jan",
            url=(
                "https://pubmed.ncbi.nlm.nih.gov/"
                "12345678/"
            ),
        )
    ]

def test_generates_answer_after_pubmed_tool_result() -> None:
    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content)
        messages = payload["messages"]

        assert messages[0] == {
            "role": "user",
            "content": "请搜索最新的房颤文献。",
        }
        assert messages[1]["role"] == "assistant"
        assert (
            messages[1]["tool_calls"][0]["id"]
            == "call-1"
        )
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call-1"

        tool_result = json.loads(
            messages[2]["content"]
        )
        assert tool_result[0]["pubmed_id"] == "12345678"
        assert (
            tool_result[0]["title"]
            == "Atrial fibrillation study."
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "找到一篇相关文献："
                                "Atrial fibrillation study. "
                                "[PMID: 12345678]"
                            ),
                        }
                    }
                ]
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
    )
    router = SiliconFlowToolRouter(
        api_key="test-key",
        client=client,
    )
    tool_call = ToolCall(
        call_id="call-1",
        name="search_pubmed",
        arguments={
            "query": "atrial fibrillation",
            "limit": 1,
        },
    )
    articles = [
        PubMedArticle(
            pubmed_id="12345678",
            title="Atrial fibrillation study.",
            authors=("Zhang Y",),
            publication_date="2026 Jan",
            url=(
                "https://pubmed.ncbi.nlm.nih.gov/"
                "12345678/"
            ),
        )
    ]

    answer = router.answer_with_tool_result(
        question="请搜索最新的房颤文献。",
        tool_call=tool_call,
        articles=articles,
    )

    assert answer == (
        "找到一篇相关文献："
        "Atrial fibrillation study. "
        "[PMID: 12345678]"
    )

class FixedToolRouter:
    def choose_tool(
        self,
        question: str,
    ) -> ToolCall | None:
        assert question == "搜索房颤文献"

        return ToolCall(
            call_id="call-1",
            name="search_pubmed",
            arguments={
                "query": "atrial fibrillation",
                "limit": 3,
            },
        )

    def answer_with_tool_result(
        self,
        question: str,
        tool_call: ToolCall,
        articles: list[PubMedArticle],
    ) -> str:
        assert question == "搜索房颤文献"
        assert tool_call.name == "search_pubmed"
        assert articles[0].pubmed_id == "12345678"

        return "找到一篇房颤相关文献。[PMID: 12345678]"


def test_runs_complete_pubmed_agent_flow() -> None:
    router = FixedToolRouter()
    pubmed = RecordingPubMedClient()

    result = run_pubmed_agent(
        question="搜索房颤文献",
        router=router,
        pubmed=pubmed,
    )

    assert result == AgentResult(
        answer="找到一篇房颤相关文献。[PMID: 12345678]",
        tool_used="search_pubmed",
        articles=(
            PubMedArticle(
                pubmed_id="12345678",
                title="Atrial fibrillation study.",
                authors=("Zhang Y",),
                publication_date="2026 Jan",
                url=(
                    "https://pubmed.ncbi.nlm.nih.gov/"
                    "12345678/"
                ),
            ),
        ),
    )