"""Agent routing for deciding when to call external tools."""

import json
from dataclasses import dataclass
import os
import httpx

from medlit_agent.http_client import post_json_with_retry
from medlit_agent.pubmed import (
    PubMedArticle,
    PubMedClient,
)

PUBMED_TOOL = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": (
            "Search PubMed for current biomedical literature. "
            "Use this when the user explicitly asks to find, "
            "search, or discover external medical papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The biomedical search query, "
                        "preferably in English."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of articles to return."
                    ),
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": [
                "query",
                "limit",
            ],
        },
    },
}


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, object]

@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_used: str | None
    articles: tuple[PubMedArticle, ...]


class SiliconFlowToolRouter:
    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        model: str = "deepseek-ai/DeepSeek-V4-Flash",
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(
            timeout=60.0,
        )
        self._model = model

    def choose_tool(
        self,
        question: str,
    ) -> ToolCall | None:
        response = post_json_with_retry(
            client=self._client,
            url=(
                "https://api.siliconflow.cn/"
                "v1/chat/completions"
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json_body={
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "tools": [PUBMED_TOOL],
                "tool_choice": "auto",
                "temperature": 0.0,
            },
        )

        data = response.json()
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return None

        first_tool_call = tool_calls[0]
        function = first_tool_call["function"]
        arguments = json.loads(
            function["arguments"]
        )

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool arguments must be a JSON object."
            )

        return ToolCall(
            call_id=first_tool_call["id"],
            name=function["name"],
            arguments=arguments,
        )

    def answer_with_tool_result(
        self,
        question: str,
        tool_call: ToolCall,
        articles: list[PubMedArticle],
    ) -> str:
        article_data = [
            {
                "pubmed_id": article.pubmed_id,
                "title": article.title,
                "authors": list(article.authors),
                "publication_date": (
                    article.publication_date
                ),
                "url": article.url,
            }
            for article in articles
        ]

        response = post_json_with_retry(
            client=self._client,
            url=(
                "https://api.siliconflow.cn/"
                "v1/chat/completions"
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json_body={
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(
                                        tool_call.arguments,
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.call_id
                        ),
                        "content": json.dumps(
                            article_data,
                            ensure_ascii=False,
                        ),
                    },
                ],
                "tools": [PUBMED_TOOL],
                "tool_choice": "auto",
                "temperature": 0.0,
            },
        )

        data = response.json()
        content = (
            data["choices"][0]["message"]["content"]
        )

        if not isinstance(content, str):
            raise ValueError(
                "The model did not return an answer."
            )

        return content.strip()

def create_siliconflow_tool_router() -> SiliconFlowToolRouter:
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not configured."
        )

    return SiliconFlowToolRouter(api_key=api_key)


def execute_tool_call(
    tool_call: ToolCall,
    pubmed: PubMedClient,
) -> list[PubMedArticle]:
    if tool_call.name != "search_pubmed":
        raise ValueError(
            f"Unsupported tool: {tool_call.name}"
        )

    query = tool_call.arguments.get("query")
    limit = tool_call.arguments.get("limit")

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "PubMed query must be a non-empty string."
        )

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
    ):
        raise ValueError(
            "PubMed limit must be an integer."
        )

    if limit < 1 or limit > 5:
        raise ValueError(
            "PubMed limit must be between 1 and 5."
        )

    return pubmed.search_articles(
        query=query.strip(),
        limit=limit,
    )

def run_pubmed_agent(
    question: str,
    router: SiliconFlowToolRouter,
    pubmed: PubMedClient,
) -> AgentResult:
    tool_call = router.choose_tool(
        question=question,
    )

    if tool_call is None:
        return AgentResult(
            answer="当前问题没有触发 PubMed 搜索工具。",
            tool_used=None,
            articles=(),
        )

    articles = execute_tool_call(
        tool_call=tool_call,
        pubmed=pubmed,
    )

    answer = router.answer_with_tool_result(
        question=question,
        tool_call=tool_call,
        articles=articles,
    )

    return AgentResult(
        answer=answer,
        tool_used=tool_call.name,
        articles=tuple(articles),
    )