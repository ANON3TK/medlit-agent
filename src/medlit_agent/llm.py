"""Interfaces and clients for generating answers with a language model."""

from typing import Protocol
import os
import httpx
from medlit_agent.http_client import (
    post_json_with_retry,
)

class ChatModel(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...


class SiliconFlowChatModel:
    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        model: str = "deepseek-ai/DeepSeek-V3.2",
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60.0)
        self._model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = post_json_with_retry(
            client=self._client,
            url=(
                "https://api.siliconflow.cn/"
                "v1/chat/completions"
            ),
            headers={
                "Authorization": (
                    f"Bearer {self._api_key}"
                ),
            },
            json_body={
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "temperature": 0.0,
            },
        )
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

def create_siliconflow_chat_model() -> SiliconFlowChatModel:
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not configured."
        )

    return SiliconFlowChatModel(api_key=api_key)