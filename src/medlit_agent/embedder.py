"""Interfaces for converting text into embedding vectors."""

from typing import Protocol
from medlit_agent.chunker import TextChunk
from medlit_agent.http_client import (
    post_json_with_retry,
)
import httpx
import os


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class SiliconFlowEmbedder:
    def __init__(
            self,
            api_key: str,
            client: httpx.Client | None = None,
            model: str = "BAAI/bge-m3",
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=30.0)
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = post_json_with_retry(
            client=self._client,
            url=(
                "https://api.siliconflow.cn/"
                "v1/embeddings"
            ),
            headers={
                "Authorization": (
                    f"Bearer {self._api_key}"
                ),
            },
            json_body={
                "model": self._model,
                "input": text,
                "encoding_format": "float",
            },
        )

        response.raise_for_status()
        data = response.json()

        return [
            float(value)
            for value in data["data"][0]["embedding"]
        ]


def create_siliconflow_embedder() -> SiliconFlowEmbedder:
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not configured."
        )

    return SiliconFlowEmbedder(api_key=api_key)

def embed_chunks(
        chunks: list[TextChunk],
        embedder: Embedder,
) -> list[list[float]]:
    return [embedder.embed(chunk.text) for chunk in chunks]
