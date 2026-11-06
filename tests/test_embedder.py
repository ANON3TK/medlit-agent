"""Tests for embedding text chunks."""
import json

import httpx

from medlit_agent.chunker import TextChunk
from medlit_agent.embedder import SiliconFlowEmbedder, embed_chunks


class LengthEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_embeds_each_chunk_in_order() -> None:
    chunks = [
        TextChunk(page_number=1, chunk_index=0, text="abc"),
        TextChunk(page_number=1, chunk_index=1, text="de"),
    ]

    vectors = embed_chunks(chunks, LengthEmbedder())

    assert vectors == [
        [3.0, 1.0],
        [2.0, 1.0],
    ]


def test_siliconflow_embedder_retries_temporary_error(
    monkeypatch,
) -> None:
    request_count = 0

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        assert request.headers[
            "Authorization"
        ] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "BAAI/bge-m3",
            "input": "hello",
            "encoding_format": "float",
        }

        if request_count == 1:
            return httpx.Response(503)

        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "embedding": [
                            0.1,
                            -0.2,
                            0.3,
                        ]
                    },
                ]
            },
        )

    monkeypatch.setattr(
        "medlit_agent.http_client.time.sleep",
        lambda _delay: None,
    )

    client = httpx.Client(
        transport=httpx.MockTransport(
            handle_request
        ),
    )
    embedder = SiliconFlowEmbedder(
        api_key="test-key",
        client=client,
    )

    assert embedder.embed("hello") == [
        0.1,
        -0.2,
        0.3,
    ]
    assert request_count == 2
