import json

import httpx

from medlit_agent.llm import SiliconFlowChatModel


def test_siliconflow_chat_model_retries_temporary_error(
    monkeypatch,
) -> None:
    request_count = 0

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        payload = json.loads(
            request.content
        )

        assert request.headers[
            "Authorization"
        ] == "Bearer test-key"
        assert payload == {
            "model": "deepseek-ai/DeepSeek-V3.2",
            "messages": [
                {
                    "role": "system",
                    "content": "只根据证据回答。",
                },
                {
                    "role": "user",
                    "content": (
                        "证据：房颤是一种心律失常。"
                    ),
                },
            ],
            "temperature": 0.0,
        }

        if request_count == 1:
            return httpx.Response(503)

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "房颤是一种心律失常。"
                            )
                        }
                    }
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
    model = SiliconFlowChatModel(
        api_key="test-key",
        client=client,
    )

    answer = model.generate(
        system_prompt="只根据证据回答。",
        user_prompt="证据：房颤是一种心律失常。",
    )

    assert answer == "房颤是一种心律失常。"
    assert request_count == 2