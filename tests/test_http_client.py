import httpx

from medlit_agent.http_client import post_json_with_retry


def test_retries_temporary_connection_errors() -> None:
    attempts = 0

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise httpx.ConnectError(
                "Temporary connection failure.",
                request=request,
            )

        return httpx.Response(
            200,
            json={"status": "ok"},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
    )

    response = post_json_with_retry(
        client=client,
        url="https://example.com/test",
        headers={"Authorization": "Bearer test-key"},
        json_body={"input": "hello"},
        max_attempts=3,
        backoff_seconds=0.0,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert attempts == 3