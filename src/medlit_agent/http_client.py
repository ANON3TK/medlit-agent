"""Shared HTTP helpers for calling external AI services."""

import time

import httpx


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def post_json_with_retry(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    json_body: dict[str, object],
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
) -> httpx.Response:
    if max_attempts <= 0:
        raise ValueError(
            "max_attempts must be positive."
        )

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(
                url,
                headers=headers,
                json=json_body,
            )

            if (
                response.status_code
                not in RETRYABLE_STATUS_CODES
            ):
                response.raise_for_status()
                return response

            if attempt == max_attempts:
                response.raise_for_status()

        except httpx.RequestError:
            if attempt == max_attempts:
                raise

        delay = backoff_seconds * (2 ** (attempt - 1))
        time.sleep(delay)

    raise RuntimeError(
        "The HTTP request ended unexpectedly."
    )