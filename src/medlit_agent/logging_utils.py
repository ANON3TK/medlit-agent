"""Helpers for writing structured JSON logs."""

import json
import logging


def log_event(
    logger: logging.Logger,
    event: str,
    **fields: object,
) -> None:
    payload = {
        "event": event,
        **fields,
    }

    message = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    logger.info(message)