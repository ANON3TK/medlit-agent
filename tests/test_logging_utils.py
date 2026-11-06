import json
import logging

from medlit_agent.logging_utils import log_event


def test_logs_event_as_json(
    caplog,
) -> None:
    logger = logging.getLogger(
        "medlit_agent.test"
    )

    with caplog.at_level(
        logging.INFO,
        logger="medlit_agent.test",
    ):
        log_event(
            logger,
            "interaction_saved",
            operation="document_ask",
            document_id="doc-1",
        )

    assert len(caplog.records) == 1

    payload = json.loads(
        caplog.records[0].getMessage()
    )

    assert payload == {
        "document_id": "doc-1",
        "event": "interaction_saved",
        "operation": "document_ask",
    }