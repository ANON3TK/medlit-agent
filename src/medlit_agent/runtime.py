"""Shared storage instances used while the application is running."""

import os

from medlit_agent.storage import (
    create_interaction_store,
    create_stores,
)


database_path = os.getenv(
    "MEDLIT_DATABASE_PATH"
)

document_store, vector_store = create_stores(
    database_path=database_path,
)

interaction_store = create_interaction_store(
    database_path=database_path,
)