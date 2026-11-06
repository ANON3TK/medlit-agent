"""FastAPI application entry point for MedLit Agent."""

import httpx

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from medlit_agent.agent_api import router as agent_router
from medlit_agent.documents import router as documents_router
from medlit_agent.interactions_api import (
    router as interactions_router,
)

app = FastAPI(
    title="MedLit Agent",
    version="0.1.0",
)

app.include_router(documents_router)
app.include_router(agent_router)
app.include_router(interactions_router)
@app.exception_handler(httpx.HTTPError)
async def handle_external_ai_error(
    request: Request,
    exc: httpx.HTTPError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "detail": "External AI service is unavailable."
        },
    )

@app.get("/health")
def health() -> dict[str, str]:
    """Return a small response proving that the service is running."""
    return {
        "status": "ok",
        "service": "medlit-agent",
    }
