"""Domain exceptions and FastAPI exception handlers."""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class OrchestratorError(Exception):
    """Base class for orchestration-layer failures."""


class RoutingError(OrchestratorError):
    """Raised when the router cannot produce a usable decision."""


class VectorStoreError(OrchestratorError):
    """Raised when the vector store cannot serve a request."""


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON error handlers so the API never leaks raw tracebacks."""

    @app.exception_handler(OrchestratorError)
    async def _orchestrator_error(_: Request, exc: OrchestratorError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": exc.__class__.__name__, "detail": str(exc)},
        )
