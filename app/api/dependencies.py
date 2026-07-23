"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Request

from app.orchestrator import Orchestrator


def get_orchestrator(request: Request) -> Orchestrator:
    """Return the orchestrator built once during app startup (lifespan)."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:  # pragma: no cover - defensive
        raise RuntimeError("Orchestrator is not initialised yet")
    return orchestrator
