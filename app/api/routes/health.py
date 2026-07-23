"""Liveness and readiness probes."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness: dependencies (orchestrator) have been wired."""
    is_ready = getattr(request.app.state, "orchestrator", None) is not None
    return {"status": "ready" if is_ready else "initialising", "ready": is_ready}
