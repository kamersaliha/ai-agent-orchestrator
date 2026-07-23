"""FastAPI application factory + lifespan.

Build/seed all dependencies once on startup and expose them via ``app.state``.
Run with: ``uvicorn app.main:app --reload``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import chat, health
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.dependencies import build_dependencies
from app.orchestrator import Orchestrator

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "Starting %s (env=%s, llm_provider=%s)",
        settings.app_name,
        settings.environment,
        settings.llm_provider,
    )

    deps = build_dependencies(settings)
    app.state.orchestrator = Orchestrator(deps)
    logger.info("Orchestrator ready — accepting requests")

    yield

    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Agentic Support Orchestrator",
        description="Hybrid (static / chit-chat / RAG / fallback) support routing "
        "with LangGraph, FastAPI and Qdrant. Streams tokens over SSE.",
        version=__version__,
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
