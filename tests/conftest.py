"""Shared pytest fixtures.

Everything runs offline: the mock LLM provider + deterministic embedder + an
in-memory Qdrant, with token delay disabled for fast tests.
"""
from __future__ import annotations

import pytest

from app.core.config import Settings
from app.dependencies import Dependencies, build_dependencies
from app.orchestrator import Orchestrator


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="mock",
        qdrant_location=":memory:",
        stream_token_delay_ms=0,
    )


@pytest.fixture
def deps(settings: Settings) -> Dependencies:
    return build_dependencies(settings)


@pytest.fixture
def orchestrator(deps: Dependencies) -> Orchestrator:
    return Orchestrator(deps)
