"""End-to-end orchestrator tests across all four routes."""
from __future__ import annotations

from app.orchestrator import Orchestrator
from app.schemas.chat import ChatRequest


async def _collect(orchestrator: Orchestrator, message: str) -> tuple[str, dict]:
    tokens: list[str] = []
    metadata: dict = {}
    async for event in orchestrator.stream(ChatRequest(message=message)):
        if event.get("type") == "token":
            tokens.append(event["data"])
        elif event.get("type") == "metadata":
            metadata = event
    return "".join(tokens), metadata


async def test_static_route_launch_date(orchestrator):
    text, meta = await _collect(orchestrator, "When is the launch date?")
    assert meta["route"] == "static"
    assert "2026" in text


async def test_chitchat_route(orchestrator):
    text, meta = await _collect(orchestrator, "Hi there, good morning!")
    assert meta["route"] == "chitchat"
    assert len(text) > 0


async def test_rag_route_password_reset(orchestrator):
    text, meta = await _collect(orchestrator, "How do I reset my password?")
    assert meta["route"] == "rag"
    assert meta["retrieved"] >= 1
    assert "password" in text.lower()


async def test_fallback_route_out_of_scope(orchestrator):
    text, meta = await _collect(orchestrator, "What's the weather today?")
    assert meta["route"] == "fallback"
    assert len(text) > 0


async def test_metadata_has_latency(orchestrator):
    _, meta = await _collect(orchestrator, "hello")
    assert "latency_ms" in meta
    assert isinstance(meta["latency_ms"], (int, float))
