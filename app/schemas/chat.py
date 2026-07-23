"""Chat I/O contracts and the streamed event envelope."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.routing import Entity, Route


class Message(BaseModel):
    """A single conversational turn (history is optional for the demo)."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Inbound payload for ``POST /chat``."""

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None)
    history: list[Message] = Field(default_factory=list)


# --- Streamed events ---------------------------------------------------------
# Every event is a small JSON object emitted as one SSE message. Token events
# carry the text the TTS layer will speak; the final metadata event carries the
# routing decision and latency for analytics.


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    data: str


class MetadataEvent(BaseModel):
    type: Literal["metadata"] = "metadata"
    route: Route
    intent: str
    confidence: float
    source: str
    entities: list[Entity] = Field(default_factory=list)
    latency_ms: float
    retrieved: int = Field(default=0, description="Number of RAG chunks used.")


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str


class ChatResponse(BaseModel):
    """Non-streaming convenience response (used by tests / debugging)."""

    answer: str
    route: Route
    intent: str
    confidence: float
    entities: list[Entity] = Field(default_factory=list)
