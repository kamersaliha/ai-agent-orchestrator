"""Shared node helpers for token streaming.

Text-generating nodes push tokens to LangGraph's *custom* stream so the API can
forward them to the client (and, later, a TTS engine). The writer is resolved
lazily and defensively so nodes also work when called directly in unit tests.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable


def get_token_writer() -> Callable[[dict], None]:
    """Return LangGraph's custom stream writer, or a no-op outside a run."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return lambda _payload: None


async def stream_and_collect(tokens: AsyncIterator[str]) -> str:
    """Forward every token to the custom stream and return the full text."""
    writer = get_token_writer()
    parts: list[str] = []
    async for token in tokens:
        parts.append(token)
        try:
            writer({"type": "token", "data": token})
        except Exception:
            # Never let a streaming hiccup break answer assembly.
            pass
    return "".join(parts)
