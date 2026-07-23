"""Streaming chat endpoint.

``POST /chat`` returns Server-Sent Events over a ``StreamingResponse``. Each SSE
message carries one JSON event: a stream of ``token`` events followed by a single
``metadata`` event. The format is TTS-friendly (raw token text) and analytics-
friendly (routing decision + latency).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_orchestrator
from app.core.logging import get_logger
from app.orchestrator import Orchestrator
from app.schemas.chat import ChatRequest

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush promptly
}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in orchestrator.stream(payload):
                yield _sse(event)
        except Exception:  # backstop: surface a generic error, never leak internals
            logger.exception("Streaming failed")
            yield _sse({"type": "error", "detail": "internal error processing request"})
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
