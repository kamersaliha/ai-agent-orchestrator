"""Fallback node — safe, bounded reply for out-of-scope / unsafe inputs."""
from __future__ import annotations

from app.core.text_stream import stream_text
from app.dependencies import Dependencies
from app.graph.nodes._common import stream_and_collect
from app.graph.state import AgentState

_FALLBACK_MESSAGE = (
    "I'm sorry, but I can only help with questions about our product, your "
    "account, and billing. Could you rephrase your question? I can also connect "
    "you with a human agent if you'd prefer."
)


async def fallback_node(state: AgentState, *, deps: Dependencies) -> dict:
    text = await stream_and_collect(
        stream_text(_FALLBACK_MESSAGE, deps.settings.stream_token_delay_ms)
    )
    return {"answer": text}
