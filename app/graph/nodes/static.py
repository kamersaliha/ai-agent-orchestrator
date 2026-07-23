"""Static node — streams a hardcoded answer for critical FAQs (no LLM, no RAG)."""
from __future__ import annotations

from app.core.text_stream import stream_text
from app.dependencies import Dependencies
from app.graph.nodes._common import stream_and_collect
from app.graph.state import AgentState

_NO_ANSWER = (
    "I have details about that on file, but I couldn't pull the exact answer. "
    "Let me connect you with a specialist who can help."
)


async def static_node(state: AgentState, *, deps: Dependencies) -> dict:
    decision = state["decision"]
    answer = deps.static_store.get(decision.intent) or _NO_ANSWER
    text = await stream_and_collect(
        stream_text(answer, deps.settings.stream_token_delay_ms)
    )
    return {"answer": text}
