"""RAG node — retrieve from Qdrant, then stream a grounded answer.

If retrieval is empty or weak (top score below the configured floor) we stream a
graceful 'no information' reply instead of hallucinating.
"""
from __future__ import annotations

import asyncio

from app.core.text_stream import stream_text
from app.dependencies import Dependencies
from app.graph.nodes._common import stream_and_collect
from app.graph.state import AgentState

_SYSTEM = (
    "You are a customer-support assistant. Answer using ONLY the provided "
    "documentation. If the answer isn't in it, say you don't have that information."
)
_NO_CONTEXT = (
    "I couldn't find that in our documentation. I'd recommend contacting our "
    "support team so a specialist can help with this specific question."
)


async def rag_node(state: AgentState, *, deps: Dependencies) -> dict:
    message = state["message"]
    # Offload the synchronous Qdrant + embedding call so it doesn't block the loop.
    chunks = await asyncio.to_thread(
        deps.vector_store.search, message, deps.settings.rag_top_k
    )
    top_score = chunks[0]["score"] if chunks else 0.0

    if not chunks or top_score < deps.settings.rag_min_score:
        text = await stream_and_collect(
            stream_text(_NO_CONTEXT, deps.settings.stream_token_delay_ms)
        )
        return {"context": chunks, "answer": text}

    text = await stream_and_collect(
        deps.llm.astream_answer(system=_SYSTEM, user=message, context=list(chunks))
    )
    return {"context": chunks, "answer": text}
