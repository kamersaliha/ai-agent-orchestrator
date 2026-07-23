"""Chit-chat node — fast conversational reply via the LLM (no retrieval)."""
from __future__ import annotations

from app.dependencies import Dependencies
from app.graph.nodes._common import stream_and_collect
from app.graph.state import AgentState

_SYSTEM = (
    "You are a warm, concise customer-support assistant. Reply in 1-2 friendly "
    "sentences. Do not invent product facts; for those, offer to look them up."
)


async def chitchat_node(state: AgentState, *, deps: Dependencies) -> dict:
    text = await stream_and_collect(
        deps.llm.astream_answer(system=_SYSTEM, user=state["message"])
    )
    return {"answer": text}
