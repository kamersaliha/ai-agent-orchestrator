"""LangGraph agent state.

A ``TypedDict`` is used (rather than a Pydantic model) because LangGraph merges
partial dict updates returned by each node into the running state. Nodes return
only the keys they change.
"""
from __future__ import annotations

from typing import TypedDict

from app.schemas.chat import Message
from app.schemas.routing import RouteDecision


class RetrievedChunk(TypedDict):
    """A single document chunk returned from the vector store."""

    id: str
    text: str
    score: float
    source: str


class AgentState(TypedDict, total=False):
    """Shared state threaded through the graph.

    ``total=False`` so nodes may populate keys incrementally.
    """

    # Inputs
    message: str
    history: list[Message]

    # Produced by the router node
    decision: RouteDecision

    # Produced by the RAG node
    context: list[RetrievedChunk]

    # Final answer assembled by a handler node (full text; tokens are streamed
    # separately via the custom stream writer).
    answer: str
