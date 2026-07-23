"""Orchestrator — runs the graph and yields a unified stream of events.

It consumes LangGraph's multiplexed stream (``custom`` token events + ``values``
state snapshots), forwards token events verbatim, and appends a final
``metadata`` event with the routing decision and latency for the caller / TTS
layer / analytics.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator

from app.core.logging import get_logger
from app.dependencies import Dependencies
from app.graph.builder import build_graph
from app.graph.state import AgentState
from app.schemas.chat import ChatRequest, MetadataEvent
from app.schemas.routing import Route, RouteDecision

logger = get_logger(__name__)


class Orchestrator:
    """Stateless-per-request driver around a compiled graph."""

    def __init__(self, deps: Dependencies) -> None:
        self._deps = deps
        self._graph = build_graph(deps)

    async def stream(self, request: ChatRequest) -> AsyncIterator[dict]:
        """Yield token events, then **always** a terminal ``metadata`` event.

        If a node raises mid-stream we emit a generic ``error`` event followed by
        a best-effort ``metadata`` event, so every turn terminates with the
        contract the client / TTS layer expects.
        """
        initial: AgentState = {"message": request.message, "history": request.history}

        start = time.perf_counter()
        final_state: dict = {}
        try:
            async for mode, chunk in self._graph.astream(
                initial, stream_mode=["custom", "values"]
            ):
                if mode == "custom":
                    yield chunk  # already a {"type": "token", "data": ...} dict
                elif mode == "values":
                    final_state = chunk
        except Exception:
            logger.exception("Graph execution failed mid-stream")
            yield {"type": "error", "detail": "internal error while generating the response"}

        yield self._build_metadata(final_state, start)

    def _build_metadata(self, final_state: dict, start: float) -> dict:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        decision: RouteDecision | None = final_state.get("decision")
        if decision is None:
            logger.warning("No decision in final state; emitting fallback metadata")
            decision = RouteDecision(
                route=Route.FALLBACK, intent="error", confidence=0.0, source="default"
            )
        return MetadataEvent(
            route=decision.route,
            intent=decision.intent,
            confidence=decision.confidence,
            source=decision.source,
            entities=decision.entities,
            latency_ms=latency_ms,
            retrieved=len(final_state.get("context") or []),
        ).model_dump(mode="json")
