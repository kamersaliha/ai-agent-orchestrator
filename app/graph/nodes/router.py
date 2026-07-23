"""Router node — the heart of the hybrid strategy.

1. Try the fast, cheap semantic router first (offloaded off the event loop).
2. If it isn't confident enough, escalate to the LLM classifier (strict JSON),
   degrading to the safe FALLBACK route on error or sub-threshold confidence.

The resulting :class:`RouteDecision` drives the conditional edge in the graph.
"""
from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.dependencies import Dependencies
from app.graph.state import AgentState
from app.schemas.routing import Route, RouteDecision

logger = get_logger(__name__)


async def router_node(state: AgentState, *, deps: Dependencies) -> dict:
    message = state["message"]

    # The semantic router is synchronous (embedding + cosine); offload it so it
    # never blocks the event loop that is serving concurrent streams.
    decision = await asyncio.to_thread(deps.semantic_router.route, message)

    if decision is None:
        logger.debug("Semantic router below threshold; escalating to LLM classifier")
        decision = await _classify_with_fallback(message, state, deps)

    logger.info(
        "Routed message | route=%s intent=%s confidence=%.2f source=%s",
        decision.route.value,
        decision.intent,
        decision.confidence,
        decision.source,
    )
    return {"decision": decision}


async def _classify_with_fallback(
    message: str, state: AgentState, deps: Dependencies
) -> RouteDecision:
    """Run the LLM classifier, degrading to FALLBACK on error or low confidence."""
    try:
        decision = await deps.llm.classify(message, state.get("history"))
    except Exception:
        # A transient classifier failure (API/network/validation) must not kill
        # the request — degrade gracefully to the safe FALLBACK route.
        logger.exception("LLM classifier failed; routing to FALLBACK")
        return RouteDecision(
            route=Route.FALLBACK,
            intent="classifier_error",
            confidence=0.0,
            source="default",
            rationale="classifier error",
        )

    floor = deps.settings.llm_router_min_confidence
    if not decision.is_confident(floor):
        logger.info(
            "LLM confidence %.2f below floor %.2f; routing to FALLBACK",
            decision.confidence,
            floor,
        )
        return RouteDecision(
            route=Route.FALLBACK,
            intent="low_confidence",
            confidence=decision.confidence,
            entities=decision.entities,
            source="default",
            rationale="below llm confidence floor",
        )
    return decision
