"""Graph factory — assembles the router + 4 handler nodes into a StateGraph."""
from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.dependencies import Dependencies
from app.graph.nodes.chitchat import chitchat_node
from app.graph.nodes.fallback import fallback_node
from app.graph.nodes.rag import rag_node
from app.graph.nodes.router import router_node
from app.graph.nodes.static import static_node
from app.graph.state import AgentState
from app.schemas.routing import Route

logger = get_logger(__name__)

_ROUTER = "router"


def _route_selector(state: AgentState) -> str:
    """Conditional-edge selector: map the decision to the target node name."""
    decision = state.get("decision")
    if decision is None:
        return Route.FALLBACK.value
    return decision.route.value


def build_graph(deps: Dependencies):
    """Compile and return the executable LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node(_ROUTER, partial(router_node, deps=deps))
    graph.add_node(Route.STATIC.value, partial(static_node, deps=deps))
    graph.add_node(Route.CHITCHAT.value, partial(chitchat_node, deps=deps))
    graph.add_node(Route.RAG.value, partial(rag_node, deps=deps))
    graph.add_node(Route.FALLBACK.value, partial(fallback_node, deps=deps))

    graph.set_entry_point(_ROUTER)
    graph.add_conditional_edges(
        _ROUTER,
        _route_selector,
        {
            Route.STATIC.value: Route.STATIC.value,
            Route.CHITCHAT.value: Route.CHITCHAT.value,
            Route.RAG.value: Route.RAG.value,
            Route.FALLBACK.value: Route.FALLBACK.value,
        },
    )
    for handler in (
        Route.STATIC.value,
        Route.CHITCHAT.value,
        Route.RAG.value,
        Route.FALLBACK.value,
    ):
        graph.add_edge(handler, END)

    compiled = graph.compile()
    logger.info("LangGraph compiled with %d nodes", 5)
    return compiled
