"""Routing contracts — the strict JSON shape the router (semantic or LLM) emits.

These models are the *single source of truth* shared by the semantic router, the
LLM classifier, the graph nodes and the fine-tuning dataset script.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    """The four distinct execution paths of the hybrid router."""

    STATIC = "static"        # Hardcoded answers for critical FAQs (e.g. launch dates).
    CHITCHAT = "chitchat"    # Fast, lightweight conversational replies (no RAG).
    RAG = "rag"              # Retrieve from the vector DB, then answer.
    FALLBACK = "fallback"    # Out-of-scope / unsafe / unclassifiable inputs.


# Where a RouteDecision came from — useful for analytics and debugging.
DecisionSource = Literal["semantic", "llm", "default"]


class Entity(BaseModel):
    """A single extracted entity (e.g. order_id, product, date)."""

    type: str = Field(..., description="Entity category, e.g. 'order_id'.")
    value: str = Field(..., description="Raw extracted value.")


class RouteDecision(BaseModel):
    """The classifier output that drives graph routing.

    This is exactly the schema serialized into the fine-tuning JSONL targets, so
    a fine-tuned model can be dropped in behind :class:`LLMProvider` later.
    """

    route: Route
    intent: str = Field(..., description="Fine-grained intent label, e.g. 'launch_date'.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: list[Entity] = Field(default_factory=list)
    source: DecisionSource = Field(default="llm")
    rationale: str | None = Field(
        default=None, description="Short, human-readable justification (optional)."
    )

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold
