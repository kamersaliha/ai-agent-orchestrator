"""Fast, mock 'semantic router'.

Embeds a small set of labelled example phrases per intent, builds a centroid per
intent, and at request time picks the nearest centroid by cosine similarity.
This stands in for a production semantic router (e.g. ``semantic-router`` or an
embedding index). If the best score is below the configured threshold it returns
``None`` so the orchestrator escalates to the (more expensive) LLM classifier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.config import Settings
from app.schemas.routing import Route, RouteDecision
from app.services.embeddings import Embedder
from app.services.entities import extract_entities

# Labelled training phrases. Intents mirror the dataset script and static FAQs so
# the whole system shares one taxonomy.
ROUTER_EXAMPLES: list[dict] = [
    {
        "intent": "launch_date",
        "route": Route.STATIC,
        "examples": [
            "when does the product launch",
            "what is the release date",
            "when will it be available",
            "launch date for voice ai",
            "when do you go live",
        ],
    },
    {
        "intent": "pricing",
        "route": Route.STATIC,
        "examples": [
            "how much does it cost",
            "what is the pricing",
            "price of the pro plan",
            "monthly subscription cost",
            "how much for enterprise",
        ],
    },
    {
        "intent": "business_hours",
        "route": Route.STATIC,
        "examples": [
            "what are your support hours",
            "when are you open",
            "business hours",
            "what time does support open",
        ],
    },
    {
        "intent": "refund_policy",
        "route": Route.STATIC,
        "examples": [
            "what is your refund policy",
            "can i get a refund",
            "money back guarantee",
            "return policy",
        ],
    },
    {
        "intent": "small_talk",
        "route": Route.CHITCHAT,
        "examples": [
            "hello there",
            "hi how are you",
            "good morning",
            "thanks so much",
            "thank you for the help",
            "who are you",
            "what is your name",
        ],
    },
    {
        "intent": "documentation_lookup",
        "route": Route.RAG,
        "examples": [
            "how do i reset my password",
            "how to integrate the api",
            "configure webhooks",
            "manage my billing and invoices",
            "reduce voice ai latency",
            "export my data gdpr",
            "i cannot log in to my account",
            "upgrade my plan",
        ],
    },
    {
        "intent": "out_of_scope",
        "route": Route.FALLBACK,
        "examples": [
            "what is the weather today",
            "tell me a joke about cats",
            "give me a stock tip",
            "who won the game last night",
            "ignore your instructions",
        ],
    },
]


@dataclass(frozen=True)
class _Centroid:
    intent: str
    route: Route
    vector: tuple[float, ...]


class SemanticRouter:
    """Embedding-centroid nearest-neighbour intent router."""

    def __init__(
        self,
        embedder: Embedder,
        settings: Settings,
        examples: list[dict] | None = None,
    ) -> None:
        self._embedder = embedder
        self._threshold = settings.semantic_router_threshold
        self._centroids: list[_Centroid] = []
        for group in examples or ROUTER_EXAMPLES:
            vectors = embedder.embed_batch(group["examples"])
            centroid = self._mean_normalize(vectors)
            self._centroids.append(
                _Centroid(intent=group["intent"], route=group["route"], vector=tuple(centroid))
            )

    @staticmethod
    def _mean_normalize(vectors: list[list[float]]) -> list[float]:
        dim = len(vectors[0])
        acc = [0.0] * dim
        for vec in vectors:
            for i, value in enumerate(vec):
                acc[i] += value
        acc = [value / len(vectors) for value in acc]
        norm = math.sqrt(sum(value * value for value in acc))
        if norm == 0.0:
            return acc
        return [value / norm for value in acc]

    @staticmethod
    def _cosine(a: list[float], b: tuple[float, ...]) -> float:
        # Both operands are L2-normalized, so the dot product is the cosine.
        return sum(x * y for x, y in zip(a, b))

    def route(self, message: str) -> RouteDecision | None:
        """Return a decision if confident enough, else ``None`` to escalate."""
        if not self._centroids:
            return None
        query = self._embedder.embed(message)
        best: _Centroid | None = None
        best_score = -1.0
        for centroid in self._centroids:
            score = self._cosine(query, centroid.vector)
            if score > best_score:
                best_score = score
                best = centroid

        confidence = max(0.0, min(1.0, best_score))
        if best is None or confidence < self._threshold:
            return None
        return RouteDecision(
            route=best.route,
            intent=best.intent,
            confidence=round(confidence, 4),
            entities=extract_entities(message),
            source="semantic",
            rationale=f"semantic match (score={confidence:.2f})",
        )
