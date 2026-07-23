"""Composition root.

Builds the singleton object graph (services) from :class:`Settings` and bundles
them in a :class:`Dependencies` container that is injected into graph nodes and
the orchestrator. This is the one place that knows concrete implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.embeddings import DeterministicEmbedder, Embedder
from app.services.knowledge_base import StaticAnswerStore, load_knowledge_base
from app.services.llm import LLMProvider, get_llm_provider
from app.services.semantic_router import SemanticRouter
from app.services.vector_store import QdrantVectorStore

logger = get_logger(__name__)


@dataclass
class Dependencies:
    """Immutable bundle of services shared across the request lifecycle."""

    settings: Settings
    embedder: Embedder
    llm: LLMProvider
    vector_store: QdrantVectorStore
    semantic_router: SemanticRouter
    static_store: StaticAnswerStore


def build_dependencies(settings: Settings) -> Dependencies:
    """Wire all services. Seeds the vector store so the demo is ready on boot."""
    embedder = DeterministicEmbedder(dim=settings.embedding_dim)
    llm = get_llm_provider(settings)

    kb = load_knowledge_base(settings.knowledge_base_path)
    static_store = StaticAnswerStore.from_faqs(kb["static_faqs"])

    vector_store = QdrantVectorStore(
        embedder=embedder,
        location=settings.qdrant_location,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )
    vector_store.seed(kb["documents"])

    semantic_router = SemanticRouter(embedder, settings)

    logger.info(
        "Dependencies built: %d static FAQs, %d RAG docs",
        len(static_store.intents),
        len(kb["documents"]),
    )
    return Dependencies(
        settings=settings,
        embedder=embedder,
        llm=llm,
        vector_store=vector_store,
        semantic_router=semantic_router,
        static_store=static_store,
    )
