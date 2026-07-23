"""Qdrant-backed vector store (Repository pattern).

Wraps ``qdrant-client`` so the rest of the app depends on a tiny, intention-
revealing API (``seed`` / ``search``) rather than the client. Defaults to the
in-memory instance (``:memory:``) so the demo needs no running server; point
``APP_QDRANT_LOCATION`` at a path for persistent local mode.
"""
from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.graph.state import RetrievedChunk
from app.services.embeddings import Embedder

logger = get_logger(__name__)


class QdrantVectorStore:
    """Thin, embedder-aware wrapper around a Qdrant collection."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        location: str = ":memory:",
        collection: str = "support_kb",
        dim: int = 256,
    ) -> None:
        self._embedder = embedder
        self._collection = collection
        self._dim = dim
        self._client = self._make_client(location)
        if location.startswith(("http://", "https://")):
            self._wait_until_ready()
        self._ensure_collection()

    def _wait_until_ready(self, attempts: int = 30, delay: float = 1.0) -> None:
        """Block until a Qdrant server accepts requests (handles Docker startup races)."""
        import time

        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                self._client.get_collections()
                return
            except Exception as exc:  # server not up yet
                last_exc = exc
                time.sleep(delay)
        raise VectorStoreError(f"Qdrant server not reachable: {last_exc}")

    @staticmethod
    def _make_client(location: str) -> QdrantClient:
        if location == ":memory:":
            return QdrantClient(location=":memory:")
        if location.startswith(("http://", "https://")):
            # Real Qdrant server (Docker / remote), e.g. http://qdrant:6333.
            # check_compatibility=False tolerates a small client/server version
            # skew (we don't control the server image's exact build).
            return QdrantClient(url=location, check_compatibility=False)
        return QdrantClient(path=location)

    def _ensure_collection(self) -> None:
        try:
            exists = self._client.collection_exists(self._collection)
        except Exception:  # older clients lack collection_exists
            exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def seed(self, documents: Sequence[dict]) -> int:
        """Embed and upsert documents. Returns the number of points written."""
        if not documents:
            return 0
        points: list[PointStruct] = []
        for i, doc in enumerate(documents):
            vector = self._embedder.embed(doc["text"])
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "doc_id": doc.get("id", str(i)),
                        "text": doc["text"],
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                    },
                )
            )
        try:
            self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"failed to seed vector store: {exc}") from exc
        logger.info("Seeded %d documents into '%s'", len(points), self._collection)
        return len(points)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-``k`` chunks for ``query`` ordered by cosine score."""
        vector = self._embedder.embed(query)
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=k,
                with_payload=True,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"vector search failed: {exc}") from exc

        chunks: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    id=str(payload.get("doc_id", point.id)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    source=str(payload.get("source", "")),
                )
            )
        return chunks
