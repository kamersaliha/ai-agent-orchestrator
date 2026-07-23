"""Initialise a (persistent) local Qdrant collection and seed it with KB docs.

The API self-seeds an in-memory Qdrant on startup, so this script is for the
*persistent* on-disk scenario: run it once, then point the API at the same path
with ``APP_QDRANT_LOCATION=./qdrant_storage``.

Usage:
    python scripts/init_qdrant.py --location ./qdrant_storage --query "reset password"
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.services.embeddings import DeterministicEmbedder  # noqa: E402
from app.services.knowledge_base import load_knowledge_base  # noqa: E402
from app.services.vector_store import QdrantVectorStore  # noqa: E402


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Seed a local Qdrant collection.")
    parser.add_argument(
        "--location",
        default="./qdrant_storage",
        help="Filesystem path for persistent mode, or ':memory:'.",
    )
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--query", default="how do I reset my password", help="Smoke-test query.")
    args = parser.parse_args()

    embedder = DeterministicEmbedder(dim=settings.embedding_dim)
    store = QdrantVectorStore(
        embedder=embedder,
        location=args.location,
        collection=args.collection,
        dim=settings.embedding_dim,
    )

    kb = load_knowledge_base(settings.knowledge_base_path)
    written = store.seed(kb["documents"])
    print(f"[init_qdrant] Seeded {written} documents into '{args.collection}' at {args.location}")

    print(f"[init_qdrant] Smoke-test query: {args.query!r}")
    for i, chunk in enumerate(store.search(args.query, k=settings.rag_top_k), start=1):
        print(f"  {i}. score={chunk['score']:.3f} source={chunk['source']} :: {chunk['text'][:80]}...")


if __name__ == "__main__":
    main()
