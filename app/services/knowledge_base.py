"""Knowledge-base loading + the static FAQ answer store (Repository pattern)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class Document(TypedDict):
    """A RAG document loaded from the seed knowledge base."""

    id: str
    title: str
    text: str
    source: str


class StaticAnswerStore:
    """Maps a fine-grained intent to a hardcoded answer (critical FAQs)."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = {k.lower(): v for k, v in answers.items()}

    @classmethod
    def from_faqs(cls, faqs: list[dict]) -> "StaticAnswerStore":
        return cls({faq["intent"]: faq["answer"] for faq in faqs})

    def get(self, intent: str | None) -> str | None:
        return self._answers.get((intent or "").lower())

    @property
    def intents(self) -> list[str]:
        return list(self._answers)


def load_knowledge_base(path: str | Path) -> dict:
    """Load and lightly validate the seed knowledge base JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Knowledge base not found at {p!s}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "static_faqs": list(data.get("static_faqs", [])),
        "documents": list(data.get("documents", [])),
    }
