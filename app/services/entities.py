"""Lightweight, regex-based entity extraction.

Shared by the semantic router and the mock LLM classifier so both emit the same
entity shape, and reused by the dataset script so fine-tuning targets match
runtime behaviour.
"""
from __future__ import annotations

import re

from app.schemas.routing import Entity

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("order_id", re.compile(r"(?:order|ord|invoice|#)\s*#?\s*(\d{4,})", re.IGNORECASE)),
    ("email", re.compile(r"([\w.\-]+@[\w\-]+\.[\w.\-]+)")),
    ("amount", re.compile(r"(\$\s?\d+(?:\.\d{2})?)")),
    ("tracking_number", re.compile(r"\b([A-Z]{2}\d{9}[A-Z]{2})\b")),
]

# Known product/feature names worth surfacing as entities.
_PRODUCTS = ("voice ai", "pro plan", "enterprise", "starter", "api", "dashboard")


def extract_entities(text: str) -> list[Entity]:
    """Extract a de-duplicated list of entities from free text."""
    found: list[Entity] = []

    for etype, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            found.append(Entity(type=etype, value=match.group(1).strip()))

    low = text.lower()
    for name in _PRODUCTS:
        if name in low:
            found.append(Entity(type="product", value=name))

    # De-duplicate on (type, normalized value) while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[Entity] = []
    for entity in found:
        key = (entity.type, entity.value.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique
