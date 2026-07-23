"""Token-streaming helper shared by mock generation and template nodes.

Yields text in small word-sized chunks (preserving whitespace so concatenation
is loss-less). An optional per-chunk delay simulates the paced output a TTS
layer would consume downstream.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

# A "token" here is one whitespace-delimited word plus its trailing whitespace.
_TOKEN_RE = re.compile(r"\S+\s*|\s+")


async def stream_text(text: str, delay_ms: int = 0) -> AsyncIterator[str]:
    """Asynchronously yield ``text`` chunk-by-chunk."""
    if not text:
        return
    for match in _TOKEN_RE.finditer(text):
        yield match.group(0)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
