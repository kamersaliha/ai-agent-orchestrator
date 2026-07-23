"""LLM provider abstraction.

Three strategies implement the same :class:`LLMProvider` Protocol:

* :class:`MockLLMProvider` — offline, deterministic. Returns strict-JSON routing
  decisions via keyword heuristics and streams canned, context-aware answers.
* :class:`ClaudeLLMProvider` — real Anthropic Claude via ``langchain-anthropic``.
* :class:`LocalLLMProvider` — a local open-source model (incl. your own
  fine-tuned router) served via an OpenAI-compatible API (Ollama / vLLM).

All third-party imports are lazy so the module loads even when those packages are
absent. The graph depends only on the Protocol; :func:`get_llm_provider` picks the
implementation from settings.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.text_stream import stream_text
from app.schemas.chat import Message
from app.schemas.routing import Route, RouteDecision
from app.services.entities import extract_entities

logger = get_logger(__name__)

# The exact contract a fine-tuned model (or Claude) must satisfy. Documented here
# so the strict-JSON shape lives next to the code that depends on it.
ROUTER_SYSTEM_PROMPT = (
    "You are a fast intent router for a customer support system. "
    "Classify the user's message into exactly one route and return STRICT JSON "
    "matching this schema: "
    '{"route": "static|chitchat|rag|fallback", "intent": "<snake_case>", '
    '"confidence": <0..1>, "entities": [{"type": "...", "value": "..."}]}. '
    "Routes: 'static' = critical canned FAQs (launch dates, pricing, hours, refund "
    "policy); 'chitchat' = greetings/small talk needing no knowledge; 'rag' = "
    "questions answerable from product documentation; 'fallback' = unsafe, abusive, "
    "or out-of-scope requests. Output JSON only."
)


@runtime_checkable
class LLMProvider(Protocol):
    """Strategy interface for classification and streamed generation."""

    async def classify(
        self, message: str, history: Sequence[Message] | None = None
    ) -> RouteDecision: ...

    def astream_answer(
        self, *, system: str, user: str, context: list[dict] | None = None
    ) -> AsyncIterator[str]: ...


# --- Mock heuristics ---------------------------------------------------------

_CHITCHAT_KW = (
    "hello", "hi ", "hey", "good morning", "good afternoon", "thanks", "thank you",
    "how are you", "what's up", "your name", "who are you", "bye", "goodbye",
)
_STATIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("launch_date", ("launch date", "release date", "when does", "when will", "available on", "go live")),
    ("pricing", ("price", "pricing", "cost", "how much", "plan cost", "subscription cost")),
    ("business_hours", ("business hours", "opening hours", "open hours", "support hours", "when are you open")),
    ("refund_policy", ("refund policy", "return policy", "money back")),
)
_RAG_KW = (
    "how do i", "how to", "set up", "setup", "configure", "integrate", "integration",
    "api", "reset", "password", "login", "log in", "error", "not working", "billing",
    "cancel", "upgrade", "downgrade", "latency", "voice", "webhook", "documentation",
    "gdpr", "data", "privacy", "export", "delete account",
)
_UNSAFE_KW = (
    "ignore previous", "ignore all previous", "system prompt", "jailbreak",
    "kill", "bomb", "hack into", "credit card number of", "ssn",
)
_OUT_OF_SCOPE_KW = ("weather", "stock tip", "tell me a joke about", "who won", "recipe")


class MockLLMProvider:
    """Deterministic offline provider used when no API key is configured."""

    def __init__(self, delay_ms: int = 0) -> None:
        self._delay = delay_ms

    async def classify(
        self, message: str, history: Sequence[Message] | None = None
    ) -> RouteDecision:
        low = message.lower().strip()
        entities = extract_entities(message)

        # 1) Safety / out-of-scope first.
        if any(kw in low for kw in _UNSAFE_KW):
            return RouteDecision(
                route=Route.FALLBACK, intent="unsafe", confidence=0.95,
                entities=entities, source="llm", rationale="matched unsafe pattern",
            )
        if any(kw in low for kw in _OUT_OF_SCOPE_KW):
            return RouteDecision(
                route=Route.FALLBACK, intent="out_of_scope", confidence=0.8,
                entities=entities, source="llm", rationale="off-topic request",
            )

        # 2) Static FAQs.
        for intent, keywords in _STATIC_RULES:
            if any(kw in low for kw in keywords):
                return RouteDecision(
                    route=Route.STATIC, intent=intent, confidence=0.86,
                    entities=entities, source="llm",
                )

        # 3) Chit-chat (short and conversational).
        if any(kw in low for kw in _CHITCHAT_KW) and len(low.split()) <= 8:
            return RouteDecision(
                route=Route.CHITCHAT, intent="small_talk", confidence=0.82,
                entities=entities, source="llm",
            )

        # 4) RAG (documentation-answerable).
        if any(kw in low for kw in _RAG_KW):
            return RouteDecision(
                route=Route.RAG, intent="documentation_lookup", confidence=0.74,
                entities=entities, source="llm",
            )

        # 5) Heuristic default: question-shaped -> try docs, else fallback.
        if low.endswith("?") or low.split()[:1] and low.split()[0] in {
            "what", "how", "why", "where", "which", "can", "does", "is",
        }:
            return RouteDecision(
                route=Route.RAG, intent="documentation_lookup", confidence=0.55,
                entities=entities, source="llm", rationale="default question handling",
            )
        return RouteDecision(
            route=Route.FALLBACK, intent="unclassified", confidence=0.5,
            entities=entities, source="llm", rationale="no rule matched",
        )

    async def astream_answer(
        self, *, system: str, user: str, context: list[dict] | None = None
    ) -> AsyncIterator[str]:
        answer = self._compose(user=user, context=context)
        async for chunk in stream_text(answer, self._delay):
            yield chunk

    @staticmethod
    def _compose(*, user: str, context: list[dict] | None) -> str:
        if context:
            top = context[0]
            snippet = top["text"].strip()
            source = top.get("source", "our documentation")
            return (
                f"Here's what I found in our documentation: {snippet} "
                f"(source: {source}). Let me know if you'd like more detail."
            )
        low = user.lower()
        if any(w in low for w in ("thank", "thanks", "appreciate")):
            return "You're very welcome! Is there anything else I can help you with today?"
        if any(w in low for w in ("hello", "hi", "hey", "morning", "afternoon")):
            return "Hi there! 👋 I'm the support assistant. How can I help you today?"
        if "your name" in low or "who are you" in low:
            return "I'm your friendly support assistant — here to help with accounts, billing, and our product."
        return "Happy to help! Could you tell me a little more about what you need?"


class ClaudeLLMProvider:
    """Real Claude provider. Lazy-imports ``langchain-anthropic``."""

    def __init__(self, settings: Settings) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - exercised only with the dep
            raise RuntimeError(
                "langchain-anthropic is required for the 'anthropic' provider. "
                "Install it or set APP_LLM_PROVIDER=mock."
            ) from exc

        self._settings = settings
        self._chat = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.3,
            streaming=True,
        )
        self._router = ChatAnthropic(
            model=settings.anthropic_router_model,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
        )

    async def classify(
        self, message: str, history: Sequence[Message] | None = None
    ) -> RouteDecision:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = self._router.with_structured_output(RouteDecision)
        decision: RouteDecision = await structured.ainvoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=message)]
        )
        decision.source = "llm"
        return decision

    async def astream_answer(
        self, *, system: str, user: str, context: list[dict] | None = None
    ) -> AsyncIterator[str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        content = user
        if context:
            joined = "\n".join(f"- {c['text']} (source: {c.get('source', 'kb')})" for c in context)
            content = f"{user}\n\nRelevant documentation:\n{joined}"

        async for chunk in self._chat.astream(
            [SystemMessage(content=system), HumanMessage(content=content)]
        ):
            text = chunk.content
            if not isinstance(text, str):
                # Anthropic can return content blocks; concatenate any text parts.
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            if text:
                yield text


class LocalLLMProvider:
    """Open-source model served via an OpenAI-compatible endpoint (Ollama / vLLM).

    This is the seam for running a local model — including your own fine-tuned
    router. Generation streams tokens; classification uses JSON mode and is parsed
    into a :class:`RouteDecision`. A parse/validation failure raises, which the
    router node catches and degrades to the FALLBACK route.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only with the dep
            raise RuntimeError(
                "langchain-openai is required for the 'local' provider. "
                "Install it: pip install langchain-openai"
            ) from exc

        self._settings = settings
        router_model = settings.local_router_model or settings.local_model
        # Ollama/vLLM accept (and ignore) any api_key.
        self._chat = ChatOpenAI(
            model=settings.local_model,
            base_url=settings.local_base_url,
            api_key="local",
            temperature=0.3,
            streaming=True,
        )
        self._router = ChatOpenAI(
            model=router_model,
            base_url=settings.local_base_url,
            api_key="local",
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def classify(
        self, message: str, history: Sequence[Message] | None = None
    ) -> RouteDecision:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await self._router.ainvoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=message)]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(content)  # bad JSON -> raises -> router degrades to FALLBACK
        data.setdefault("confidence", 0.6)
        data.setdefault("entities", [])
        data["source"] = "llm"
        return RouteDecision.model_validate(data)

    async def astream_answer(
        self, *, system: str, user: str, context: list[dict] | None = None
    ) -> AsyncIterator[str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        content = user
        if context:
            joined = "\n".join(
                f"- {c['text']} (source: {c.get('source', 'kb')})" for c in context
            )
            content = f"{user}\n\nRelevant documentation:\n{joined}"

        async for chunk in self._chat.astream(
            [SystemMessage(content=system), HumanMessage(content=content)]
        ):
            text = chunk.content
            if isinstance(text, str) and text:
                yield text


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Factory: pick the provider strategy from ``settings.llm_provider``."""
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        if settings.anthropic_api_key:
            logger.info("Using ClaudeLLMProvider (model=%s)", settings.anthropic_model)
            return ClaudeLLMProvider(settings)
        logger.warning("llm_provider=anthropic but no API key set; falling back to mock")
    elif provider == "local":
        logger.info(
            "Using LocalLLMProvider (model=%s, router=%s, url=%s)",
            settings.local_model,
            settings.local_router_model or settings.local_model,
            settings.local_base_url,
        )
        return LocalLLMProvider(settings)
    logger.info("Using MockLLMProvider (offline, deterministic)")
    return MockLLMProvider(delay_ms=settings.stream_token_delay_ms)
