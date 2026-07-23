"""Application configuration (12-factor, via environment / .env).

A single :class:`Settings` instance is the source of truth for all tunables.
It is intentionally free of any heavy imports so it can be loaded in scripts,
tests, and the API process alike.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/core/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    # --- Service metadata ---
    app_name: str = "agentic-support-orchestrator"
    environment: str = Field(default="local", description="local|staging|production")
    log_level: str = Field(default="INFO")

    # --- Routing behaviour ---
    semantic_router_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Min cosine confidence for the fast semantic router to win; below this we "
        "escalate to the LLM JSON classifier. CALIBRATED FOR DeterministicEmbedder "
        "(legitimate messages score >= ~0.20, noise <= ~0.06): 0.25 sends ~45% of traffic "
        "down the fast path at ~98% routing accuracy, with no noise leaking through. "
        "Raise it toward 0.6+ when swapping in a real sentence-embedding model — that is "
        "the scale those cosines live on.",
    )
    llm_router_min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Floor confidence accepted from the LLM classifier before fallback.",
    )

    # --- LLM provider ---
    llm_provider: str = Field(
        default="mock",
        description="mock|anthropic|local. 'anthropic' needs anthropic_api_key; "
        "'local' talks to an OpenAI-compatible server (Ollama/vLLM) at local_base_url.",
    )
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")
    anthropic_router_model: str = Field(
        default="claude-haiku-4-5",
        description="Cheaper/faster model for the strict-JSON routing fallback.",
    )

    # --- Local (open-source) LLM via Ollama / vLLM (OpenAI-compatible API) ---
    local_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible endpoint. Ollama's default is shown.",
    )
    local_model: str = Field(
        default="llama3.2:1b",
        description="Model tag used for generation (chitchat/RAG). Small = fast on CPU.",
    )
    local_router_model: str = Field(
        default="",
        description="Model used for the strict-JSON router. Empty -> use local_model. "
        "Point this at your fine-tuned router model once trained.",
    )

    # --- Embeddings / vector store ---
    embedding_dim: int = Field(default=256, ge=16, le=4096)
    qdrant_location: str = Field(
        default=":memory:",
        description="':memory:' (ephemeral), a filesystem path (persistent local), "
        "or a server URL like 'http://qdrant:6333' (Docker / remote).",
    )
    qdrant_collection: str = Field(default="support_kb")
    rag_top_k: int = Field(default=3, ge=1, le=20)
    rag_min_score: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Below this top retrieval score, RAG yields a graceful 'no info' answer. "
        "Calibrated for DeterministicEmbedder (relevant >= ~0.22, irrelevant <= ~0.16); "
        "raise it when swapping in a real semantic embedder.",
    )

    # --- Streaming ---
    stream_token_delay_ms: int = Field(
        default=12,
        ge=0,
        le=1000,
        description="Artificial per-token delay for the mock generator (simulates TTS-paced output).",
    )

    # --- Data paths ---
    knowledge_base_path: Path = Field(
        default=PROJECT_ROOT / "data" / "seed" / "knowledge_base.json"
    )
    generated_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "generated")

    @property
    def use_real_llm(self) -> bool:
        return self.llm_provider.lower() == "anthropic" and bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor — import this everywhere instead of constructing Settings."""
    return Settings()
