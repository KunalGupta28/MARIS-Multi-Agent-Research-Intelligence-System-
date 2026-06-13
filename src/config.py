"""
MARIS Configuration — Centralized settings loaded from environment variables.

Uses pydantic-settings for type-safe, validated configuration. All secrets
are loaded from a .env file and never hardcoded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

# Project root is the directory containing pyproject.toml
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """
    Application settings loaded from .env file.

    All configuration is centralized here to avoid scattered os.getenv() calls.
    Secrets are loaded from .env and validated at startup.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Providers ──────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key for fallback LLM and embeddings")
    groq_api_key: str = Field(default="", description="Groq API key for primary LLM reasoning")

    # ── LangSmith Observability ────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=True, description="Enable LangSmith tracing")
    langchain_api_key: str = Field(default="", description="LangSmith API key for observability")
    langchain_project: str = Field(default="maris-research-agent", description="LangSmith project name")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )

    # ── Model Configuration ────────────────────────────────────────
    primary_llm_provider: str = Field(default="groq", description="Primary LLM provider ('groq' or 'openai')")
    primary_llm_model: str = Field(default="llama-3.3-70b-versatile", description="Primary LLM model name")
    fallback_llm_provider: str = Field(default="groq", description="Fallback LLM provider")
    fallback_llm_model: str = Field(default="llama-3.1-8b-instant", description="Fallback LLM model name")
    embedding_provider: str = Field(default="huggingface", description="Embedding provider ('huggingface' or 'openai')")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")

    # ── Vector Store ───────────────────────────────────────────────
    qdrant_path: str = Field(default="./data/qdrant_db", description="Path to Qdrant local disk storage")
    qdrant_collection_name: str = Field(default="maris_papers", description="Qdrant collection name")

    # ── Local Storage ──────────────────────────────────────────────
    sqlite_db_path: str = Field(default="./data/maris.db", description="Path to SQLite database file")

    # ── ArXiv Settings ─────────────────────────────────────────────
    arxiv_max_results: int = Field(default=10, description="Maximum arXiv search results per query")
    arxiv_rate_limit: int = Field(default=3, description="arXiv API concurrent request limit")

    @model_validator(mode="after")
    def _validate_api_keys(self) -> "Settings":
        """Warn if no LLM provider API keys are configured."""
        has_groq = bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")
        has_openai = bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")

        if not has_groq and not has_openai:
            logger.warning(
                "No LLM API keys configured. Set GROQ_API_KEY or OPENAI_API_KEY in .env. "
                "The system will fail when attempting LLM calls."
            )
        elif not has_groq and self.primary_llm_provider == "groq":
            logger.warning(
                "Primary provider is 'groq' but GROQ_API_KEY is not set. "
                "Falling back to OpenAI if available."
            )
        return self

    def get_qdrant_abs_path(self) -> Path:
        """Resolve the Qdrant path relative to the project root."""
        p = Path(self.qdrant_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_sqlite_abs_path(self) -> Path:
        """Resolve the SQLite path relative to the project root."""
        p = Path(self.sqlite_db_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
