"""
MARIS Configuration — Centralized settings loaded from environment variables.

Uses pydantic-settings for type-safe, validated configuration. All secrets
are loaded from a .env file and never hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Project root is the directory containing pyproject.toml
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Providers ──────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    groq_api_key: str = Field(default="", description="Groq API key")

    # ── LangSmith Observability ────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=True)
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_project: str = Field(default="maris-research-agent")

    # ── Model Configuration ────────────────────────────────────────
    primary_llm_provider: str = Field(default="groq")
    primary_llm_model: str = Field(default="llama-3.3-70b-versatile")
    fallback_llm_provider: str = Field(default="groq")
    fallback_llm_model: str = Field(default="llama-3.1-8b-instant")
    embedding_provider: str = Field(default="huggingface")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # ── Vector Store ───────────────────────────────────────────────
    qdrant_path: str = Field(default="./data/qdrant_db")
    qdrant_collection_name: str = Field(default="maris_papers")

    # ── Local Storage ──────────────────────────────────────────────
    sqlite_db_path: str = Field(default="./data/maris.db")

    # ── ArXiv Settings ─────────────────────────────────────────────
    arxiv_max_results: int = Field(default=10)
    arxiv_rate_limit: int = Field(default=3)

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
