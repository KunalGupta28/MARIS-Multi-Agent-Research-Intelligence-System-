"""
MARIS LLM Factory — Centralized LLM provider initialization.

Supports Groq (primary, fast + free) and OpenAI (fallback, high quality).
All LLM instances are configured with LangSmith tracing enabled automatically
via environment variables.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Get an LLM instance configured for the specified provider.

    Args:
        provider: "groq" or "openai". Defaults to primary provider from settings.
        model: Model name. Defaults to primary model from settings.
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        A LangChain chat model instance with LangSmith tracing.
    """
    settings = get_settings()

    if provider is None:
        provider = settings.primary_llm_provider
    if model is None:
        model = (
            settings.primary_llm_model
            if provider == settings.primary_llm_provider
            else settings.fallback_llm_model
        )

    logger.info(f"Initializing LLM: {provider}/{model}")

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Use 'groq' or 'openai'."
        )


def get_fallback_llm(temperature: float = 0.1, max_tokens: int = 4096) -> BaseChatModel:
    """Get the fallback LLM (OpenAI GPT-4o-mini by default)."""
    settings = get_settings()
    return get_llm(
        provider=settings.fallback_llm_provider,
        model=settings.fallback_llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
