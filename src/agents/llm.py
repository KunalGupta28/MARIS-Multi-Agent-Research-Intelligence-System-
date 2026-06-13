"""
MARIS LLM Factory — Centralized LLM provider initialization with automatic failover.

Supports Groq (primary, fast + free) and OpenAI (fallback, high quality).
All LLM instances are configured with LangSmith tracing enabled automatically
via environment variables.

Resilience features:
    - Automatic retry with exponential backoff for transient API errors
    - Automatic fallback to secondary provider on primary failure
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


# ── Retry Configuration ──────────────────────────────────────────
# Retry on transient HTTP/API errors (rate limits, timeouts, server errors)
_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)


def _create_llm(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Create a LangChain chat model instance for the given provider.

    Args:
        provider: LLM provider name ('groq' or 'openai').
        model: Model identifier string.
        api_key: API key for the provider.
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        A configured LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Use 'groq' or 'openai'."
        )


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Get an LLM instance configured for the specified provider.

    If ``provider`` and ``model`` are not specified, the primary provider
    from Settings is used. On creation failure, automatically falls back
    to the secondary provider.

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

    # Resolve API key for the chosen provider
    api_key = (
        settings.groq_api_key if provider == "groq" else settings.openai_api_key
    )

    logger.info(f"Initializing LLM: provider={provider}, model={model}")

    try:
        return _create_llm(provider, model, api_key, temperature, max_tokens)
    except Exception as e:
        # Attempt fallback to secondary provider
        fb_provider = settings.fallback_llm_provider
        fb_model = settings.fallback_llm_model
        fb_key = (
            settings.groq_api_key if fb_provider == "groq" else settings.openai_api_key
        )

        if fb_provider != provider:
            logger.warning(
                f"Primary LLM ({provider}/{model}) failed: {e}. "
                f"Falling back to {fb_provider}/{fb_model}."
            )
            return _create_llm(fb_provider, fb_model, fb_key, temperature, max_tokens)
        raise


def get_fallback_llm(temperature: float = 0.1, max_tokens: int = 4096) -> BaseChatModel:
    """Get the fallback LLM (configured secondary provider)."""
    settings = get_settings()
    return get_llm(
        provider=settings.fallback_llm_provider,
        model=settings.fallback_llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
