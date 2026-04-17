"""
Provider defaults and helpers for OpenAI-compatible LLM backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional, cast


LLMProviderName = Literal["openai", "deepseek", "gemini"]
DEFAULT_LLM_PROVIDER: LLMProviderName = "deepseek"


@dataclass(frozen=True)
class LLMProviderSettings:
    """Resolved connection settings for one provider."""

    provider: LLMProviderName
    base_url: str
    model: str
    chat_completions_path: str = "/v1/chat/completions"


PROVIDER_DEFAULTS: dict[LLMProviderName, LLMProviderSettings] = {
    "openai": LLMProviderSettings(
        provider="openai",
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
    ),
    "deepseek": LLMProviderSettings(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    ),
    "gemini": LLMProviderSettings(
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-2.5-flash",
        chat_completions_path="/chat/completions",
    ),
}

PROVIDER_ENV_PREFIXES: dict[LLMProviderName, str] = {
    "openai": "OPENAI",
    "deepseek": "DEEPSEEK",
    "gemini": "GEMINI",
}


def supported_llm_providers() -> tuple[LLMProviderName, ...]:
    """Return supported provider names in UI/API order."""

    return tuple(PROVIDER_DEFAULTS.keys())


def resolve_llm_provider_settings(
    provider: Optional[str],
    *,
    model: Optional[str] = None,
) -> LLMProviderSettings:
    """Resolve provider defaults with optional model and env overrides."""

    normalized = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if normalized not in PROVIDER_DEFAULTS:
        supported = ", ".join(supported_llm_providers())
        raise ValueError(f"Unsupported llm_provider '{provider}'. Expected one of: {supported}.")

    typed_provider = cast(LLMProviderName, normalized)
    default = PROVIDER_DEFAULTS[typed_provider]
    env_prefix = PROVIDER_ENV_PREFIXES[typed_provider]

    base_url = os.getenv(f"{env_prefix}_BASE_URL", default.base_url).strip() or default.base_url
    resolved_model = (model or os.getenv(f"{env_prefix}_MODEL") or default.model).strip() or default.model

    return LLMProviderSettings(
        provider=typed_provider,
        base_url=base_url,
        model=resolved_model,
        chat_completions_path=default.chat_completions_path,
    )
