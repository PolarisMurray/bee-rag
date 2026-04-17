import pytest

from beekeeper_intel.llm.provider_config import resolve_llm_provider_settings


def test_resolve_llm_provider_settings_for_gemini_defaults():
    settings = resolve_llm_provider_settings("gemini")

    assert settings.provider == "gemini"
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert settings.model == "gemini-2.5-flash"
    assert settings.chat_completions_path == "/chat/completions"


def test_resolve_llm_provider_settings_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.invalid/gemini-openai")

    settings = resolve_llm_provider_settings("gemini")

    assert settings.model == "gemini-2.5-pro"
    assert settings.base_url == "https://example.invalid/gemini-openai"


def test_resolve_llm_provider_settings_rejects_unknown_provider():
    with pytest.raises(ValueError):
        resolve_llm_provider_settings("unknown-provider")
