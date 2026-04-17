"""
LLM provider adapters for the Beekeeper Research Intelligence Platform.

LLM adapters are intentionally optional and injected/used only during synthesis.
"""

from .openai_compatible_client import OpenAICompatibleChatClient  # noqa: F401
from .provider_config import resolve_llm_provider_settings, supported_llm_providers  # noqa: F401
