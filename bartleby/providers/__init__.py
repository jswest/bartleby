"""Provider clients for ingest-time summarization.

Each provider returns a validated ``DocumentSummary`` (Pydantic) — never raw
text. Adding a fourth provider means writing one class plus an entry here.
"""

from __future__ import annotations

from bartleby.lib.consts import ALLOWED_PROVIDERS
from bartleby.providers.base import (
    DocumentSummary, ImageAnalysis, Provider, VlmDescription,
)


def get_provider(name: str, *, ollama_base_url: str | None = None) -> Provider:
    if name == "anthropic":
        from bartleby.providers.anthropic import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from bartleby.providers.openai import OpenAIProvider
        return OpenAIProvider()
    if name == "ollama":
        from bartleby.providers.ollama import OllamaProvider
        return OllamaProvider(base_url=ollama_base_url)
    if name == "wsjpt":
        from bartleby.providers.wsjpt import WsjptProvider
        return WsjptProvider()
    raise ValueError(
        f"Unknown provider {name!r}; expected one of {ALLOWED_PROVIDERS}"
    )


__all__ = [
    "ALLOWED_PROVIDERS",
    "DocumentSummary", "ImageAnalysis", "Provider", "VlmDescription",
    "get_provider",
]
