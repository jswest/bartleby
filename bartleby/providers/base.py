"""Provider interface and shared types for LLM summarization."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    """Pydantic schema enforced across all three providers (SPEC §5.3.1)."""

    text: str


class Provider(Protocol):
    name: str

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
    ) -> DocumentSummary: ...
