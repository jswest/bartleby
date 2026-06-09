"""wsjpt provider — routes through WSJ's parsing toolkit (Pydantic-AI under the hood).

Optional install: ``pip install bartleby[wsjpt]``. The provider talks to
Gemini via wsjpt's ``ModelConfig(provider="google", model=<alias>)`` so model
aliases (``fast`` / ``smart`` / ``smartest``) resolve centrally and bartleby
doesn't have to track concrete model names. Anthropic/OpenAI/Bedrock are also
reachable through wsjpt if a future config wants them; this provider only
exercises the Google path because that's the driver Rob and John asked for.

Authentication: when ``GEMINI_API_KEY`` is in env (set by ``ensure_provider_env``
from ``wsjpt_api_key`` in config), wsjpt uses the Gemini API BYO-key path.
Without it, wsjpt falls back to Vertex AI via Application Default Credentials.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    SUMMARY_INSTRUCTIONS,
)


class WsjptProvider:
    name = "wsjpt"

    def __init__(self) -> None:
        try:
            from wsjpt import Jpt, ModelConfig
        except ImportError as e:
            raise RuntimeError(
                "The wsjpt provider requires the 'wsjpt' package, which is "
                "installed out-of-band (it is not part of the locked dependency "
                "set). Install with: "
                "uv pip install 'git+ssh://git@github.dowjones.net/data/wsjpt.git'"
            ) from e
        self._Jpt = Jpt
        self._ModelConfig = ModelConfig
        self._api_key = os.environ.get("GEMINI_API_KEY") or None

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> DocumentSummary:
        # temperature and reasoning_effort are intentionally ignored — wsjpt owns
        # model settings centrally so callers can't drift from the toolkit defaults.
        jpt = self._Jpt(
            DocumentSummary,
            model_config=self._model_config(model),
            custom_instructions=SUMMARY_INSTRUCTIONS,
        )
        return jpt.parse(input_text=f"DOCUMENT:\n{document_text}")

    def classify(
        self,
        prompt: str,
        *,
        model: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        # temperature ignored (see summarize) — wsjpt owns model settings.
        # The prompt is self-contained, so no custom_instructions.
        jpt = self._Jpt(schema, model_config=self._model_config(model))
        return jpt.parse(input_text=prompt)

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        jpt = self._Jpt(
            VlmDescription,
            model_config=self._model_config(model),
            custom_instructions=IMAGE_DESCRIPTION_INSTRUCTIONS,
        )
        return jpt.parse(binary_files=[(image_bytes, media_type)])

    def _model_config(self, model: str):
        return self._ModelConfig(
            provider="google",
            model=model,
            api_key=self._api_key,
        )
