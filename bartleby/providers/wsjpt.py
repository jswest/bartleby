"""wsjpt provider — routes through WSJ's parsing toolkit (Pydantic-AI under the hood).

Out-of-band install: there is **no** ``bartleby[wsjpt]`` extra — the git source
is WSJ-internal and unreachable outside WSJ, so it can't live in the locked
deps. Inject it into the installed tool's environment with
``uv tool install '.[...]' --with 'git+ssh://git@github.dowjones.net/data/wsjpt.git' --force``
(see the README install section). The provider talks to
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
import threading

# Must land before wsjpt/pydantic_ai import below: pydantic-ai v2 prints a
# Logfire advertisement banner on the first agent run unless this is set (or
# observability is configured). setdefault so a user who's actually wired up
# Logfire is unaffected.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

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
                "set). Install from the bartleby checkout with: "
                "uv tool install '.[docling,sec2md]' --with "
                "'git+ssh://git@github.dowjones.net/data/wsjpt.git' --force"
            ) from e
        self._Jpt = Jpt
        self._ModelConfig = ModelConfig
        self._api_key = os.environ.get("GEMINI_API_KEY") or None
        # Each Jpt.__init__ builds a fresh genai.Client (open sockets); building
        # one per call piled up past the OS fd limit on long scribe runs (#755).
        # Cache one Jpt per (schema, model, instructions), as the other providers
        # build one client in __init__. Shared across the caption/summarize pools:
        # calls interleave on wsjpt's single background loop, and the only
        # per-call instance state (last_parse_duration, last_retry_count) is
        # never read by bartleby (it can mislabel wsjpt's own history rows).
        self._jpt_cache: dict[tuple, object] = {}
        self._jpt_cache_lock = threading.Lock()

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
        jpt = self._get_jpt(DocumentSummary, model, SUMMARY_INSTRUCTIONS)
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
        jpt = self._get_jpt(schema, model, None)
        return jpt.parse(input_text=prompt)

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        temperature: float,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        # temperature intentionally ignored (see summarize) — wsjpt owns model
        # settings centrally so callers can't drift from the toolkit defaults.
        jpt = self._get_jpt(VlmDescription, model, IMAGE_DESCRIPTION_INSTRUCTIONS)
        return jpt.parse(binary_files=[(image_bytes, media_type)])

    def _get_jpt(
        self,
        schema: type[BaseModel],
        model: str,
        custom_instructions: str | None,
    ):
        key = (schema, model, custom_instructions)
        with self._jpt_cache_lock:
            jpt = self._jpt_cache.get(key)
            if jpt is None:
                jpt = self._jpt_cache[key] = self._Jpt(
                    schema,
                    model_config=self._model_config(model),
                    custom_instructions=custom_instructions,
                )
            return jpt

    def _model_config(self, model: str):
        return self._ModelConfig(
            provider="google",
            model=model,
            api_key=self._api_key,
        )
