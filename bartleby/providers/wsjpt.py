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
import re

from pydantic import BaseModel

from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    SUMMARY_INSTRUCTIONS,
)

# wsjpt's Vertex AI/ADC auth path calls GoogleProvider(vertexai=..., project=...,
# location=...). Those kwargs only exist in pydantic-ai [0.3.0, 2.0.0) — earlier
# releases lack the unified google module, 2.0+ dropped the vertexai kwarg. wsjpt
# itself declares pydantic-ai>=0.0.45 with no upper bound, so an unconstrained
# install can silently resolve an incompatible version.
_PYDANTIC_AI_MIN = (0, 3)
_PYDANTIC_AI_MAX_EXCLUSIVE = (2, 0)
_WSJPT_REINSTALL_CMD = (
    "uv tool install '.[docling,sec2md]' --with 'pydantic-ai>=1,<2' "
    "--with 'git+ssh://git@github.dowjones.net/data/wsjpt.git' --force"
)


def _check_pydantic_ai_version() -> None:
    """Raise if the installed pydantic-ai is outside wsjpt's compatible window.

    Fails open (returns silently) whenever the version can't be determined —
    pydantic_ai isn't a bartleby dependency, so a missing import or an
    unparseable __version__ means "can't introspect," not "incompatible."
    """
    try:
        import pydantic_ai

        version = pydantic_ai.__version__
        major, minor = (
            int(re.match(r"\d+", part).group()) for part in version.split(".")[:2]
        )
    except Exception:
        return

    if not (_PYDANTIC_AI_MIN <= (major, minor) < _PYDANTIC_AI_MAX_EXCLUSIVE):
        raise RuntimeError(
            f"The wsjpt provider requires pydantic-ai>=1,<2 (found {version}). "
            "wsjpt's Vertex AI/ADC auth path calls GoogleProvider(vertexai=...), "
            "a kwarg that only exists in pydantic-ai [0.3.0, 2.0.0) — pydantic-ai "
            "2.0 removed it and pydantic-ai <0.3 never had it. Reinstall with: "
            f"{_WSJPT_REINSTALL_CMD}"
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
        _check_pydantic_ai_version()
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
        temperature: float,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        # temperature intentionally ignored (see summarize) — wsjpt owns model
        # settings centrally so callers can't drift from the toolkit defaults.
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
