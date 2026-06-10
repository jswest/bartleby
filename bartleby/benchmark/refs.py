"""Model references and cell-file naming for the benchmark stores.

A model is identified as ``<provider>:<model>`` in the YAML configs (parse on
the *first* colon — Ollama model names carry their own colons, e.g.
``ollama:gemma4:e2b``) and as ``<provider>/<model>`` when passed on the
command line (parse on the first slash, so the model keeps its colons).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROVIDERS = ("ollama", "openai")

# Providers whose timings are measured on this machine and therefore comparable
# on the leaderboard's speed axis. Everything else is a reference row: quality
# only, excluded from the Pareto frontier.
LOCAL_PROVIDERS = ("ollama",)


@dataclass(frozen=True, order=True)
class ModelRef:
    provider: str
    model: str

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise SystemExit(
                f"Unknown provider {self.provider!r} in model reference "
                f"{self.provider}:{self.model} (known: {', '.join(PROVIDERS)})"
            )
        if not self.model:
            raise SystemExit(f"Empty model name in reference {self.provider!r}")

    @classmethod
    def from_yaml(cls, raw) -> "ModelRef":
        """Parse the YAML form ``provider:model`` (first colon splits)."""
        # A non-string means YAML parsed the entry itself — `- ollama: gemma4`
        # (colon-space) yields a dict, the most natural slip to make here.
        if not isinstance(raw, str) or ":" not in raw:
            raise SystemExit(
                f"Model reference {raw!r} must be <provider>:<model> in YAML "
                f"(e.g. ollama:gemma4:e2b — no space after the colon)"
            )
        provider, _, model = raw.partition(":")
        return cls(provider, model)

    @classmethod
    def from_flag(cls, raw: str) -> "ModelRef":
        """Parse the CLI form ``provider/model`` (first slash splits)."""
        provider, sep, model = raw.partition("/")
        if not sep:
            raise SystemExit(
                f"Model reference {raw!r} must be <provider>/<model> on the "
                f"command line (e.g. ollama/gemma4:e2b)"
            )
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    @property
    def slug(self) -> str:
        """Filename-safe form: ``provider_model`` with ``:`` and ``/`` → ``-``."""
        return f"{self.provider}_{re.sub(r'[:/]', '-', self.model)}"


def parse_flag_refs(raw: str | None) -> list[ModelRef] | None:
    """Comma-separated ``provider/model`` list from a CLI flag (None passes through)."""
    if raw is None:
        return None
    refs = [ModelRef.from_flag(part.strip()) for part in raw.split(",") if part.strip()]
    if not refs:
        raise SystemExit("Empty model list")
    return refs


def check_slug_collisions(refs: list[ModelRef]) -> None:
    """Normalization maps ``:``/``/`` to ``-``, so distinct refs can collide on
    the filename slug; refuse up front rather than silently sharing a store."""
    by_slug: dict[str, ModelRef] = {}
    for i, ref in enumerate(refs):
        if ref in refs[:i]:
            raise SystemExit(f"Duplicate model reference {ref} in models.yaml")
        other = by_slug.setdefault(ref.slug, ref)
        if other != ref:
            raise SystemExit(
                f"Model references {other} and {ref} collide on file slug "
                f"{ref.slug!r}; rename one in models.yaml"
            )
