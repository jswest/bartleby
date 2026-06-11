"""Provider and worker-count resolution for the scribe ingest run.

Pulled out of ``commands/scribe.py`` (#306): these turn the loaded config (plus
any CLI overrides) into the concrete providers and pool sizes ``main`` hands to
the parse/caption/summarize phases. Pure config→value functions — no DB, no
parsing — so they sit below the phase modules and are unit-testable on their own.
"""

from __future__ import annotations

import os

from bartleby.config import ensure_provider_env
from bartleby.lib import console
from bartleby.lib.consts import (
    DEFAULT_CAPTION_WORKERS,
    DEFAULT_SUMMARIZE_WORKERS,
    DOCLING_HF_REPOS,
    EMBEDDING_MODEL,
    PER_WORKER_GB,
    RESERVED_CORES,
)
from bartleby.providers import Provider, get_provider


def _resolve_llm_provider(
    config: dict,
    *,
    provider_override: str | None,
    model_override: str | None,
) -> tuple[Provider | None, str | None]:
    if config.get("summary_depth", "none") != "one-shot":
        return None, None

    name = provider_override or config.get("provider")
    model = model_override or config.get("model")
    if not name or not model:
        console.warn(
            "summary_depth=one-shot but no provider/model configured; "
            "skipping summaries."
        )
        return None, None

    ensure_provider_env(name, config)
    return get_provider(name, ollama_base_url=config.get("ollama_base_url")), model


def _resolve_vision_provider(
    config: dict,
) -> tuple[Provider | None, str | None]:
    name = config.get("vision_provider")
    model = config.get("vision_model")
    if not name or not model:
        return None, None
    ensure_provider_env(name, config)
    return get_provider(name, ollama_base_url=config.get("ollama_base_url")), model


def _resolve_max_workers(config: dict, *, timings: bool) -> int:
    """Resolve how many parse workers to run.

    An explicit ``max_workers`` in config wins (warning when it exceeds what the
    machine can safely hold); otherwise auto-pick ``min(cpu_count -
    RESERVED_CORES, free_ram_gb // PER_WORKER_GB)``, floored at 1 (see those
    consts for the why). ``--timings`` forces 1: the per-stage breakdown it
    produces is a sequential baseline, meaningless once documents parse
    concurrently and overlap.
    """
    if timings:
        if int(config.get("max_workers") or 1) > 1:
            console.warn("--timings runs sequentially (max_workers=1) for a clean baseline.")
        return 1

    import psutil

    cores = os.cpu_count() or 1
    free_gb = psutil.virtual_memory().available / 1024**3
    usable = max(1, cores - RESERVED_CORES)
    auto = max(1, min(usable, int(free_gb // PER_WORKER_GB)))

    configured = config.get("max_workers")
    if configured is None:
        return auto
    n = max(1, int(configured))
    if n > auto:
        console.warn(
            f"max_workers={n} exceeds what this machine can comfortably run "
            f"({cores} cores − {RESERVED_CORES} reserved, ~{free_gb:.0f}GB free "
            f"→ {auto}); proceeding, but watch for memory pressure."
        )
    return n


def _resolve_io_workers(
    config: dict,
    *,
    key: str,
    default: int,
    provider: str | None,
    verb: str,
    unit: str,
    timings: bool,
) -> int:
    """Resolve the worker count for a network/IO-bound post-parse stage.

    Captioning and summarization share this resolver (#441): unlike RAM-bound
    parse workers (auto-sized), both are a plain configured count (``config[key]``)
    defaulting to ``default``. ``--timings`` forces 1 for a sequential baseline,
    meaningless once the stage's work overlaps (same rationale as ``max_workers``).

    A local Ollama ``provider`` clamps to 1 regardless (#243): with
    ``OLLAMA_NUM_PARALLEL`` defaulting to 1 a single model serializes requests, so
    an explicit ``config[key]`` > 1 is ignored (with a warning), not honored.

    ``provider`` is the provider the stage *actually* runs against — the caller
    threads in the already-resolved name (``config['vision_provider']`` for
    captions; the ``--provider`` override folded in for summaries, not bare
    ``config['provider']`` (#314)) so the clamp keys off the real backend.
    ``verb``/``unit`` only shape the warning strings.
    """
    configured = int(config.get(key) or default)
    if timings:
        if configured > 1:
            console.warn(
                f"--timings {verb} sequentially ({key}=1) for a clean baseline."
            )
        return 1
    if provider == "ollama":
        # Re-read the raw value (not `configured`, which has the default folded
        # in) so the warning fires only on an explicit count, not the default.
        if int(config.get(key) or 0) > 1:
            console.warn(
                f"{key} > 1 ignored — Ollama serializes requests "
                f"(OLLAMA_NUM_PARALLEL defaults to 1); {unit} at a time."
            )
        return 1
    return max(1, configured)


def _required_hf_models(pdf_converter: str, html_converter: str) -> tuple[str, ...]:
    """HF repos this ingest run will load, used to gate offline mode (#88).

    The embedding model is always needed; docling's layout/table models only
    when a docling converter is active for PDFs or HTML. The gate stays online
    until every model here is cached, so lazy downloads succeed.
    """
    models = [EMBEDDING_MODEL]
    if "docling" in (pdf_converter, html_converter):
        models.extend(DOCLING_HF_REPOS)
    return tuple(models)
