"""Silence the worst noise from Docling / RapidOCR / transformers / HF Hub.

Call :func:`setup_quiet_third_party` as early as possible — before any of the
ML libraries are imported — so env vars take effect. Idempotent.

Set the env var ``BARTLEBY_VERBOSE=1`` (or call with ``verbose=True``) to
restore the libraries' default output. The CLI's ``--verbose`` flag does this.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterable


_NOISY_LOGGERS = (
    "rapidocr",
    "rapidocr_onnxruntime",
    "transformers",
    "huggingface_hub",
    "sentence_transformers",
    "docling",
    "docling_core",
    "docling_ibm_models",
    "safetensors",
)


# Shown when a model fetch is blocked by the offline default (issue #88).
OFFLINE_HINT = (
    "A required model is missing from the Hugging Face cache and Bartleby ran "
    "with offline mode on, so it could not be downloaded. Re-run with "
    "HF_HUB_OFFLINE=0 to fetch the missing model(s)."
)


def _hf_cache_root() -> str:
    return (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )


def _model_cached(repo_id: str) -> bool:
    """True if ``repo_id`` has a non-empty snapshot in the HF Hub cache.

    The Hub stores ``org/name`` under ``models--org--name/snapshots/<rev>/``.
    """
    folder = "models--" + repo_id.replace("/", "--")
    snapshots = os.path.join(_hf_cache_root(), folder, "snapshots")
    try:
        return os.path.isdir(snapshots) and bool(os.listdir(snapshots))
    except OSError:
        return False


def offline_blocked(exc: BaseException) -> bool:
    """True if ``exc`` looks like an HF fetch blocked by our offline default.

    Lets callers append :data:`OFFLINE_HINT` to the surfaced error so a user
    who hits the surprise (a required model genuinely missing while offline is
    on) gets the ``HF_HUB_OFFLINE=0`` remedy instead of a bare HF traceback.
    """
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        return False
    msg = str(exc)
    return (
        "outgoing traffic has been disabled" in msg
        or "Cannot find an appropriate cached snapshot" in msg
    )


def setup_quiet_third_party(
    verbose: bool = False,
    required_models: Iterable[str] = (),
) -> None:
    # Offline mode (no Hub HEAD request on every process start) is about
    # behavior, not noise — set it regardless of --verbose. But it is safe only
    # once EVERY model this run needs is already cached: enabling it while a
    # required model is still missing turns its lazy on-demand download into a
    # hard failure (issue #88 — docling's layout/table models download only on
    # the first conversion, often the Nth file in a batch). If anything required
    # is absent we stay online so it can fetch; offline kicks in on the next run
    # once all are cached. An explicit HF_HUB_OFFLINE in the environment always
    # wins. With no required_models declared we stay online — never guess that
    # the cache is complete.
    required = list(required_models)
    if required and all(_model_cached(m) for m in required):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if verbose:
        os.environ["BARTLEBY_VERBOSE"] = "1"
        return

    # Env vars must be set before the corresponding lib is imported.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["BARTLEBY_VERBOSE"] = "0"

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def is_verbose() -> bool:
    return os.environ.get("BARTLEBY_VERBOSE") == "1"
