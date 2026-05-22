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


def _hf_cache_is_populated() -> bool:
    """True if the HF Hub cache already contains at least one downloaded model.

    We use this as the trigger for offline mode: if anything is cached we
    assume the user has done at least one successful run, so we can safely
    skip the per-process update check that prints the unauthenticated-request
    warning. A fresh install has an empty cache and stays online so the first
    download succeeds.
    """
    cache_root = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )
    if not os.path.isdir(cache_root):
        return False
    try:
        entries = os.listdir(cache_root)
    except OSError:
        return False
    for entry in entries:
        if not entry.startswith("models--"):
            continue
        snapshots = os.path.join(cache_root, entry, "snapshots")
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            return True
    return False


def setup_quiet_third_party(verbose: bool = False) -> None:
    # Offline mode is about behavior (no Hub HEAD requests on every process
    # start), not noise — set it regardless of --verbose. An explicit
    # HF_HUB_OFFLINE in the environment always wins.
    if _hf_cache_is_populated():
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
