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


def setup_quiet_third_party(verbose: bool = False) -> None:
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
