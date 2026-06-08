"""Embedding model wrapper.

One SentenceTransformer instance per process. Loading is lazy because the
first ``encode`` call costs ~5–10s and we don't want to pay it just because
someone ran ``bartleby --help``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from bartleby.db.schema import EMBEDDING_DIM
from bartleby.lib.consts import EMBEDDING_MODEL

# SentenceTransformer's tokenizer worker pool fights with our own loops on macOS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def prewarm() -> None:
    """Load the embedding model now (idempotent; cached per process).

    The parse pool calls this in each worker's initializer so the model load is
    paid once at startup rather than on the worker's first document.
    """
    _model()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of 768-dim float lists, one per input text.

    Embeddings are L2-normalized to keep cosine and dot products equivalent.
    """
    if not texts:
        return []
    model = _model()
    arr = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding model produced {arr.shape[1]} dims; "
            f"schema requires {EMBEDDING_DIM}."
        )
    return [row.tolist() for row in arr]
