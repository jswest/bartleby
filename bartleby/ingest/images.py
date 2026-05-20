"""Image ingestion pipeline — hash, scale, archive, VLM-analyze, chunk.

Pure functions and IO helpers. No DB writes — the scribe orchestrator handles
those so the polymorphic-chunks invariant stays at one chokepoint.

The flow per image:
  1. ``hash_bytes`` on the *original* bytes for dedup identity.
  2. ``prepare_image`` to scale + re-encode as JPEG. Use the resulting bytes
     for both the VLM call and the archive (so the analysis matches what's
     on disk).
  3. ``archive_image`` writes them to ``<archive_root>/images/<hash>.jpg``.
  4. ``analyze`` calls the provider's ``analyze_image``.
  5. ``analysis_to_chunk_inputs`` builds typed ChunkInput rows ready for
     ``insert_image_chunks``.

The caller wires DB reads (dedup check) and writes (``images``,
``document_images``, chunks via the helper) around these.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from bartleby.db.chunks import ChunkInput
from bartleby.ingest.embed import embed_texts
from bartleby.providers import ImageAnalysis, Provider


JPEG_QUALITY = 82


@dataclass
class PreparedImage:
    hash: str            # sha256 of the *original* bytes (dedup identity)
    jpeg_bytes: bytes    # scaled, re-encoded JPEG — fed to VLM + archived
    width: int           # post-scale dimensions
    height: int


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_image(image_bytes: bytes, *, max_dimension: int) -> PreparedImage:
    """Scale the image so its long edge is ``max_dimension`` and re-encode as JPEG.

    Hash is computed over the original bytes. ``jpeg_bytes`` is what gets
    shown to the VLM and written to the archive.
    """
    if max_dimension <= 0:
        raise ValueError(f"max_dimension must be positive, got {max_dimension}")

    image_hash = hash_bytes(image_bytes)

    img = Image.open(io.BytesIO(image_bytes))
    # JPEG can't encode alpha or palette modes; flatten to RGB.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    long_edge = max(img.width, img.height)
    if long_edge > max_dimension:
        scale = max_dimension / long_edge
        new_size = (max(1, int(img.width * scale)),
                    max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return PreparedImage(
        hash=image_hash,
        jpeg_bytes=buf.getvalue(),
        width=img.width,
        height=img.height,
    )


def archive_image(prepared: PreparedImage, archive_root: Path) -> Path:
    """Write the prepared JPEG to ``<archive_root>/images/<hash>.jpg``.

    Idempotent: if the file already exists with the same hash, no rewrite.
    Returns the absolute path on disk.
    """
    dest = archive_root / "images" / f"{prepared.hash}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(prepared.jpeg_bytes)
    return dest


def analyze(
    provider: Provider,
    prepared: PreparedImage,
    *,
    model: str,
) -> ImageAnalysis:
    return provider.analyze_image(
        prepared.jpeg_bytes, model=model, media_type="image/jpeg",
    )


def analysis_to_chunk_inputs(analysis: ImageAnalysis) -> list[ChunkInput]:
    """Embed and wrap the non-empty fields of an ImageAnalysis as ChunkInputs.

    Returns up to two rows: an ``image_ocr`` chunk if ``text`` is non-empty
    and an ``image_description`` chunk if ``description`` is non-empty. Both
    share the same image source (the caller passes ``image_id`` to
    ``insert_image_chunks``).
    """
    payloads: list[tuple[str, str]] = []
    if analysis.text and analysis.text.strip():
        payloads.append(("image_ocr", analysis.text.strip()))
    if analysis.description and analysis.description.strip():
        payloads.append(("image_description", analysis.description.strip()))

    if not payloads:
        return []

    embeddings = embed_texts([text for _, text in payloads])
    return [
        ChunkInput(
            text=text,
            embedding=emb,
            chunk_index=i,
            section_heading=None,
            content_type=content_type,
        )
        for i, ((content_type, text), emb) in enumerate(zip(payloads, embeddings))
    ]
