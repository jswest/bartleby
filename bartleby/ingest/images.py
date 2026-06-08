"""Image ingestion pipeline — hash, scale, archive, OCR-or-VLM, chunk.

Pure functions and IO helpers. No DB writes — the scribe orchestrator handles
those so the polymorphic-chunks invariant stays at one chokepoint.

The flow per image:
  1. ``hash_bytes`` on the *original* bytes for dedup identity.
  2. ``prepare_image`` to scale + re-encode as JPEG. Use the resulting bytes
     for both the VLM call and the archive (so the analysis matches what's
     on disk).
  3. ``archive_image`` writes them to ``<archive_root>/images/<hash>.jpg``.
  4. ``analyze`` decides text-image vs scene-image: Tesseract first; if it
     clears the text-image threshold the VLM is skipped, otherwise the VLM
     is called for a bounded description.
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
from bartleby.ingest import ocr as ocr_module
from bartleby.ingest.embed import embed_texts
from bartleby.lib import console
from bartleby.providers import ImageAnalysis, Provider


JPEG_QUALITY = 82

# Whether we've already warned that Tesseract is unusable this process. The OCR
# pass is only a cheap text-image classifier; when it's broken every image just
# routes to the VLM, so we surface that once rather than once per image.
_ocr_degraded_warned = False

# Tesseract-on-image dispositioning thresholds. Stricter than the page-level
# sparse threshold (100 chars / 30 confidence) so a labeled chart — which has
# real OCR-able text but isn't text-dominated — still routes to the VLM for a
# proper description.
IMAGE_TEXT_MIN_CHARS = 300
IMAGE_TEXT_MIN_CONFIDENCE = 60


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


def is_below_vlm_minimum(prepared: PreparedImage, *, min_dimension: int) -> bool:
    """True when either edge is under ``min_dimension``.

    VLM image processors (e.g. qwen3-vl) tile images into patches and crash
    when an edge is smaller than the patch factor. Such images — thin rules,
    banners, sliver crops — carry no describable scene, so the scribe skips
    them entirely rather than sending them to the model. The check is on the
    *post-scale* dimensions, since downscaling a tall-thin image can itself
    push an edge below the minimum.
    """
    return min(prepared.width, prepared.height) < min_dimension


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
    """Decide text-image vs scene-image and produce the merged analysis.

    Runs Tesseract first. If the OCR clears the text-image threshold,
    classification is ``'text'`` and the VLM is skipped entirely. Otherwise
    classification is ``'scene'`` and the VLM produces a bounded description.

    OCR is only a classifier, not the captioner — so if Tesseract raises (a
    locked-down ``TMPDIR``, a broken install), we degrade to "couldn't classify
    → not a text image" and fall through to the VLM rather than dropping the
    image. A busted Tesseract just means everything routes to the VLM.
    """
    try:
        ocr_result = ocr_module.run(prepared.jpeg_bytes)
    except Exception as exc:
        _warn_ocr_degraded_once(exc)
        ocr_result = None
    if ocr_result is not None and _is_text_image(ocr_result):
        return ImageAnalysis(
            kind="text",
            text=ocr_result.text,
            description="",
            notes="",
        )
    vlm = provider.analyze_image(
        prepared.jpeg_bytes, model=model, media_type="image/jpeg",
    )
    return ImageAnalysis(
        kind="scene",
        text="",
        description=vlm.description,
        notes=vlm.notes,
    )


def _warn_ocr_degraded_once(exc: Exception) -> None:
    """Warn once per process that OCR is down and images now go straight to VLM."""
    global _ocr_degraded_warned
    if _ocr_degraded_warned:
        return
    _ocr_degraded_warned = True
    console.warn(
        f"OCR classification unavailable ({exc}); captioning every image via "
        "the VLM instead. Images are still described — only the "
        "text-image shortcut is lost."
    )


def _is_text_image(ocr_result: ocr_module.OcrResult) -> bool:
    return (len(ocr_result.text) >= IMAGE_TEXT_MIN_CHARS
            and ocr_result.avg_confidence >= IMAGE_TEXT_MIN_CONFIDENCE)


def analysis_to_chunk_inputs(analysis: ImageAnalysis) -> list[ChunkInput]:
    """Embed the populated field of an ImageAnalysis as a single ChunkInput.

    Under the binary classification each image yields at most one chunk:
    ``image_ocr`` for text-images, ``image_description`` for scene-images.
    An empty payload (no OCR text *and* no VLM description) returns an empty
    list — the image row still exists, it just has no searchable chunks.
    """
    if analysis.kind == "text" and analysis.text.strip():
        content_type, text = "image_ocr", analysis.text.strip()
    elif analysis.kind == "scene" and analysis.description.strip():
        content_type, text = "image_description", analysis.description.strip()
    else:
        return []

    [embedding] = embed_texts([text])
    return [ChunkInput(
        text=text, embedding=embedding, chunk_index=0,
        section_heading=None, content_type=content_type,
    )]
