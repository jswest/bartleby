"""Phase 2 of the scribe ingest: caption every uncaptioned image.

Pulled out of ``commands/scribe.py`` (#306). Decoupled from parse (#166): parse
only records image rows; this stage analyzes them (OCR + VLM, fanned out across
caption workers) and joins the captions back through the single Writer, whose
embedding + DB write stay on the calling thread.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from bartleby.ingest import images as image_pipeline
from bartleby.ingest.parse import DocUnit
from bartleby.ingest.progress import _ProgressTally
from bartleby.ingest.writer import (
    MAX_INGEST_ATTEMPTS,
    ImageCaption,
    PendingImage,
    Writer,
)
from bartleby.lib import console
from bartleby.providers import Provider


def _analyze_image(
    pending: PendingImage,
    *,
    vision_provider: Provider,
    vision_model: str,
) -> image_pipeline.ImageAnalysis:
    """Run the §7 image pipeline (Tesseract → VLM) on one recorded image row.

    Reloads the prepared JPEG from the archive and classifies text-image vs
    scene-image (VLM only for scenes). Pure analysis: no embedding, no DB — the
    OCR subprocess and VLM network call both release the GIL, so this is the
    half of captioning that runs off the writer thread in the caption pool.
    """
    prepared = image_pipeline.PreparedImage(
        hash=pending.file_hash,
        jpeg_bytes=Path(pending.file_path).read_bytes(),
        width=pending.width,
        height=pending.height,
    )
    return image_pipeline.analyze(vision_provider, prepared, model=vision_model)

def _caption_from_analysis(
    pending: PendingImage,
    analysis: image_pipeline.ImageAnalysis,
    vision_model: str,
) -> ImageCaption:
    """Package an analysis into a writer-ready caption, embedding its chunk.

    Runs on the writer thread: ``analysis_to_chunk_inputs`` embeds via the one
    per-process SentenceTransformer, which isn't safe to call from several
    caption threads at once — so embedding stays here, next to the DB write.
    """
    return ImageCaption(
        image_id=pending.image_id,
        analysis_json=analysis.model_dump_json(),
        analysis_model=vision_model,
        chunks=image_pipeline.analysis_to_chunk_inputs(analysis),
    )

def _caption_all(
    writer: Writer,
    units: list[DocUnit],
    *,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_enabled: bool,
    caption_workers: int,
    timings: bool,
    on_progress: Callable[[int, int], None] | None,
    on_lane: Callable[[object, str, str], None] | None = None,
) -> None:
    """Phase 2: caption every still-uncaptioned image across all parsed units.

    Captioning is decoupled from parse (#166): parse only records image rows
    (deduped by byte-hash, ``analysis_json IS NULL``); this stage analyzes them
    and joins the captions back through the single Writer. Rows are keyed by id,
    so one image shared by several documents is captioned once. Idempotent: only
    rows still lacking a caption are touched, so a resumed run fills gaps.

    Analysis (OCR + VLM) runs concurrently across ``caption_workers`` threads —
    the network-bound work the GIL releases — while embedding and the DB write
    stay on this thread, the Writer's sole owner. At ``caption_workers <= 1`` the
    work runs inline: the path ``--timings`` forces, for a clean sequential
    baseline with per-document caption seconds.

    Mutates nothing on the units except their ``stages`` (under timings); each
    document's remaining incompleteness is recomputed from the DB in phase 3.
    """
    # Unique uncaptioned rows across every unit; the first unit referencing a row
    # owns its timing. A row shared across documents appears once.
    pending: dict[int, PendingImage] = {}
    owner: dict[int, DocUnit] = {}
    for unit in units:
        for pi in writer.uncaptioned_images(unit.document_id):
            if pi.image_id not in pending:
                pending[pi.image_id] = pi
                owner[pi.image_id] = unit
    if not pending:
        return
    if not vision_enabled:
        console.warn(
            f"Skipping {len(pending)} uncaptioned image(s) — no vision provider "
            f"configured."
        )
        return

    progress = _ProgressTally(len(pending), on_progress)

    # Capped rows (failed MAX_INGEST_ATTEMPTS× already) are skipped, not retried.
    to_caption: dict[int, PendingImage] = {}
    for image_id, pi in pending.items():
        if writer.is_capped(pi.file_hash, "caption"):
            console.warn(
                f"{owner[image_id].file_name}: skipping image — failed "
                f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
            )
            progress.advance()
        else:
            to_caption[image_id] = pi

    def _persist(image_id: int, analysis: image_pipeline.ImageAnalysis) -> None:
        pi = to_caption[image_id]
        writer.persist_caption(_caption_from_analysis(pi, analysis, vision_model))

    def _fail(image_id: int, exc: Exception) -> None:
        unit = owner[image_id]
        console.warn(f"{unit.file_name}: image analysis failed: {exc}")
        writer.record_failure(
            to_caption[image_id].file_hash, unit.file_name, "caption", exc
        )

    def _analyze(image_id: int, pi: PendingImage) -> image_pipeline.ImageAnalysis:
        # Claim this thread's lane for the owning document, then run the analysis.
        # Keyed by thread id, so the pool's N threads map to N sticky lanes.
        if on_lane is not None:
            on_lane(threading.get_ident(), owner[image_id].file_name, "captioning")
        return _analyze_image(
            pi, vision_provider=vision_provider, vision_model=vision_model,
        )

    if caption_workers <= 1:
        # Inline: one image at a time on this thread. Same analyze/persist calls
        # as the pool, but with clean per-document caption seconds for --timings.
        for image_id, pi in to_caption.items():
            t0 = time.perf_counter() if timings else None
            try:
                _persist(image_id, _analyze(image_id, pi))
            except Exception as e:
                _fail(image_id, e)
            unit = owner[image_id]
            if t0 is not None and unit.stages is not None:
                unit.stages["caption"] = (
                    unit.stages.get("caption", 0.0) + (time.perf_counter() - t0)
                )
            progress.advance()
        return

    with ThreadPoolExecutor(max_workers=caption_workers) as pool:
        futures = {
            pool.submit(_analyze, image_id, pi): image_id
            for image_id, pi in to_caption.items()
        }
        for fut in as_completed(futures):
            image_id = futures[fut]
            try:
                _persist(image_id, fut.result())
            except Exception as e:
                _fail(image_id, e)
            progress.advance()
