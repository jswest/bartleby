"""pdfplumber-based PDF converter.

Replaces Docling as the default backend (Docling stays available as opt-in).
Cheap text extraction; embedded images come out via page-render-crop, which
sidesteps the PDF image-codec dance (FlateDecode raw rasters etc.) at the
cost of one page render per page.

Returns enough information for ``scribe`` to:
  - chunk text-extractable pages via the character chunker
  - pump embedded images through the image pipeline
  - fall back to Tesseract or the VLM on sparse pages (using the saved page
    render bytes — no second render needed)
  - crop vector figure regions (charts, diagrams drawn as PDF path operators)
    that never appear in page.images but are visible in the page render
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from PIL import Image
from shapely import box, unary_union

from bartleby.ingest import ocr as ocr_module
from bartleby.lib import console


PAGE_RENDER_DPI = 150


class NotAPdfError(ValueError):
    """A file dispatched as PDF whose bytes are not a PDF.

    Upstream scrapers sometimes save an HTTP error page (``text/html``) with a
    ``.pdf`` extension; without this guard pdfminer dies deep with the cryptic
    ``No /Root object! - Is this really a PDF?``.
    """


# HTML error pages saved as `.pdf` reach the PDF backend and die deep in
# pdfminer with "No /Root object! - Is this really a PDF?". Peek the head and
# reject HTML with an actionable reason instead. We only reject content that
# *clearly* looks like HTML — a head merely missing the `%PDF` marker is left
# to the backend, which tolerates real-but-slightly-malformed PDFs.
_HTML_HEAD_MARKERS = (b"<!doctype html", b"<html")


def reject_if_html(path: Path) -> None:
    """Raise :class:`NotAPdfError` if ``path`` is an HTML page, not a PDF.

    The mirror image of #78: that *recovered* a mislabeled file into its true
    type; here a `.pdf` whose bytes are an HTML error page is rejected outright
    rather than rerouted into the HTML pipeline, which would ingest a portal
    error page as a first-class document and pollute the corpus + embeddings.
    """
    with path.open("rb") as fh:
        stripped = fh.read(1024).lstrip()
    if stripped.startswith(b"%PDF"):
        return
    lowered = stripped[:64].lower()
    if any(lowered.startswith(marker) for marker in _HTML_HEAD_MARKERS):
        raise NotAPdfError(
            "not a PDF — file contains an HTML page "
            "(likely a failed download or portal error page)"
        )

# OCR'd-scan PDFs embed a single page-sized raster on each page with the OCR
# text floating above it as a selectable layer. pdfplumber sees that raster as
# an "embedded image" and we'd otherwise crop it and pump it through the VLM —
# which would just transcribe text we already extracted. Skip embedded images
# whose bbox covers nearly the whole page; if the page is genuinely image-only,
# the sparse-page render fallback still captures it.
PAGE_SUBSTRATE_AREA_RATIO = 0.9

# Embedded images smaller than this are almost always decorative — bullet
# icons, checkmarks, page-number ornaments. ~0.85" on the long edge at our
# 150 DPI render. Combined with the area floor, the rule also catches thin
# decorative strips (e.g. 300×15 = 4,500 px²) that pass the long-edge check.
# Real signatures, letterheads, tax stamps, and figures clear both bars.
MIN_EMBEDDED_IMAGE_LONG_EDGE_PX = 128
MIN_EMBEDDED_IMAGE_AREA_PX = 5000

# Vector figure detection: proximity-merge tolerance (PDF points) and minimum
# merged-cluster area (PDF points²). tol=3 bridges small gaps between adjacent
# path primitives in the same figure; min_area=2500 (~50×50 pt) drops slivers,
# hairlines, and decoration that form isolated clusters too small to be figures.
_VECTOR_MERGE_TOL = 3.0
_VECTOR_MIN_AREA = 2500


@dataclass
class EmbeddedImage:
    image_index_on_page: int   # 1-indexed; 0 reserved for full-page render
    png_bytes: bytes           # PNG of the rendered crop


@dataclass
class PdfPage:
    page_number: int           # 1-indexed
    text: str                  # extracted text (may be empty, OCR, or raw PDF text)
    content_type: str | None   # 'text' | 'ocr' | None (caller routes None pages to VLM)
    page_render_png: bytes | None  # populated only when the page is sparse
    embedded_images: list[EmbeddedImage] = field(default_factory=list)


@dataclass
class PdfResult:
    full_text: str             # all page texts joined with form-feed (\f)
    page_count: int
    pages: list[PdfPage]


def convert(
    path: Path,
    *,
    sparse_text_threshold: int,
    ocr_min_confidence: int,
    vector_ink_threshold: int = 0,
) -> PdfResult:
    """Extract text + embedded images from a PDF, with OCR fallback for sparse pages.

    OCR runs inline per-page so each page is fully resolved before the caller
    sees it — the page is never reported done while Tesseract grinds in the
    background.

    ``vector_ink_threshold``: pages with fewer than this many vector primitives
    (curves + lines + rects combined) skip the vector-figure pass entirely.
    0 (the default) never skips.
    """
    pages: list[PdfPage] = []
    full_text_parts: list[str] = []

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)

        for ix, page in enumerate(pdf.pages):
            page_number = ix + 1
            raw_text = page.extract_text() or ""
            text = raw_text.strip()
            is_sparse = len(text) < sparse_text_threshold

            page_render_png = None
            embedded_images: list[EmbeddedImage] = []

            # Decide upfront whether this page has enough vector ink to bother
            # running the proximity-merge pass. The threshold is an ink-primitive
            # count, not a figure count — it gates the shapely work only.
            ink_count = len(page.curves) + len(page.lines) + len(page.rects)
            has_vector_ink = ink_count >= vector_ink_threshold and ink_count > 0

            needs_render = is_sparse or page.images or has_vector_ink
            rendered = None
            if needs_render:
                rendered = page.to_image(resolution=PAGE_RENDER_DPI).original
                if is_sparse:
                    page_render_png = _to_png_bytes(rendered)
                if page.images:
                    embedded_images = _crop_embedded_images(rendered, page)
                if has_vector_ink:
                    vector_crops = _crop_vector_figures(rendered, page)
                    embedded_images.extend(vector_crops)

            ocr_result: ocr_module.OcrResult | None = None
            if is_sparse:
                # Run OCR on the page render and decide whether it clears the
                # same length + confidence bar we apply to native PDF text.
                # If not, the caller routes the page render through the VLM.
                #
                # OCR is only the classifier here, not the captioner — so if
                # Tesseract raises (missing binary, locked TMPDIR), we degrade
                # to "couldn't classify → route to VLM" rather than failing the
                # whole PDF parse and discarding every non-sparse page. Mirrors
                # images.analyze's deliberate call for the identical situation.
                if page_render_png is not None:
                    try:
                        ocr_result = ocr_module.run(page_render_png)
                    except Exception as exc:
                        console.warn_once(
                            "ocr_degraded",
                            f"OCR classification unavailable ({exc}); routing "
                            "sparse PDF pages straight to the VLM instead. Pages "
                            "are still captured — only the cheap Tesseract "
                            "shortcut is lost.",
                        )
                        ocr_result = None
                if (ocr_result
                        and len(ocr_result.text) >= sparse_text_threshold
                        and ocr_result.avg_confidence >= ocr_min_confidence):
                    text, content_type = ocr_result.text, "ocr"
                else:
                    text, content_type = "", None
            else:
                content_type = "text"

            full_text_parts.append(text or raw_text)
            pages.append(PdfPage(
                page_number=page_number,
                text=text,
                content_type=content_type,
                page_render_png=page_render_png,
                embedded_images=embedded_images,
            ))

    return PdfResult(
        full_text="\n\f\n".join(full_text_parts),
        page_count=page_count,
        pages=pages,
    )


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _crop_pdf_bbox(
    rendered: Image.Image, x0: float, top: float, x1: float, bottom: float,
    scale: float,
) -> Image.Image | None:
    """Crop a PDF-point bbox from a rendered page, returning None if the crop
    is degenerate (zero dimension) or too small to be meaningful.

    Used by both raster-embed and vector-figure croppers so the pixel-coord
    computation and size-filter constants live in one place.
    """
    page_w, page_h = rendered.size
    ix0 = max(0, int(x0 * scale))
    iy0 = max(0, int(top * scale))
    ix1 = min(page_w, int(x1 * scale))
    iy1 = min(page_h, int(bottom * scale))
    crop_w, crop_h = ix1 - ix0, iy1 - iy0
    if crop_w < 1 or crop_h < 1:
        return None
    if (max(crop_w, crop_h) < MIN_EMBEDDED_IMAGE_LONG_EDGE_PX
            or crop_w * crop_h < MIN_EMBEDDED_IMAGE_AREA_PX):
        return None
    return rendered.crop((ix0, iy0, ix1, iy1))


def _crop_embedded_images(rendered: Image.Image, page) -> list[EmbeddedImage]:
    """Crop each pdfplumber `page.images` entry out of the rendered page.

    pdfplumber image bboxes are in PDF points (1 pt = 1/72 in); the rendered
    image is at PAGE_RENDER_DPI, so points → pixels = DPI/72. Page-substrate
    images (full-page scans underneath an OCR text overlay) are dropped — see
    ``PAGE_SUBSTRATE_AREA_RATIO``.
    """
    scale = PAGE_RENDER_DPI / 72.0
    page_area_pt = max(1.0, page.width * page.height)
    out: list[EmbeddedImage] = []
    for i, im in enumerate(page.images):
        bbox_area_pt = max(0.0, im["x1"] - im["x0"]) * max(0.0, im["bottom"] - im["top"])
        if bbox_area_pt / page_area_pt >= PAGE_SUBSTRATE_AREA_RATIO:
            continue
        crop = _crop_pdf_bbox(rendered, im["x0"], im["top"], im["x1"], im["bottom"], scale)
        if crop is None:
            continue
        out.append(EmbeddedImage(
            image_index_on_page=i + 1,
            png_bytes=_to_png_bytes(crop),
        ))
    return out


def _vector_figure_bboxes(page, tol: float = _VECTOR_MERGE_TOL,
                          min_area: float = _VECTOR_MIN_AREA) -> list[tuple]:
    """Return figure bounding boxes derived from vector ink via proximity-merge.

    Uses shapely to buffer-and-union all curves/lines/rects into clusters, then
    subtracts table bboxes (table gridlines are rects/lines but are not figures).
    Returns bboxes as (x0, top, x1, bottom) tuples in PDF point coordinates.

    Never includes ``page.chars`` — text is never vector ink for this purpose.
    """
    prims = page.curves + page.lines + page.rects
    if not prims:
        return []

    merged = unary_union([
        box(p["x0"], p["top"], p["x1"], p["bottom"]).buffer(tol / 2)
        for p in prims
    ])
    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]

    # Collect table bboxes to exclude table-gridline clusters. find_tables()
    # is pdfplumber's purpose-built table detector — the right tool here rather
    # than rolling a heuristic.
    table_boxes = [box(*tbl.bbox) for tbl in page.find_tables()]

    results = []
    for g in geoms:
        if g.area < min_area:
            continue
        # Drop clusters that substantially overlap a table bbox. A cluster is
        # "table-like" when its centroid falls inside any table bbox — simpler
        # and cheaper than a full intersection-over-union check, and correct for
        # the common case where the table gridlines form a cluster fully inside
        # the table bbox.
        cx, cy = g.centroid.x, g.centroid.y
        is_table = any(
            tb.bounds[0] <= cx <= tb.bounds[2] and tb.bounds[1] <= cy <= tb.bounds[3]
            for tb in table_boxes
        )
        if is_table:
            continue
        b = g.bounds   # (minx, miny, maxx, maxy) = (x0, top, x1, bottom)
        results.append((b[0], b[1], b[2], b[3]))

    return results


def _crop_vector_figures(rendered: Image.Image, page) -> list[EmbeddedImage]:
    """Crop vector figure regions from the rendered page.

    Vector figures (matplotlib/Illustrator charts drawn as PDF path operators)
    never appear in page.images but are visible in the rendered raster. This
    crops each proximity-merged cluster from the EXISTING render — no second
    render needed.

    image_index_on_page uses a high base (1000 + cluster_index) to avoid
    colliding with raster-embed indices (which are 1-indexed from page.images).
    """
    bboxes = _vector_figure_bboxes(page)
    if not bboxes:
        return []

    scale = PAGE_RENDER_DPI / 72.0
    out: list[EmbeddedImage] = []
    for cluster_idx, (x0, top, x1, bottom) in enumerate(bboxes):
        crop = _crop_pdf_bbox(rendered, x0, top, x1, bottom, scale)
        if crop is None:
            continue
        out.append(EmbeddedImage(
            # High base avoids collision with raster-embed indices (1-indexed).
            image_index_on_page=1000 + cluster_idx,
            png_bytes=_to_png_bytes(crop),
        ))
    return out
