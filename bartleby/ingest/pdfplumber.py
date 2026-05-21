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
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pdfplumber
from PIL import Image

from bartleby.ingest import ocr as ocr_module


PAGE_RENDER_DPI = 150

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


@dataclass
class EmbeddedImage:
    image_index_on_page: int   # 1-indexed; 0 reserved for full-page render
    png_bytes: bytes           # PNG of the rendered crop


@dataclass
class PdfPage:
    page_number: int           # 1-indexed
    text: str                  # extracted text (may be empty, OCR, or raw PDF text)
    content_type: str | None   # 'text' | 'ocr' | None (caller routes None pages to VLM)
    is_sparse: bool
    page_render_png: bytes | None  # populated only when is_sparse
    embedded_images: list[EmbeddedImage] = field(default_factory=list)
    # The raw Tesseract output when OCR was run on the page render. Surfaced
    # so the image pipeline can reuse it instead of re-Tesseract'ing the same
    # bytes when a sparse page's render gets routed through as an image.
    ocr_result: ocr_module.OcrResult | None = None


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
    on_progress: Callable[[int, int], None] | None = None,
) -> PdfResult:
    """Extract text + embedded images from a PDF, with OCR fallback for sparse pages.

    OCR runs inline per-page so each page is fully resolved before its
    ``on_progress`` tick fires — the progress bar never lies about being done
    while Tesseract grinds in the background.

    ``on_progress(pages_done, total)`` is called once with ``(0, total)`` after
    the PDF is opened so callers can size a progress bar, then again after
    every page with ``(page_number, total)``.
    """
    pages: list[PdfPage] = []
    full_text_parts: list[str] = []

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        if on_progress is not None:
            on_progress(0, page_count)

        for ix, page in enumerate(pdf.pages):
            page_number = ix + 1
            raw_text = page.extract_text() or ""
            text = raw_text.strip()
            is_sparse = len(text) < sparse_text_threshold

            page_render_png = None
            embedded_images: list[EmbeddedImage] = []

            if is_sparse or page.images:
                rendered = page.to_image(resolution=PAGE_RENDER_DPI).original
                if is_sparse:
                    page_render_png = _to_png_bytes(rendered)
                if page.images:
                    embedded_images = _crop_embedded_images(rendered, page)

            ocr_result: ocr_module.OcrResult | None = None
            if is_sparse:
                # Run OCR on the page render and decide whether it clears the
                # same length + confidence bar we apply to native PDF text.
                # If not, the caller routes the page render through the VLM.
                if page_render_png is not None:
                    ocr_result = ocr_module.run(page_render_png)
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
                is_sparse=is_sparse,
                page_render_png=page_render_png,
                embedded_images=embedded_images,
                ocr_result=ocr_result,
            ))

            if on_progress is not None:
                on_progress(page_number, page_count)

    return PdfResult(
        full_text="\n\f\n".join(full_text_parts),
        page_count=page_count,
        pages=pages,
    )


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _crop_embedded_images(rendered: Image.Image, page) -> list[EmbeddedImage]:
    """Crop each pdfplumber `page.images` entry out of the rendered page.

    pdfplumber image bboxes are in PDF points (1 pt = 1/72 in); the rendered
    image is at PAGE_RENDER_DPI, so points → pixels = DPI/72. Page-substrate
    images (full-page scans underneath an OCR text overlay) are dropped — see
    ``PAGE_SUBSTRATE_AREA_RATIO``.
    """
    scale = PAGE_RENDER_DPI / 72.0
    page_w, page_h = rendered.size
    page_area_pt = max(1.0, page.width * page.height)
    out: list[EmbeddedImage] = []
    for i, im in enumerate(page.images):
        bbox_area_pt = max(0.0, im["x1"] - im["x0"]) * max(0.0, im["bottom"] - im["top"])
        if bbox_area_pt / page_area_pt >= PAGE_SUBSTRATE_AREA_RATIO:
            continue
        # Truncate to integer pixel coords (PIL crop does this anyway). Skip
        # crops that degenerate to zero in either dimension — happens when
        # pdfplumber registers a thin horizontal/vertical PDF rule as an
        # "image" with a sub-pixel bbox, and downstream JPEG encoding bails
        # with "cannot write empty image."
        ix0 = max(0, int(im["x0"] * scale))
        iy0 = max(0, int(im["top"] * scale))
        ix1 = min(page_w, int(im["x1"] * scale))
        iy1 = min(page_h, int(im["bottom"] * scale))
        crop_w, crop_h = ix1 - ix0, iy1 - iy0
        if crop_w < 1 or crop_h < 1:
            continue
        if (max(crop_w, crop_h) < MIN_EMBEDDED_IMAGE_LONG_EDGE_PX
                or crop_w * crop_h < MIN_EMBEDDED_IMAGE_AREA_PX):
            continue
        crop = rendered.crop((ix0, iy0, ix1, iy1))
        out.append(EmbeddedImage(
            image_index_on_page=i + 1,
            png_bytes=_to_png_bytes(crop),
        ))
    return out
