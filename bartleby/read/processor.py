from __future__ import annotations
from collections import Counter
import hashlib
import io
from pathlib import Path
import shutil
import uuid


import fitz
from PIL import Image
import pytesseract

from bartleby.lib.consts import DEFAULT_PDF_PAGES_TO_SUMMARIZE, DEFAULT_PDF_PAGE_IMAGE_DPI
from bartleby.lib.embeddings import embed_chunk
from bartleby.read.llm import summarize_document
from bartleby.read.nlp import chunk_page_body, clean_block_text, extract_body_from_pdf_page

HASH_ALGORITHM = "sha256"
HASH_CHUNK_SIZE = 8_192
OCR_CHARACTER_THRESHOLD = 10


def page_to_png_bytes(page: fitz.Page, dpi: int = DEFAULT_PDF_PAGE_IMAGE_DPI) -> bytes:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def ocr_from_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        text = pytesseract.image_to_string(image, lang=lang)
    return text or ""


def save_chunk(
    embedding_model,
    cursor,
    body: str,
    index: int,
    page_id: str,
):
    chunk_id = str(uuid.uuid4())
    cursor.execute(
        """
        INSERT INTO chunks(
            chunk_id,
            body,
            chunk_index,
            page_id
        ) VALUES (?, ?, ?, ?)
        """,
        (chunk_id, body, index, page_id)
    )

    chunk_embedding = embed_chunk(embedding_model, body)
    cursor.execute(
        """
        INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)
        """,
        (chunk_id, chunk_embedding.astype("float32").tobytes())
    )


def process_pdf(
    connection,
    pdf_path: Path,
    db_path: Path,
    archive_path: Path,
    embedding_model,
    model_id: str = "",
    llm_has_vision: bool = True,
    pdf_pages_to_summarize: int = DEFAULT_PDF_PAGES_TO_SUMMARIZE
):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found at: {archive_path}")

    hasher = hashlib.new(HASH_ALGORITHM)
    with pdf_path.open("rb") as in_file:
        for chunk in iter(lambda: in_file.read(HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)

    document_id = hasher.hexdigest()
    document_archive_path = archive_path / document_id / f"{document_id}.pdf"
    document_archive_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pdf_path, document_archive_path)

    with fitz.open(str(document_archive_path)) as pdf:
        pages_count = pdf.page_count

        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO documents (
                document_id,
                origin_file_path,
                pages_count
            ) VALUES (?, ?, ?)
            """,
            (document_id, str(pdf_path), pages_count)
        )

        # Collect page text for document summary
        summary_page_texts = []
        first_page_image_bytes = None

        for page_index in range(pages_count):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            page_id = str(uuid.uuid4())
            body = extract_body_from_pdf_page(page)

            needs_ocr = len((body or "").strip()) < OCR_CHARACTER_THRESHOLD
            page_image_bytes = None
            if needs_ocr:
                page_image_bytes = page_to_png_bytes(page)
                raw_page_body = ocr_from_bytes(page_image_bytes)
                body = clean_block_text(raw_page_body)

            cursor.execute(
                """
                INSERT INTO pages(
                    page_id,
                    body,
                    document_id,
                    page_number
                ) VALUES (?, ?, ?, ?)
                """,
                (page_id, body, document_id, page_number)
            )

            # Collect text from first N pages for the document summary
            if page_number <= pdf_pages_to_summarize:
                summary_page_texts.append(body or "")
                # Capture first page image for vision-based summarization
                if page_number == 1 and llm_has_vision:
                    first_page_image_bytes = page_image_bytes or page_to_png_bytes(page)

            for chunk_index, chunk_body in enumerate(chunk_page_body(body)):
                save_chunk(embedding_model, cursor, chunk_body, chunk_index, page_id=page_id)

        # Generate document-level summary after processing all pages
        if pdf_pages_to_summarize > 0 and summary_page_texts:
            pages_text = "\n\n".join(summary_page_texts)
            summary = summarize_document(
                model_id,
                pages_text,
                llm_has_vision=llm_has_vision,
                first_page_image_bytes=first_page_image_bytes,
            )
            if summary:
                cursor.execute(
                    """
                    INSERT INTO summaries (
                        document_id,
                        title,
                        subtitle,
                        body
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (document_id, summary.title, summary.subtitle, summary.body)
                )


# --- Docling pipeline ---


_DOCLING_LABEL_MAP = {
    "TABLE": "table",
    "CODE": "code",
    "FORMULA": "formula",
    "PICTURE": "picture",
    "LIST_ITEM": "list",
}


def _get_chunk_page_number(chunk) -> int:
    """Extract page number from Docling chunk provenance metadata."""
    page_numbers = []
    for item in getattr(chunk.meta, "doc_items", []):
        for prov in getattr(item, "prov", []):
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                page_numbers.append(page_no)
    return min(page_numbers) if page_numbers else 1


def _get_chunk_content_type(chunk) -> str:
    """Determine dominant content type from Docling chunk's doc_items labels."""
    labels = []
    for item in getattr(chunk.meta, "doc_items", []):
        label = getattr(item, "label", None)
        if label is not None:
            labels.append(str(label).split(".")[-1])  # e.g. DocItemLabel.TABLE -> TABLE
    if not labels:
        return "text"
    most_common = Counter(labels).most_common(1)[0][0]
    return _DOCLING_LABEL_MAP.get(most_common, "text")


def save_chunk_docling(
    embedding_model,
    cursor,
    body: str,
    section_heading: str | None,
    content_type: str | None,
    index: int,
    page_id: str,
    embed_text: str | None = None,
):
    """Save a chunk from the Docling pipeline with heading and content type metadata."""
    chunk_id = str(uuid.uuid4())
    cursor.execute(
        """
        INSERT INTO chunks(
            chunk_id,
            body,
            chunk_index,
            page_id,
            section_heading,
            content_type
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, body, index, page_id, section_heading, content_type)
    )

    # Use contextualized text for embedding; fall back to raw body if too long
    text_to_embed = embed_text or body
    try:
        chunk_embedding = embed_chunk(embedding_model, text_to_embed)
    except ValueError:
        chunk_embedding = embed_chunk(embedding_model, body)

    cursor.execute(
        """
        INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)
        """,
        (chunk_id, chunk_embedding.astype("float32").tobytes())
    )


def process_pdf_docling(
    connection,
    pdf_path: Path,
    db_path: Path,
    archive_path: Path,
    embedding_model,
    converter,
    chunker,
    model_id: str = "",
    llm_has_vision: bool = True,
    pdf_pages_to_summarize: int = DEFAULT_PDF_PAGES_TO_SUMMARIZE,
):
    """Process a PDF using Docling for layout-aware conversion and structural chunking."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found at: {archive_path}")

    # Hash and archive
    hasher = hashlib.new(HASH_ALGORITHM)
    with pdf_path.open("rb") as in_file:
        for file_chunk in iter(lambda: in_file.read(HASH_CHUNK_SIZE), b""):
            hasher.update(file_chunk)

    document_id = hasher.hexdigest()
    document_archive_path = archive_path / document_id / f"{document_id}.pdf"
    document_archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, document_archive_path)

    # Get page count via fitz (lightweight metadata only)
    with fitz.open(str(document_archive_path)) as pdf:
        pages_count = pdf.page_count
        first_page_image_bytes = None
        if llm_has_vision and pages_count > 0:
            first_page_image_bytes = page_to_png_bytes(pdf.load_page(0))

    # Convert with Docling
    doc = converter.convert(str(document_archive_path))
    doc_result = doc.document

    # Chunk with Docling
    chunks = list(chunker.chunk(doc_result))

    cursor = connection.cursor()

    # Insert document record
    cursor.execute(
        """
        INSERT INTO documents (
            document_id,
            origin_file_path,
            pages_count
        ) VALUES (?, ?, ?)
        """,
        (document_id, str(pdf_path), pages_count)
    )

    # Create page records (body="" — meaningful content lives in chunks)
    page_ids = {}
    for page_number in range(1, pages_count + 1):
        page_id = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO pages(
                page_id,
                body,
                document_id,
                page_number
            ) VALUES (?, ?, ?, ?)
            """,
            (page_id, "", document_id, page_number)
        )
        page_ids[page_number] = page_id

    # Process each chunk
    for chunk_index, chunk in enumerate(chunks):
        page_number = _get_chunk_page_number(chunk)
        # Clamp to valid page range
        page_number = max(1, min(page_number, pages_count))
        page_id = page_ids[page_number]

        # Get heading from chunk metadata
        headings = getattr(chunk.meta, "headings", None)
        section_heading = " > ".join(headings) if headings else None

        # Determine content type
        content_type = _get_chunk_content_type(chunk)

        # Contextualize for embedding (prepends heading hierarchy)
        try:
            embed_text = chunker.contextualize(chunk)
        except Exception:
            embed_text = None

        save_chunk_docling(
            embedding_model,
            cursor,
            body=chunk.text,
            section_heading=section_heading,
            content_type=content_type,
            index=chunk_index,
            page_id=page_id,
            embed_text=embed_text,
        )

    # Generate summary using Docling's markdown export
    if pdf_pages_to_summarize > 0:
        summary_text = doc_result.export_to_markdown()
        if summary_text:
            summary = summarize_document(
                model_id,
                summary_text,
                llm_has_vision=llm_has_vision,
                first_page_image_bytes=first_page_image_bytes,
            )
            if summary:
                cursor.execute(
                    """
                    INSERT INTO summaries (
                        document_id,
                        title,
                        subtitle,
                        body
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (document_id, summary.title, summary.subtitle, summary.body)
                )
