from __future__ import annotations
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
