# Schema v3 — image pipeline

added `images` (one row per unique image, deduped on `file_hash`) and `document_images` (one row per occurrence, with `page_number` + `image_index_on_page`). `chunks.source_kind` gains `'image'`. New content_type values: `'ocr'` for Tesseract-recovered scanned-page text (stored under `source_kind='document'`); `'image_ocr'` and `'image_description'` for VLM outputs (stored under `source_kind='image'`). Default PDF backend swapped from Docling to pdfplumber for ~10x faster text-PDF ingest.
