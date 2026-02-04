import base64
import json
from dataclasses import dataclass
from typing import Optional

import litellm
from loguru import logger

MAX_INPUT_CHARACTERS = 10000  # ~2,000 words

SUMMARIZE_PROMPT = """\
Given the following pages from a document, provide a JSON object with these fields:
- "title": the document's title (infer from content if not explicit)
- "subtitle": a short subtitle or tagline (null if not applicable)
- "body": a concise summary covering the document's topic, key points, and structure

Respond with ONLY the JSON object, no other text.

---

{text}"""

SUMMARIZE_PROMPT_VISION = """\
Given the following pages from a document (and an image of the first page), provide a JSON object with these fields:
- "title": the document's title (infer from content and/or the image if not explicit)
- "subtitle": a short subtitle or tagline (null if not applicable)
- "body": a concise summary covering the document's topic, key points, and structure

Respond with ONLY the JSON object, no other text."""


@dataclass
class DocumentSummary:
    title: str
    subtitle: Optional[str]
    body: str


def _parse_summary_response(text: str) -> DocumentSummary:
    """Parse the LLM response into a DocumentSummary."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines[1:] if not l.strip() == "```"]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        return DocumentSummary(
            title=data.get("title", "Untitled"),
            subtitle=data.get("subtitle"),
            body=data.get("body", ""),
        )
    except (json.JSONDecodeError, AttributeError):
        # Fallback: treat entire response as body
        return DocumentSummary(
            title="Untitled",
            subtitle=None,
            body=text.strip(),
        )


def summarize_document(
    model_id: str,
    pages_text: str,
    llm_has_vision: bool = True,
    first_page_image_bytes: Optional[bytes] = None,
) -> Optional[DocumentSummary]:
    """Generate a document-level summary from concatenated page text.

    Args:
        model_id: LiteLLM model identifier.
        pages_text: Concatenated text from the first N pages.
        llm_has_vision: Whether the model supports vision.
        first_page_image_bytes: Optional PNG bytes of the first page.

    Returns:
        DocumentSummary with title, subtitle, and body, or None on failure.
    """
    if not model_id:
        return None

    if len(pages_text) > MAX_INPUT_CHARACTERS:
        pages_text = pages_text[:MAX_INPUT_CHARACTERS]

    if first_page_image_bytes and llm_has_vision:
        image_base64 = base64.b64encode(first_page_image_bytes).decode("utf-8")
        content = [
            {"type": "text", "text": SUMMARIZE_PROMPT_VISION},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            {"type": "text", "text": pages_text},
        ]
    else:
        content = SUMMARIZE_PROMPT.format(text=pages_text)

    try:
        response = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": content}],
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None
        return _parse_summary_response(raw)
    except Exception as e:
        logger.error(f"Document summarization failed ({type(e).__name__}): {e}")
        return None
