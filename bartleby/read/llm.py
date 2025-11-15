import base64
from typing import Optional

from langchain_core.messages import HumanMessage
from loguru import logger

MAX_INPUT_CHARACTERS = 10000 # ~2,000 words

def summarize_pdf_page(
    llm,
    body: str,
    llm_has_vision: bool = True,
    page_image_bytes: Optional[bytes] = None
) -> str:

    if len(body) > MAX_INPUT_CHARACTERS:
        body = body[:MAX_INPUT_CHARACTERS]

    if page_image_bytes and llm_has_vision:
        page_image_base64 = base64.b64encode(page_image_bytes).decode("utf-8")
        content = [
            {"type": "text", "text": f"Provide a fulsome summary of this page of a PDF, using both the image and text."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image_base64}"}},
            {"type": "text", "text": body},
        ]
    else:
        content = f"Provide a fulsome summary of this page of a PDF:\n\n{body}"

    try:
        response = llm.invoke([HumanMessage(content=content)])
        return (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.error(f"Summarization failed with ({type(e).__name__}): {e}")
        return ""