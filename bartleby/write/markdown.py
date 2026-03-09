"""Light markdown renderer — no Rich Markdown panels/borders."""

from __future__ import annotations

import re

from rich.console import Console
from rich.text import Text


def render_markdown(console: Console, text: str) -> None:
    """Render markdown text to the console with light formatting.

    Line-by-line processing:
    - # headers -> bold text
    - **text** -> bold inline
    - *text* -> italic inline
    - `code` -> dim inline
    - - / * list items -> 2-space indent
    - 1. numbered items -> 2-space indent
    - ``` code blocks -> 4-space indent, no borders
    - Paragraphs separated by blank lines
    """
    lines = text.split("\n")
    in_code_block = False
    buffer: list[Text | str] = []

    def flush():
        for item in buffer:
            console.print(item)
        buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code block toggle
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                i += 1
                continue
            else:
                in_code_block = False
                i += 1
                continue

        if in_code_block:
            buffer.append(Text(f"    {line}", style="dim"))
            i += 1
            continue

        # Blank line — paragraph break
        if not line.strip():
            buffer.append("")
            i += 1
            continue

        # Headers
        header_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if header_match:
            header_text = header_match.group(2)
            buffer.append(Text(header_text, style="bold"))
            i += 1
            continue

        # Unordered list items
        list_match = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if list_match:
            indent = list_match.group(1)
            content = list_match.group(2)
            rendered = _render_inline(f"  {indent}- {content}")
            buffer.append(rendered)
            i += 1
            continue

        # Ordered list items
        ol_match = re.match(r"^(\s*)(\d+\.)\s+(.*)", line)
        if ol_match:
            indent = ol_match.group(1)
            number = ol_match.group(2)
            content = ol_match.group(3)
            rendered = _render_inline(f"  {indent}{number} {content}")
            buffer.append(rendered)
            i += 1
            continue

        # Regular paragraph text
        buffer.append(_render_inline(line))
        i += 1

    flush()


def _render_inline(text: str) -> Text:
    """Apply inline formatting: bold, italic, inline code."""
    result = Text()
    i = 0
    chars = text

    while i < len(chars):
        # Inline code: `...`
        if chars[i] == "`" and not (i > 0 and chars[i - 1] == "\\"):
            end = chars.find("`", i + 1)
            if end != -1:
                result.append(chars[i + 1 : end], style="dim")
                i = end + 1
                continue

        # Bold: **...**
        if chars[i : i + 2] == "**":
            end = chars.find("**", i + 2)
            if end != -1:
                result.append(chars[i + 2 : end], style="bold")
                i = end + 2
                continue

        # Italic: *...*  (but not **)
        if chars[i] == "*" and (i + 1 >= len(chars) or chars[i + 1] != "*"):
            end = chars.find("*", i + 1)
            if end != -1 and (end + 1 >= len(chars) or chars[end + 1] != "*"):
                result.append(chars[i + 1 : end], style="italic")
                i = end + 1
                continue

        result.append(chars[i])
        i += 1

    return result
