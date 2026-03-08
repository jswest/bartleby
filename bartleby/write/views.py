"""Rich rendering for source references and browse views."""

import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bartleby.write.references import ReferenceRegistry


def render_sources_table(console: Console, registry: ReferenceRegistry):
    """Render a sources table after an agent answer."""
    refs = registry.all_refs()
    if not refs:
        return

    table = Table(title="Sources", show_lines=False, pad_edge=False)
    table.add_column("#", style="bold", width=4)
    table.add_column("Document", ratio=3)
    table.add_column("Page", width=6)
    table.add_column("Section", ratio=2)

    for entry in refs:
        origin = entry.get("origin_file_path") or ""
        filename = os.path.basename(origin) if origin else entry.get("document_id", "")
        page = str(entry["page_number"]) if entry.get("page_number") is not None else ""
        section = entry.get("section_heading") or ""

        table.add_row(
            f"[{entry['ref']}]",
            filename,
            page,
            section,
        )

    console.print(table)
    console.print("[dim]Use /browse <#> to see the full passage in context.[/dim]\n")


def render_browse_view(console: Console, ref_entry: dict, window_data: dict):
    """Render a Rich Panel showing a chunk window with the target highlighted."""
    origin = window_data.get("origin_file_path") or ""
    filename = os.path.basename(origin) if origin else ref_entry.get("document_id", "")
    ref_num = ref_entry["ref"]
    center_index = window_data.get("center_chunk_index")

    # Build subtitle from page and section
    parts = []
    page = ref_entry.get("page_number")
    if page is not None:
        parts.append(f"p. {page}")
    section = ref_entry.get("section_heading")
    if section:
        parts.append(section)
    subtitle = " | ".join(parts) if parts else None

    # Build the body text
    body = Text()
    chunks = window_data.get("chunks", [])
    for i, chunk in enumerate(chunks):
        chunk_index = chunk.get("chunk_index")
        text = chunk.get("body", "")
        is_target = chunk_index == center_index

        if i > 0:
            body.append("\n\n")

        if is_target:
            body.append(text, style="bold")
        else:
            body.append(text, style="dim")

    panel = Panel(
        body,
        title=f"[{ref_num}] {filename}",
        subtitle=subtitle,
        expand=True,
        padding=(1, 2),
    )
    console.print(panel)
