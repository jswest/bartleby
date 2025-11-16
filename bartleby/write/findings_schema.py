"""Schema for structured Search Agent findings with markdown rendering."""

from typing import List
from pydantic import BaseModel, Field


class SearchFindings(BaseModel):
    """Complete structured findings from a Search Agent delegation."""

    task: str = Field(description="The research task that was delegated")
    key_findings: List[str] = Field(
        description="Main discoveries, insights, or conclusions (bullet points)"
    )
    documents_cited: List[str] = Field(
        description="List of document IDs referenced in the research"
    )
    summary: str = Field(
        description="Brief summary of findings to return to Primary Agent"
    )


def render_findings_to_markdown(findings: SearchFindings) -> str:
    """
    Render SearchFindings to a consistent markdown format.

    Args:
        findings: Structured findings from Search Agent

    Returns:
        Markdown-formatted research report
    """
    lines = []

    # Title
    lines.append(f"# {findings.task}")
    lines.append("")

    # Key Findings
    if findings.key_findings:
        lines.append("## Key Findings")
        lines.append("")
        for finding in findings.key_findings:
            lines.append(f"- {finding}")
        lines.append("")

    # Documents Cited
    if findings.documents_cited:
        lines.append("## Documents Cited")
        lines.append("")
        for doc_id in findings.documents_cited:
            lines.append(f"- `{doc_id}`")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(findings.summary)
    lines.append("")

    return "\n".join(lines)
