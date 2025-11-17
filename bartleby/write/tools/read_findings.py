"""Tool for reading all research findings files at synthesis time."""

from pathlib import Path

from langchain_core.tools import tool


def create_read_findings_tool(findings_dir: Path, run_uuid: str):
    """
    Create findings reading tool for Primary Agent synthesis phase.

    Args:
        findings_dir: Directory containing findings files
        run_uuid: UUID of current run

    Returns:
        LangChain tool instance
    """

    @tool
    def read_findings() -> str:
        """
        Read all research findings from delegated Search Agent tasks.

        This tool is only available in the final synthesis phase. It reads all findings
        files generated during the research phase and returns them as a single corpus
        for you to synthesize into your final report.

        Returns:
            Complete research findings from all Search Agent delegations
        """
        if not findings_dir.exists():
            return "No findings directory found. No research has been conducted yet."

        # Glob for all findings files matching this run's UUID
        pattern = f"{run_uuid}-*.md"
        findings_files = sorted(findings_dir.glob(pattern))

        if not findings_files:
            return f"No findings files found for this run (pattern: {pattern})"

        # Read and concatenate all findings
        findings_parts = []
        for i, findings_file in enumerate(findings_files, 1):
            content = findings_file.read_text(encoding="utf-8")
            findings_parts.append(
                f"# Research Task {i}\n"
                f"*Source: {findings_file.name}*\n"
                f"\n"
                f"{content}\n"
                f"\n"
                f"---\n"
            )

        summary_header = (
            "# Complete Research Findings\n"
            "\n"
            f"Total research tasks completed: {len(findings_files)}\n"
            "\n"
            "---\n"
            "\n"
        )

        return summary_header + "".join(findings_parts)

    return read_findings
