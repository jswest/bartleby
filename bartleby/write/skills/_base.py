"""Shared utilities for skill definitions."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SkillMeta:
    """Metadata parsed from a skill.md frontmatter."""
    name: str
    description: str
    inputs: dict
    output_type: str
    agents: list[str] = field(default_factory=list)


def load_skill_meta(tool_file: str) -> SkillMeta:
    """Load skill metadata from the skill.md next to the given tool.py.

    Parses YAML frontmatter (between --- delimiters) for structured metadata,
    and uses the body text as the tool description.
    """
    skill_path = Path(tool_file).parent / "skill.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"No skill.md found at {skill_path}")

    raw = skill_path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw)

    return SkillMeta(
        name=frontmatter.get("name", skill_path.parent.name),
        description=body.strip(),
        inputs=frontmatter.get("inputs", {}),
        output_type=frontmatter.get("output_type", "string"),
        agents=frontmatter.get("agents", []),
    )


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into YAML frontmatter dict and body text."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    return frontmatter, body


def slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe kebab-case slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]
