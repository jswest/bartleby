"""Skill registry — discovers and instantiates tools by agent name."""

import importlib
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills._base import _parse_frontmatter


_SKILLS_DIR = Path(__file__).parent


def collect_tools(agent_name: str, context: dict) -> list[Tool]:
    """Discover and instantiate all tools assigned to the given agent.

    Scans subdirectories of skills/ for skill.md files, checks if the
    agent is listed in the frontmatter's 'agents' field, and if so,
    imports the tool module and calls its create(context) factory.
    """
    tools = []
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue

        skill_md = skill_dir / "skill.md"
        if not skill_md.exists():
            continue

        raw = skill_md.read_text(encoding="utf-8")
        frontmatter, _ = _parse_frontmatter(raw)

        agents = frontmatter.get("agents", [])
        if agent_name not in agents:
            continue

        # Import the tool module and call create()
        module_name = f"bartleby.write.skills.{skill_dir.name}.tool"
        module = importlib.import_module(module_name)
        tool = module.create(context)
        tools.append(tool)

    return tools
