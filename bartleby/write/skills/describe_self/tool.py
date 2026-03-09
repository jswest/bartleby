"""Describe the agent's own capabilities and available tools."""

from collections import defaultdict
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills._base import _parse_frontmatter, load_skill_meta

meta = load_skill_meta(__file__)


class DescribeSelfTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

    def __init__(self):
        super().__init__()
        self._skills_dir = Path(__file__).parent.parent

    def forward(self) -> str:
        # Load static about section
        about_path = Path(__file__).parent / "about.md"
        about = about_path.read_text(encoding="utf-8").strip()

        # Scan all skills and group by agent
        by_agent: dict[str, list[str]] = defaultdict(list)
        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue
            skill_md = skill_dir / "skill.md"
            if not skill_md.exists():
                continue
            raw = skill_md.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(raw)
            name = frontmatter.get("name", skill_dir.name)
            short_desc = body.strip().split("\n")[0] if body.strip() else ""
            for agent in frontmatter.get("agents", []):
                by_agent[agent].append(f"- **{name}**: {short_desc}")

        # Build tool inventory
        sections = [about, ""]
        for agent_name in ["research", "search_expert"]:
            label = "Your tools" if agent_name == "research" else "Search expert's tools"
            sections.append(f"## {label}\n")
            sections.extend(by_agent.get(agent_name, ["- (none)"]))
            sections.append("")

        return "\n".join(sections)


def create(context: dict) -> DescribeSelfTool:
    return DescribeSelfTool()
