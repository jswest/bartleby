"""Skill discovery and loading for the write command."""

import importlib
import inspect
from pathlib import Path

from loguru import logger
from smolagents import Tool

from bartleby.write.skills.base import Skill


_EXCLUDED_MODULES = {"__init__", "base"}


def load_skills(context: dict) -> list[Tool]:
    """
    Auto-discover and load all skills from this directory.

    1. Scan for .py files (exclude __init__.py, base.py)
    2. Import each module
    3. Find Skill subclass
    4. Call get_tools(context) to collect tools
    5. Return flat list of all tools

    Args:
        context: Shared resources dict (db_path, model, etc.)

    Returns:
        Flat list of all Tool instances from all skills
    """
    skills_dir = Path(__file__).parent
    all_tools: list[Tool] = []
    loaded_skills: list[str] = []

    for py_file in sorted(skills_dir.glob("*.py")):
        module_name = py_file.stem
        if module_name in _EXCLUDED_MODULES:
            continue

        try:
            module = importlib.import_module(f"bartleby.write.skills.{module_name}")
        except Exception as e:
            logger.warning(f"Failed to import skill module '{module_name}': {e}")
            continue

        # Find Skill subclass in module
        skill_class = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Skill) and obj is not Skill:
                skill_class = obj
                break

        if skill_class is None:
            logger.debug(f"No Skill subclass found in '{module_name}', skipping")
            continue

        try:
            skill_instance = skill_class()
            tools = skill_instance.get_tools(context)
            all_tools.extend(tools)
            loaded_skills.append(skill_instance.name or module_name)
            logger.debug(
                f"Loaded skill '{skill_instance.name or module_name}' "
                f"with {len(tools)} tool(s)"
            )
        except Exception as e:
            logger.warning(f"Failed to load skill '{module_name}': {e}")

    logger.info(f"Loaded {len(loaded_skills)} skills with {len(all_tools)} total tools")
    return all_tools
