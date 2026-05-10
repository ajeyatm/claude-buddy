"""
Skill router: routes user input to appropriate skills and composes system prompts.
"""

from app.skills import detect_skill, get_skill_context, SKILLS_CATALOG
from app.tools import build_generic_system_prompt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def build_skill_aware_system_prompt(skill_name: str) -> str:
    """
    Build a system prompt that includes skill-specific instructions.
    
    Args:
        skill_name: Name of the skill to apply (or "default")
    
    Returns:
        Complete system prompt with skill context appended
    """
    base_prompt = build_generic_system_prompt()
    skill_context = get_skill_context(skill_name)
    
    if skill_context:
        # Append skill context to the base prompt
        return f"{base_prompt}\n\n[ACTIVE SKILL: {skill_name.upper()}]\n{skill_context}"
    
    return base_prompt


def route_user_input(user_input: str) -> dict:
    """
    Route user input to the appropriate skill and get routing info.
    
    Args:
        user_input: The user's message
    
    Returns:
        Dict with:
        - skill: detected skill name
        - skill_info: skill description and keywords
        - system_prompt: skill-aware system prompt
        - is_default: True if default skill was used
    """
    skill_name = detect_skill(user_input)
    is_default = skill_name == "default"
    
    # Get skill metadata
    skill_info = None
    if not is_default and skill_name in SKILLS_CATALOG:
        skill = SKILLS_CATALOG[skill_name]
        skill_info = {
            "name": skill.name,
            "description": skill.description,
            "keywords": skill.keywords,
        }
    
    return {
        "skill": skill_name,
        "skill_info": skill_info,
        "system_prompt": build_skill_aware_system_prompt(skill_name),
        "is_default": is_default,
    }


def log_skill_routing(skill_name: str, user_input: str) -> str:
    """
    Format a log message for skill routing.
    
    Args:
        skill_name: Name of detected skill
        user_input: The user's input
    
    Returns:
        Formatted log message
    """
    if skill_name == "default":
        return ""  # Don't log default skill routing
    
    skill = SKILLS_CATALOG.get(skill_name)
    if skill:
        return f"[dim]🎯 Skill: {skill_name} ({skill.description})[/dim]"
    
    return ""
