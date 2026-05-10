"""
Skills catalog and router for the conversational agent.

A skill is a specialized behavioral mode that the agent can enter based on user intent.
Each skill has:
- name: identifier for the skill
- keywords: list of trigger keywords to detect the skill
- description: what the skill does
- context: skill-specific system instruction to append
"""

from typing import NamedTuple

class Skill(NamedTuple):
    """Definition of a single agent skill."""
    name: str
    keywords: list[str]
    description: str
    context: str


# Define initial skills catalog
SKILLS_CATALOG = {
    "explain": Skill(
        name="explain",
        keywords=["explain", "tell me about", "what is", "how does", "describe", "definition"],
        description="Educational mode: provide clear explanations with examples",
        context="""When in EXPLAIN mode:
- Provide clear, educational explanations
- Use examples and analogies where helpful
- Break down complex concepts into digestible parts
- Define technical terms as needed
- End with key takeaways if appropriate"""
    ),
    
    "code-edit": Skill(
        name="code-edit",
        keywords=["code", "edit", "implement", "fix", "refactor", "write", "create"],
        description="Coding mode: write and modify code efficiently",
        context="""When in CODE-EDIT mode:
- Provide working code examples
- Include necessary imports and dependencies
- Add inline comments for non-obvious logic
- Suggest improvements or optimizations
- Ask clarifying questions if requirements are ambiguous"""
    ),
    
    "debug": Skill(
        name="debug",
        keywords=["debug", "error", "bug", "fix this", "broken", "crash", "issue"],
        description="Debugging mode: diagnose and fix issues systematically",
        context="""When in DEBUG mode:
- Ask for error messages, logs, and context
- Reproduce the issue mentally before suggesting fixes
- Explain root causes, not just symptoms
- Provide step-by-step debugging instructions
- Suggest preventive measures for the future"""
    ),
    
    "bash-help": Skill(
        name="bash-help",
        keywords=["bash", "shell", "terminal", "command", "script", "zsh", "sh"],
        description="Terminal mode: help with shell commands and scripts",
        context="""When in BASH-HELP mode:
- Explain shell syntax and command behavior
- Provide safe, tested shell commands
- Warn about potentially destructive operations
- Show common patterns and best practices
- Include examples of command usage"""
    ),
}


def detect_skill(user_input: str) -> str:
    """
    Detect which skill to use based on user input.
    
    Args:
        user_input: The user's message
    
    Returns:
        Skill name (e.g., "explain", "code-edit", "debug", "bash-help")
        Returns "default" if no skill matches
    """
    user_lower = user_input.lower()
    
    # Score each skill based on keyword matches
    skill_scores = {}
    for skill_name, skill in SKILLS_CATALOG.items():
        score = sum(1 for keyword in skill.keywords if keyword in user_lower)
        if score > 0:
            skill_scores[skill_name] = score
    
    # Return highest scoring skill, or "default" if none match
    if skill_scores:
        return max(skill_scores, key=skill_scores.get)
    return "default"


def get_skill_context(skill_name: str) -> str:
    """
    Get the system instruction context for a skill.
    
    Args:
        skill_name: Name of the skill (or "default")
    
    Returns:
        The skill's context string, or empty string for "default"
    """
    if skill_name == "default" or skill_name not in SKILLS_CATALOG:
        return ""
    return SKILLS_CATALOG[skill_name].context


def get_skill_info(skill_name: str) -> dict:
    """
    Get full information about a skill.
    
    Args:
        skill_name: Name of the skill
    
    Returns:
        Dict with name, description, keywords
    """
    if skill_name == "default":
        return {
            "name": "default",
            "description": "Standard conversational mode",
            "keywords": [],
        }
    
    skill = SKILLS_CATALOG.get(skill_name)
    if not skill:
        return {"name": skill_name, "description": "Unknown skill", "keywords": []}
    
    return {
        "name": skill.name,
        "description": skill.description,
        "keywords": skill.keywords,
    }
