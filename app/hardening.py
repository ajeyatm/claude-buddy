"""
Phase 6 Regression Checklist and Optional Features

This module provides:
1. Regression checks to ensure no regressions
2. Optional toggles for runtime configuration
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# Optional toggles (configure via environment variables)
SHOW_USAGE = os.getenv("SHOW_USAGE", "true").lower() == "true"
CLI_THEME = os.getenv("CLI_THEME", "default")  # minimal, high-contrast, default
COMPACTION_ENABLED = os.getenv("COMPACTION_ENABLED", "true").lower() == "true"


class RegressionChecklist:
    """Regression checks to validate agent behavior."""
    
    @staticmethod
    def check_no_invalid_tool_calls(messages: list) -> bool:
        """
        Check that no assistant messages have invalid tool_calls structure.
        
        Returns:
            True if all tool_calls are valid, False otherwise
        """
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls is not None:
                    # tool_calls must be a non-empty list with proper structure
                    if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                        return False
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            return False
                        if not all(k in tc for k in ("id", "type", "function")):
                            return False
                        if not isinstance(tc["id"], str) or not tc["id"].strip():
                            return False
        return True
    
    @staticmethod
    def check_session_token_tracking(session_tokens: int) -> bool:
        """
        Check that session token usage is tracked.
        
        Returns:
            True if tokens > 0, False otherwise
        """
        return session_tokens > 0
    
    @staticmethod
    def check_no_tool_execution_crash(execution_result: dict) -> bool:
        """
        Check that tool execution didn't crash (has result or error message).
        
        Returns:
            True if execution has result or error, False if both missing
        """
        return "result" in execution_result or "error" in execution_result
    
    @staticmethod
    def validate_all(messages: list, session_tokens: int) -> dict:
        """
        Run all regression checks.
        
        Returns:
            Dict with check results
        """
        return {
            "no_invalid_tool_calls": RegressionChecklist.check_no_invalid_tool_calls(messages),
            "session_token_tracking": RegressionChecklist.check_session_token_tracking(session_tokens),
        }


class ToggleConfig:
    """Runtime configuration toggles."""
    
    @staticmethod
    def should_show_usage() -> bool:
        """Check if usage statistics should be shown."""
        return SHOW_USAGE
    
    @staticmethod
    def get_cli_theme() -> str:
        """Get the current CLI theme."""
        if CLI_THEME not in ("minimal", "high-contrast", "default"):
            return "default"
        return CLI_THEME
    
    @staticmethod
    def is_compaction_enabled() -> bool:
        """Check if message compaction is enabled."""
        return COMPACTION_ENABLED
    
    @staticmethod
    def log_config() -> str:
        """Log current configuration."""
        return (
            f"Config: SHOW_USAGE={SHOW_USAGE}, "
            f"CLI_THEME={CLI_THEME}, "
            f"COMPACTION_ENABLED={COMPACTION_ENABLED}"
        )
