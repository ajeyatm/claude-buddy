import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Token budget configuration (configurable via environment variables)
# Soft limit: triggers message compaction as a precaution
# Hard limit: absolute maximum tokens we'll allow in a session
SOFT_TOKEN_LIMIT = int(os.getenv("SOFT_TOKEN_LIMIT", "100000"))
HARD_TOKEN_LIMIT = int(os.getenv("HARD_TOKEN_LIMIT", "120000"))

def estimate_tokens(text: str) -> int:
    """
    Simple heuristic to estimate token count from text.
    Uses the rough approximation: 1 token ≈ 4 characters (OpenAI standard).
    This is a quick estimate; actual token count may vary.
    """
    # Standard OpenAI approximation: ~4 chars per token
    estimated = len(text) / 4
    return max(1, int(estimated))


def estimate_message_tokens(messages: list["ChatCompletionMessageParam"]) -> int:
    """
    Estimate total token count for a list of messages.
    Sum up individual message token estimates.
    """
    total_tokens = 0
    
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_tokens += estimate_tokens(content)
        
        # Handle assistant messages with tool_calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Each tool call has structure: id, type, function.name, function.arguments
            # Rough estimate: ~50 tokens per tool call metadata + arguments
            for tc in tool_calls:
                if isinstance(tc, dict):
                    function_dict = tc.get("function", {})
                    args_str = function_dict.get("arguments", "")
                    # Estimate: tool call overhead + argument tokens
                    total_tokens += 50 + estimate_tokens(args_str)
    
    return total_tokens


def is_over_soft_limit(messages: list["ChatCompletionMessageParam"]) -> bool:
    """Check if message context exceeds soft token limit."""
    estimated = estimate_message_tokens(messages)
    return estimated > SOFT_TOKEN_LIMIT


def is_over_hard_limit(messages: list["ChatCompletionMessageParam"]) -> bool:
    """Check if message context exceeds hard token limit."""
    estimated = estimate_message_tokens(messages)
    return estimated > HARD_TOKEN_LIMIT


def log_budget_status(messages: list["ChatCompletionMessageParam"]) -> None:
    """Log current token budget status."""
    estimated = estimate_message_tokens(messages)
    soft_pct = (estimated / SOFT_TOKEN_LIMIT) * 100 if SOFT_TOKEN_LIMIT > 0 else 0
    hard_pct = (estimated / HARD_TOKEN_LIMIT) * 100 if HARD_TOKEN_LIMIT > 0 else 0
    
    from app.ui import LOG_CONSOLE
    
    status = f"[dim]Budget: {estimated} tokens"
    if soft_pct > 100:
        status += f" (OVER soft limit {SOFT_TOKEN_LIMIT})"
    if hard_pct > 100:
        status += f" (OVER hard limit {HARD_TOKEN_LIMIT})"
    status += f" ({soft_pct:.1f}% of soft limit)[/dim]"
    
    LOG_CONSOLE.print(status)
