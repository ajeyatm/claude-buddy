import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam

# Configuration for sliding-window compaction
# COMPACTION_WINDOW: Number of most recent turns to keep (each turn = user + assistant messages)
# A turn is a user message followed by one or more assistant/tool messages
COMPACTION_WINDOW = int(os.getenv("COMPACTION_WINDOW", "10"))


def count_turns(messages: list["ChatCompletionMessageParam"]) -> int:
    """
    Count the number of conversation turns (user-initiated rounds).
    System messages and tool messages are not counted as turns.
    """
    turn_count = 0
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            turn_count += 1
    return turn_count


def compact_messages(
    messages: list["ChatCompletionMessageParam"],
    keep_last_n_turns: int = COMPACTION_WINDOW,
    client: "OpenAI | None" = None,
    model: str = "",
) -> tuple[list["ChatCompletionMessageParam"], dict]:
    """
    Implement sliding-window compaction: keep system message + most recent K turns.
    Drops oldest turns when the conversation exceeds the window.
    Optionally generates a summary of dropped messages.
    
    Args:
        messages: List of conversation messages
        keep_last_n_turns: Number of recent turns to keep
        client: OpenAI client (optional, for summary generation)
        model: Model name to use for summarization
    
    Returns:
        - Compacted message list
        - Metrics dict with before/after counts and token estimates
    """
    from app.budget import estimate_message_tokens
    from app.ui import LOG_CONSOLE
    
    # Record metrics before compaction
    before_count = len(messages)
    before_tokens = estimate_message_tokens(messages)
    
    # Always keep the system message (first message)
    if not messages or messages[0].get("role") != "system":
        return messages, {
            "before_count": before_count,
            "before_tokens": before_tokens,
            "after_count": before_count,
            "after_tokens": before_tokens,
            "turns_dropped": 0,
        }
    
    system_message = messages[0]
    remaining_messages = messages[1:]
    
    # Count user turns in remaining messages
    user_indices = []
    for i, msg in enumerate(remaining_messages):
        if msg.get("role") == "user":
            user_indices.append(i)
    
    # If we have more turns than desired, keep only the most recent ones
    dropped_messages = []
    if len(user_indices) > keep_last_n_turns:
        # Calculate how many turns to drop
        turns_to_drop = len(user_indices) - keep_last_n_turns
        
        # Find the index of the last user message to drop
        last_index_to_drop = user_indices[turns_to_drop - 1]
        
        # Keep all messages after (and including) the first message of the turn to keep
        # We need to find where the next turn starts (the next user message after drop point)
        if turns_to_drop < len(user_indices):
            next_turn_start = user_indices[turns_to_drop]
            # Extract dropped messages for potential summarization
            dropped_messages = remaining_messages[:next_turn_start]
            kept_messages = remaining_messages[next_turn_start:]
        else:
            kept_messages = []
            dropped_messages = remaining_messages
        
        turns_dropped = turns_to_drop
    else:
        kept_messages = remaining_messages
        dropped_messages = []
        turns_dropped = 0
    
    # Rebuild the message list with system message + kept messages
    compacted = [system_message] + kept_messages
    
    # Generate summary if messages were dropped and client is available
    if dropped_messages and client and model and turns_dropped > 0:
        from app.summary import generate_summary, update_summary_message, has_summary
        
        # Only summarize if we don't already have a summary (avoid redundant summaries)
        if not has_summary(compacted):
            summary_text = generate_summary(client, model, dropped_messages)
            if summary_text.strip():
                update_summary_message(compacted, summary_text)
                LOG_CONSOLE.print(f"[dim]💾 Summary added to conversation[/dim]")
    
    # Record metrics after compaction
    after_count = len(compacted)
    after_tokens = estimate_message_tokens(compacted)
    
    # Log compaction metrics
    metrics = {
        "before_count": before_count,
        "before_tokens": before_tokens,
        "after_count": after_count,
        "after_tokens": after_tokens,
        "turns_dropped": turns_dropped,
    }
    
    reduction_pct = ((before_tokens - after_tokens) / before_tokens * 100) if before_tokens > 0 else 0
    
    LOG_CONSOLE.print(
        f"[dim]💾 Compaction: {before_count} msgs → {after_count} msgs | "
        f"{before_tokens} tokens → {after_tokens} tokens ({reduction_pct:.1f}% reduction) | "
        f"Dropped {turns_dropped} turns[/dim]"
    )
    
    return compacted, metrics


def should_compact(messages: list["ChatCompletionMessageParam"], soft_limit: int) -> bool:
    """
    Determine if messages should be compacted based on token count.
    Returns True if we're at or over the soft limit.
    """
    from app.budget import is_over_soft_limit
    return is_over_soft_limit(messages)
