import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam

SUMMARY_MARKER = "[CONVERSATION_SUMMARY]"
SUMMARY_ROLE = "user"  # Store as user message so model can read it but won't confuse the conversation flow


def build_summary_prompt() -> str:
    """
    Create a prompt to summarize the conversation history.
    The prompt asks the model to extract key information from older turns.
    """
    return f"""Based on the conversation history below, create a concise summary covering:
1. Main goals and objectives discussed
2. Key constraints or limitations mentioned
3. Important decisions or conclusions reached
4. Pending tasks or questions to follow up on

Format as bullet points. Be concise but capture all essential context.
Only summarize messages that were marked for compaction (older turns).

Conversation to summarize:
"""


def generate_summary(
    client: "OpenAI",
    model: str,
    dropped_messages: list["ChatCompletionMessageParam"],
) -> str:
    """
    Generate a summary of dropped messages using the LLM.
    
    Args:
        client: OpenAI client instance
        model: Model name to use for summarization
        dropped_messages: Messages that were dropped during compaction
    
    Returns:
        A formatted summary string
    """
    from app.ui import LOG_CONSOLE
    
    if not dropped_messages:
        return ""
    
    # Build the conversation text from dropped messages
    conv_text = ""
    for msg in dropped_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            conv_text += f"[{role.upper()}] {content}\n\n"
    
    if not conv_text.strip():
        return ""
    
    # Create summary request
    summary_prompt = build_summary_prompt() + conv_text
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise summaries of conversations.",
                },
                {
                    "role": "user",
                    "content": summary_prompt,
                }
            ],
            temperature=0.5,  # Lower temperature for more consistent summaries
            max_tokens=500,  # Limit summary length
        )
        
        summary_text = response.choices[0].message.content or ""
        LOG_CONSOLE.print(f"[dim]📝 Summary generated ({len(summary_text)} chars)[/dim]")
        return summary_text
    except Exception as e:
        LOG_CONSOLE.print(f"[yellow]⚠️ Failed to generate summary: {e}[/yellow]")
        return ""


def create_summary_message(summary_text: str) -> "ChatCompletionMessageParam":
    """
    Create a summary message to insert into the conversation.
    This message preserves the summary for the next API request.
    """
    return {
        "role": "user",
        "content": f"{SUMMARY_MARKER}\n{summary_text}",
    }


def has_summary(messages: list["ChatCompletionMessageParam"]) -> bool:
    """Check if a summary message already exists in the conversation."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and SUMMARY_MARKER in content:
            return True
    return False


def update_summary_message(
    messages: list["ChatCompletionMessageParam"],
    new_summary: str,
) -> None:
    """
    Update or insert a summary message in the conversation.
    Replaces existing summary if present, otherwise inserts after system message.
    """
    # Find and remove existing summary message (if any)
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, str) and SUMMARY_MARKER in content:
            messages.pop(i)
            break
    
    # Insert new summary after system message (at index 1)
    summary_msg = create_summary_message(new_summary)
    if len(messages) > 0:
        # Insert after system message
        messages.insert(1, summary_msg)
    else:
        # Shouldn't happen, but handle gracefully
        messages.append(summary_msg)
