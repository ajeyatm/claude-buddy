import argparse
import os
from typing import TYPE_CHECKING, Literal, cast

from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from app.tools import TOOL_SPECS, execute_tool_calls, build_generic_system_prompt, build_dynamic_system_prompt
from app.ui import APP_CONSOLE, LOG_CONSOLE, ASSISTANT_HEADER_STYLE, USER_PROMPT_STYLE, print_usage
from app.models import FatalAgentError, RecoverableAgentError, validate_and_append_message
from app.budget import (
    is_over_soft_limit, is_over_hard_limit, 
    estimate_message_tokens, log_budget_status,
    SOFT_TOKEN_LIMIT, HARD_TOKEN_LIMIT
)
from app.compaction import compact_messages, should_compact, COMPACTION_WINDOW
from app.router import route_user_input, log_skill_routing

load_dotenv()

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageFunctionToolCall,
        ChatCompletionMessageParam,
    )


# API_KEY = os.getenv("OPENROUTER_API_KEY")
# BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
# MODEL = os.getenv("OPENROUTER_MODEL", default="anthropic/claude-haiku-4.5")
API_KEY = os.getenv("MY_GEN_ASSIST_TOKEN")
BASE_URL = os.getenv("MY_GEN_ASSIST_BASE_URL")
MODEL = os.getenv("MY_GEN_ASSIST_MODEL", "myGenAssist,claude-sonnet-4.6-azure")

MAX_CONSECUTIVE_RECOVERABLE_ERRORS = 3

# System prompt selection:
# Option A (Dynamic): Automatically generated from TOOL_SPECS, updates when tools change
#   SYSTEM_PROMPT = build_dynamic_system_prompt(TOOL_SPECS)
# Option B (Generic): Concise, lower token cost, still guides tool usage
#   SYSTEM_PROMPT = build_generic_system_prompt()

# Currently using Option B (Generic prompt)
SYSTEM_PROMPT = build_generic_system_prompt()

def build_initial_messages() -> list[ChatCompletionMessageParam]:
    """Create a fresh conversation history containing only the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def my_agent(client: OpenAI, messages: list[ChatCompletionMessageParam]) -> tuple[int, int, int]:
    """Run one full agent turn, including tool-call iterations, and return token totals."""

    turn_prompt_tokens = 0
    turn_completion_tokens = 0
    turn_total_tokens = 0
    
    # Preflight budget check before making API request
    estimated_tokens = estimate_message_tokens(messages)
    
    if is_over_hard_limit(messages):
        raise FatalAgentError(f"Hard token limit ({HARD_TOKEN_LIMIT}) exceeded. Session context too large. Consider starting a new session.")
    
    # Check if compaction is needed (soft limit exceeded)
    if should_compact(messages, SOFT_TOKEN_LIMIT):
        LOG_CONSOLE.print(f"[yellow]⚠️ Soft token limit ({SOFT_TOKEN_LIMIT}) exceeded. Compacting message history...[/yellow]")
        messages_to_use, metrics = compact_messages(messages, keep_last_n_turns=COMPACTION_WINDOW, client=client, model=MODEL)
        # Update the messages list in-place
        messages.clear()
        messages.extend(messages_to_use)
        estimated_tokens = metrics["after_tokens"]
    elif is_over_soft_limit(messages):
        LOG_CONSOLE.print(f"[yellow]⚠️ Warning: Context approaching soft limit ({SOFT_TOKEN_LIMIT} tokens). Currently at ~{estimated_tokens} tokens.[/yellow]")
        LOG_CONSOLE.print("[dim]Next message may trigger automatic compaction to manage memory.[/dim]")
    
    log_budget_status(messages)
 
    #agentic loop
    while True:
        # Each iteration is one model round-trip; tools may extend the same user turn.
        try:
            chat = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SPECS
            )
        except Exception as e:
            raise RecoverableAgentError(f"model request failed: {e}") from e

        if chat.usage:
            prompt_tokens = chat.usage.prompt_tokens or 0
            completion_tokens = chat.usage.completion_tokens or 0
            total_tokens = chat.usage.total_tokens or 0

            turn_prompt_tokens += prompt_tokens
            turn_completion_tokens += completion_tokens
            turn_total_tokens += total_tokens

            # usage({
            #     "usage_type": "turn",
            #     "prompt_tokens": prompt_tokens,
            #     "completion_tokens": completion_tokens,
            #     "total_tokens": total_tokens,
            # })


        if not chat.choices or len(chat.choices) == 0:
            raise FatalAgentError("no choices in response")
        
        response_message = chat.choices[0].message

        if not response_message:
            raise FatalAgentError("no message content in top choice")
        
        '''
        Whatever message the model returns, add it to your messages array. 
        If the model wants to use a tool, the response will contain a tool_calls array
        '''
        
        message_dict: dict[str, object] = {
            "role": response_message.role,
            "content": response_message.content,
        }

        #Tool calls are optional, so only add them to the message dict if they are present in the response
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            serialized_tool_calls: list[dict[str, object]] = []
            for tc in response_message.tool_calls:
                if tc.type != "function":
                    continue
                function_tc = cast("ChatCompletionMessageFunctionToolCall", tc)
                serialized_tool_calls.append(
                    {
                        "id": function_tc.id,
                        "type": function_tc.type,
                        "function": {
                            "name": function_tc.function.name,
                            "arguments": function_tc.function.arguments,
                        },
                    }
                )
            if serialized_tool_calls:
                # Only include tool_calls when present to avoid invalid empty arrays in history.
                message_dict["tool_calls"] = serialized_tool_calls

        # Persist assistant output so the next model call has full conversational context.
        validate_and_append_message(messages, message_dict)

        #Record the assistant's response --> END
        if not message_dict.get("tool_calls"):
            '''
             Continue the loop until the model responds without requesting any tools (when tool_calls is missing or empty).
             At this point, print the final message content to stdout and exit.
            '''
            break

        # Delegate concrete tool execution to tools module; it appends tool messages/errors.
        try:
            execute_tool_calls(response_message, messages)
        except Exception as e:
            raise RecoverableAgentError(f"tool execution failed: {e}") from e

    APP_CONSOLE.rule(f"[{ASSISTANT_HEADER_STYLE}]Assistant[/{ASSISTANT_HEADER_STYLE}]", style="dim")
    APP_CONSOLE.print(response_message.content)
    print_usage({
        "usage_type": "turn_summary",
        "prompt_tokens": turn_prompt_tokens,
        "completion_tokens": turn_completion_tokens,
        "total_tokens": turn_total_tokens,
    })
    return turn_prompt_tokens, turn_completion_tokens, turn_total_tokens

def main():
    """Run interactive chat loop and maintain session-level token usage totals."""
    if not API_KEY:
        raise RuntimeError("API_KEY is not set")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0
    consecutive_recoverable_errors = 0
    messages = build_initial_messages()

    while True:
        # Prompt user for the next turn in the ongoing session.
        query = APP_CONSOLE.input(f"[{USER_PROMPT_STYLE}]You[/{USER_PROMPT_STYLE}] [dim]> [/dim]")
        
        query = query.strip()
        if not query:
            continue

        normalized_query = query.lower()
        if normalized_query in ("exit", "quit", "q"):
            break

        try:
            _msg: dict[str, object] = {"role": "user", "content": query}
            validate_and_append_message(messages, _msg)
            
            # Route to skill and update system prompt with skill context
            routing_result = route_user_input(query)
            skill_log = log_skill_routing(routing_result["skill"], query)
            if skill_log:
                LOG_CONSOLE.print(skill_log)
            
            # Temporarily update system message with skill-aware version
            base_system_msg = messages[0]["content"]
            messages[0]["content"] = routing_result["system_prompt"]
            
            # Run agent with skill-aware system prompt
            turn_prompt_tokens, turn_completion_tokens, turn_total_tokens = my_agent(client, messages)
            
            # Restore base system message for next turn
            messages[0]["content"] = base_system_msg
            
            session_prompt_tokens += turn_prompt_tokens
            session_completion_tokens += turn_completion_tokens
            session_total_tokens += turn_total_tokens
            # Keep a running token summary for full-session observability.
            print_usage({
                "usage_type": "session_summary",
                "prompt_tokens": session_prompt_tokens,
                "completion_tokens": session_completion_tokens,
                "total_tokens": session_total_tokens,
            })
            consecutive_recoverable_errors = 0
        except RecoverableAgentError as e:
            consecutive_recoverable_errors += 1
            LOG_CONSOLE.print(f"[yellow]Recoverable error:[/yellow] {e}")
            if consecutive_recoverable_errors >= MAX_CONSECUTIVE_RECOVERABLE_ERRORS:
                LOG_CONSOLE.print("[bold red]Too many recoverable errors in a row. Resetting session history.[/bold red]")
                messages = build_initial_messages()
                consecutive_recoverable_errors = 0
            continue
        except FatalAgentError as e:
            LOG_CONSOLE.print(f"[bold red]Fatal validation/protocol error:[/bold red] {e}")
            LOG_CONSOLE.print("[bold red]Session history was reset. Please retry your request.[/bold red]")
            messages = build_initial_messages()
            consecutive_recoverable_errors = 0
            continue

if __name__ == "__main__":
    main()
