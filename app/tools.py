import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast
import subprocess

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageFunctionToolCall,
        ChatCompletionMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolUnionParam,
    )

BASH_TIMEOUT_SECONDS = int(os.getenv("BASH_TIMEOUT_SECONDS", "15"))

TOOL_SPECS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the content of a file. The input should be a valid file path. The output will be the content of the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file. The input should be a valid file path and the content to write. The output will be a success message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a bash command. The input should be a valid bash command. The output will be the result of the command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def append_tool_error(messages: list[ChatCompletionMessageParam], tool_call_id: str, tool_name: str, reason: str) -> None:
    """Append a standardized tool error message to conversation history."""
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": f"TOOL_ERROR: {tool_name}: {reason}",
        }
    )


def execute_tool_calls(response_message: object, messages: list[ChatCompletionMessageParam]) -> None:
    """Execute model-requested tool calls and append tool outputs/errors to messages."""
    tool_calls = getattr(response_message, "tool_calls", None) or []

    # Process each requested tool call independently so one failure does not break others.
    for tc in tool_calls:
        # Only function tool calls are supported in this agent.
        if tc.type != "function":
            append_tool_error(messages, tc.id, tc.type, "tool call is not a function call")
            continue

        function_tc = cast("ChatCompletionMessageFunctionToolCall", tc)
        tool_name = function_tc.function.name

        # Guardrail for unknown tool names.
        if tool_name not in ["Read", "Write", "Bash"]:
            append_tool_error(messages, function_tc.id, tool_name, f"tool {tool_name} is not supported")
            continue

        # Tool arguments are passed as JSON strings by the model.
        try:
            args = json.loads(function_tc.function.arguments)
        except json.JSONDecodeError as e:
            # Invalid JSON should be surfaced to the model as a tool error, not a hard crash.
            append_tool_error(messages, function_tc.id, tool_name, f"invalid JSON arguments: {e}")
            continue

        if tool_name == "Read":
            # Read file content from disk and return it as tool output.
            file_path = args.get("file_path")

            if not file_path:
                append_tool_error(messages, function_tc.id, tool_name, "no file_path argument provided for Read tool call")
                continue

            if not os.path.isfile(file_path):
                append_tool_error(messages, function_tc.id, tool_name, f"file_path {file_path} does not exist or is not a file")
                continue

            try:
                # Return raw file contents so the model can reason over exact text.
                with open(file_path) as f:
                    result: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": function_tc.id,
                        "content": f.read(),
                    }
                    messages.append(result)
            except Exception as e:
                append_tool_error(messages, function_tc.id, tool_name, f"error reading file_path {file_path}: {e}")
                continue

        elif tool_name == "Write":
            # Write content to target file, creating parent directories when needed.
            file_path_str = args.get("file_path")
            content = args.get("content")

            if not file_path_str:
                append_tool_error(messages, function_tc.id, tool_name, "no file_path argument provided for Write tool call")
                continue

            if not content:
                append_tool_error(messages, function_tc.id, tool_name, "no content argument provided for Write tool call")
                continue

            try:
                file_path = Path(file_path_str)
                # Ensure parent directories exist before writing nested files.
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, "w") as f:
                    f.write(content)

                    result: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": function_tc.id,
                        "content": f"Successfully wrote to {file_path}",
                    }
                    messages.append(result)
            except Exception as e:
                append_tool_error(messages, function_tc.id, tool_name, f"error writing to file_path {file_path_str}: {e}")
                continue

        elif tool_name == "Bash":
            # Execute shell command and return combined stdout/stderr.
            command = args.get("command")

            if not command:
                append_tool_error(messages, function_tc.id, tool_name, "no command argument provided for Bash tool call")
                continue

            try:
                # Use check=True so non-zero exit codes are handled as explicit tool failures.
                resp = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout=BASH_TIMEOUT_SECONDS)
                # Return combined stdout/stderr so the model sees the full command result.
                output = resp.stdout + resp.stderr

                result: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": function_tc.id,
                    "content": output,
                }
                messages.append(result)
            except subprocess.CalledProcessError as e:
                append_tool_error(messages, function_tc.id, tool_name, f"Command failed with error: {e}")
            except subprocess.TimeoutExpired as e:
                append_tool_error(messages, function_tc.id, tool_name, f"Command timed out after {BASH_TIMEOUT_SECONDS}s")
            except Exception as e:
                append_tool_error(messages, function_tc.id, tool_name, f"Error executing command: {e}")


def build_dynamic_system_prompt(tool_specs: list) -> str:
    """
    OPTION A: Build a system prompt dynamically from available tool specs.
    This ensures the prompt always stays in sync with actual tools.
    Auto-updates when tools change without code modifications.
    """
    tools_description = []
    
    for tool in tool_specs:
        if tool["type"] != "function":
            continue
        
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        params = func["parameters"].get("properties", {})
        required = func["parameters"].get("required", [])
        
        # Build parameter list
        param_info = []
        for param_name, param_spec in params.items():
            param_type = param_spec.get("type", "unknown")
            param_desc = param_spec.get("description", "")
            is_required = param_name in required
            req_marker = "(required)" if is_required else "(optional)"
            param_info.append(f"   - {param_name} ({param_type}) {req_marker}: {param_desc}")
        
        param_str = "\n".join(param_info) if param_info else "   - No parameters"
        
        tools_description.append(f"""
**{name}**
  Description: {description}
  Parameters:
{param_str}""")
    
    tools_section = "\n".join(tools_description)
    
    return f"""You are an expert programming assistant with access to powerful tools for code analysis and execution.

**Available Tools:**
{tools_section}

**Usage Guidelines:**
- Use tools proactively to understand the codebase or execute commands
- Read files before suggesting changes
- Execute bash commands to verify system state
- For file operations, always provide complete, well-formatted content
- Provide concise, actionable responses with clear explanations
- End with relevant follow-up questions the user can ask for deeper learning

**Safety & Best Practices:**
- Bash commands have a {BASH_TIMEOUT_SECONDS}-second timeout
- Always validate file paths before reading/writing
- Handle errors gracefully and provide helpful error messages
"""


def build_generic_system_prompt() -> str:
    """
    OPTION B: Build an enhanced generic system prompt.
    More concise than Option A, guides tool usage without explicit tool listing.
    Lower token cost while maintaining strong tool awareness.
    """
    return f"""You are a highly knowledgeable programming assistant with access to powerful tools for file operations and command execution.

You have the ability to:
- Read and analyze file contents to understand code and configuration
- Write and create files to implement solutions
- Execute shell commands to verify system state and run scripts

When answering questions:
- Use tools proactively to examine files, run commands, and verify system state when helpful
- Read relevant files before suggesting changes to understand context fully
- Provide clear, concise, and actionable explanations
- Suggest follow-up questions at the end of your response for deeper learning

Key constraints:
- Bash commands have a {BASH_TIMEOUT_SECONDS}-second timeout for safety
- Always validate paths and handle errors gracefully
- Be strategic about tool use—don't overuse when a direct answer suffices
"""