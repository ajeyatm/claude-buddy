import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
import subprocess

from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageFunctionToolCall,
        ChatCompletionMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolUnionParam,
        ChatCompletionAssistantMessageParam
    )


# API_KEY = os.getenv("OPENROUTER_API_KEY")
# BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
# MODEL = os.getenv("OPENROUTER_MODEL", default="anthropic/claude-haiku-4.5")
API_KEY = os.getenv("MY_GEN_ASSIST_TOKEN")
BASE_URL = os.getenv("MY_GEN_ASSIST_BASE_URL")
MODEL = os.getenv("MY_GEN_ASSIST_MODEL", "myGenAssist,claude-sonnet-4.6-azure")

REPEAT_CHAR = "-"
REPEAT_COUNT = 40
APP_CONSOLE = Console()
LOG_CONSOLE = Console(stderr=True)

USER_PROMPT_STYLE = "bold bright_cyan"
ASSISTANT_HEADER_STYLE = "bold bright_green"
USAGE_LABEL_STYLE = "bold yellow"
USAGE_PROMPT_STYLE = "bright_cyan"
USAGE_COMPLETION_STYLE = "bright_green"
USAGE_TOTAL_STYLE = "bold white"
USAGE_SEPARATOR_STYLE = "dim"

def usage(usage : dict[str, int | str]) -> None:
    """Prints usage information to the console in a formatted way. 
    Expects a dictionary with keys 'usage_type', 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
    """
    usage_type = usage.get("usage_type", "N/A")
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    LOG_CONSOLE.print(f"[{USAGE_SEPARATOR_STYLE}]{REPEAT_CHAR * REPEAT_COUNT}[/{USAGE_SEPARATOR_STYLE}]")
    LOG_CONSOLE.print(
        f"[{USAGE_LABEL_STYLE}]usage_{usage_type}[/{USAGE_LABEL_STYLE}] "
        f"prompt=[{USAGE_PROMPT_STYLE}]{prompt_tokens}[/{USAGE_PROMPT_STYLE}] "
        f"completion=[{USAGE_COMPLETION_STYLE}]{completion_tokens}[/{USAGE_COMPLETION_STYLE}] "
        f"total=[{USAGE_TOTAL_STYLE}]{total_tokens}[/{USAGE_TOTAL_STYLE}]"
    )
    LOG_CONSOLE.print(f"[{USAGE_SEPARATOR_STYLE}]{REPEAT_CHAR * REPEAT_COUNT}[/{USAGE_SEPARATOR_STYLE}]")

def append_tool_error(messages: list[ChatCompletionMessageParam], tool_call_id: str, tool_name: str, reason: str) -> None:
    """
    append standardized tool errors:
    append_tool_error(messages, tool_call_id, tool_name, reason)
    Content format:
    TOOL_ERROR: tool_name: reason
    """
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": f"TOOL_ERROR: {tool_name}: {reason}"
    })

TOOL_SPECS : list[ChatCompletionToolUnionParam] = [
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
                                    "description": "The path to the file to read."
                                }
                            },
                            "required": ["file_path"]
                        }
                    }
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
                                    "description": "The path to the file to write."
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file."
                                }
                            },
                            "required": ["file_path", "content"]
                        }
                    }
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
                                    "description": "The bash command to execute."
                                }
                            },
                            "required": ["command"]
                        }
                    }
                }
            ]


def my_agent(client: OpenAI, messages: list[ChatCompletionMessageParam]) -> tuple[int, int, int]:

    turn_prompt_tokens = 0
    turn_completion_tokens = 0
    turn_total_tokens = 0
 
    #agentic loop
    while True:
        chat = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SPECS
        )

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
            raise RuntimeError("no choices in response")
        
        response_message = chat.choices[0].message

        if not response_message:
            raise RuntimeError("no message content in top choice")
        
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
                message_dict["tool_calls"] = serialized_tool_calls

        messages.append(cast("ChatCompletionMessageParam", message_dict))

        #Record the assistant's response --> END
        if not message_dict.get("tool_calls"):
            '''
             Continue the loop until the model responds without requesting any tools (when tool_calls is missing or empty).
             At this point, print the final message content to stdout and exit.
            '''
            break

        
        #Execute tool calls if there are any
        for tc in response_message.tool_calls or []:
            if tc.type != "function":
                append_tool_error(messages, tc.id, tc.type, "tool call is not a function call")
                continue
            
            function_tc = cast("ChatCompletionMessageFunctionToolCall", tc)
            

            if function_tc.function.name not in ['Read', 'Write', 'Bash']:
                append_tool_error(messages, function_tc.id, function_tc.function.name, f"tool {function_tc.function.name} is not supported")
                continue
            
            try:
                args = json.loads(function_tc.function.arguments)
            except json.JSONDecodeError as e:
                append_tool_error(messages, function_tc.id, function_tc.function.name, f"invalid JSON arguments: {e}")
                continue

            # Execute each requested tool (but do not print their result to stdout)
            if function_tc.function.name == "Read":
                file_path = args.get("file_path")

                if not file_path:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, "no file_path argument provided for Read tool call")
                    continue

                if not os.path.isfile(file_path):
                    append_tool_error(messages, function_tc.id, function_tc.function.name, f"file_path {file_path} does not exist or is not a file")
                    continue
                
                try:
                    with open(file_path) as f:
                        #Add each tool call result to your messages array
                        result : ChatCompletionToolMessageParam = {
                            "role": 'tool',
                            "tool_call_id": function_tc.id,
                            "content": f.read()
                        }
                        messages.append(result)
                except Exception as e:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, f"error reading file_path {file_path}: {e}")
                    continue

            elif function_tc.function.name == "Write":
                file_path_str = args.get("file_path")
                content = args.get("content")

                if not file_path_str:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, "no file_path argument provided for Write tool call")
                    continue
                
                if not content:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, "no content argument provided for Write tool call")
                    continue
                
                try:
                    file_path = Path(file_path_str)
                    # Create directories if they don't exist
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(file_path, "w") as f:
                        f.write(content)
                    
                        result: ChatCompletionToolMessageParam = {
                            "role": "tool",
                            "tool_call_id": function_tc.id,
                            "content": f"Successfully wrote to {file_path}"
                        }
                        messages.append(result)
                except Exception as e:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, f"error writing to file_path {file_path_str}: {e}")
                    continue

            elif function_tc.function.name == "Bash":
                command = args.get("command")

                if not command:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, "no command argument provided for Bash tool call")
                    continue
                
                try:
                    #Execute the bash command and capture the output
                    resp = subprocess.run(command, shell=True, capture_output=True, text=True, check=True) #timeout=30 -->avoid hanging indefinitely
                    output = resp.stdout + resp.stderr

                    result: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": function_tc.id,
                        "content": output
                    }
                    messages.append(result)
                except subprocess.CalledProcessError as e:
                    append_tool_error(messages, function_tc.id, function_tc.function.name, f"Command failed with error: {e}")

    APP_CONSOLE.rule(f"[{ASSISTANT_HEADER_STYLE}]Assistant[/{ASSISTANT_HEADER_STYLE}]", style="dim")
    APP_CONSOLE.print(response_message.content)
    usage({
        "usage_type": "turn_summary",
        "prompt_tokens": turn_prompt_tokens,
        "completion_tokens": turn_completion_tokens,
        "total_tokens": turn_total_tokens,
    })
    return turn_prompt_tokens, turn_completion_tokens, turn_total_tokens

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can use tools to answer the user's question. At the end of your final respose, provide suggestions for follow up questions the user can ask to learn more about the topic."
    }
]
def main():
    if not API_KEY:
        raise RuntimeError("API_KEY is not set")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    while True:
        query = APP_CONSOLE.input(f"[{USER_PROMPT_STYLE}]You[/{USER_PROMPT_STYLE}] [dim]> [/dim]")
        if query.lower() == "exit":
            break
        _msg: ChatCompletionMessageParam = {"role": "user", "content": query}
        messages.append(_msg)
        turn_prompt_tokens, turn_completion_tokens, turn_total_tokens = my_agent(client, messages)
        session_prompt_tokens += turn_prompt_tokens
        session_completion_tokens += turn_completion_tokens
        session_total_tokens += turn_total_tokens
        usage({
            "usage_type": "session_summary",
            "prompt_tokens": session_prompt_tokens,
            "completion_tokens": session_completion_tokens,
            "total_tokens": session_total_tokens,
        })

if __name__ == "__main__":
    main()
