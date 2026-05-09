import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import subprocess

from openai import OpenAI
from dotenv import load_dotenv

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()


    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    #Initialize the conversation
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": args.p}]

    #agentic loop
    while True:

        chat = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SPECS
        )


        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")
        
        response_message = chat.choices[0].message

        if not response_message:
            raise RuntimeError("no message content in top choice")
        
        #Record the assistant's response --> START
        '''
        Whatever message the model returns, add it to your messages array. 
        If the model wants to use a tool, the response will contain a tool_calls array
        '''
        
        message_dict = response_message.model_dump(exclude_unset=True)

        #POSSIBLE ISSUE AND WORK-AROUND: we should be able to just do messages.append(response_message.model_dump()) here, but if for some reason the tool_calls field is not being included in the model_dump output, even though it is present in the response_message object the we'll manually construct the message dict to include the tool_calls field.
        
        # message_dict = {
        #     "role": response_message.role,
        #     "content": response_message.content,
        # }

        # if hasattr(response_message, "tool_calls") and response_message.tool_calls:
        #     message_dict["tool_calls"] = [tc.model_dump() for tc in response_message.tool_calls]
        #     # message_dict["tool_calls"] = [
        #     #     {
        #     #         "id": tc.id,
        #     #         "type": tc.type,
        #     #         "function": {
        #     #             "name": tc.function.name,
        #     #             "arguments": tc.function.arguments,
        #     #         },
        #     #     }
        #     #     for tc in response_message.tool_calls
        #     # ]
        
        messages.append(message_dict)

        #Record the assistant's response --> END
        
        if not message_dict.get("tool_calls"):
            '''
             Continue the loop until the model responds without requesting any tools (when tool_calls is missing or empty).
             At this point, print the final message content to stdout and exit.
            '''
            break

        
        #Execute tool calls if there are any
        for tc in response_message.tool_calls or []:
            if not tc.type == "function":
                raise RuntimeError("tool call is not a function call")
            
            
            # Execute each requested tool (but do not print their result to stdout)
            if tc.function.name == "Read":
                args = json.loads(tc.function.arguments)
                file_path = args.get("file_path")

                if not file_path:
                    raise RuntimeError("no file_path argument in Read function call")

                if not os.path.isfile(file_path):
                    raise RuntimeError(f"file_path {file_path} does not exist or is not a file")
                
                with open(file_path) as f:
                    #Add each tool call result to your messages array
                    result : ChatCompletionToolMessageParam = {
                        "role": 'tool',
                        "tool_call_id": tc.id,
                        "content": f.read()
                    }
                    messages.append(result)

            elif tc.function.name == "Write":
                args = json.loads(tc.function.arguments)
                file_path_str = args.get("file_path")
                content = args.get("content")

                if not file_path_str:
                    raise RuntimeError("no file_path argument in Write function call")
                
                if not content:
                    raise RuntimeError("no content argument in Write function call")
                
                file_path = Path(file_path_str)
                # Create directories if they don't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, "w") as f:
                    f.write(content)
                
                    result: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Successfully wrote to {file_path}"
                    }
                    messages.append(result)

            elif tc.function.name == "Bash":
                args = json.loads(tc.function.arguments)
                command = args.get("command")

                if not command:
                    raise RuntimeError("no command argument in Bash function call")
                
                try:
                    #Execute the bash command and capture the output
                    resp = subprocess.run(command, shell=True, capture_output=True, text=True, check=True) #timeout=30 -->avoid hanging indefinitely
                    output = resp.stdout + resp.stderr

                    result: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output
                    }
                    messages.append(result)
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e}")
                

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    print(response_message.content)

if __name__ == "__main__":
    main()
