import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[
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
                    "name": "ListFiles",
                    "description": "List the files in a directory. The input should be a valid directory path. The output will be a list of file names in the directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory_path": {
                                "type": "string",
                                "description": "The path to the directory to list files from."
                            }
                        },
                        "required": ["directory_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Delete",
                    "description": "Delete a file. The input should be a valid file path. The output will be a success message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to delete."
                            }
                        },
                        "required": ["file_path"]
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
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")
    
    top_choice = chat.choices[0]
    if not top_choice.message:
        raise RuntimeError("no message content in top choice")
    
    
    content = top_choice.message.content
    tool_calls = top_choice.message.tool_calls

    if not tool_calls and content:
        print(content)

    # if not tool_calls or len(tool_calls) == 0:
    #     raise RuntimeError("no tool calls in message")
    
    for tc in tool_calls or []:
        if not tc.type == "function":
            raise RuntimeError("tool call is not a function call")
        
        
        if tc.function.name == "Read":
            args = json.loads(tc.function.arguments)
            file_path = args.get("file_path")

            if not file_path:
                raise RuntimeError("no file_path argument in Read function call")

            if not os.path.isfile(file_path):
                raise RuntimeError(f"file_path {file_path} does not exist or is not a file")
            
            with open(file_path) as f:
                print(f.read())

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)



if __name__ == "__main__":
    main()
