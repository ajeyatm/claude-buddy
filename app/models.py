from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

class FatalAgentError(Exception):
    """Raised when session state may be corrupted and must be reset."""

class RecoverableAgentError(Exception):
    """Raised for transient issues where the session can safely continue."""

def validate_and_append_message(messages: list["ChatCompletionMessageParam"], message: dict[str, object]) -> None:
    """Validate message shape before adding it to conversation history."""
    if not isinstance(message, dict):
        raise FatalAgentError(f"MESSAGE_VALIDATION_ERROR: expected dict, got {type(message)}")

    role = message.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        raise FatalAgentError(f"MESSAGE_VALIDATION_ERROR: invalid role {role}")

    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                raise FatalAgentError("MESSAGE_VALIDATION_ERROR: assistant tool_calls must be a non-empty list when present")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call entry must be a dict")
                if not all(k in tc for k in ("id", "type", "function")):
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call missing id/type/function")
                if not isinstance(tc["id"], str) or not tc["id"].strip():
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call id must be a non-empty string")
                if tc["type"] != "function":
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call type must be 'function'")
                function = tc["function"]
                if not isinstance(function, dict):
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call function must be a dict")
                if not all(k in function for k in ("name", "arguments")):
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call function missing name/arguments")
                if not isinstance(function["name"], str) or not function["name"].strip():
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call function.name must be a non-empty string")
                if not isinstance(function["arguments"], str):
                    raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool_call function.arguments must be a string")
        else:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise FatalAgentError("MESSAGE_VALIDATION_ERROR: assistant content must be a non-empty string when no tool_calls")

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        content = message.get("content")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool message requires non-empty tool_call_id")
        if not isinstance(content, str) or not content.strip():
            raise FatalAgentError("MESSAGE_VALIDATION_ERROR: tool message content must be a non-empty string")

    messages.append(cast("ChatCompletionMessageParam", message))
