from rich.console import Console

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

def print_usage(usage_data: dict[str, int | str]) -> None:
    """Prints usage information to the console in a formatted way."""
    usage_type = usage_data.get("usage_type", "N/A")
    prompt_tokens = usage_data.get("prompt_tokens", 0)
    completion_tokens = usage_data.get("completion_tokens", 0)
    total_tokens = usage_data.get("total_tokens", 0)
    
    LOG_CONSOLE.print(f"[{USAGE_SEPARATOR_STYLE}]{REPEAT_CHAR * REPEAT_COUNT}[/{USAGE_SEPARATOR_STYLE}]")
    LOG_CONSOLE.print(
        f"[{USAGE_LABEL_STYLE}]usage_{usage_type}[/{USAGE_LABEL_STYLE}] "
        f"prompt=[{USAGE_PROMPT_STYLE}]{prompt_tokens}[/{USAGE_PROMPT_STYLE}] "
        f"completion=[{USAGE_COMPLETION_STYLE}]{completion_tokens}[/{USAGE_COMPLETION_STYLE}] "
        f"total=[{USAGE_TOTAL_STYLE}]{total_tokens}[/{USAGE_TOTAL_STYLE}]"
    )
    LOG_CONSOLE.print(f"[{USAGE_SEPARATOR_STYLE}]{REPEAT_CHAR * REPEAT_COUNT}[/{USAGE_SEPARATOR_STYLE}]")
