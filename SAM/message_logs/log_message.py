from pathlib import Path
from datetime import datetime

from discord_functions.discord_message_helpers import session_chat_cache
from utility_scripts.system_logging import setup_logger

# configure logging
logger = setup_logger(__name__)


def get_timestamp() -> str:
    return datetime.now().strftime("%m-%d__%H_%M_%S")


def build_role_message(entry: str) -> dict:
    if entry.startswith("SAM:"):
        return {
            "role": "assistant",
            "content": entry.removeprefix("SAM: ").lstrip(),
        }

    return {
        "role": "user",
        "content": entry,
    }


def format_details(summary: str, content: str) -> str:
    return (
        "<details>\n"
        f"   <summary>{summary}</summary>\n"
        f"{content}\n"
        "</details>\n"
    )


async def log_message(sent_message, thinking: str, user_message: dict) -> None:
    save_dir = Path(__file__).resolve().parent / "logs"
    save_dir.mkdir(exist_ok=True)

    path = save_dir / f"{get_timestamp()}__{sent_message.id}.md"

    chat_log = session_chat_cache()
    chat_history = [build_role_message(entry) for entry in chat_log]

    full_chat_history = "\n".join(
        f'{m["role"]}: {m["content"]}\n' for m in chat_history
    )

    output = (
        f"User Message ID:    {user_message['id']}  "
        "\n"
        f"User:    {user_message['name']}  "
        "\n"
        f"{format_details('Message Content', user_message['content'])}"
        "\n"
        f"Message ID:    {sent_message.id}  "
        f"{format_details('Thinking', thinking)}"
        "\n"
        f"{format_details('Response', sent_message.content)}"
        "\n"
        f"{format_details('Full History', full_chat_history)}"
    )

    path.write_text(output, encoding="utf-8")

    logger.info(f"Logged: {sent_message.id}")
