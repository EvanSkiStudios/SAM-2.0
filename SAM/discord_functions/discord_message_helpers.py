import os
from collections import deque

import discord
from utility_scripts.system_logging import setup_logger
from dotenv import load_dotenv
from types import SimpleNamespace


# configure logging
logger = setup_logger(__name__)


def ns(d: dict) -> SimpleNamespace:
    """Convert dict into a dot-accessible namespace (recursively)."""
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})


# Load Env
load_dotenv()

config_dict = {
    "THREADS_ALLOW": {
        "GMCD_CHANNEL_ID": os.getenv("GMCD_CHANNEL_ID"),
        "TEST_THREAD_ID": os.getenv("TEST_THREAD_ID")
    },
    "THREADS_DENY": {
        "DISCUSSION": os.getenv("GMCD_NOT_ALLOWED_THREAD_D"),
        "NO_CONTEXT": os.getenv("GMCD_NOT_ALLOWED_THREAD_NC"),
        "DANEEL": os.getenv("GMCD_DANEEL_STINKY")
    },
    "BOTS": {
        "SCUNGEONMASTER": os.getenv("BOT_ID_SCUNGE")
    }
}
CONFIG = ns(config_dict)
channels_blacklist = [int(v) for v in CONFIG.THREADS_DENY.__dict__.values()]
bots_blacklist = [int(b) for b in CONFIG.BOTS.__dict__.values()]
channels_whitelist = [int(t) for t in CONFIG.THREADS_ALLOW.__dict__.values()]


def should_ignore_message(client, message):
    if message.author in bots_blacklist:
        return True
    if message.channel.id not in channels_whitelist:
        return True
    if message.type == discord.MessageType.chat_input_command:
        return True
    if message.mention_everyone:
        return True
    if message.author == client.user:
        return True
    return False


# used for conversations
current_session_chat_cache = deque(maxlen=40)
current_turn_number = 0


def session_chat_cache():
    global current_session_chat_cache
    return current_session_chat_cache


def clear_chat_cache():
    global current_session_chat_cache, current_turn_number
    current_session_chat_cache.clear()
    current_turn_number = 0


def next_turn():
    global current_turn_number
    current_turn_number += 1
    return current_turn_number


async def message_history_cache(client, message):
    global current_session_chat_cache

    def build_user_prompt(author, nick, content, reply_to=None):
        """Helper to format user messages (with optional reply)."""
        if reply_to == client.user.name:
            reply_to = None

        turn = next_turn()

        if reply_to:
            base = f'[turn: {turn}] {author} ({nick}): (Replying to: {reply_to}) "{content}"'
        else:
            base = f'[turn: {turn}] {author} ({nick}): "{content}"'

        return {
            "role": "user",
            "content": base
        }

    def build_assistant_prompt(content=""):
        """Helper to format assistant messages."""
        turn = next_turn()
        return {
            "role": "assistant",
            "content": f'[turn: {turn}] {content}'
        }

    async def process_message(msg):
        """Process a single Discord message into prompts."""
        # Skip irrelevant messages
        if msg.type in {discord.MessageType.chat_input_command, discord.MessageType.thread_created}:
            return []
        if msg.author in bots_blacklist:
            return []

        author_name = msg.author.name
        author_nick = msg.author.display_name
        content = msg.clean_content

        # Assistant (bot) message
        if msg.author.id == client.user.id:
            return [build_assistant_prompt(content)]

        # Reply message
        if msg.type == discord.MessageType.reply and msg.reference:
            try:
                referenced = await msg.channel.fetch_message(msg.reference.message_id)
            except discord.NotFound:
                referenced = None  # message was deleted
            except discord.Forbidden:
                referenced = None  # missing permissions
            except discord.HTTPException:
                referenced = None  # network or other fetch error

            if referenced is None:
                return []

            prompts = [
                build_user_prompt(author_name, author_nick, content, referenced.author.name)
            ]
            return prompts

        # Regular user message
        prompts = [build_user_prompt(author_name, author_nick, content)]
        return prompts

    # First-time cache build
    if not current_session_chat_cache:
        logger.debug("Session Cache Not Found -- Creating")
        channel = client.get_channel(message.channel.id)
        history_prompts = []
        async for past_message in channel.history(limit=20):
            history_prompts.extend(await process_message(past_message))

        current_session_chat_cache.extend(reversed(history_prompts))
        logger.debug("Session Cache Created")
        return

    # Incremental update (no assistant prompt for user messages here)
    new_prompts = await process_message(message)
    current_session_chat_cache.extend(new_prompts)

