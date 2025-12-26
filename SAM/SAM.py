import asyncio
import os
import sys

from dotenv import load_dotenv
from ollama import Client, chat

from sam_config import SAM_personality
from discord_functions.discord_message_helpers import session_chat_cache
from tools.vision.gemma_vision import vision_image_cleanup
from utility_scripts.system_logging import setup_logger
from utility_scripts.utility import split_response

from pathlib import Path

# configure logging
logger = setup_logger(__name__)

load_dotenv()
os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API")

# model settings for easy swapping
sam_model_name = 'SAM'
sam_ollama_model = 'huihui_ai/deepseek-r1-abliterated'
sam_vision_model = 'gemma3'


def sam_create():
    try:
        client = Client()
        response = client.create(
            model=sam_model_name,
            from_=sam_ollama_model,
            system=SAM_personality,
            stream=False,
        )
        # print(f"# Client: {response.status}")
        logger.info(f"# Client: {response.status}")
        return

    except ConnectionError as e:
        logger.error('Ollama is not running!')
        sys.exit(1)  # Exit program with error code 1

    except Exception as e:
        # Catches any other unexpected errors
        logger.error("❌ An unexpected error occurred:", e)
        sys.exit(1)


# === Main Entry Point ===
async def sam_message(message_author_name, message_author_nickname, message_content, image_file=None, message_attachments=None):
    if image_file is None:
        llm_response = await sam_converse(message_author_name, message_author_nickname, message_content)
    else:
        llm_response = await sam_converse_image(message_author_name, message_author_nickname, message_content, image_file ,message_attachments)

    cleaned = llm_response.replace("'", "\\'")
    return split_response(cleaned)


chat_history_system_prompt = f"""
Input:
- You will receive ONE message in the format: [turn: (turn number)] Username (nickname): content
- The turn number is to help keep the order of the messages.

Input Example:
[turn: 7] Bob (KingBobby): How are you feeling today?

Behavior:
- Reply to the content of the message. Use the username only if it improves clarity.
- Do not invent server history, quotes, or prior messages you cannot see.
- Do not impersonate other users.

Output:
- Respond with ONLY the message content.
- Do NOT include turn numbers or your username.
- If any of the above appear, the response is invalid.

Output Example:
How are you feeling today?

- If your draft response contains any input formatting, rewrite it to remove all input style formatting before returning it.
"""


async def sam_converse(user_name, user_nickname, user_input):
    current_session_chat_cache = session_chat_cache()

    chat_log = []
    for message in current_session_chat_cache:
        chat_log.append(message)

    # Get everything except the last entry
    chat_log_all_but_last = chat_log[:-1]
    turn_number = len(chat_log_all_but_last) + 1

    current_prompt = [{"role": "user", "content": f'[turn: {turn_number}] {user_name} ({user_nickname}): "{user_input}"'}]

    full_prompt = (
        [
            {"role": "system", "content": SAM_personality},
            {"role": "system", "content": chat_history_system_prompt}
        ] +
        chat_log_all_but_last +
        current_prompt
    )

    response = await asyncio.to_thread(
        chat,
        model=sam_model_name,
        messages=full_prompt,
        options={
            "num_ctx": 8192,
            'temperature': 0.5
        }
    )

    # return the message to main script
    return response.message.content


async def sam_converse_image(user_name, user_nickname, user_input, image_file, message_attachments):
    # Go one directory up
    parent_dir = Path(__file__).resolve().parent
    path = parent_dir / 'tools/vision/images_temp' / image_file

    attachments = message_attachments[0]["attachments"]

    # print(attachments)
    # print(f'Analyzing image ({image_file_name})...')
    logger.debug(f"Attachments: {attachments}")
    logger.info(f'Analyzing image ({image_file})...')

    response = await asyncio.to_thread(
        chat,
        model=sam_vision_model,
        messages=(
            [
                {"role": "system", "content": SAM_personality},
                {"role": "user", "content": user_input, 'images': [path]}
            ]
        ),
        options={
            "num_ctx": 8192
        }
    )

    # clean up image
    vision_image_cleanup(image_file)

    # return the message to main script
    return response.message.content

