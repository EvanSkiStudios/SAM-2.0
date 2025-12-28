import asyncio
import os
import sys

from dotenv import load_dotenv
from ollama import Client, chat

from sam_config import SAM_personality, chat_history_system_prompt
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
sam_model_name = 'SAM-deepseek-r1'
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
async def sam_message(message_author_name, message_author_nickname, message_content, image_file=None,
                      message_attachments=None):
    llm_response = await sam_converse(
        message_author_name,
        message_author_nickname,
        message_content,
        image_file=image_file,
        message_attachments=message_attachments
    )
    cleaned = llm_response.replace("'", "\\'")
    return split_response(cleaned)


async def sam_converse(user_name, user_nickname, user_input, image_file=None, message_attachments=None):
    current_session_chat_cache = session_chat_cache()
    chat_log = list(current_session_chat_cache)

    # Keep assistant vs user turns distinct (LLMs are trained on role-tagged conversations)
    def build_role_message(entry: str):
        role = "assistant" if entry.startswith("SAM:") else "user"
        return {"role": role, "content": entry}

    full_prompt = [
        {"role": "system", "content": SAM_personality},
        {"role": "system", "content": chat_history_system_prompt},
        *[build_role_message(entry) for entry in chat_log]  # Store an array of role-tagged turns
    ]

    model_to_use = sam_model_name

    if image_file:
        model_to_use = sam_vision_model
        # Go one directory up
        parent_dir = Path(__file__).resolve().parent
        path = parent_dir / 'tools/vision/images_temp' / image_file

        full_prompt[-1]["images"] = [path]
        attachments = message_attachments[0]["attachments"]

        logger.debug(f"Attachments: {attachments}")
        logger.info(f'Analyzing image ({image_file})...')

    response = await asyncio.to_thread(
        chat,
        model=model_to_use,
        messages=full_prompt,
        options={
            "num_ctx": 8192,
            'temperature': 0.5,
            'think': True
        },
        stream=False
    )

    if image_file:
        vision_image_cleanup(image_file)

    # logger.info(response.message.thinking)

    # Log history + assistant's reply
    logger.info(
        "\n".join(f'{m["role"]}: {m["content"]}' for m in full_prompt))  # place each message item on its own line
    logger.info(response.message.content)

    # logger.info(full_prompt)

    # return response
    return response.message.content
