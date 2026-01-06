import asyncio
import re

from ollama import ChatResponse, chat

import sam_config
from tools.weather_search.weather_api import get_weather
from utility_scripts.utility import split_response
from utility_scripts.system_logging import setup_logger

# configure logging
logger = setup_logger(__name__)

tool_model = 'huihui_ai/llama3.2-abliterate'
chat_model = 'SAM-deepseek-r1'


def get_the_weather(city, state):
    return get_weather(city, state)


available_functions = {
    'get_the_weather': get_the_weather,
}

system_prompt = f"""
{sam_config.SAM_personality}

You will be given the results of a weather_search search.
Respond with the full results in detail.
Always try to provide a neutral and informative response.
"""


async def weather_search(message):
    messages = [
        {'role': 'user', 'content': message}
    ]
    response: ChatResponse = await asyncio.to_thread(
        chat,
        tool_model,
        messages=messages,
        # tools=[search_the_web, search_wikipedia],
        tools=[get_the_weather],
        options={'temperature': 0.2},  # Make responses less or more deterministic
        stream=False
    )

    if response.message.tool_calls:
        # There may be multiple tool calls in the response
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_to_call := available_functions.get(tool.function.name):
                debug_print = (
                    f'Calling function: {tool.function.name}' + '\n'
                    f'Arguments: {tool.function.arguments}'
                )
                logger.info(debug_print)

                output = function_to_call(**tool.function.arguments)
                logger.info(f'Function output: {output}')
            else:
                logger.error(f'Function {tool.function.name} not found')

    # Only needed to chat with the model using the tool call results
    if response.message.tool_calls:
        # Add the function response to messages for the model to use
        messages.append(response.message)
        messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})

        # Get final response from model with function outputs
        final_response = chat(chat_model, stream=False, messages=[{'role': 'system', 'content': system_prompt}] + messages)
        # print('Final response:', final_response.message.content)
    else:
        logger.info(f'No tool calls returned from model')
        final_response = chat(chat_model, stream=False, messages=[{'role': 'system', 'content': system_prompt}] + messages)

    output = final_response.message.content
    output = re.sub(r'\bEvanski_\b', 'Evanski', output, flags=re.IGNORECASE)

    logger.info(response)
    logger.info(final_response)
    debug_print = (f"""
    ===================================
    CONTENT:  {message}\n
    RESPONSE:  {output}
    ===================================
    """)
    logger.info(debug_print)

    cleaned = output.replace("'", "\\'")
    return {
        "content": split_response(cleaned),
        "message": final_response
    }


async def main():
    query = "whats the weather_search in north dekota?"
    response = await weather_search(query)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
