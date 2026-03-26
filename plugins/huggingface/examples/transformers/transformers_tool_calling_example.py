"""
Transformers Tool Calling Example

Demonstrates tool calling with local HuggingFace models. Both the LLM and VLM
can register functions, detect when the model wants to invoke them, and feed
the results back for a follow-up response.

This example runs standalone without any edge/TTS/STT infrastructure.

Usage:
    python transformers_tool_calling_example.py

Requirements:
    - torch (with MPS, CUDA, or CPU)
    - transformers
    - First run downloads the model (~3 GB for Qwen2.5-1.5B-Instruct)
"""

import asyncio
import logging

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.huggingface import TransformersLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"Loading model: {model_id}")

    llm = TransformersLLM(
        model=model_id,
        max_new_tokens=200,
    )

    # Load the model
    resources = await llm.on_warmup()
    llm.on_warmed_up(resources)

    # Set up conversation
    conversation = InMemoryConversation(
        system_prompt="",
        messages=[],
    )
    llm.set_conversation(conversation)
    llm.set_instructions(
        "You are a helpful assistant with access to tools. "
        "When a user asks about the weather, use the get_weather tool. "
        "When a user asks about the time, use the get_time tool. "
        "After receiving tool results, summarize the answer for the user."
    )

    # Register tools
    @llm.register_function("get_weather", description="Get current weather for a city")
    async def get_weather(city: str) -> str:
        logger.info(f"  [tool] get_weather called with city={city}")
        return f"Sunny, 22°C in {city}"

    @llm.register_function("get_time", description="Get current time in a timezone")
    async def get_time(timezone: str) -> str:
        logger.info(f"  [tool] get_time called with timezone={timezone}")
        return f"14:30 in {timezone}"

    # Listen for events
    @llm.events.subscribe
    async def on_completed(event: LLMResponseCompletedEvent) -> None:
        logger.info(f"  [event] Response completed: {event.text[:80]}...")

    # Run a conversation that should trigger tool calling
    prompts = [
        "What's the weather like in Paris?",
        "And what time is it in Tokyo?",
    ]

    for prompt in prompts:
        logger.info(f"\nUser: {prompt}")
        response = await llm.simple_response(text=prompt)
        await llm.events.wait(1)
        logger.info(f"Assistant: {response.text}")

    llm.unload()
    logger.info("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
