"""
Voice AI with Gemma 4 - Local Tool-Calling Agent

A real-time voice assistant powered by Gemma 4 running entirely on your hardware.
Demonstrates how to build a voice AI agent with tool calling using local models:

- Gemma 4 E2B for text generation and tool calling (~5GB)
- Deepgram for speech-to-text and text-to-speech
- GetStream for real-time communication

The user speaks naturally and the agent responds with voice, calling tools
to look up live information (weather, web searches, unit conversions, etc).

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
- GPU with ~6GB VRAM recommended (or Apple Silicon with 16GB+ unified memory)

First run will download Gemma 4 E2B (~5GB).

Note: Gemma 4 also supports vision (VLM) via TransformersVLM for camera-based
use-cases, but the multimodal variant (E4B) requires ~16GB VRAM / 32GB unified
memory. See TransformersVLM in the huggingface plugin for details.
"""

import asyncio
import logging
import urllib.parse

import aiohttp
from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, huggingface

logger = logging.getLogger(__name__)

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful voice assistant running on a local Gemma 4 model. "
    "You can search the web and check the weather using your tools. "
    "Always use a tool when the user asks a factual question you're unsure about. "
    "Speak naturally, as if having a conversation. No lists or formatting. "
    "Never use emojis or special characters. Keep responses under 50 words."
)

async def get_weather(location: str) -> str:
    """Get current weather for a location using wttr.in."""
    logger.info(f"  [tool] get_weather({location!r})")
    encoded = urllib.parse.quote_plus(location)
    url = f"https://wttr.in/{encoded}?format=%C+%t+%h+%w"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                return await resp.text()
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.error(f"Weather lookup failed: {e}")
        return f"Weather lookup failed: {e}"


async def create_agent(**kwargs) -> Agent:
    """Create a voice AI agent with Gemma 4 and tool calling."""
    llm = huggingface.TransformersLLM(
        model="google/gemma-4-E2B-it",
    )

    llm.register_function(
        "get_weather",
        description=(
            "Get current weather for a location. "
            "Pass a city name like 'San Francisco' or 'London'."
        ),
    )(get_weather)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Voice Assistant", id="agent"),
        instructions=SYSTEM_PROMPT,
        llm=llm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Voice Assistant...")

    async with agent.join(call):
        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="Greet the user briefly. Tell them you can help answer questions, search the web, and check the weather.",
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
