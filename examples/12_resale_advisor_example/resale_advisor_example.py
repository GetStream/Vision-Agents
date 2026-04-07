"""
Vision AI with Gemma 4 - Local VLM Agent (MLX)

A real-time vision + voice assistant powered by Gemma 4 E4B running on Apple
Silicon via MLX.  Demonstrates how to build a multimodal AI agent that can see
the user's video feed and respond with voice:

- Gemma 4 E4B (8-bit quantized) via mlx-vlm for vision-language inference
- Deepgram for speech-to-text and text-to-speech
- GetStream for real-time communication

The user speaks naturally and the agent responds with voice, describing what
it sees and answering questions about the video feed.

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
- Apple Silicon Mac with 16GB+ unified memory

First run will download the MLX model (~8GB).
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, huggingface

logger = logging.getLogger(__name__)

load_dotenv()

SYSTEM_PROMPT = (
    "You are a vision assistant running on a local Gemma 4 model. "
    "You can see the user's camera feed. Describe what you see concisely. "
    "Speak naturally, as if having a conversation. No lists or formatting. "
    "Never use emojis or special characters. Keep responses under 50 words."
)


async def create_agent(**kwargs) -> Agent:
    """Create a vision AI agent with Gemma 4 VLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Vision Assistant", id="agent"),
        instructions=SYSTEM_PROMPT,
        llm=huggingface.MlxVLM(
            model="mlx-community/gemma-4-e4b-it-8bit",
            max_new_tokens=150,
        ),
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Vision Assistant...")

    async with agent.join(call):
        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="Greet the user briefly. Tell them you can see their camera and can describe what you see.",
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
