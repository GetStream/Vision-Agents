"""
Baseten VLM Example

Demonstrates using a vision-language model hosted on Baseten with the
HuggingFace plugin's base_url parameter.

Creates an agent that uses:
- HuggingFace VLM pointed at a Baseten endpoint for vision + text
- Deepgram for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- BASETEN_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
"""

import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, huggingface

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with a Baseten-hosted VLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Baseten VLM Agent", id="agent"),
        instructions=(
            "You are a vision assistant that can see the user's video feed. "
            "Describe what you see concisely. Respond in one or two sentences. "
            "Never use lists, markdown or special formatting."
        ),
        llm=huggingface.VLM(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            base_url=os.environ["BASETEN_BASE_URL"],
            api_key=os.environ["BASETEN_API_KEY"],
            fps=1,
            frame_buffer_seconds=3,
        ),
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Baseten VLM Agent...")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
