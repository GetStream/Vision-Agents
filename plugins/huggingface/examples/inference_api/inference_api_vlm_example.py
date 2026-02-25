"""
HuggingFace Inference API VLM Example

Demonstrates HuggingFace Inference Providers integration with Vision Agents
using a vision-language model that can see the user's video feed.

Creates an agent that uses:
- HuggingFace VLM (via Inference Providers API) for vision + text
- Deepgram for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- HF_TOKEN environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, huggingface

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with a HuggingFace VLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="HuggingFace VLM Agent", id="agent"),
        instructions=(
            "You are a vision assistant that can see the user's video feed. "
            "Describe what you see concisely. Respond in one or two sentences. "
            "Never use lists, markdown or special formatting."
        ),
        llm=huggingface.VLM(
            model="Qwen/Qwen3-VL-32B-Instruct",
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

    logger.info("Starting HuggingFace VLM Agent...")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
