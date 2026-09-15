"""
Smallest AI Example

This example creates a voice AI agent using Smallest AI for both ends of the
voice pipeline:
- Smallest AI Pulse for speech-to-text
- Smallest AI Lightning for text-to-speech
- OpenAI for chat completions
- GetStream for real-time edge communication
- Smart Turn for turn detection (Pulse doesn't emit turn-boundary events)

Requirements:
- SMALLEST_API_KEY environment variable
- OPENAI_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, openai, smallest, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Smallest AI STT and TTS."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Smallest AI Agent", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        stt=smallest.STT(language="en"),
        tts=smallest.TTS(voice_id="magnus"),
        llm=openai.LLM(model="gpt-4o-mini"),
        turn_detection=smart_turn.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Smallest AI Agent...")

    async with agent.join(call):
        logger.info("Joined call")

        await asyncio.sleep(5)
        await agent.simple_response(text="Hello! How can I help you today?")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
