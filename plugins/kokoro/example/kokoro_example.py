"""
Kokoro TTS Example

This example demonstrates Kokoro TTS integration with Vision Agents.

Kokoro is an open-source, offline TTS model that runs locally without requiring API keys.

This example creates an agent that uses:
- Kokoro for text-to-speech (TTS) - runs locally
- OpenAI for LLM
- GetStream for edge/real-time communication

Requirements:
- espeak-ng installed (brew install espeak-ng on macOS)
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- OPENAI_API_KEY environment variable
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, kokoro, openai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Kokoro TTS."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="TTS Bot", id="tts-bot"),
        instructions="I'm a TTS bot that greets users when they join.",
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=kokoro.TTS(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
