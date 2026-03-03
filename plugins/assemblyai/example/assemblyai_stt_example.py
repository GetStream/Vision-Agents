"""
AssemblyAI STT Example

This example demonstrates AssemblyAI streaming STT integration with Vision Agents.

Uses:
- AssemblyAI Universal-3 Pro for speech-to-text (STT) via async WebSocket
- GetStream for edge/real-time communication
- Gemini for LLM

Requirements:
- ASSEMBLYAI_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- GOOGLE_API_KEY environment variable (for Gemini)
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import assemblyai, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with AssemblyAI STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AssemblyAI Agent", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        stt=assemblyai.STT(),
        llm=gemini.LLM(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting AssemblyAI STT Agent...")

    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Hello! How can I help you today?")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
