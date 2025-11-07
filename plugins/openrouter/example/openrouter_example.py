"""
OpenRouter LLM Example

This example demonstrates how to use the OpenRouter plugin with a Vision Agent.
OpenRouter provides access to multiple LLM providers through a unified API.

Set OPENROUTER_API_KEY environment variable before running.
"""

import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import openrouter, getstream, elevenlabs, deepgram, smart_turn


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with OpenRouter LLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="OpenRouter AI", id="agent"),
        instructions="Be helpful and friendly to the user",
        llm=openrouter.LLM(
            model="openai/gpt-4o",  # Can also use other models like anthropic/claude-3-opus
        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        turn_detection=smart_turn.TurnDetection(
            buffer_in_seconds=2.0, confidence_threshold=0.5
        )
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting OpenRouter Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")
        
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Hello! I'm powered by OpenRouter.")
        
        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))

