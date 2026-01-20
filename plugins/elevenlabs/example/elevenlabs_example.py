"""
ElevenLabs TTS and STT Example

This example demonstrates ElevenLabs TTS and Scribe v2 STT integration with Vision Agents.

This example creates an agent that uses:
- ElevenLabs for text-to-speech (TTS)
- ElevenLabs Scribe v2 for speech-to-text (STT)
- GetStream for edge/real-time communication
- Smart Turn for turn detection

Requirements:
- ELEVENLABS_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import elevenlabs, gemini, getstream, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with ElevenLabs TTS and STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI", id="agent"),
        instructions="You're a friendly voice AI assistant. Keep your replies conversational",
        tts=elevenlabs.TTS(),  # Uses ElevenLabs for text-to-speech
        stt=elevenlabs.STT(),  # Uses ElevenLabs Scribe v2 for speech-to-text
        llm=gemini.LLM("gemini-2.5-flash-lite"),
        turn_detection=smart_turn.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting ElevenLabs Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        await agent.simple_response("tell me something interesting in a short sentence")
        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
