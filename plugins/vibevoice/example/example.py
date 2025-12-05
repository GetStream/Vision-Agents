"""
VibeVoice TTS and STT Example

This example demonstrates VibeVoice TTS and Scribe v2 STT integration with Vision Agents.

This example creates an agent that uses:
- VibeVoice for text-to-speech (TTS)
- GetStream for edge/real-time communication
- Smart Turn for turn detection

Requirements:
- VIBEVOICE_URL environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import vibevoice, getstream, smart_turn, gemini, deepgram


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with VibeVoice TTS."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI", id="agent"),
        instructions="You're a friendly voice AI assistant. Keep your replies conversational",
        tts=vibevoice.TTS(),  # Uses VibeVoice for text-to-speech
        stt=deepgram.STT(),  # Uses Deepgram for speech-to-text
        llm=gemini.LLM("gemini-2.5-flash-lite"),
        turn_detection=smart_turn.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting VibeVoice Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        await agent.simple_response("tell me something interesting in a short sentence")
        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
