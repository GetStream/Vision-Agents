import logging
import os

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, gemini
from vision_agents.plugins.vibevoice.tts import VibeVoiceTTS

logger = logging.getLogger(__name__)

load_dotenv()

"""
VibeVoice Agent Example

This example demonstrates how to use VibeVoice TTS with the Vision Agents framework.
It uses:
- Deepgram for STT (Speech-to-Text)
- Gemini for LLM (Language Model)
- VibeVoice for TTS (Text-to-Speech)
- Stream for Edge Network (Audio/Video Transport)

Usage:
    uv run plugins/vibevoice/example/agent_example.py

Configuration in .env:
    GETSTREAM_API_KEY=...
    GETSTREAM_SECRET_KEY=...
    DEEPGRAM_API_KEY=...
    GEMINI_API_KEY=...
    # Optional:
    VIBEVOICE_DEVICE=cpu  # or cuda, mps
"""


async def create_agent(**kwargs) -> Agent:
    # Initialize plugins
    llm = gemini.LLM("gemini-2.5-flash-lite")

    # You can specify a custom voice path if you have a .pt file
    # voice_path = "/path/to/custom_voice.pt"
    # tts = VibeVoiceTTS(voice_path=voice_path)

    # Or use a built-in preset (will download automatically)
    tts = VibeVoiceTTS(
        voice="en-Carter_man", device=os.environ.get("VIBEVOICE_DEVICE", "cpu")
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="VibeVoice Agent", id="agent_vibevoice"),
        instructions="You are a helpful AI assistant with a natural sounding voice. Keep responses concise.",
        llm=llm,
        tts=tts,
        stt=deepgram.STT(eager_turn_detection=True),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    logger.info(f"Joining call: {call_type}:{call_id}")
    with await agent.join(call):
        # Trigger an initial greeting
        await agent.simple_response("Say hello and introduce yourself briefly.")

        # Keep the agent running until the call ends
        await agent.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
