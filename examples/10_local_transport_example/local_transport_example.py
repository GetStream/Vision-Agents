"""
Local Transport Example

Demonstrates using LocalTransport for local audio/video I/O with vision agents.
This enables running agents using your microphone, speakers, and camera without
cloud-based edge infrastructure.

Usage:
    uv run python local_transport_example.py run

Requirements:
    - Working microphone and speakers
    - Optional: Camera for video input
    - API keys for Gemini, Deepgram, and ElevenLabs in .env file
"""

import logging
from typing import Any

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, gemini
from vision_agents.plugins.local import LocalEdge
from vision_agents.plugins.local.devices import (
    select_audio_input_device,
    select_audio_output_device,
    select_video_device,
)

logger = logging.getLogger(__name__)

load_dotenv()

INSTRUCTIONS = (
    "You're a helpful voice AI assistant running on the user's local machine. "
    "Keep responses short and conversational. Don't use special characters or "
    "formatting. Be friendly and helpful."
)


def setup_llm(model: str = "gemini-3.1-flash-lite-preview") -> gemini.LLM:
    llm = gemini.LLM(model)

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> dict[str, Any]:
        return await get_weather_by_location(location)

    return llm


async def create_agent() -> Agent:
    llm = setup_llm()

    if input_device is None:
        raise RuntimeError("No audio input device available")
    if output_device is None:
        raise RuntimeError("No audio output device available")

    logger.info(f"Using input: {input_device.name} ({input_device.sample_rate}Hz)")
    logger.info(f"Using output: {output_device.name} ({output_device.sample_rate}Hz)")
    if video_device:
        logger.info(f"Using video device: {video_device.name}")

    transport = LocalEdge(
        audio_input=input_device,
        audio_output=output_device,
        video_input=video_device,
    )

    agent = Agent(
        edge=transport,
        agent_user=User(name="Local AI Assistant", id="local-agent"),
        instructions=INSTRUCTIONS,
        processors=[],
        llm=llm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
    call = await agent.edge.create_call(call_id)
    async with agent.join(call=call, participant_wait_timeout=0):
        await agent.simple_response("Greet the user briefly")
        await agent.finish()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Local Transport Voice Agent")
    print("=" * 60)
    print("\nThis agent uses your local microphone, speakers, and optionally camera.")

    input_device = select_audio_input_device()
    output_device = select_audio_output_device()
    video_device = select_video_device()

    print("Speak into your microphone to interact with the AI.")
    if video_device:
        print("Camera is enabled for video input.")
    print("Press Ctrl+C to stop.\n")

    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
