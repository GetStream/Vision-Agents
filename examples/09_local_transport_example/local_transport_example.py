"""
Local Transport Example

Demonstrates using LocalTransport for local audio/video I/O with vision agents.
This enables running agents using your microphone, speakers, and camera without
cloud-based edge infrastructure.

Usage:
    uv run python local_transport_example.py

Requirements:
    - Working microphone and speakers
    - Optional: Camera for video input
    - API keys for Gemini, Deepgram, and ElevenLabs in .env file
"""

import asyncio
import logging
import signal
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from vision_agents.core import Agent, User
from vision_agents.core.edge.local_transport import (
    LocalTransport,
    get_device_sample_rate,
    select_audio_devices,
    select_video_device,
)
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    video_device: Optional[str] = None,
) -> Agent:
    """Create an agent with LocalTransport for local audio/video I/O."""
    llm = gemini.LLM("gemini-2.5-flash-lite")

    # Get device sample rates to ensure proper configuration
    input_sample_rate = get_device_sample_rate(input_device, is_input=True)
    output_sample_rate = get_device_sample_rate(output_device, is_input=False)

    logger.info(f"Using input sample rate: {input_sample_rate}Hz")
    logger.info(f"Using output sample rate: {output_sample_rate}Hz")
    if video_device:
        logger.info(f"Using video device: {video_device}")

    transport = LocalTransport(
        sample_rate=output_sample_rate,  # Use output device's native rate
        input_device=input_device,
        output_device=output_device,
        video_device=video_device,
    )

    agent = Agent(
        edge=transport,
        agent_user=User(name="Local AI Assistant", id="local-agent"),
        instructions=(
            "You're a helpful voice AI assistant running on the user's local machine. "
            "Keep responses short and conversational. Don't use special characters or "
            "formatting. Be friendly and helpful."
        ),
        processors=[],
        llm=llm,
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Register a sample function
    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent


async def run_agent(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    video_device: Optional[str] = None,
):
    """Run the agent with local audio/video transport."""
    agent = await create_agent(input_device, output_device, video_device)

    # Handle shutdown gracefully
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        logger.info("Starting local transport agent...")
        logger.info("Speak into your microphone. Press Ctrl+C to stop.")

        # For LocalTransport, we don't need a real call object
        # The join method will start the microphone capture
        async with agent.join(call=None, participant_wait_timeout=0):
            # Send initial greeting
            await agent.simple_response("Greet the user briefly")

            # Wait until shutdown signal or call ends
            await shutdown_event.wait()

    except asyncio.CancelledError:
        logger.info("Agent task cancelled")
    finally:
        logger.info("Shutting down agent...")
        await agent.close()
        logger.info("Agent stopped")


def main():
    """Entry point for the local transport example."""
    print("\n" + "=" * 60)
    print("Local Transport Voice Agent")
    print("=" * 60)
    print("\nThis agent uses your local microphone, speakers, and optionally camera.")

    # Let user select audio devices
    input_device, output_device = select_audio_devices()

    # Let user select video device (optional)
    video_device = select_video_device()

    print("Speak into your microphone to interact with the AI.")
    if video_device:
        print("Camera is enabled for video input.")
    print("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(run_agent(input_device, output_device, video_device))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
