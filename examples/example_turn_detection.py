#!/usr/bin/env python3
"""
Example: AI Agent with FAL Smart-Turn Detection

This example demonstrates how to use the FalTurnDetection class
with Stream's Agent to detect when speakers complete their turns.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4
import webbrowser
from urllib.parse import urlencode

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from getstream.models import UserRequest
from getstream.stream import Stream
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from getstream.plugins.deepgram.stt import DeepgramSTT

from agents import Agent
from turn_detection import FalTurnDetection

# Set up logging with more detail for turn detection
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

# Enable debug logging for turn detection
logging.getLogger("FalTurnDetection").setLevel(logging.DEBUG)


def create_user(client: Stream, id: str, name: str) -> None:
    """Create a user with a unique Stream ID."""
    user_request = UserRequest(id=id, name=name)
    client.upsert_users(user_request)


def open_browser(api_key: str, token: str, call_id: str) -> str:
    """Helper function to open browser with Stream call link."""
    base_url = f"{os.getenv('EXAMPLE_BASE_URL')}/join/"
    params = {"api_key": api_key, "token": token, "skip_lobby": "true"}

    url = f"{base_url}{call_id}?{urlencode(params)}"
    print(f"Opening browser to: {url}")

    try:
        webbrowser.open(url)
        print("Browser opened successfully!")
    except Exception as e:
        print(f"Failed to open browser: {e}")
        print(f"Please manually open this URL: {url}")

    return url


class MockLLM:
    """Simple LLM implementation for testing turn detection responses."""

    def __init__(self):
        self.response_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a simple response to test turn detection."""
        self.response_count += 1

        responses = [
            "I heard you speaking! Thanks for sharing.",
            "That's interesting - turn detection is working well.",
            "Got it! I can tell when you finish speaking.",
            "Thanks for testing the turn detection with me.",
            "Perfect! The smart-turn model detected your completion.",
        ]

        return responses[self.response_count % len(responses)]


async def main() -> None:
    """Create a video call and test FAL turn detection."""

    load_dotenv()

    # Check for required API keys
    if not os.getenv("FAL_KEY"):
        print("❌ FAL_KEY environment variable is required!")
        print("Please set your FAL API key: export FAL_KEY=your_fal_api_key")
        return

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("⚠️  ELEVENLABS_API_KEY not found - TTS will not work")

    if not os.getenv("DEEPGRAM_API_KEY"):
        print("⚠️  DEEPGRAM_API_KEY not found - STT will not work")

    client = Stream.from_env()

    human_id = f"user-{uuid4()}"
    create_user(client, human_id, "Human")

    token = client.create_token(human_id, expiration=3600)

    call_id = str(uuid4())
    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": human_id})

    logging.info("📞 Call ready: %s", call_id)

    open_browser(client.api_key, token, call_id)

    # Create services
    tts = ElevenLabsTTS() if os.getenv("ELEVENLABS_API_KEY") else None
    stt = DeepgramSTT() if os.getenv("DEEPGRAM_API_KEY") else None
    llm = MockLLM()

    # Create FAL turn detection with custom settings
    turn_detection = FalTurnDetection(
        buffer_duration=3.0,  # Process 3 seconds of audio at a time
        prediction_threshold=0.7,  # Higher threshold for more confident detections
        mini_pause_duration=0.5,
        max_pause_duration=2.0,
    )

    # Create agent with turn detection
    agent = Agent(
        llm=llm,
        stt=stt,
        tts=tts,
        turn_detection=turn_detection,
        name="Turn Detection Bot",
    )

    def create_bot_user(bot_id: str, bot_name: str):
        create_user(client, bot_id, bot_name)
        logging.info("Created bot user: %s (%s)", bot_id, bot_name)

    async def on_agent_connected(agent, connection):
        """Called when agent connects to the call."""
        await asyncio.sleep(2)  # Wait before greeting

        if agent.tts:
            await agent.say(
                "Hello! I'm testing turn detection with the smart-turn model. "
                "Try speaking to me and I'll respond when you finish talking."
            )
        else:
            logging.info("🤖 Turn Detection Bot connected (no TTS available)")

        # Set up turn detection event handlers for logging
        def on_turn_started(event_data):
            user_id = (
                event_data.custom.get("user_id", "unknown")
                if event_data.custom
                else "unknown"
            )
            logging.info(f"🎤 Turn STARTED by user {user_id}")

        def on_turn_ended(event_data):
            user_id = (
                event_data.custom.get("user_id", "unknown")
                if event_data.custom
                else "unknown"
            )
            prediction = (
                event_data.custom.get("prediction", "N/A")
                if event_data.custom
                else "N/A"
            )
            confidence = event_data.confidence or 0.0
            logging.info(
                f"🛑 Turn ENDED by user {user_id} (prediction: {prediction}, confidence: {confidence:.3f})"
            )

        # Register event handlers
        turn_detection.on("turn_started", on_turn_started)
        turn_detection.on("turn_ended", on_turn_ended)

    try:
        print("\n" + "=" * 60)
        print("🎯 FAL TURN DETECTION TEST")
        print("=" * 60)
        print("1. Join the call in your browser")
        print("2. Grant microphone permissions")
        print("3. Start speaking - the bot will respond when you finish")
        print("4. Check the logs to see turn detection events")
        print("5. Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        await agent.join(
            call,
            user_creation_callback=create_bot_user,
            on_connected_callback=on_agent_connected,
        )
    except asyncio.CancelledError:
        logging.info("Stopping agent...")
    finally:
        # Cleanup
        client.delete_users([human_id, agent.bot_id])
        logging.info("✅ Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
