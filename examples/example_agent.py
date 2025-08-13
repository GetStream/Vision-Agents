#!/usr/bin/env python3
"""
Example: AI Agent with Stream video call integration

This example shows how to use the new Agent class with the existing Stream call setup.
It demonstrates the requested syntax for creating agents with tools, pre-processors, and AI services.
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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


# Example tool implementation
class DotaAPI:
    """Example Dota API tool."""

    def __init__(self, game_id: str):
        self.game_id = game_id

    def __call__(self, *args, **kwargs):
        # Mock implementation - in reality this would call the Dota API
        return {
            "game_id": self.game_id,
            "player_stats": {
                "kills": 5,
                "deaths": 8,
                "assists": 12,
                "last_hits": 150,
                "gpm": 420,
            },
            "match_duration": "35:42",
        }


# Example pre-processor implementation
class Roboflow:
    """Example Roboflow pre-processor."""

    def process(self, data):
        # Mock implementation - in reality this would process computer vision data
        return {
            "original_input": data,
            "detected_objects": ["hero", "creep", "tower"],
            "confidence_scores": [0.95, 0.87, 0.92],
        }


def dota_api(game_id: str):
    """Factory function for creating Dota API tool."""
    return DotaAPI(game_id)


async def main() -> None:
    """Create a video call and let an AI agent join with tools and pre-processors."""

    load_dotenv()

    client = Stream.from_env()

    human_id = f"user-{uuid4()}"

    create_user(client, human_id, "Human")

    token = client.create_token(human_id, expiration=3600)

    call_id = str(uuid4())
    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": human_id})

    logging.info("📞 Call ready: %s", call_id)

    open_browser(client.api_key, token, call_id)

    # Create TTS service
    tts = ElevenLabsTTS()  # API key picked from ELEVENLABS_API_KEY
    stt = DeepgramSTT()

    # Create agent with the requested syntax
    agent = Agent(
        instructions="Roast my in-game performance in a funny but encouraging manner",
        tools=[dota_api("game123")],
        pre_processors=[Roboflow()],
        # model=None,  # Would be set to your AI model
        stt=stt,  # Would be set to your STT service
        tts=tts,
        # turn_detection=None,  # Would be set to your turn detection service
        name="Dota Roast Bot",
    )

    # Create user creation callback
    def create_bot_user(bot_id: str, bot_name: str):
        create_user(client, bot_id, bot_name)
        logging.info("Created bot user: %s (%s)", bot_id, bot_name)

    async def on_agent_connected(agent, connection):
        await asyncio.sleep(
            2
        )  # Wait two seconds after joining before saying the greeting
        await agent.say("Hello there, I am a dota 2 bot")

    try:
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
        logging.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
