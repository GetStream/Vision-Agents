#!/usr/bin/env python3
"""
Example: Real-time Call Transcription with Wizper STT

This example demonstrates how to:
1. Create an Agent with STT capabilities for translation
2. Join a Stream video call
3. Transcribe audio in real-time using Wizper STT
4. Open a browser link for users to join the call

Usage:
    python main.py

Requirements:
    - Create a .env file with your Stream and FAL.ai credentials (see env.example)
    - Install dependencies: pip install -e .
"""

import asyncio
import logging
import os
import time
import uuid
import webbrowser
from urllib.parse import urlencode

from dotenv import load_dotenv

from getstream.models import UserRequest
from getstream.stream import Stream
from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.edge.types import User
from vision_agents.plugins import wizper, silero, getstream, openai
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent
from vision_agents.core.vad.events import VADAudioEvent, VADErrorEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_user(client: Stream, id: str, name: str) -> None:
    """
    Create a user with a unique Stream ID.

    Args:
        client: Stream client instance
        id: Unique user ID
        name: Display name for the user
    """
    user_request = UserRequest(id=id, name=name)
    client.upsert_users(user_request)


def open_browser(api_key: str, token: str, call_id: str) -> str:
    """
    Helper function to open browser with Stream call link.

    Args:
        api_key: Stream API key
        token: JWT token for the user
        call_id: ID of the call

    Returns:
        The URL that was opened
    """
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


load_dotenv()

# Global client
client = Stream.from_env()


async def create_agent(**kwargs) -> Agent:
    """Main example function."""
    print("🎙️  Stream + Fal Real-time Transcription Example")
    print("=" * 55)
    print("\n🤖 Starting transcription bot...")
    print(
        "The bot will join the call and transcribe all audio it receives, optionally translating it to French."
    )
    print("Join the call in your browser and speak to see transcriptions appear here!")
    print("\nPress Ctrl+C to stop the transcription bot.\n")

    # Create agent with STT and VAD for transcription
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Transcription Bot", id="transcription-bot"),
        instructions="I transcribe speech and translate it to French.",
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=wizper.STT(target_language="fr"),
        vad=silero.VAD(),
    )

    # Subscribe to VAD events for speech detection
    @agent.subscribe
    async def handle_speech_detected(event: VADAudioEvent):
        user_info = "unknown"
        if event.participant:
            user = event.participant
            user_info = user.name if user.name else str(user)
        print(
            f"{time.time()} Speech detected from user: {user_info} duration {event.duration_ms:.2f}ms"
        )

    # Subscribe to transcript events
    @agent.subscribe
    async def handle_transcript(event: STTTranscriptEvent):
        timestamp = time.strftime("%H:%M:%S")
        user_info = "unknown"
        if event.participant:
            user = event.participant
            user_info = user.name if user.name else str(user)

        print(f"[{timestamp}] {user_info}: {event.text}")
        if event.confidence:
            print(f"    └─ confidence: {event.confidence:.2%}")
        if event.processing_time_ms:
            print(f"    └─ processing time: {event.processing_time_ms:.1f}ms")

    # Subscribe to STT error events
    @agent.subscribe
    async def handle_stt_error(event: STTErrorEvent):
        print(f"\n❌ STT Error: {event.error_message}")
        if event.context:
            print(f"    └─ context: {event.context}")

    # Subscribe to VAD error events
    @agent.subscribe
    async def handle_vad_error(event: VADErrorEvent):
        print(f"\n❌ VAD Error: {event.error_message}")
        if event.context:
            print(f"    └─ context: {event.context}")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # Create a demo user for browser testing
    user_id = f"user-{uuid.uuid4()}"
    create_user(client, user_id, "My User")
    logging.info("👤 Created user: %s", user_id)

    user_token = client.create_token(user_id, expiration=3600)
    logging.info("🔑 Created token for user: %s", user_id)

    # Open browser for users to join with the user token
    open_browser(client.api_key, user_token, call_id)

    try:
        # ensure the agent user is created
        await agent.create_user()
        # Create a call
        call = await agent.create_call(call_type, call_id)

        # Join call and start transcription
        with await agent.join(call):
            print("🎧 Listening for audio... (Press Ctrl+C to stop)")
            await agent.finish()
    except asyncio.CancelledError:
        print("\n⏹️  Stopping transcription bot...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.delete_users([user_id])
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
