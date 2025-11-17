#!/usr/bin/env python3
"""
Example: Real-time Call Video Moderation with Deepgram STT

This example demonstrates how to:
1. Create an Agent with STT capabilities for video moderation
2. Join a Stream video call
3. Transcribe audio in real-time and moderate content
4. Open a browser link for users to join the call

Usage:
    python main.py

Requirements:
    - Create a .env file with your Stream and Deepgram credentials (see env.example)
    - Install dependencies: pip install -e .
"""

import argparse
import asyncio
import logging
import warnings
import os
import time
import uuid
import webbrowser
from urllib.parse import urlencode

from dotenv import load_dotenv

from getstream.models import UserRequest
from getstream.stream import Stream
from getstream.models import CheckResponse, ModerationPayload
from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.edge.types import User
from vision_agents.plugins import deepgram, getstream, openai
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Suppress dataclasses_json missing value RuntimeWarnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="dataclasses_json.core"
)


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


def moderate(client: Stream, text: str, user_name: str) -> CheckResponse:
    """Moderate a transcript using Stream Moderation.

    This helper is synchronous on purpose so it can be executed in a background
    thread with ``asyncio.to_thread`` from async code without blocking the event
    loop.
    """

    return client.moderation.check(
        config_key="custom:python-ai-test",  # your moderation config key
        entity_creator_id=user_name,
        entity_id=str(uuid.uuid4()),
        entity_type="transcript",
        moderation_payload=ModerationPayload(texts=[text]),
    ).data


load_dotenv()

# Global client for moderation
client = Stream.from_env()


async def create_agent(**kwargs) -> Agent:
    print("\n🤖 Starting moderation bot...")
    print("The bot will join the call and moderate all audio it receives.")
    print(
        "Join the call in your browser and speak to see moderation results appear here!"
    )
    print("\nPress Ctrl+C to stop the moderation bot.\n")

    # Create agent with STT for moderation
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Moderation Bot", id="moderation-bot"),
        instructions="I moderate audio content in real-time.",
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(interim_results=True),
    )

    # Subscribe to transcript events for moderation
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

        # Moderation check (executed in a background thread to avoid blocking)
        moderation = await asyncio.to_thread(moderate, client, event.text, user_info)
        print(
            f"    └─ moderation recommended action: {moderation.recommended_action} for transcript: {event.text}"
        )

    # Subscribe to STT error events
    @agent.subscribe
    async def handle_stt_error(event: STTErrorEvent):
        print(f"\n❌ STT Error: {event.error_message}")
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

        # Join call and start moderation
        with await agent.join(call):
            print("🎧 Listening for audio... (Press Ctrl+C to stop)")
            await agent.finish()
    except asyncio.CancelledError:
        print("\n⏹️  Stopping moderation bot...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.delete_users([user_id])
        print("🧹 Cleanup completed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream Real-time Audio Moderation Example"
    )
    parser.add_argument("--setup", action="store_true", help="Setup moderation config")
    return parser.parse_args()


def setup_moderation_config(client: Stream):
    try:
        # Create moderation config
        config = {
            "key": "custom:python-ai-test",
            "ai_text_config": {
                "rules": [
                    {"label": "INSULT", "action": "flag"},
                ],
            },
        }
        client.moderation.upsert_config(**config)
    except Exception as e:
        print(f"Could not create moderation config. Error: {e}")


if __name__ == "__main__":
    print("🎙️  Stream Real-time Video Moderation Example")
    print("=" * 55)

    args = parse_args()

    if args.setup:
        setup_moderation_config(client)
    else:
        cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
