#!/usr/bin/env python3
"""
Example: Real-time Call Transcription with Deepgram STT

This example demonstrates how to:
1. Join a Stream video call
2. Transcribe audio in real-time using Deepgram
3. Open a browser link for users to join the call

Usage:
    python main.py

Requirements:
    - Create a .env file with your Stream and Deepgram credentials (see env.example)
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

from getstream.models import CallRequest, UserRequest
from getstream.stream import Stream
from getstream.video import rtc
from getstream.video.rtc.track_util import PcmData
from getstream.plugins.deepgram.stt import DeepgramSTT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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


async def main():
    """Main example function."""
    print("🎙️  Stream + Deepgram Real-time Transcription Example")
    print("=" * 55)

    # Load environment variables
    load_dotenv()

    # Initialize Stream client from ENV
    client = Stream.from_env()

    # Create a unique call ID for this session
    call_id = str(uuid.uuid4())
    print(f"📞 Call ID: {call_id}")

    user_id = f"user-{uuid.uuid4()}"
    create_user(client, user_id, "My User")
    logging.info("👤 Created user: %s", user_id)

    user_token = client.create_token(user_id, expiration=3600)
    logging.info("🔑 Created token for user: %s", user_id)

    bot_user_id = f"transcription-bot-{uuid.uuid4()}"
    create_user(client, bot_user_id, "Transcription Bot")
    logging.info("🤖 Created bot user: %s", bot_user_id)

    # Create the call
    call = client.video.call("default", call_id)
    call.get_or_create(data=CallRequest(created_by_id=bot_user_id))
    print(f"📞 Call created: {call_id}")

    # Open browser for users to join with the user token
    open_browser(client.api_key, user_token, call_id)

    print("\n🤖 Starting transcription bot...")
    print("The bot will join the call and transcribe all audio it receives.")
    print("Join the call in your browser and speak to see transcriptions appear here!")
    print("\nPress Ctrl+C to stop the transcription bot.\n")

    # Initialize Deepgram STT (api_key comes from .env)
    stt = DeepgramSTT()

    try:
        async with await rtc.join(call, bot_user_id) as connection:
            print(f"✅ Bot joined call: {call_id}")

            # Set up transcription handlers
            @connection.on("audio")
            async def on_audio(pcm: PcmData, user):
                # Process audio through Deepgram STT with user metadata
                user_metadata = {"user": user} if user else None
                await stt.process_audio(pcm, user_metadata)

            @stt.on("transcript")
            async def on_transcript(event):
                timestamp = time.strftime("%H:%M:%S")
                user_info = "unknown"
                if event.user_metadata and "user" in event.user_metadata:
                    user = event.user_metadata["user"]
                    user_info = user.name if hasattr(user, "name") else str(user)
                print(f"[{timestamp}] {user_info}: {event.text}")
                if hasattr(event, "confidence") and event.confidence:
                    print(f"    └─ confidence: {event.confidence:.2%}")
                if hasattr(event, "processing_time_ms") and event.processing_time_ms:
                    print(f"    └─ processing time: {event.processing_time_ms:.1f}ms")

            @stt.on("partial_transcript")
            async def on_partial_transcript(event):
                if event.text.strip():  # Only show non-empty partial transcripts
                    user_info = "unknown"
                    if event.user_metadata and "user" in event.user_metadata:
                        user = event.user_metadata["user"]
                        user_info = user.name if hasattr(user, "name") else str(user)
                    print(
                        f"    {user_info} (partial): {event.text}", end="\r"
                    )  # Overwrite line

            @stt.on("error")
            async def on_stt_error(event):
                print(f"\n❌ STT Error: {event.error_message}")
                if hasattr(event, "context") and event.context:
                    print(f"    └─ context: {event.context}")

            # Keep the connection alive and wait for audio
            print("🎧 Listening for audio... (Press Ctrl+C to stop)")
            await connection.wait()

    except asyncio.CancelledError:
        print("\n⏹️  Stopping transcription bot...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await stt.close()
        client.delete_users([user_id, bot_user_id])
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
