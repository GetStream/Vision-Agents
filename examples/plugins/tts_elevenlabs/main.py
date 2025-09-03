#!/usr/bin/env python3
"""
Example: Text-to-Speech bot with ElevenLabs

This minimal example shows how to:
1. Spin up a Stream video call
2. Attach an ElevenLabs TTS bot that can speak into the call

Run it, join the call in your browser, and hear the bot greet you 🗣️

Usage::
    python main.py

The script looks for the following env vars (see `env.example`):
    STREAM_API_KEY / STREAM_API_SECRET
    ELEVENLABS_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

from dotenv import load_dotenv

from getstream.models import UserRequest
from getstream.plugins import ElevenLabsTTS
from getstream.stream import Stream
from getstream.video import rtc
from getstream.video.rtc import audio_track

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


async def main() -> None:
    """Create a video call and let a TTS bot greet participants."""

    load_dotenv()

    client = Stream.from_env()

    human_id = f"user-{uuid4()}"
    bot_id = f"tts-bot-{uuid4()}"

    create_user(client, human_id, "Human")
    create_user(client, bot_id, "TTS Bot")

    logging.info("Created users: %s (human) / %s (bot)", human_id, bot_id)

    token = client.create_token(human_id, expiration=3600)

    call_id = str(uuid4())
    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": bot_id})

    logging.info("📞 Call ready: %s", call_id)

    open_browser(client.api_key, token, call_id)

    track = audio_track.AudioStreamTrack(framerate=16000)
    tts = ElevenLabsTTS()  # API key picked from ELEVENLABS_API_KEY
    tts.set_output_track(track)

    greeting = "Hello there! I'm an ElevenLabs TTS bot speaking inside this call. As this is a minimal example, I'll stop speaking now."

    try:
        async with await rtc.join(call, bot_id) as connection:
            logging.info("🤖 Bot joined call: %s", call_id)

            # Greeting control to ensure we only greet once
            greeted = asyncio.Event()
            greeting_lock = asyncio.Lock()

            async def send_greeting_once():
                """Send greeting once, with concurrency protection."""
                async with greeting_lock:
                    if greeted.is_set():
                        return

                    greeted.set()  # Set immediately to prevent concurrent calls

                    # Wait for publisher connection to be ready
                    if connection.publisher_pc is not None:
                        await connection.publisher_pc.wait_for_connected()

                    try:
                        await tts.send(greeting)
                        logging.info("🤖 Sent greeting via TTS")
                    except Exception as e:
                        logging.error(f"Failed to send greeting: {e}")
                        greeted.clear()  # Reset so we can retry

            # Publish our audio track
            await connection.add_tracks(audio=track)
            logging.info("🤖 Bot ready to speak")

            # Listen for new participants via track_published events
            async def on_track_published(event):
                if hasattr(event, "participant") and event.participant:
                    user_id = getattr(event.participant, "user_id", None)
                    if user_id and user_id != bot_id:
                        logging.info(f"👋 New participant joined: {user_id}")
                        await send_greeting_once()

            connection._ws_client.on_event("track_published", on_track_published)

            # Check for existing participants and greet if any
            existing_participants = [
                p
                for p in connection.participants_state._participant_by_prefix.values()
                if getattr(p, "user_id", None) != bot_id
            ]

            if existing_participants:
                logging.info(
                    f"👋 Found {len(existing_participants)} existing participants"
                )
                await send_greeting_once()

            logging.info("🎧 Bot is idle - press Ctrl+C to stop")
            await connection.wait()

    except asyncio.CancelledError:
        logging.info("Stopping TTS bot…")
    finally:
        client.delete_users([human_id, bot_id])
        logging.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
