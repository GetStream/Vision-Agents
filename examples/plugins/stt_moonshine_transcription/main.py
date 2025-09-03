#!/usr/bin/env python3
"""
Example: Real-time Call Transcription with Moonshine STT

Uses the local Moonshine model via the Moonshine plugin in this repo.

Steps:
1. Create two temporary Stream users (human + bot)
2. Spin up a new call and open it in your browser
3. The bot joins and pipes every audio frame to Moonshine STT
4. Final transcripts are printed with timestamps

Run:
    python main.py
"""

from __future__ import annotations

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
from getstream.video import rtc
from getstream.video.rtc.track_util import PcmData
from getstream.plugins import MoonshineSTT, SileroVAD


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


async def main() -> None:  # noqa: D401
    print("🌙  Stream + Moonshine Real-time Transcription Example")
    print("=" * 60)

    load_dotenv()

    client = Stream.from_env()

    call_id = str(uuid.uuid4())
    print(f"📞 Call ID: {call_id}")

    user_id = f"user-{uuid.uuid4()}"
    create_user(client, user_id, "My User")
    logging.info("👤 Created user: %s", user_id)

    user_token = client.create_token(user_id, expiration=3600)

    bot_user_id = f"moonshine-bot-{uuid.uuid4()}"
    create_user(client, bot_user_id, "Moonshine Bot")
    logging.info("🤖 Created bot user: %s", bot_user_id)

    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": bot_user_id})
    print(f"📞 Call created: {call_id}")

    open_browser(client.api_key, user_token, call_id)

    print("\n🤖 Starting transcription bot…")
    print("Speak in the browser and see transcripts below. Press Ctrl+C to stop.\n")

    stt = MoonshineSTT()

    # Initialize Silero VAD for speech detection
    print("🔊 Initializing Silero VAD...")
    vad = SileroVAD()
    print("✅ Audio processing pipeline ready: VAD → Moonshine STT")

    try:
        async with await rtc.join(call, bot_user_id) as connection:
            print(f"✅ Bot joined call: {call_id}")

            @connection.on("audio")
            async def _on_audio(pcm: PcmData, user):
                await vad.process_audio(pcm, user)

            @vad.on("audio")
            async def _on_speech_detected(pcm: PcmData, user):
                print(
                    f"🎤 Speech detected from user: {user.name}, duration: {pcm.duration:.2f}s"
                )
                # Process audio through STT with user metadata
                user_metadata = {"user": user} if user else None
                await stt.process_audio(pcm, user_metadata)

            @stt.on("transcript")
            async def _on_transcript(event):
                ts = time.strftime("%H:%M:%S")
                user_info = "unknown"
                if event.user_metadata and "user" in event.user_metadata:
                    user = event.user_metadata["user"]
                    user_info = str(user)
                print(f"[{ts}] {user_info}: {event.text}")
                if hasattr(event, 'confidence') and event.confidence:
                    print(f"    └─ confidence: {event.confidence:.2%}")
                if hasattr(event, 'processing_time_ms') and event.processing_time_ms:
                    print(f"    └─ processing time: {event.processing_time_ms:.1f}ms")

            @stt.on("error")
            async def _on_error(event):
                print(f"\n❌ STT Error: {event.error_message}")
                if hasattr(event, 'context') and event.context:
                    print(f"    └─ context: {event.context}")

            print("🎧 Listening for audio… (Press Ctrl+C to stop)")
            await connection.wait()

    except asyncio.CancelledError:
        print("\n⏹️  Stopping transcription bot…")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await stt.close()
        client.delete_users([user_id, bot_user_id])
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
