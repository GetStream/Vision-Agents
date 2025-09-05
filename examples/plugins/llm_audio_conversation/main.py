#!/usr/bin/env python3
"""
Example: AI Voice Conversation Bot with Stream, Deepgram STT, ElevenLabs TTS, and OpenAI

This example demonstrates how to:
1. Create a Stream video call and add an AI bot
2. Use VAD (Voice Activity Detection) to detect when users speak
3. Transcribe user speech in real-time using Deepgram STT
4. Send transcribed text to OpenAI's LLM for intelligent responses
5. Convert LLM responses to speech using ElevenLabs TTS
6. Enable natural voice conversations between users and AI

Usage:
    python main.py

Requirements:
    - Create a .env file with your Stream, Deepgram, ElevenLabs, and OpenAI credentials
    - Install dependencies: pip install -e .
"""

import asyncio
import os
import time
from uuid import uuid4
import webbrowser
from urllib.parse import urlencode

from dotenv import load_dotenv
from openai import OpenAI

from getstream.models import UserRequest
from getstream.plugins.deepgram.stt import DeepgramSTT
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from getstream.plugins.silero.vad import SileroVAD
from getstream.stream import Stream
from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.track_util import PcmData


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
    load_dotenv()

    print("🎙️  Stream + VAD + Deepgram STT + ElevenLabs TTS Example")
    print("=" * 60)

    # Initialize Stream client from environment variables
    client = Stream.from_env()

    # Create a user
    id = f"example-user-{str(uuid4())}"
    create_user(client, id, "Me")
    print(f"👤 Created user: {id}")
    user_token = client.create_token(id, expiration=3600)
    print(f"🔑 Created token for user: {id}")

    # Create a bot user
    bot_user_id = f"llm-audio-bot-{str(uuid4())}"
    create_user(client, bot_user_id, "LLM Bot")
    print(f"🤖 Created bot user: {bot_user_id}")

    # Create a unique call ID for this session
    call_id = str(uuid4())
    print(f"📞 Call ID: {call_id}")

    # Create the call
    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": bot_user_id})
    print(f"📞 Call created: {call_id}")

    open_browser(client.api_key, user_token, call_id)

    # Initialize components
    audio = audio_track.AudioStreamTrack(framerate=16000)
    vad = SileroVAD()
    stt = DeepgramSTT()

    # Use "Arnold" voice - a default ElevenLabs voice available to all users
    tts_instance = ElevenLabsTTS(voice_id="VR6AewLTigWG4xSOukaG")  # ID of default voice
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tts_instance.set_output_track(audio)

    print("\n🤖 Starting LLM audio conversation bot...")
    print("The bot will join the call, detect speech, transcribe it, and echo it back.")
    print("\nPress Ctrl+C to stop the bot.\n")

    try:
        async with await rtc.join(call, bot_user_id) as connection:
            await connection.add_tracks(
                audio=audio,
            )

            print(f"✅ Bot joined call: {call_id}")

            await tts_instance.send("welcome my friend, how's it going today?")

            @connection.on("audio")
            async def on_audio(pcm: PcmData, user):
                await vad.process_audio(pcm, user)

            # TODO: add interruption logic to flush the audio queue when the user speaks to interrupt the LLM
            @vad.on("audio")
            async def on_speech_detected(pcm: PcmData, user):
                print(
                    f"{time.time()} Speech detected from user: {user} duration {pcm.duration}"
                )
                # Process audio through STT with user metadata
                user_metadata = {"user": user} if user else None
                await stt.process_audio(pcm, user_metadata)

            @stt.on("transcript")
            async def on_transcript(event):
                user_info = "unknown"
                if event.user_metadata and "user" in event.user_metadata:
                    user = event.user_metadata["user"]
                    user_info = str(user)
                print(
                    f"{time.time()} got text from user {user_info}, with metadata {event.to_dict()}"
                    f"will send the transcript to the LLM: {event.text}"
                )

                response = openai_client.responses.create(
                    model="gpt-4o",
                    input=f"You are a helpful and friendly assistant. Respond to the user's message in a conversational way. Here's the message: {event.text}",
                )

                llm_response = response.output_text
                print(f"{time.time()} LLM response: {llm_response}")
                await tts_instance.send(llm_response)

            # Keep the connection alive
            print("🎧 Listening for audio... (Press Ctrl+C to stop)")
            await connection.wait()

    except KeyboardInterrupt:
        print("\n⏹️  Stopping LLM audio conversation bot...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Delete users
        client.delete_users([id, bot_user_id])

        # Clean up services
        await stt.close()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
