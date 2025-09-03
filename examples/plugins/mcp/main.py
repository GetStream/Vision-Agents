import logging
import os
import uuid
import webbrowser
from urllib.parse import urlencode

from dotenv import load_dotenv

from getstream.models import UserRequest
from getstream.stream import Stream
from getstream.video import rtc
from getstream.video.call import Call
from getstream.plugins import DeepgramSTT, ElevenLabsTTS
from getstream.video.rtc import audio_track
from getstream.video.rtc.track_util import PcmData
from agent import chat_with_tools
import asyncio
import fastmcp

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


async def run_bot(call: Call, bot_user_id: str):
    """Join the call as a bot, convert speech→text, call MCP tools
    when the transcript is final, then speak the answer back."""

    stt = DeepgramSTT()
    tts = ElevenLabsTTS()
    track = audio_track.AudioStreamTrack(framerate=16000)
    tts.set_output_track(track)

    try:
        # The bot itself joins as a participant so we can receive audio
        async with await rtc.join(call, bot_user_id) as connection:
            await connection.add_tracks(audio=track)
            logging.info("🤖 Bot joined the call and is now listening")

            async with fastmcp.Client("transport.py") as mcp_client:

                @connection.on("audio")
                async def on_audio(pcm: PcmData, user):
                    """Pipe raw PCM into the STT engine."""
                    # Process audio through STT with user metadata
                    user_metadata = {"user": user} if user else None
                    await stt.process_audio(pcm, user_metadata)

                @stt.on("transcript")
                async def on_transcript(event):
                    user_info = "unknown"
                    if event.user_metadata and "user" in event.user_metadata:
                        user = event.user_metadata["user"]
                        user_info = str(user)
                    logging.info("🗣️  %s: %s", user_info, event.text)

                    # Ask the LLM; it may decide to call an MCP tool.
                    answer = await chat_with_tools(event.text, mcp_client)
                    if answer:
                        logging.info("🤖 Bot reply: %s", answer)
                        await tts.send(answer)

                @stt.on("error")
                async def on_stt_error(event):
                    logging.error("STT error: %s", event.error_message)
                    if hasattr(event, 'context') and event.context:
                        logging.error("Context: %s", event.context)

                logging.info("🎧 Bot is listening… (Ctrl-C to stop)")
                await connection.wait()  # run until the program is stopped
    except asyncio.CancelledError:
        print("\n⏹️  Stopping transcription bot...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await stt.close()


async def main():
    """Entry point for the Stream + FastMCP example."""
    print("🎙️  Stream MCP Example")
    print("=" * 55)

    load_dotenv()

    client = Stream.from_env()

    call_id = str(uuid.uuid4())
    print(f"📞 Call ID: {call_id}")

    # Create a regular human user
    user_id = f"user-{uuid.uuid4()}"
    create_user(client, user_id, "Human User")
    user_token = client.create_token(user_id, expiration=3600)

    # Create the bot user that will join
    bot_user_id = f"mcp-bot-{uuid.uuid4()}"
    create_user(client, bot_user_id, "MCP Bot")

    # Initialise the call
    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": bot_user_id})
    print(f"📞 Call created: {call_id}")

    # Open browser for users to join with the user token
    open_browser(client.api_key, user_token, call_id)

    try:
        await run_bot(call, bot_user_id)
    finally:
        client.delete_users([user_id, bot_user_id])
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
