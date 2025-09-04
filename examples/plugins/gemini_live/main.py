import asyncio
import logging
import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

from dotenv import load_dotenv

from getstream import Stream
from getstream.models import CallRequest, UserRequest
from getstream.plugins import GeminiLive
from getstream.video import rtc
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import (
    SubscriptionConfig,
    TrackSubscriptionConfig,
    TrackType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any previous basicConfig calls
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


async def main():
    """Run a demo call with a Gemini Live Speech-to-Speech agent attached."""

    load_dotenv()

    client = Stream.from_env()

    user_id = f"user-{uuid4()}"
    create_user(client, user_id, "My User")
    logging.info("👤 Created user: %s", user_id)

    user_token = client.create_token(user_id, expiration=3600)
    logging.info("🔑 Created token for user: %s", user_id)

    bot_user_id = f"gemini-live-speech-to-speech-bot-{uuid4()}"
    create_user(client, bot_user_id, "Gemini Live Speech to Speech Bot")
    logging.info("🤖 Created bot user: %s", bot_user_id)

    call_id = str(uuid4())
    logging.info("📞 Call ID: %s", call_id)

    call = client.video.call("default", call_id)
    call.get_or_create(data=CallRequest(created_by_id=bot_user_id))
    logging.info("📞 Call created: %s", call_id)

    # Open demo browser so you can join from the UI
    open_browser(client.api_key, user_token, call_id)

    gemini_live = GeminiLive(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-live-2.5-flash-preview",
    )

    try:
        logging.info("Connecting to Gemini Live...")

        if not os.getenv("GOOGLE_API_KEY"):
            logging.error("GOOGLE_API_KEY not found in environment")
            return

        subscription = SubscriptionConfig(
            default=TrackSubscriptionConfig(track_types=[TrackType.TRACK_TYPE_VIDEO])
        )

        async with await rtc.join(
            call, bot_user_id, subscription_config=subscription
        ) as connection:
            await connection.add_tracks(audio=gemini_live.output_track)

            @connection.on("track_added")
            async def _on_track_added(track_id, kind, user):
                try:
                    if kind == "video" and connection.subscriber_pc:
                        track = connection.subscriber_pc.add_track_subscriber(track_id)
                    if track:
                        await gemini_live.start_video_sender(track, fps=1)
                        logging.info("🎥 Started forwarding video for user %s", user)
                except Exception as e:
                    logging.error("❌ Failed to start video sender: %s", e)

            @connection.on("audio")
            async def on_audio(pcm: PcmData, user):
                try:
                    await gemini_live.send_audio_pcm(pcm, target_rate=48000)
                    logging.debug("✅ Audio sent to Gemini Live from user %s", user)
                except Exception as e:
                    logging.error("❌ Failed to send audio to Gemini: %s", e)

            await gemini_live.send_text("Give a greeting to the user.")

            logging.info("🎧 Listening for responses... (Press Ctrl+C to stop)")

            # Keep the example running until interrupted
            while True:
                await asyncio.sleep(1)

    except asyncio.CancelledError:  # noqa
        logging.info("\n⏹️  Stopping Gemini Live Speech to Speech bot…")
    except Exception as e:  # noqa: BLE001
        logging.exception("❌ Error: %s", e)
    finally:
        logging.info("Cleaning up...")
        client.delete_users([user_id, bot_user_id])
        await gemini_live.stop_video_sender()
        await gemini_live.close()
        logging.info("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
