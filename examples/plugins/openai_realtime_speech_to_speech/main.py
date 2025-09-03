import asyncio
import logging
import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

from dotenv import load_dotenv

from getstream import Stream
from getstream.models import CallRequest, StartClosedCaptionsResponse, UserRequest
from getstream.plugins.openai.sts import OpenAIRealtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any previous basicConfig calls
)

# Enable verbose logging for the OpenAI Realtime plugin
logging.getLogger("getstream_openai.sts").setLevel(logging.INFO)


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
    """Run a demo call with an OpenAI Speech-to-Speech agent attached."""

    load_dotenv()

    client = Stream.from_env()

    user_id = f"user-{uuid4()}"
    create_user(client, user_id, "My User")
    logging.info("👤 Created user: %s", user_id)

    user_token = client.create_token(user_id, expiration=3600)
    logging.info("🔑 Created token for user: %s", user_id)

    bot_user_id = f"openai-realtime-speech-to-speech-bot-{uuid4()}"
    create_user(client, bot_user_id, "OpenAI Realtime Speech to Speech Bot")
    logging.info("🤖 Created bot user: %s", bot_user_id)

    call_id = str(uuid4())
    logging.info("📞 Call ID: %s", call_id)

    call = client.video.call("default", call_id)
    call.get_or_create(data=CallRequest(created_by_id=bot_user_id))
    logging.info("📞 Call created: %s", call_id)

    # Open demo browser so you can join from the UI
    open_browser(client.api_key, user_token, call_id)

    sts_bot = OpenAIRealtime(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_REALTIME_MODEL") or "gpt-realtime",
        instructions="You are a friendly assistant; reply verbally in a short sentence of maximum 5 words.",
        voice="alloy",
    )

    try:
        logging.info("Connecting to OpenAI Realtime...")

        if not os.getenv("OPENAI_API_KEY"):
            logging.error("❌ OPENAI_API_KEY not found in environment")
            return

        async with await sts_bot.connect(call, agent_user_id=bot_user_id) as connection:
            tools = [
                {
                    "type": "function",
                    "name": "start_closed_captions",
                    "description": "start closed captions for the call",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
            ]

            await sts_bot.send_user_message("Give a very short greeting to the user.")

            logging.info("🎧 Listening for responses... (Press Ctrl+C to stop)")
            logging.info(
                "💡 Try speaking in the browser – ask it something like 'start closed captions' to trigger the function call."
            )

            async def start_closed_captions() -> StartClosedCaptionsResponse:
                """Helper that starts closed captions for the call."""
                return call.start_closed_captions().data

            # One-time session update after server confirms session creation
            session_updated_once = False

            async for event in connection:
                logging.info("🔔 Event received: %s", event.type)
                if event.type == "session.created" and not session_updated_once:
                    try:
                        await sts_bot.update_session(
                            modalities=["audio"],
                            turn_detection={
                                "type": "semantic_vad",
                                "eagerness": "low",
                                "create_response": True,
                                "interrupt_response": True,
                            },
                            voice="alloy",
                            speed=1.1,
                            tools=tools,
                        )
                        session_updated_once = True
                    except Exception:
                        logging.exception("Failed to update session")
                if event.type == "session.updated":
                    try:
                        logging.info(
                            "🛠 Session updated: %s", getattr(event, "session", None)
                        )
                        logging.info("🛠🛠🛠🛠🛠🛠🛠 Using model: %s", event.session.model)
                    except Exception:
                        pass

                if event.type == "session.created":
                    logging.info(
                        "🛠 Session created: %s", getattr(event, "session", None)
                    )
                    logging.info(
                        "session_id %s", getattr(event.session, "session_id", None)
                    )
                    logging.info("model %s", getattr(event.session, "model", None))

                if (
                    event.type == "response.done"
                    and event.response.output is not None
                    and len(event.response.output) > 0
                    and event.response.output[0].type == "function_call"
                ):
                    tool_call_id = event.response.output[0].call_id

                    if event.response.output[0].name == "start_closed_captions":
                        logging.info("🛠  Assistant requested start_closed_captions()")

                        result = await start_closed_captions()
                        await sts_bot.send_function_call_output(
                            tool_call_id, result.to_json()
                        )
                        await sts_bot.request_assistant_response()
                        logging.info("🛠  Replied to tool call with result: %s", result)

    except asyncio.CancelledError:  # noqa
        logging.info("\n⏹️  Stopping OpenAI Realtime Speech to Speech bot…")
    except Exception as e:  # noqa: BLE001
        logging.exception("❌ Error: %s", e)
    finally:
        logging.info("Cleaning up...")
        client.delete_users([user_id, bot_user_id])
        logging.info("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
