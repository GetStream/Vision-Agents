import asyncio
import logging
import os
import uuid

import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from getstream.video import rtc
from getstream.video.rtc import PcmData
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from twilio.rest import Client

from vision_agents.core import Agent, User
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream, twilio, openai

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NGROK_URL = os.environ["NGROK_URL"]

app = FastAPI()
call_registry = twilio.TwilioCallRegistry()

"""
TODO
- secret/auth for media endpoint
"""


async def create_agent() -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(id="ai-agent", name="AI Assistant"),
        instructions="Speak english. Keep your replies short. You're calling a restaurant and want to make a dinner reservation. for 4, today between 18:00 or 19:00 pm. ideally 18:30",
        llm=gemini.Realtime(),
    )


async def initiate_outbound_call(from_number: str, to_number: str) -> str:
    """Initiate an outbound call via Twilio. Returns the call_id."""
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

    call_id = str(uuid.uuid4())
    url = f"wss://{NGROK_URL}/twilio/media/{call_id}"

    async def prepare_call():
        agent = await create_agent()
        phone_user = User(name=f"Outbound call {call_id[:8]}", id=f"phone-{call_id}")

        # Create both users in a single API call
        await agent.edge.create_users([agent.agent_user, phone_user])

        stream_call = await agent.create_call("default", call_id)
        agent_session = await agent.join(stream_call)
        return agent, phone_user, stream_call, agent_session

    call_registry.create(call_id, prepare=prepare_call)

    client.calls.create(
        twiml=twilio.create_media_stream_twiml(url),
        to=to_number,
        from_=from_number,
    )
    logger.info(f"📞 Initiated call {call_id} from {from_number} to {to_number}")
    return call_id


@app.websocket("/twilio/media/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """Receive real-time audio stream from Twilio."""
    twilio_call = call_registry.require(call_sid)

    logger.info(f"🔗 Media stream connected for call {call_sid}")

    twilio_stream = twilio.TwilioMediaStream(websocket)
    await twilio_stream.accept()
    twilio_call.twilio_stream = twilio_stream

    try:
        # Wait for the prepare task to complete
        agent, phone_user, stream_call, agent_session = await twilio_call.await_prepare()
        twilio_call.stream_call = stream_call

        await attach_phone_to_call(stream_call, twilio_stream, phone_user)

        with agent_session:
            await agent.llm.simple_response(
                text="Ask to reserve and answer any follow up questions as needed"
            )
            await twilio_stream.run()
    finally:
        call_registry.remove(call_sid)


async def attach_phone_to_call(
    call, twilio_stream: twilio.TwilioMediaStream, phone_user: User
) -> None:
    """Join a call and bridge audio between Twilio and Stream."""
    subscription_config = SubscriptionConfig(
        default=TrackSubscriptionConfig(track_types=[TrackType.TRACK_TYPE_AUDIO])
    )

    connection = await rtc.join(call, phone_user.id, subscription_config=subscription_config)

    @connection.on("audio")
    async def on_audio_received(pcm: PcmData):
        await twilio_stream.send_audio(pcm)

    await connection.__aenter__()
    await connection.add_tracks(audio=twilio_stream.audio_track, video=None)

    logger.info(f"{phone_user.name} is now attached to the call")


async def run_with_server(from_number: str, to_number: str):
    """Start the server and initiate the outbound call once ready."""
    config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Start server in background task
    server_task = asyncio.create_task(server.serve())

    # Wait for server to be ready
    while not server.started:
        await asyncio.sleep(0.1)

    logger.info("🚀 Server ready, initiating outbound call...")

    # Initiate the outbound call
    await initiate_outbound_call(from_number, to_number)

    # Keep running until server shuts down
    await server_task


@click.command()
@click.option("--from", "from_number", required=True, help="The phone number to call from. Needs to be active in your Twilio account")
@click.option("--to", "to_number", required=True, help="The phone number to call")
def main(from_number: str, to_number: str):
    asyncio.run(run_with_server(from_number, to_number))


if __name__ == "__main__":
    main()