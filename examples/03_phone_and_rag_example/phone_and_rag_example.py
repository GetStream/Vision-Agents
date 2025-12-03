import base64
import json
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from getstream.video import rtc
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from twilio.twiml.voice_response import VoiceResponse, Start

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini

from .utils import mulaw_to_pcm, pcm_to_mulaw, TWILIO_SAMPLE_RATE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Flow
1. Twilio triggers webhook for phone number on /twilio/voice
2. Start a bi directional stream using start.stream which goes to /twilio/media
3. Start a call on Stream's edge network
4. Create a participant for the phone call and join the call
5. Create the AI, and have the AI join the call

Notes, twilio uses ulaw audio encoding at 8khz

TODO/ simplification
* Cleanup the ulaw audio utils
* Create some sort of session object to combine the Twilio Connection, Call object (or is call alone enough?)

"""

load_dotenv()

NGROK_URL = "dc29ca3d3d70.ngrok-free.app"

app = FastAPI()


@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls and start media streaming."""
    response = VoiceResponse()
    response.say("Hi! Thanks for calling. I'm now listening.", voice="alice")

    # Start streaming audio to our WebSocket endpoint
    start = Start()
    start.stream(url=f"wss://{NGROK_URL}/media")
    response.append(start)

    # Keep the call alive with a pause
    response.pause(length=60)

    return Response(content=str(response), media_type="application/xml")


@app.websocket("/twilio/media")
async def media_stream(websocket: WebSocket):
    """Receive real-time audio stream from Twilio and write to QueuedAudioTrack."""
    global audio_track, twilio_websocket, twilio_stream_sid

    await websocket.accept()
    twilio_websocket = websocket
    logger.info("WebSocket connection accepted")

    # Create audio track for this call
    audio_track = AudioStreamTrack(
        sample_rate=TWILIO_SAMPLE_RATE,
        channels=1,
        format="s16",
    )

    has_seen_media = False
    message_count = 0

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            match data["event"]:
                case "connected":
                    logger.info(f"Connected: {data}")
                case "start":
                    # Store the stream SID for sending audio back
                    twilio_stream_sid = data["streamSid"]
                    logger.info(f"Stream started, streamSid={twilio_stream_sid}")
                case "media":
                    # Decode base64 mulaw audio
                    payload = data["media"]["payload"]
                    mulaw_bytes = base64.b64decode(payload)

                    # Convert to PCM and write to audio track
                    pcm = mulaw_to_pcm(mulaw_bytes)
                    await audio_track.write(pcm)

                    if not has_seen_media:
                        logger.info(f"Receiving audio: {len(mulaw_bytes)} bytes/chunk, writing to audio track")
                        has_seen_media = True
                case "stop":
                    logger.info("Stream stopped")
                    break

            message_count += 1
    except Exception as e:
        logger.info(f"WebSocket closed: {e}")
    finally:
        twilio_websocket = None
        twilio_stream_sid = None

    logger.info(f"Connection closed. Received {message_count} messages, wrote to audio track")


# Audio track to hold incoming Twilio audio
audio_track: AudioStreamTrack | None = None

# Active Twilio websocket connection for sending audio back
twilio_websocket: WebSocket | None = None
twilio_stream_sid: str | None = None


async def send_audio_to_twilio(pcm: PcmData) -> None:
    """Send PCM audio back to Twilio as mulaw."""
    global twilio_websocket, twilio_stream_sid
    
    if twilio_websocket is None or twilio_stream_sid is None:
        return
    
    # Convert PCM to mulaw and base64 encode
    mulaw_bytes = pcm_to_mulaw(pcm)
    payload = base64.b64encode(mulaw_bytes).decode("ascii")
    
    # Send media message per Twilio docs
    message = {
        "event": "media",
        "streamSid": twilio_stream_sid,
        "media": {
            "payload": payload
        }
    }
    
    await twilio_websocket.send_json(message)





async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),  # use stream for edge video transport
        agent_user=User(name="AI"),
        instructions="Assist the user, short answers only please",
        llm=gemini.Realtime(),  # Share video with gemini
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    """
    TODO:
    - when someone speaks on the phone. they should join the call as a participant..
    
    """
    user = await agent.edge.create_user(User(name="phone call from123"))
    subscription_config = SubscriptionConfig(
        default=TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_AUDIO,
            ]
        )
    )

    # phone connection -> publish to the call
    connection = await rtc.join(
        call, user, subscription_config=subscription_config
    )
    await connection.add_tracks(audio=audio_track, video=None)

    @connection.on("audio")
    async def on_audio_received(pcm: PcmData):
        # Forward audio from the call to Twilio
        await send_audio_to_twilio(pcm)


    # join the call and open a demo env
    with await agent.join(call):
        await agent.llm.simple_response(
            text="Say hi. After the user does their golf swing offer helpful feedback."
        )
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    # Run the FastAPI server to receive Twilio webhooks
    uvicorn.run(app, host="localhost", port=8000)
    # cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
