import asyncio
import base64
import json
import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from getstream.video import rtc
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini

try:
    from .utils import mulaw_to_pcm, pcm_to_mulaw, TWILIO_SAMPLE_RATE
except ImportError:
    from utils import mulaw_to_pcm, pcm_to_mulaw, TWILIO_SAMPLE_RATE

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

TODO
- Call identifiers/ parsing the twilio data
- Fix ulaw processing
- There is some delay... Why? (think it selects frankfurt somehow...)
- pass session id from voice to media
- maybe start stream call earlier

Outbound example
"""

load_dotenv()

NGROK_URL = os.environ['NGROK_URL']

app = FastAPI()


class TwilioMediaStream:
    """
    Manages a Twilio Media Stream WebSocket connection.
    
    Handles:
    - Audio track for incoming audio
    - WebSocket connection to Twilio
    - Parsing Twilio WebSocket messages
    - Sending audio back to Twilio
    """
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stream_sid: str | None = None
        self.audio_track = AudioStreamTrack(
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format="s16",
        )
        self._connected = False
    
    async def accept(self):
        """Accept the WebSocket connection."""
        await self.websocket.accept()
        self._connected = True
        logger.info("TwilioMediaStream: WebSocket connection accepted")
    
    async def run(self):
        """
        Process incoming Twilio WebSocket messages.
        
        Parses messages and writes audio to the audio track.
        Returns when the stream ends.
        """
        has_seen_media = False
        message_count = 0
        
        try:
            while True:
                message = await self.websocket.receive_text()
                data = json.loads(message)
                
                match data["event"]:
                    case "connected":
                        logger.info(f"TwilioMediaStream: Connected: {data}")
                    case "start":
                        self.stream_sid = data["streamSid"]
                        logger.info(f"TwilioMediaStream: Stream started, streamSid={self.stream_sid}")
                    case "media":
                        # Decode base64 mulaw audio
                        payload = data["media"]["payload"]
                        mulaw_bytes = base64.b64decode(payload)
                        
                        # Convert to PCM and write to audio track
                        pcm = mulaw_to_pcm(mulaw_bytes)
                        await self.audio_track.write(pcm)
                        
                        if not has_seen_media:
                            logger.info(f"TwilioMediaStream: Receiving audio: {len(mulaw_bytes)} bytes/chunk")
                            has_seen_media = True
                    case "stop":
                        logger.info("TwilioMediaStream: Stream stopped")
                        break
                
                message_count += 1
        except Exception as e:
            logger.info(f"TwilioMediaStream: WebSocket closed: {e}")
        finally:
            self._connected = False
        
        logger.info(f"TwilioMediaStream: Connection closed. Received {message_count} messages")
    
    async def send_audio(self, pcm: PcmData) -> None:
        """Send PCM audio back to Twilio as mulaw."""
        if not self._connected or self.stream_sid is None:
            return
        
        # Convert PCM to mulaw and base64 encode
        mulaw_bytes = pcm_to_mulaw(pcm)
        payload = base64.b64encode(mulaw_bytes).decode("ascii")
        
        # Send media message per Twilio docs
        message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": payload
            }
        }
        
        await self.websocket.send_json(message)
    
    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is still connected."""
        return self._connected


@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls and start media streaming."""
    response = VoiceResponse()

    # Use <Connect><Stream> to keep the call open for the duration of the stream
    # Unlike <Start><Stream>, <Connect> blocks until the stream ends
    url = f"wss://{NGROK_URL}/twilio/media"
    connect = Connect()
    connect.stream(url=url)
    logger.info(f"Connecting to media stream on {url}")
    response.append(connect)

    twiml = str(response)
    logger.info(f"TWIML {twiml}")

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio/media")
async def media_stream(websocket: WebSocket):
    """Receive real-time audio stream from Twilio."""
    
    # Create the Twilio media stream connection
    twilio_stream = TwilioMediaStream(websocket)
    await twilio_stream.accept()
    
    # Create an agent
    agent = await create_agent()
    
    # Create the agent's user first (required for server-side auth)
    await agent.create_user()
    
    # Create the call on Stream
    call = await agent.create_call("default", str(uuid.uuid4()))
    
    # Join the call with the Twilio stream
    await join_call(agent, call, twilio_stream)


def outbound_example():
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    client = Client(account_sid, auth_token)
    # TODO: replace with media stream
    call = client.calls.create(
        twiml="<Response><Say>Ahoy, World</Say></Response>",
        to="+14155551212",
        from_="+172505482175",
    )



async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),  # use stream for edge video transport
        agent_user=User(id="ai-agent", name="AI"),
        instructions="You're a sales person, explaining to customers why they should use Stream's chat, video, feed and moderation products. As a voice agent use short replies",
        llm=gemini.Realtime(),  # Share video with gemini
    )
    return agent


async def join_call(agent: Agent, call, twilio_stream: TwilioMediaStream) -> None:
    """
    Join a call and bridge audio between Twilio and Stream.
    
    Args:
        agent: The AI agent
        call: The Stream call object
        twilio_stream: The Twilio media stream connection
    """
    # Create a user for the phone caller
    user = User(name="phone caller", id="phonenumber")
    full_user = await agent.edge.create_user(user=user)
    
    subscription_config = SubscriptionConfig(
        default=TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_AUDIO,
            ]
        )
    )

    # Phone connection -> publish to the call
    connection = await rtc.join(
        call, user.id, subscription_config=subscription_config
    )


    @connection.on("audio")
    async def on_audio_received(pcm: PcmData):
        # Forward audio from the call to Twilio
        await twilio_stream.send_audio(pcm)

    await (
        connection.__aenter__()
    )
    await asyncio.sleep(1.0)
    await connection.add_tracks(audio=twilio_stream.audio_track, video=None)

    logger.info("phone caller joined the call, agent is joining next")

    # Join the call with the agent
    with await agent.join(call):
        await agent.llm.simple_response(
            text="Say hi. After the user does their golf swing offer helpful feedback."
        )
        
        # Run the Twilio stream (blocks until stream ends)
        await twilio_stream.run()


if __name__ == "__main__":
    # Run the FastAPI server to receive Twilio webhooks
    uvicorn.run(app, host="localhost", port=8000)
