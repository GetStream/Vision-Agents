import base64
import json
import logging
import os

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData, AudioFormat
from twilio.twiml.voice_response import VoiceResponse, Start

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

NGROK_URL = "dc29ca3d3d70.ngrok-free.app"
TWILIO_SAMPLE_RATE = 8000  # Twilio streams mulaw at 8kHz

app = FastAPI()

# Audio track to hold incoming Twilio audio
audio_track: AudioStreamTrack | None = None

# Precompute mulaw decoding table (ITU-T G.711)
MULAW_DECODE_TABLE = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0,
], dtype=np.int16)


def mulaw_to_pcm(mulaw_bytes: bytes) -> PcmData:
    """Convert mulaw audio bytes to PcmData using lookup table."""
    # Convert bytes to numpy array of uint8 indices
    mulaw_samples = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    
    # Decode using lookup table
    samples = MULAW_DECODE_TABLE[mulaw_samples]
    
    return PcmData(
        samples=samples,
        sample_rate=TWILIO_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )


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


@app.websocket("/media")
async def media_stream(websocket: WebSocket):
    """Receive real-time audio stream from Twilio and write to QueuedAudioTrack."""
    global audio_track
    
    await websocket.accept()
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
                    logger.info(f"Stream started: {data}")
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
    
    logger.info(f"Connection closed. Received {message_count} messages, wrote to audio track")


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
