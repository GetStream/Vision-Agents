import asyncio
import logging
import os
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from getstream.video import rtc
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, twilio, elevenlabs, deepgram

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Path to knowledge directory
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# Global file search store (initialized on startup)
file_search_store: gemini.FileSearchStore | None = None

"""
Flow
1. Twilio triggers webhook for phone number on /twilio/voice
2. Start a bi directional stream using start.stream which goes to /twilio/media
3. Start a call on Stream's edge network
4. Create a participant for the phone call and join the call
5. Create the AI, and have the AI join the call

Notes, twilio uses ulaw audio encoding at 8khz

TODO:
- Readme excellence
- Fix that it connects to frankfurt
- Code cleanup
- Nicer logging
- Separate outbound example
- Turbopuffer + langchain example
"""

load_dotenv()

NGROK_URL = os.environ['NGROK_URL']

app = FastAPI()

# Global registry instance
call_registry = twilio.TwilioCallRegistry()


@app.on_event("startup")
async def startup_event():
    """Initialize the Gemini File Search store with knowledge documents."""
    global file_search_store
    
    if KNOWLEDGE_DIR.exists():
        logger.info(f"📚 Initializing File Search store from {KNOWLEDGE_DIR}")
        file_search_store = await gemini.create_file_search_store(
            name="stream-product-knowledge",
            knowledge_dir=KNOWLEDGE_DIR,
            extensions=[".md"],
        )
        logger.info(f"✅ File Search store ready with {len(file_search_store._uploaded_files)} documents")
    else:
        logger.warning(f"Knowledge directory not found: {KNOWLEDGE_DIR}")


# =============================================================================
# FastAPI Endpoints
# =============================================================================

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls and start media streaming."""
    form = await request.form()
    form_data = dict(form)
    call_sid = form_data.get("CallSid", str(uuid.uuid4()))
    
    # Log caller information
    caller = form_data.get("Caller", "unknown")
    called_city = form_data.get("CalledCity", "unknown location")
    logger.info(f"📞 Received call {caller} calling from {called_city}")

    call_registry.create(call_sid, form_data)
    response = VoiceResponse()

    # start a media stream to receive and send audio
    url = f"wss://{NGROK_URL}/twilio/media/{call_sid}"
    connect = Connect()
    connect.stream(url=url)
    logger.info(f"Forwarding to media stream on {url}")
    response.append(connect)
    twiml = str(response)

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio/media/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """Receive real-time audio stream from Twilio."""
    
    # Look up the TwilioCall from the registry
    twilio_call = call_registry.get(call_sid)
    if not twilio_call:
        raise ValueError(f"Unknown call_sid: {call_sid}. Call must be registered via /twilio/voice first.")
    
    # Log connection info
    caller = twilio_call.form_data.get("Caller", "unknown")
    called_city = twilio_call.form_data.get("CalledCity", "unknown location")
    logger.info(f"🔗 Media stream connecting for {caller} from {called_city}")
    
    # Create the Twilio media stream connection
    twilio_stream = twilio.TwilioMediaStream(websocket)
    await twilio_stream.accept()
    
    # Associate the stream with the call
    twilio_call.twilio_stream = twilio_stream

    try:
        # Create an agent
        agent = await create_agent()
        
        # Create the agent's user first (required for server-side auth)
        await agent.create_user()
        
        # Create a user for the phone caller
        phone_number = twilio_call.from_number or "unknown"
        # Sanitize phone number for user ID (only a-z, 0-9, @, _, - allowed)
        sanitized_number = phone_number.replace("+", "").replace(" ", "").replace("(", "").replace(")", "")
        phone_user = User(name=f"Call from {phone_number}", id=f"phone-{sanitized_number}")
        await agent.edge.create_user(user=phone_user)
        
        # Create the call on Stream
        stream_call = await agent.create_call("default", call_sid)
        twilio_call.stream_call = stream_call
        
        # Join the call with the Twilio stream
        await join_call(agent, stream_call, twilio_stream, phone_user)
    finally:
        # Clean up the registry when the call ends
        call_registry.remove(call_sid)


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
    """Create an agent with RAG capabilities using Gemini File Search."""
    instructions = """You're a sales person for Stream, helping customers understand Stream's products:
- Chat API: Real-time messaging with offline support and edge network
- Video API: WebRTC-based video calling and streaming  
- Feeds API: Activity feeds and social features
- Moderation: AI-powered content moderation

Use the file_search tool to find detailed product information when answering questions.
As a voice agent, keep your replies concise and conversational."""

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(id="ai-agent", name="AI"),
        instructions=instructions,
        tts=elevenlabs.TTS(voice_id="FGY2WhTYpPnrIDTdsKH5"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM("gemini-2.5-flash-lite", file_search_store=file_search_store),
    )
    return agent


async def join_call(agent: Agent, call, twilio_stream: twilio.TwilioMediaStream, phone_user: User) -> None:
    """
    Join a call and bridge audio between Twilio and Stream.
    
    Args:
        agent: The AI agent
        call: The Stream call object
        twilio_stream: The Twilio media stream connection
        phone_user: The user representing the phone caller
    """
    subscription_config = SubscriptionConfig(
        default=TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_AUDIO,
            ]
        )
    )

    # Phone connection -> publish to the call
    connection = await rtc.join(
        call, phone_user.id, subscription_config=subscription_config
    )


    @connection.on("audio")
    async def on_audio_received(pcm: PcmData):
        # Forward audio from the call to Twilio
        await twilio_stream.send_audio(pcm)

    await (
        connection.__aenter__()
    )
    await connection.add_tracks(audio=twilio_stream.audio_track, video=None)

    logger.info(f"{phone_user.name} joined the call, agent is joining next")

    # Join the call with the agent
    with await agent.join(call):
        await agent.llm.simple_response(
            text="Greet the caller warmly and ask what kind of app they're building. Use your knowledge base to provide relevant product recommendations."
        )
        
        # Run the Twilio stream (blocks until stream ends)
        await twilio_stream.run()


if __name__ == "__main__":
    # Run the FastAPI server to receive Twilio webhooks
    uvicorn.run(app, host="localhost", port=8000)
