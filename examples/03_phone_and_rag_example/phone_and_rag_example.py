"""
Phone + RAG Example

A voice AI agent that answers phone calls via Twilio with RAG capabilities.

RAG Backend Configuration (via RAG_BACKEND environment variable):
- "gemini" (default): Uses Gemini's built-in File Search
- "turbopuffer": Uses TurboPuffer + LangChain with function calling

Flow:
1. Twilio triggers webhook for phone number on /twilio/voice
2. Start a bi directional stream using start.stream which goes to /twilio/media
3. Start a call on Stream's edge network
4. Create a participant for the phone call and join the call
5. Create the AI, and have the AI join the call

Notes: Twilio uses ulaw audio encoding at 8kHz.
"""

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

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, twilio, elevenlabs, deepgram

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

NGROK_URL = os.environ["NGROK_URL"]
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# RAG backend: "gemini" or "turbopuffer"
RAG_BACKEND = os.environ.get("RAG_BACKEND", "gemini").lower()

# Global RAG state (initialized on startup)
file_search_store: gemini.FileSearchStore | None = None  # For Gemini
rag = None  # For TurboPuffer

app = FastAPI()
call_registry = twilio.TwilioCallRegistry()


# =============================================================================
# Startup - Initialize RAG
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG backend based on RAG_BACKEND environment variable."""
    global file_search_store, rag

    if not KNOWLEDGE_DIR.exists():
        logger.warning(f"Knowledge directory not found: {KNOWLEDGE_DIR}")
        return

    if RAG_BACKEND == "turbopuffer":
        await _init_turbopuffer_rag()
    else:
        await _init_gemini_rag()


async def _init_gemini_rag():
    """Initialize Gemini File Search RAG."""
    global file_search_store

    logger.info(f"📚 Initializing Gemini File Search from {KNOWLEDGE_DIR}")
    file_search_store = await gemini.create_file_search_store(
        name="stream-product-knowledge",
        knowledge_dir=KNOWLEDGE_DIR,
        extensions=[".md"],
    )
    logger.info(f"✅ Gemini RAG ready with {len(file_search_store._uploaded_files)} documents")


async def _init_turbopuffer_rag():
    """Initialize TurboPuffer + LangChain RAG."""
    global rag

    from rag_turbopuffer import create_rag

    logger.info(f"📚 Initializing TurboPuffer RAG from {KNOWLEDGE_DIR}")
    rag = await create_rag(
        namespace="stream-product-knowledge",
        knowledge_dir=KNOWLEDGE_DIR,
        extensions=[".md"],
    )
    logger.info(f"✅ TurboPuffer RAG ready with {len(rag._indexed_files)} documents indexed")


# =============================================================================
# FastAPI Endpoints
# =============================================================================


@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls and start media streaming."""
    form = await request.form()
    form_data = dict(form)
    call_sid = form_data.get("CallSid", str(uuid.uuid4()))

    caller = form_data.get("Caller", "unknown")
    called_city = form_data.get("CalledCity", "unknown location")
    logger.info(f"📞 Received call {caller} calling from {called_city}")

    call_registry.create(call_sid, form_data)
    response = VoiceResponse()

    url = f"wss://{NGROK_URL}/twilio/media/{call_sid}"
    connect = Connect()
    connect.stream(url=url)
    logger.info(f"Forwarding to media stream on {url}")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


@app.websocket("/twilio/media/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """Receive real-time audio stream from Twilio."""
    twilio_call = call_registry.get(call_sid)
    if not twilio_call:
        raise ValueError(f"Unknown call_sid: {call_sid}")

    caller = twilio_call.form_data.get("Caller", "unknown")
    called_city = twilio_call.form_data.get("CalledCity", "unknown location")
    logger.info(f"🔗 Media stream connecting for {caller} from {called_city}")

    twilio_stream = twilio.TwilioMediaStream(websocket)
    await twilio_stream.accept()
    twilio_call.twilio_stream = twilio_stream

    try:
        agent = await create_agent()
        await agent.create_user()

        phone_number = twilio_call.from_number or "unknown"
        sanitized_number = phone_number.replace("+", "").replace(" ", "").replace("(", "").replace(")", "")
        phone_user = User(name=f"Call from {phone_number}", id=f"phone-{sanitized_number}")
        await agent.edge.create_user(user=phone_user)

        stream_call = await agent.create_call("default", call_sid)
        twilio_call.stream_call = stream_call

        await join_call(agent, stream_call, twilio_stream, phone_user)
    finally:
        call_registry.remove(call_sid)


# =============================================================================
# Agent Creation
# =============================================================================


async def create_agent(**kwargs) -> Agent:
    """Create an agent with RAG capabilities."""
    if RAG_BACKEND == "turbopuffer":
        return await _create_agent_turbopuffer()
    else:
        return await _create_agent_gemini()


async def _create_agent_gemini() -> Agent:
    """Create agent with Gemini File Search RAG."""
    instructions = """You're a sales person for Stream, helping customers understand Stream's products:
- Chat API: Real-time messaging with offline support and edge network
- Video API: WebRTC-based video calling and streaming  
- Feeds API: Activity feeds and social features
- Moderation: AI-powered content moderation

Use the file_search tool to find detailed product information when answering questions.
As a voice agent, keep your replies concise and conversational."""

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(id="ai-agent", name="AI"),
        instructions=instructions,
        tts=elevenlabs.TTS(voice_id="FGY2WhTYpPnrIDTdsKH5"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM("gemini-2.5-flash-lite", file_search_store=file_search_store),
    )


async def _create_agent_turbopuffer() -> Agent:
    """Create agent with TurboPuffer RAG via function calling."""
    instructions = """You're a sales person for Stream, helping customers understand Stream's products:
- Chat API: Real-time messaging with offline support and edge network
- Video API: WebRTC-based video calling and streaming  
- Feeds API: Activity feeds and social features
- Moderation: AI-powered content moderation

IMPORTANT: When answering questions about Stream's products, use the search_knowledge 
function to find accurate information from our knowledge base.
As a voice agent, keep your replies concise and conversational."""

    llm = gemini.LLM("gemini-2.5-flash-lite")

    # Register RAG search as a callable function
    @llm.register_function(
        description="Search Stream's product knowledge base for detailed information about Chat, Video, Feeds, and Moderation APIs."
    )
    async def search_knowledge(query: str) -> str:
        """Search the knowledge base for relevant product information."""
        if rag is None:
            return "Knowledge base not available."
        return await rag.search(query, top_k=3)

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(id="ai-agent", name="AI"),
        instructions=instructions,
        tts=elevenlabs.TTS(voice_id="FGY2WhTYpPnrIDTdsKH5"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=llm,
    )


# =============================================================================
# Call Handling
# =============================================================================


async def join_call(
    agent: Agent, call, twilio_stream: twilio.TwilioMediaStream, phone_user: User
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

    logger.info(f"{phone_user.name} joined the call, agent is joining next")

    with await agent.join(call):
        await agent.llm.simple_response(
            text="Greet the caller warmly and ask what kind of app they're building. Use your knowledge base to provide relevant product recommendations."
        )
        await twilio_stream.run()


if __name__ == "__main__":
    logger.info(f"Starting with RAG_BACKEND={RAG_BACKEND}")
    uvicorn.run(app, host="localhost", port=8000)
