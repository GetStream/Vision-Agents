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

TODO/ to fix:
- Things should prep when creating the call in the voice endpoint
- Auth for stream endpoint
- Ulaw audio bugs
- Frankfurt connection bug
- Study best practices for Gemini RAG
- Study Turbopuffer Rag
- Add an outbound calling example
- See if there is a nicer diff approach to rag indexing
- Write docs about Rag
"""
import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, WebSocket
from getstream.video import rtc
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, twilio, elevenlabs, deepgram

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


NGROK_URL = os.environ["NGROK_URL"]
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# RAG backend: "gemini" or "turbopuffer"
RAG_BACKEND = os.environ.get("RAG_BACKEND", "gemini").lower()

# Global RAG state (initialized on startup)
file_search_store: gemini.FileSearchStore | None = None  # For Gemini
rag = None  # For TurboPuffer

app = FastAPI()
call_registry = twilio.TwilioCallRegistry()

"""
Twilio call webhook points here. Signature is validated and we start the media stream
"""
@app.post("/twilio/voice")
async def twilio_voice_webhook(
        _: None = Depends(twilio.verify_twilio_signature),
        data: twilio.CallWebhookInput = Depends(twilio.CallWebhookInput.as_form),
):
    url = f"wss://{NGROK_URL}/twilio/media/{data.call_sid}"
    logger.info(f"📞 Call from {data.caller} ({data.caller_city or 'unknown location'}) forwarding to {url}")

    call_registry.create(data.call_sid, data)

    return twilio.create_media_stream_response(url)


"""
Twilio media stream endpoint
"""
@app.websocket("/twilio/media/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """Receive real-time audio stream from Twilio."""
    twilio_call = call_registry.require(call_sid)

    logger.info(f"🔗 Media stream connecting for {twilio_call.caller} from {twilio_call.caller_city or 'unknown location'}")

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
        namespace="stream-product-knowledge-gemini",
        knowledge_dir=KNOWLEDGE_DIR,
        extensions=[".md"],
    )
    logger.info(f"✅ TurboPuffer RAG ready with {len(rag._indexed_files)} documents indexed")



async def create_agent(**kwargs) -> Agent:
    """Create an agent with RAG capabilities."""
    if RAG_BACKEND == "turbopuffer":
        return await _create_agent_turbopuffer()
    else:
        return await _create_agent_gemini()


async def _create_agent_gemini() -> Agent:
    """Create agent with Gemini File Search RAG."""
    instructions = """Read the instructions in @instructions.md"""

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
    instructions = """Read the instructions in @instructions.md"""

    llm = gemini.LLM("gemini-2.5-flash-lite")

    # Register RAG search as a callable function
    @llm.register_function(
        description="Search Stream's product knowledge base for detailed information about Chat, Video, Feeds, and Moderation APIs."
    )
    async def search_knowledge(query: str) -> str:
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
    asyncio.run(startup_event())
    logger.info(f"Starting with RAG_BACKEND={RAG_BACKEND}")
    uvicorn.run(app, host="localhost", port=8000)
