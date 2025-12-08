"""
Phone + RAG Example

A voice AI agent that answers phone calls via Twilio with RAG capabilities.

RAG Backend Configuration (via RAG_BACKEND environment variable):
- "gemini" (default): Uses Gemini's built-in File Search
- "turbopuffer": Uses TurboPuffer + LangChain with function calling

Flow:
1. Twilio triggers webhook on /twilio/voice, which starts preparing the call
2. Start a bi-directional stream using start.stream which goes to /twilio/media
3. When media stream connects, await the prepared call and attach the phone user
4. Run the agent session until the call ends

Notes: Twilio uses mulaw audio encoding at 8kHz.
"""

import asyncio
import logging
import os
import traceback
import uuid
from pathlib import Path

import uvicorn
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse

from vision_agents.core import User, Agent
from vision_agents.plugins import (
    getstream,
    gemini,
    twilio,
    elevenlabs,
    deepgram,
    turbopuffer,
)

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
# Trust proxy headers from ngrok so Twilio signature validation works (https vs http)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])
call_registry = twilio.TwilioCallRegistry()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.post("/twilio/voice")
async def twilio_voice_webhook(
    _: None = Depends(twilio.verify_twilio_signature),
    data: twilio.CallWebhookInput = Depends(twilio.CallWebhookInput.as_form),
):
    """Twilio call webhook. Validates signature and starts the media stream."""
    logger.info(
        f"📞 Call from {data.caller} ({data.caller_city or 'unknown location'})"
    )
    call_id = str(uuid.uuid4())

    async def prepare_call():
        agent = await create_agent()
        await agent.create_user()

        phone_number = data.from_number or "unknown"
        sanitized_number = (
            phone_number.replace("+", "")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
        )
        phone_user = User(
            name=f"Call from {phone_number}", id=f"phone-{sanitized_number}"
        )
        await agent.edge.create_user(user=phone_user)

        stream_call = await agent.create_call("default", call_id=call_id)
        agent_session = await agent.join(stream_call, wait_for_participant=False)
        return agent, phone_user, stream_call, agent_session

    twilio_call = call_registry.create(call_id, data, prepare=prepare_call)
    url = f"wss://{NGROK_URL}/twilio/media/{call_id}/{twilio_call.token}"
    logger.info("twilio redirect to %s", url)

    return twilio.create_media_stream_response(url)


@app.websocket("/twilio/media/{call_id}/{token}")
async def media_stream(websocket: WebSocket, call_id: str, token: str):
    """Receive real-time audio stream from Twilio."""
    twilio_call = call_registry.validate(call_id, token)

    logger.info(f"🔗 Media stream connected for {twilio_call.caller}")

    twilio_stream = twilio.TwilioMediaStream(websocket)
    await twilio_stream.accept()
    twilio_call.twilio_stream = twilio_stream

    try:
        (
            agent,
            phone_user,
            stream_call,
            agent_session,
        ) = await twilio_call.await_prepare()
        twilio_call.stream_call = stream_call

        await twilio.attach_phone_to_call(stream_call, twilio_stream, phone_user.id)

        with agent_session:
            await agent.llm.simple_response(
                text="Greet the caller warmly and ask what kind of app they're building. Use your knowledge base to provide relevant product recommendations."
            )
            await twilio_stream.run()
    finally:
        call_registry.remove(call_id)


async def create_rag_from_directory():
    """Initialize the RAG backend based on RAG_BACKEND environment variable."""
    global file_search_store, rag

    if not KNOWLEDGE_DIR.exists():
        logger.warning(f"Knowledge directory not found: {KNOWLEDGE_DIR}")
        return

    if RAG_BACKEND == "turbopuffer":
        logger.info(f"📚 Initializing TurboPuffer RAG from {KNOWLEDGE_DIR}")
        rag = await turbopuffer.create_rag(
            namespace="stream-product-knowledge-gemini",
            knowledge_dir=KNOWLEDGE_DIR,
            extensions=[".md"],
        )
        logger.info(
            f"✅ TurboPuffer RAG ready with {len(rag._indexed_files)} documents indexed"
        )
    else:
        logger.info(f"📚 Initializing Gemini File Search from {KNOWLEDGE_DIR}")
        file_search_store = await gemini.create_file_search_store(
            name="stream-product-knowledge",
            knowledge_dir=KNOWLEDGE_DIR,
            extensions=[".md"],
        )
        logger.info(
            f"✅ Gemini RAG ready with {len(file_search_store._uploaded_files)} documents"
        )


async def create_agent() -> Agent:
    """Create an agent with RAG capabilities."""
    instructions = """Read the instructions in @instructions.md"""

    if RAG_BACKEND == "turbopuffer":
        llm = gemini.LLM("gemini-2.5-flash-lite")

        @llm.register_function(
            description="Search Stream's product knowledge base for detailed information about Chat, Video, Feeds, and Moderation APIs."
        )
        async def search_knowledge(query: str) -> str:
            return await rag.search(query, top_k=3)

    else:
        llm = gemini.LLM("gemini-2.5-flash-lite", file_search_store=file_search_store)

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(id="ai-agent", name="AI"),
        instructions=instructions,
        tts=elevenlabs.TTS(voice_id="FGY2WhTYpPnrIDTdsKH5"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=llm,
    )


if __name__ == "__main__":
    asyncio.run(create_rag_from_directory())
    logger.info(f"Starting with RAG_BACKEND={RAG_BACKEND}")
    uvicorn.run(app, host="localhost", port=8000)
