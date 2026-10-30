"""
Sales Assistant — real-time meeting coach powered by Vision Agents.

Run as an HTTP server, on this machine only:
    uv run main.py serve --port 8000

The Flutter overlay app calls POST /sessions to start a coaching session.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import Depends, Query
from fastapi.responses import JSONResponse
from getstream import Stream
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.runner import allow_local
from vision_agents.plugins import assemblyai, gemini
from vision_agents.plugins import getstream as getstream_edge

logger = logging.getLogger(__name__)
load_dotenv()

# Meeting context set by the Flutter app before starting a session.
_meeting_context: str = ""

# How long a token handed to the Flutter app lasts, in seconds.
TOKEN_VALIDITY = 3600


async def create_agent(**kwargs) -> Agent:
    """Create the sales assistant coaching agent.

    No TTS — the agent writes coaching suggestions to Stream Chat,
    it doesn't need to speak.  This also keeps the WebRTC setup simple
    (only a subscriber PC, no publisher audio track).
    """
    agent = Agent(
        edge=getstream_edge.Edge(),
        agent_user=User(name="Sales Assistant", id="sales-assistant-agent"),
        instructions="Read @instructions.md",
        llm=gemini.LLM("gemini-flash-lite-latest"),
        stt=assemblyai.STT(speaker_labels=True),
    )

    return agent


async def join_call(
    agent: Agent,
    call_type: str,
    call_id: str,
    **kwargs,
) -> None:
    """Join the Stream Video call and coach until it ends."""
    logger.info("join_call invoked for %s/%s", call_type, call_id)

    call = await agent.create_call(call_type, call_id)

    logger.info("Sales assistant joining call %s/%s", call_type, call_id)

    async with agent.join(call):
        logger.info("Sales assistant is live — coaching started")

        # LLM responses are automatically synced to Stream Chat
        # on the messaging:{call_id} channel by the Vision Agents SDK.
        prompt = (
            "Listen to the conversation. "
            "Provide real-time coaching suggestions — tell the user what to say next. "
            "Keep every suggestion to 1-3 sentences."
        )
        if _meeting_context:
            prompt = f"Meeting context: {_meeting_context}\n\n{prompt}"

        await agent.responses.create(prompt)

        # Keep the agent alive — STT + turn detection will trigger
        # additional responses.create calls automatically.
        await agent.finish()


if __name__ == "__main__":
    runner = Runner(
        AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
        )
    )

    # Expose /auth/token so the Flutter app can get credentials for the same
    # Stream application the agent uses.
    _stream_client = Stream()
    _api_key = os.environ["STREAM_API_KEY"]

    # Both routes answer only the app on this machine. /auth/token signs for whoever it is
    # asked for, because a demo has nobody signed in; a real application signs for the user
    # its own sign-in says the caller is.
    @runner.fast_api.put("/context", dependencies=[Depends(allow_local)])
    async def set_context(request: dict) -> JSONResponse:
        """Set meeting context for the next coaching session."""
        global _meeting_context
        _meeting_context = request.get("context", "")
        logger.info(
            "Meeting context updated: %s",
            _meeting_context[:80] if _meeting_context else "(cleared)",
        )
        return JSONResponse({"ok": True})

    @runner.fast_api.get("/auth/token", dependencies=[Depends(allow_local)])
    async def create_token(user_id: str = Query(...)) -> JSONResponse:
        """Generate a Stream user token for the Flutter client, valid for an hour."""
        token = _stream_client.create_token(user_id, expiration=TOKEN_VALIDITY)
        return JSONResponse({"token": token, "apiKey": _api_key})

    runner.cli()
