"""
Sales Assistant — real-time meeting coach powered by Vision Agents.

Run as an HTTP server:
    uv run main.py serve --host 0.0.0.0 --port 8000

The Flutter overlay app calls POST /sessions to start a coaching session.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import Query
from fastapi.responses import JSONResponse
from getstream import Stream

from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import deepgram, gemini, getstream as getstream_edge

logger = logging.getLogger(__name__)
load_dotenv()


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
        llm=gemini.LLM("gemini-2.5-flash"),
        stt=deepgram.STT(),
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

    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("Sales assistant joining call %s/%s", call_type, call_id)

    async with agent.join(call):
        logger.info("Sales assistant is live — coaching started")

        # LLM responses are automatically synced to Stream Chat
        # on the messaging:{call_id} channel by the Vision Agents SDK.
        await agent.simple_response(
            "Listen to the conversation. "
            "Provide real-time coaching suggestions — tell the user what to say next. "
            "Keep every suggestion to 1-3 sentences."
        )

        # Keep the agent alive — STT + turn detection will trigger
        # additional simple_response calls automatically.
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

    @runner.fast_api.get("/auth/token")
    async def create_token(user_id: str = Query(...)) -> JSONResponse:
        """Generate a Stream user token for the Flutter client."""
        token = _stream_client.create_token(user_id)
        return JSONResponse({"token": token, "apiKey": _api_key})

    runner.cli()
