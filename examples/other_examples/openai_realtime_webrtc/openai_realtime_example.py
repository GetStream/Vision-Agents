"""
OpenAI STS (Speech-to-Speech) Example

This example demonstrates using OpenAI's Realtime API for speech-to-speech conversation.
The agent uses WebRTC to establish a peer connection with OpenAI's servers, enabling
real-time bidirectional audio streaming.
"""

import asyncio
import logging
from contextvars import ContextVar
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import openai, getstream
from stream_agents.core.agents import Agent
from stream_agents.core.cli import start_dispatcher
from getstream import Stream

# Enable info-level logs to surface track subscription and forwarding diagnostics
call_id_ctx: ContextVar[str] = ContextVar("call_id", default="-")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s",
)
previous_factory = logging.getLogRecordFactory()


def contextual_record_factory(*args, **kwargs) -> logging.LogRecord:
    """Attach the call ID from context to every log record."""

    record = previous_factory(*args, **kwargs)
    record.call_id = call_id_ctx.get()
    return record


logging.setLogRecordFactory(contextual_record_factory)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # Set the call ID here to be used in the logging
    call_id = str(uuid4())
    token = call_id_ctx.set(call_id)
    
    try:
        # create a stream client and a user object
        client = Stream.from_env()
        agent_user = client.create_user(name="My happy AI friend")

        # Create the agent
        agent = Agent(
            edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
            agent_user=agent_user,  # the user object for the agent (name, image etc)
            instructions=("""
You are a voice assistant.
- Greet the user once when asked, then wait for the next user input.
- Speak English only.
- If you see images/video, describe them when asked. Don't hallucinate.
- If you don't see images/video, say you don't see them.
- Keep responses natural and conversational.
- Only respond to clear audio or text.
- If the user's audio is not clear (e.g., ambiguous input/background noise/silent/unintelligible) or you didn't fully understand, ask for clarification.
"""
            ),
            # Enable video input and set a conservative default frame rate for realtime responsiveness
            llm=openai.Realtime(),
            processors=[],  # processors can fetch extra data, check images/audio data or transform video
        )

        # Create a call
        call = client.video.call("default", call_id)
        # Ensure the call exists server-side before joining
        call.get_or_create(data={"created_by_id": agent.agent_user.id})

        logger.info("🤖 Starting OpenAI Realtime Agent...")

        # Have the agent join the call/room
        with await agent.join(call):
            print("Joining call")
            agent.edge.open_demo(call)
            print("LLM ready")
            #await agent.llm.request_session_info()
            print("Requested session info")
            # Wait for a human to join the call before greeting
            print("Waiting for human to join the call")
            await agent.llm.simple_response(text="Please greet the user.")
            print("Greeted the user")

            await agent.finish()  # run till the call ends
    finally:
        call_id_ctx.reset(token)


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
