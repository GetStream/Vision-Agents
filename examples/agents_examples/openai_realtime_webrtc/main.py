"""
OpenAI STS (Speech-to-Speech) Example

This example demonstrates using OpenAI's Realtime API for speech-to-speech conversation.
The agent uses WebRTC to establish a peer connection with OpenAI's servers, enabling
real-time bidirectional audio streaming.
"""

import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import openai, getstream
from stream_agents.core.agents import Agent
from stream_agents.core.cli import start_dispatcher
from getstream import Stream

# Enable info-level logs to surface track subscription and forwarding diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")

    # Create the agent
    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions=("""
You are a voice assistant.
- Greet the user once when asked, then wait for the next input.
- Speak English only.
- If you see images/video, describe them when asked. Don't hallucinate.
- If you don't see images/video, say you don't see them.
- Keep responses natural and conversational.
- Only respond to clear audio or text.
- If the user's audio is not clear (e.g., ambiguous input/background noise/silent/unintelligible) or you didn't fully understand, ask for clarification.
"""
        ),
        # Enable video input and set a conservative default frame rate for realtime responsiveness
        llm=openai.Realtime(send_video=True),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))
    # Ensure the call exists server-side before joining
    call.get_or_create(data={"created_by_id": agent.agent_user.id})

    # Open the demo UI
    agent.edge.open_demo(call)

    logger.info("🤖 Starting OpenAI Realtime Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        await agent.llm.connect()
        print("Joining call")
        # Ensure the LLM realtime connection is ready (should already be awaited internally)
        await agent.llm.wait_until_ready(timeout=10.0)  # type: ignore[attr-defined]
        print("LLM ready")
        # Wait for a human to join the call before greeting
        print("Waiting for human to join the call")
        await agent.llm.simple_response(text="Please greet the user.")
        print("Greeted the user")

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
