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
from getstream import AsyncStream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # Set the call ID here to be used in the logging
    call_id = str(uuid4())
    
    # create a stream client and a user object
    client = AsyncStream()
    agent_user = await client.create_user(name="My happy AI friend")

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
    await call.get_or_create(data={"created_by_id": agent.agent_user.id})

    logger.info("🤖 Starting OpenAI Realtime Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")
        await agent.edge.open_demo(call)
        logger.info("LLM ready")
        #await agent.llm.request_session_info()
        logger.info("Requested session info")
        # Wait for a human to join the call before greeting
        logger.info("Waiting for human to join the call")
        await agent.llm.simple_response(text="Please greet the user.")
        logger.info("Greeted the user")

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
