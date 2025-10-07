"""
Roboflow Example

This example demonstrates using Roboflow to process images/video.
The agent uses Roboflow to process images/video and pass the results to OpenAI Realtime.
"""

import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from getstream import AsyncStream
from vision_agents.plugins import openai, getstream
from vision_agents.core.agents import Agent
from vision_agents.core.cli import start_dispatcher
from vision_agents.core.edge.events import TrackAddedEvent
from vision_agents.core.logging_utils import configure_call_id_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

configure_call_id_logging(False)

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

        @agent.subscribe
        async def handle_screen_share(event: TrackAddedEvent):
            logger.info(f"Screen share event: {event.track_id} {event.track_type} {event.user}")

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
