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

from stream_agents.plugins.openai import Realtime
from stream_agents.core.agents import Agent
from stream_agents.core.edge import StreamEdge
from stream_agents.core.cli import start_dispatcher
from stream_agents.core.utils import open_demo
from getstream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")

    # Create the agent
    agent = Agent(
        edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        llm=Realtime(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    open_demo(call)

    logger.info("🤖 Starting OpenAI STS Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        print("Joining call")
        await agent.llm.simple_response(text="Please greet the user.")
        print("Greeted the user")
        # img_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        # await agent.llm.create_response(
        #     input=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "input_text",
        #                     "text": "Tell me a short poem about this image",
        #                 },
        #                 {"type": "input_image", "image_url": f"{img_url}"},
        #             ],
        #         }
        #     ]
        # )
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
