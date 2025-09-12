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
from getstream import Stream

# Temporarily suppress most logs to surface audio-track prints
logging.basicConfig(level=logging.WARNING)
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
        instructions=("""
You are a voice assistant. Speak English only. Keep responses short, natural, and conversational.
- Always respond in the same language the user is speaking in, if intelligible.
- Only respond to clear audio or text.
- If the user's audio is not clear (e.g., ambiguous input/background noise/silent/unintelligible) or if you did not fully hear or understand the user, ask for clarification.
"""

        ),
        llm=Realtime(instructions="You are a voice assistant. Speak English only."),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    agent.edge.open_demo(call)

    logger.info("🤖 Starting OpenAI STS Agent...")

    async def wait_for_human_participant(timeout: float = 60.0) -> None:
        """Wait until a non-agent participant is in the call (WebRTC-side)."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            connection = getattr(agent, "_connection", None)
            if connection is not None:
                try:
                    # poll participants from WebRTC state
                    participants = list(
                        connection.participants_state._participant_by_prefix.values()  # type: ignore[attr-defined]
                    )
                    for p in participants:
                        user_id = getattr(p, "user_id", None)
                        if user_id and user_id != agent.agent_user.id:
                            return
                except Exception:
                    pass
            await asyncio.sleep(0.2)
        raise TimeoutError("No human participant detected in time")

    # Have the agent join the call/room
    with await agent.join(call):
        print("Joining call")
        # Ensure the LLM realtime connection is ready (should already be awaited internally)
        try:
            await agent.llm.wait_until_ready(timeout=10.0)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Wait for a human to join the call before greeting
        await wait_for_human_participant(timeout=60.0)

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
