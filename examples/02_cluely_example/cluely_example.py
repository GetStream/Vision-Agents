import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv
from getstream import Stream
from getstream.models import UserRequest
from stream_agents.plugins import deepgram, elevenlabs
from stream_agents.processors import YOLOPoseProcessor

# TODO: imports are not nice
from stream_agents.turn_detection import FalTurnDetection

from stream_agents.edge.edge_transport import StreamEdge
from stream_agents.utils import open_demo
from stream_agents.llm import OpenAILLM
from stream_agents.agents.agents import Agent
from stream_agents.cli import start_dispatcher


async def main() -> None:
    """Create a simple agent and join a call."""

    load_dotenv()

    # TODO this user creation flow is ugly.
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    client = Stream.from_env()
    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    # Create the agent
    turn_detection = FalTurnDetection(
        buffer_duration=3.0,  # Process 3 seconds of audio at a time
        prediction_threshold=0.7,  # Higher threshold for more confident detections
        mini_pause_duration=0.5,
        max_pause_duration=2.0,
    )

    # TODO: LLM class
    agent = Agent(
        edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user name etc for the agent
        # tts, llm, stt more. see the realtime example for sts
        llm=OpenAILLM(
            name="gpt-4o",
            instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        # turn keeping
        turn_detection=turn_detection,
        # processors can fetch extra data, check images/audio data or transform video
        processors=[YOLOPoseProcessor()],
    )

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        # Open the demo env
        open_demo(call)

        # have the agent join a call/room
        await agent.join(call)
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")

        # run till the call is ended
        await agent.finish()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main))
