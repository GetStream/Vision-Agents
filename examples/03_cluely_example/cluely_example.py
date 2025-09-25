import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv
from getstream import Stream
from getstream.models import UserRequest
from stream_agents.plugins import deepgram, elevenlabs, openai, ultralytics
from stream_agents.core import edge, agents, cli

# Main feats:
# 1. API endpoints to create a sessions, end session
# 2. Generate 2 - 4 sentiments for new chat messages

load_dotenv()


async def main() -> None:
    """Create a simple agent and join a call."""
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    client = Stream.from_env()
    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    # TODO: LLM class
    agent = agents.Agent(
        edge=edge.StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user name etc for the agent
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        # tts, llm, stt more. see the realtime example for sts
        llm=openai.LLM(
            model="gpt-4o",

        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        # processors can fetch extra data, check images/audio data or transform video
        processors=[ultralytics.YOLOPoseProcessor()],
    )

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        # Open the demo env
        agent.edge.open_demo(call)

        # have the agent join a call/room
        await agent.join(call)
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")

        # run till the call is ended
        await agent.finish()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(main))
