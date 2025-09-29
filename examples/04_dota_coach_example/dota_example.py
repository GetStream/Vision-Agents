import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv
from stream_agents.plugins import deepgram, elevenlabs
from stream_agents.core.llm import OpenAILLM
from stream_agents.core.agents.agents import Agent
from stream_agents.core.edge.edge_transport import StreamEdge
from stream_agents.core.cli import start_dispatcher
from stream_agents.core.utils import open_demo
from getstream import Stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    call_id = str(uuid4())

    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")

    # Create the agent
    agent = Agent(
        edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        # tts, llm, stt more. see the realtime example for sts
        llm=OpenAILLM(
            model="gpt-4o",
        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        # turn_detection=FalTurnDetection(),
        # Soccer: processors=[MatchStatistics()]
        # DOTA: processors=[GameStats(), Yolo(), Image()]
    )

    # Create a call
    call = client.video.call("default", call_id)

    # Open the demo UI
    open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
