import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.core.processors import YOLOPoseProcessor
from stream_agents.plugins import gemini, getstream
from stream_agents.core.agents import Agent
from stream_agents.core.cli import start_dispatcher
from getstream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    client = Stream.from_env()
    agent_user = client.create_user(name="AI golf coach")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=agent_user,
        instructions="Read @golf_coach.md",
        llm=gemini.Realtime(),
        processors=[YOLOPoseProcessor()],
    )

    call = client.video.call("default", str(uuid4()))

    agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Say hi. After the user does their golf swing offer helpful feedback.")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
