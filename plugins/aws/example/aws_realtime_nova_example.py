import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import aws, getstream

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Story Teller AI"),
        instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
        llm=aws.Realtime(),
    )
    await agent.create_user()

    call = agent.edge.client.video.call("default", str(uuid4()))
    await agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Say hi and start the story")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())
