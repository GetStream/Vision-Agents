import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv
from getstream import AsyncStream
from vision_agents.core.agents import Agent
from vision_agents.plugins import gemini, getstream

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    client = AsyncStream()

    agent_user = await client.create_user(name="My happy AI friend")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="Read @voice-agent.md",
        llm=gemini.Realtime(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    call = client.video.call("default", str(uuid4()))

    await agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Describe what you see and say hi")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_agent())
