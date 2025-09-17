import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import gemini, getstream
from stream_agents.core.agents import Agent
from stream_agents.core.cli import start_dispatcher
from getstream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        llm=gemini.Realtime2(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    call = client.video.call("default", str(uuid4()))

    agent.edge.open_demo(call)

    with await agent.join(call):
        await agent.llm.simple_response(text="Please greet the user.")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
