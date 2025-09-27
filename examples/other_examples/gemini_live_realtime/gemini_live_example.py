import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import gemini, getstream
from stream_agents.core.agents import Agent
from stream_agents.core.cli import start_dispatcher
from stream_agents.core import logging_utils
from getstream import Stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logging_utils.initialize_logging_context()
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    call_id = str(uuid4())
    token = logging_utils.set_call_context(call_id)

    try:
        client = Stream.from_env()
        agent_user = client.create_user(name="My happy AI friend")

        agent = Agent(
            edge=getstream.Edge(),
            agent_user=agent_user,  # the user object for the agent (name, image etc)
            instructions="Read @voice-agent.md",
            llm=gemini.Realtime(),
            processors=[],  # processors can fetch extra data, check images/audio data or transform video
        )

        call = client.video.call("default", call_id)

        agent.edge.open_demo(call)

        with await agent.join(call):
            await asyncio.sleep(5)
            await agent.llm.simple_response(text="Describe what you see and say hi")
            await agent.finish()  # run till the call ends
    finally:
        logging_utils.clear_call_context(token)


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
