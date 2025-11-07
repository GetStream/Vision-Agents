import logging

from dotenv import load_dotenv

from vision_agents.core.edge.types import User
from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.plugins import gemini, getstream

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="My happy AI friend"
        ),  # the user object for the agent (name, image etc)
        instructions="Read @voice-agent.md",
        llm=gemini.Realtime(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    with await agent.join(call):
        await agent.llm.simple_response(text="Describe what you see and say hi")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
