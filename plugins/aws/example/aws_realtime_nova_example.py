"""
AWS Bedrock Nova Sonic 2 Example

- Shows how to use Nova 2 in realtime mode
- With function calling
"""

import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import Runner, User, Agent
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import aws, getstream


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = aws.Realtime()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Story Teller AI", id="agent"),
        instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
        llm=llm,
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    async with agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(
            text="Tell me a short story about a dragon and a princess"
        )

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
