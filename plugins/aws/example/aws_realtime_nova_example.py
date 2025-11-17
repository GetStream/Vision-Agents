"""
AWS Bedrock Realtime Nova Example

This example demonstrates using AWS Bedrock Realtime with the Nova model.
"""

import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import aws, getstream


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with AWS Bedrock Realtime."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Story Teller AI", id="agent"),
        instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
        llm=aws.Realtime(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting AWS Realtime Nova Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Say hi and start the story")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
