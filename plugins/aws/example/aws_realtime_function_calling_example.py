"""
AWS Bedrock Realtime with Function Calling Example

This example creates an agent that can call custom functions to get
weather information and perform calculations.
"""

import asyncio
import logging
from typing import Dict

from dotenv import load_dotenv
from typing_extensions import Any
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import aws, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with AWS Bedrock Realtime."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Weather Assistant AI", id="agent"),
        instructions="""You are a helpful weather assistant. When users ask about weather,
        use the get_weather function to fetch current conditions. You can also help with
        simple calculations using the calculate function.""",
        llm=aws.Realtime(
            model="amazon.nova-2-sonic-v1:0",
            region_name="us-east-1",
        ),
    )

    # Register custom functions that the LLM can call
    @agent.llm.register_function(
        name="get_weather", description="Get the current weather for a given city"
    )
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    @agent.llm.register_function(
        name="calculate", description="Perform a mathematical calculation"
    )
    def calculate(operation: str, a: float, b: float) -> dict:
        """Perform a calculation.

        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number

        Returns:
            Result of the calculation
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None,
        }

        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}

        result = operations[operation](a, b)
        if result is None:
            return {"error": "Cannot divide by zero"}

        return {"operation": operation, "a": a, "b": b, "result": result}

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting AWS Bedrock Realtime Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        # Give the agent a moment to connect
        await asyncio.sleep(2)

        await agent.llm.simple_response(
            text="What's the weather like in Boulder? Please use the get_weather function."
        )
        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
