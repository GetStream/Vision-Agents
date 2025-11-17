"""
AWS Bedrock LLM with Function Calling Example

This example demonstrates using AWS Bedrock LLM with streaming and function calling.
The agent can call custom functions to get weather information and perform calculations.
"""

import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import aws, getstream, cartesia, deepgram


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with AWS Bedrock LLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Weather Bot", id="agent"),
        instructions="You are a helpful weather bot. Use the provided tools to answer questions.",
        llm=aws.LLM(
            model="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1"
        ),
        tts=cartesia.TTS(),
        stt=deepgram.STT(),
    )

    # Register custom functions
    @agent.llm.register_function(
        name="get_weather", description="Get the current weather for a given city"
    )
    def get_weather(city: str) -> dict:
        """Get weather information for a city."""
        logger.info(f"Tool: get_weather called for city: {city}")
        if city.lower() == "boulder":
            return {"city": city, "temperature": 72, "condition": "Sunny"}
        return {"city": city, "temperature": "unknown", "condition": "unknown"}

    @agent.llm.register_function(
        name="calculate", description="Performs a mathematical calculation"
    )
    def calculate(expression: str) -> dict:
        """Performs a mathematical calculation."""
        logger.info(f"Tool: calculate called with expression: {expression}")
        try:
            result = eval(
                expression
            )  # DANGER: In a real app, use a safer math evaluator!
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting AWS Bedrock LLM Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        # Give the agent a moment to connect
        await asyncio.sleep(5)

        # Test function calling with weather
        logger.info("Testing weather function...")
        await agent.llm.simple_response(
            text="What's the weather like in Boulder? Please use the get_weather function."
        )

        await asyncio.sleep(5)

        # Test function calling with calculation
        logger.info("Testing calculation function...")
        await agent.llm.simple_response(
            text="Can you calculate 25 multiplied by 4 using the calculate function?"
        )

        await asyncio.sleep(5)

        # Wait a bit before finishing
        await asyncio.sleep(5)
        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
