"""
OpenAI STS (Speech-to-Speech) Example

This example demonstrates using OpenAI's Realtime API for speech-to-speech conversation.
The agent uses WebRTC to establish a peer connection with OpenAI's servers, enabling
real-time bidirectional audio streaming.
"""

import logging

from dotenv import load_dotenv
from typing import Dict, Any

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import openai, getstream
from vision_agents.core.utils.examples import get_weather_by_location

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = openai.Realtime()

    # MCP and function calling are supported. see https://visionagents.ai/guides/mcp-tool-calling
    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    # Create the agent
    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(
            name="My happy AI friend", id="agent"
        ),  # the user object for the agent (name, image etc)
        instructions=(
            "You are a voice assistant. Keep your responses short and friendly. Speak english plz"
        ),
        # Enable video input and set a conservative default frame rate for realtime responsiveness
        llm=llm,
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting OpenAI Realtime Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Agent is now joining the call")
        await agent.llm.simple_response(text="Please greet the user.")
        logger.info("Greeted the user")

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
