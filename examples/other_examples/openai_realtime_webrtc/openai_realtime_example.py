"""
OpenAI STS (Speech-to-Speech) Example

This example demonstrates using OpenAI's Realtime API for speech-to-speech conversation.
The agent uses WebRTC to establish a peer connection with OpenAI's servers, enabling
real-time bidirectional audio streaming.
"""

import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import openai, getstream


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
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
        llm=openai.Realtime(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )
    return agent



async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = agent.edge.client.video.call(call_type, call_id)
    # Ensure the call exists server-side before joining
    await call.get_or_create(data={"created_by_id": agent.agent_user.id})

    logger.info("🤖 Starting OpenAI Realtime Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")

        #TODO: should open demo be done by the CLI instead of the example?
        await agent.edge.open_demo(call)
        logger.info("LLM ready")
        # Wait for a human to join the call before greeting
        logger.info("Waiting for human to join the call")
        await agent.llm.simple_response(text="Please greet the user.")
        logger.info("Greeted the user")

        await agent.finish()  # run till the call ends

if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
