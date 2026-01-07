import asyncio
import logging
from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, elevenlabs, moondream
from vision_agents.core.events import CallSessionParticipantJoinedEvent
import os

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = moondream.CloudVLM(
        api_key=os.getenv("MOONDREAM_API_KEY"),
    )
    # create an agent to run with Stream's edge, openAI llm
    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(name="My happy AI friend", id="agent"),
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(2)
            await agent.simple_response("Describe what you currently see")

    # Have the agent join the call/room
    async with agent.join(call):
        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
