import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.events import CallSessionParticipantJoinedEvent
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    vlm = gemini.VLM()

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Gemini Vision Agent", id="gemini-vision-agent"),
        instructions="Describe what you see in one sentence.",
        llm=vlm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "gemini-vision-agent":
            await asyncio.sleep(2)
            await agent.simple_response("Describe the scene.")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
