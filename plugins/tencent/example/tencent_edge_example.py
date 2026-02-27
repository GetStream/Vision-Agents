import asyncio
import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, gemini, tencent

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    sdk_app_id = int(os.environ["TENCENT_SDKAppID"])
    secret_key = os.environ["TENCENT_SDKSecretKey"]

    agent = Agent(
        edge=tencent.Edge(sdk_app_id=sdk_app_id, key=secret_key),
        agent_user=User(name="Tencent Voice Agent", id="tencent-voice-agent"),
        instructions="You are a helpful voice assistant. Respond concisely.",
        llm=gemini.Realtime(),
        # tts=elevenlabs.TTS(),
        # stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.authenticate()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
