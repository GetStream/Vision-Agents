import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import cartesia, elevenlabs, gemini, tencent

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    sdk_app_id = int(os.environ["TENCENT_SDKAppID"])
    secret_key = os.environ["TENCENT_SDKSecretKey"]

    agent = Agent(
        edge=tencent.Edge(sdk_app_id=sdk_app_id, key=secret_key),
        agent_user=User(name="Tencent Voice Agent", id="tencent-voice-agent"),
        instructions="You are a helpful voice assistant. Respond concisely.",
        llm=gemini.LLM(model="gemini-3.1-flash-lite-preview"),
        tts=cartesia.TTS(),
        stt=elevenlabs.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.authenticate()
    # Room ID must match the one shown in the Tencent TRTC quick demo form
    # (see plugins/tencent/README.md). The browser needs to be in the room
    # before the agent connects so the STT websocket stays warm.
    room_id = os.environ["TENCENT_TEST_ROOM_ID"]
    call = await agent.create_call(call_type, room_id)

    async with agent.join(call, participant_wait_timeout=None):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
