import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import elevenlabs, openai, tencent

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    sdk_app_id = int(os.environ["TENCENT_SDK_APP_ID"])
    secret_key = os.environ["TENCENT_SDK_SECRET_KEY"]

    agent = Agent(
        edge=tencent.Edge(sdk_app_id=sdk_app_id, key=secret_key),
        agent_user=User(name="Tencent Voice Agent", id="tencent-voice-agent"),
        instructions="You are a helpful voice assistant. Respond concisely.",
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        stt=elevenlabs.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.authenticate()
    # `call_id` is passed via `--call-id` (see docker-compose.yml). It
    # must match the RoomID(String) shown in the Tencent TRTC quick demo
    # form so the browser-side participant is in the same room before
    # the agent connects — otherwise STT idles out (see README).
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call, participant_wait_timeout=None):
        await agent.say("Hi! How can I help you today?")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
