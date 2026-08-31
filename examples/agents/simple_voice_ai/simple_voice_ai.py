import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
A voice agent whose whole pipeline runs in the Go acceleration backend.

Gemini transcribes, Gemini Flash Lite holds the conversation, ElevenLabs speaks it, and
anything worth thinking about is handed to Sol while the conversation carries on. The
config names what the agent is; the targets below name who does the work.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""

NAME = "simple_voice_ai"


async def create_agent(**kwargs) -> Agent:
    await acceleration.sync_agent(NAME)

    return Agent(
        config=NAME,
        llm=acceleration.Accelerated(
            config=NAME,
            stt="gemini/gemini-3.5-transcribe-live",
            tts="elevenlabs/eleven_v3_conversational",
            model="gemini/gemini-3.5-flash-lite",
            subagent="openai/gpt-5.6-sol",
        ),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("greet the user in one short sentence")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
