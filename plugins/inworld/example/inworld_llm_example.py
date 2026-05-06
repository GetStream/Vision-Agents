"""Inworld AI LLM router example.

Uses ``inworld.LLM`` — the OpenAI-compatible chat completions endpoint at
``https://api.inworld.ai/v1`` — as the brain for a voice agent. Inworld
routes upstream across providers; this example asks the router to pick
the lowest-latency model with a fallback chain so that voice latency stays
predictable even when the primary upstream is slow.

For full-duplex speech-to-speech (no STT/TTS hop), use ``inworld.Realtime``
instead — that path is strictly lower-latency than STT → LLM → TTS.

Set the following before running:
- ``INWORLD_API_KEY``
- ``STREAM_API_KEY`` / ``STREAM_API_SECRET``
- ``DEEPGRAM_API_KEY``
"""

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, inworld

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = inworld.LLM(
        model="auto",
        sort_by=["latency"],
        ttft_timeout="500ms",
        fallback_models=[
            "openai/gpt-4o-mini",
            "google-ai-studio/gemini-2.5-flash",
        ],
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Inworld Router Agent", id="agent"),
        instructions=(
            "You are a friendly assistant. Keep replies to one or two sentences."
        ),
        llm=llm,
        stt=deepgram.STT(),
        tts=inworld.TTS(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
