"""Inworld AI LLM router example — tuned for low voice latency.

Uses ``inworld.LLM`` — the OpenAI-compatible chat completions endpoint at
``https://api.inworld.ai/v1`` — as the brain for a voice agent.

Routing notes:

- We pin ``model="openai/gpt-4o-mini"`` rather than ``"auto"``. ``auto``
  asks Inworld to pick an upstream per request based on ``sort_by``, which
  adds a routing-decision hop on top of the selected upstream's TTFT. For
  voice, that variance is felt directly. A pinned fast model gives a
  flatter, lower TTFT.
- ``fallback_models`` keeps a vision/text-capable second choice in case
  the primary times out per ``ttft_timeout``.
- ``ttft_timeout="500ms"`` is Inworld's enforced minimum (anything lower
  returns a 502 ``Upstream server unavailable`` from the router): cut
  over to the fallback aggressively if the primary is slow to first
  token, but no faster than the gateway allows.

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
        model="openai/gpt-4o-mini",
        ttft_timeout="500ms",
        fallback_models=[
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
