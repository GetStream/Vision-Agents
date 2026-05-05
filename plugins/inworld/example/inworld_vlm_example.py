"""Inworld AI VLM router example — low-latency video Q&A agent.

The agent watches the participant's camera feed and answers spoken
questions about it. Inworld routes upstream across vision-capable
providers; we ask for ``model="auto"`` sorted by latency, with a tight
``ttft_timeout`` and a fallback chain to two small, fast vision models.

Performance tuning notes — these matter a lot for video latency:

- ``frame_width=512``, ``frame_height=384`` — small frames cut JPEG
  encoding cost, request bandwidth, and upstream input-token cost.
  800×600 (the default) is fine for stills but 4× the bytes.
- ``frame_buffer_seconds=3`` — only the most-recent 3 frames go to
  the model per request. Short-horizon video Q&A doesn't need 10s of
  history; longer history quadratically inflates input tokens.
- ``fps=1`` — one frame/sec is enough for typical "what do you see?"
  questions. Bumping fps mostly increases token spend without making
  answers more accurate.
- ``sort_by=["latency"]`` + ``ttft_timeout="500ms"`` + a fallback chain
  of ``google-ai-studio/gemini-2.5-flash`` and ``openai/gpt-4o-mini``
  (both vision-capable, fast) keeps TTFT in the sub-second range.

If you need full-duplex audio (no STT/TTS hop) and don't need video
understanding, use ``inworld.Realtime`` instead — it's strictly
lower-latency for voice-only.

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
    vlm = inworld.VLM(
        model="google-ai-studio/gemini-2.5-flash",
        ttft_timeout="500ms",
        fallback_models=[
            "openai/gpt-4o-mini",
        ],
        fps=1,
        frame_buffer_seconds=3,
        frame_width=512,
        frame_height=384,
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Inworld Vision Agent", id="agent"),
        instructions=(
            "You are a video assistant. You can see the user's camera. "
            "Keep replies to one or two short sentences. Describe what you "
            "actually observe, not what you assume — if the frame is unclear, "
            "say so."
        ),
        llm=vlm,
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
