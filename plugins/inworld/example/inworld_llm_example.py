"""Inworld AI LLM router example — tuned for low voice latency.

Uses ``inworld.LLM`` — the OpenAI-compatible chat completions endpoint at
``https://api.inworld.ai/v1`` — as the brain for a voice agent.

Routing notes:

- Each *turn* picks a random primary from ``ROUTER_MODELS``, mirroring
  the [router quickstart](https://docs.inworld.ai/router/quickstart)'s
  weighted-variant demo but client-side so no portal config is needed.
  The other two models become fallbacks for that same turn, so a single
  upstream wobble does not drop the turn. The resolved model is logged
  on every turn via the ``[LLM response final]`` line.
- ``ttft_timeout="2s"`` gives the primary room to actually deliver
  first token before we cut over. 500ms (Inworld's floor) trips the
  fallback on essentially every request because real-world TTFT
  typically lands at 1–2s; that turns the fallback into the de-facto
  primary and amplifies a single outage into a hard failure.

For full-duplex speech-to-speech (no STT/TTS hop), use ``inworld.Realtime``
instead — that path is strictly lower-latency than STT → LLM → TTS.

Set the following before running:
- ``INWORLD_API_KEY``
- ``STREAM_API_KEY`` / ``STREAM_API_SECRET``
- ``DEEPGRAM_API_KEY``
"""

import logging
import random
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.plugins import deepgram, getstream, inworld

load_dotenv()

logger = logging.getLogger(__name__)

ROUTER_MODELS = [
    "openai/gpt-5.4-nano",
    "google-ai-studio/gemini-3.1-flash-lite-preview",
    "deepinfra/deepseek-v4-flash",
]


class RotatingInworldLLM(inworld.LLM):
    """Demo-only LLM that picks a fresh primary on every request.

    Mirrors the router quickstart's weighted-variant pattern client-side:
    each turn ``random.choice``-s a primary out of ``models``, and the
    rest become per-request fallbacks via ``extra_body.models``. Used by
    the example to make the router's model-switching behavior visible.
    """

    def __init__(self, models: list[str], **kwargs: Any) -> None:
        if not models:
            raise ValueError("RotatingInworldLLM requires at least one model")
        super().__init__(model=models[0], fallback_models=models[1:], **kwargs)
        self._rotation = list(models)

    async def _create_response_internal(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        primary = random.choice(self._rotation)
        self.model = primary
        self._extra_body["models"] = [m for m in self._rotation if m != primary]
        logger.info(f"🎲 Routing this turn: primary={primary}")
        async for item in super()._create_response_internal(messages, tools, **kwargs):
            yield item


async def create_agent(**kwargs) -> Agent:
    llm = RotatingInworldLLM(
        models=ROUTER_MODELS,
        ttft_timeout="2s",
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
