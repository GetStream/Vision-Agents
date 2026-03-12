import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()

"""
Deploy example - similar to 01_simple_agent_example but containerized.

Eager turn taking STT, LLM, TTS workflow
- deepgram for optimal latency
- eleven labs for TTS
- gemini-3.1-flash-lite-preview for fast responses
- stream's edge network for video transport
"""

# ── Prometheus metrics ──────────────────────────────────────────────

ACTIVE_SESSIONS = Gauge("ai_demo_active_sessions", "Number of active agent sessions")
STT_LATENCY = Gauge("ai_demo_stt_latency_seconds", "Average STT latency")
LLM_LATENCY = Gauge("ai_demo_llm_latency_seconds", "Average LLM latency")
TTS_LATENCY = Gauge("ai_demo_tts_latency_seconds", "Average TTS latency")
LLM_TTFT = Gauge("ai_demo_llm_ttft_seconds", "Average LLM time-to-first-token")
TURN_DURATION = Gauge("ai_demo_turn_duration_seconds", "Average turn duration")
LLM_INPUT_TOKENS = Gauge("ai_demo_llm_input_tokens", "Total LLM input tokens")
LLM_OUTPUT_TOKENS = Gauge("ai_demo_llm_output_tokens", "Total LLM output tokens")
TTS_CHARACTERS = Gauge("ai_demo_tts_characters", "Total TTS characters")

_AVG_METRICS = [
    ("stt_latency_ms__avg", STT_LATENCY),
    ("llm_latency_ms__avg", LLM_LATENCY),
    ("tts_latency_ms__avg", TTS_LATENCY),
    ("llm_time_to_first_token_ms__avg", LLM_TTFT),
    ("turn_duration_ms__avg", TURN_DURATION),
]
_SUM_METRICS = [
    ("llm_input_tokens__total", LLM_INPUT_TOKENS),
    ("llm_output_tokens__total", LLM_OUTPUT_TOKENS),
    ("tts_characters__total", TTS_CHARACTERS),
]


def _collect(launcher: AgentLauncher) -> None:
    sessions = launcher._sessions  # noqa: SLF001
    ACTIVE_SESSIONS.set(len(sessions))

    for key, gauge in _AVG_METRICS:
        values = []
        for s in sessions.values():
            try:
                v = s.agent.metrics.to_dict().get(key)
                if v is not None:
                    values.append(v)
            except Exception:
                continue
        gauge.set(sum(values) / len(values) / 1000 if values else 0)

    for key, gauge in _SUM_METRICS:
        total = 0
        for s in sessions.values():
            try:
                v = s.agent.metrics.to_dict().get(key)
                if v is not None:
                    total += int(v)
            except Exception:
                continue
        gauge.set(total)


# ── Agent setup ─────────────────────────────────────────────────────


async def create_agent(**kwargs) -> Agent:
    llm = gemini.LLM("gemini-3.1-flash-lite-preview")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        processors=[],
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("tell me something interesting in a short sentence")
        await agent.finish()


# ── Launcher + Runner ───────────────────────────────────────────────

redis_url = os.environ.get("REDIS_URL")
registry = None

if redis_url:
    from vision_agents.core.agents.session_registry import (
        RedisSessionKVStore,
        SessionRegistry,
    )

    store = RedisSessionKVStore(url=redis_url)
    registry = SessionRegistry(store=store)
    logger.info("Using Redis session registry: %s", redis_url.split("@")[-1])

launcher = AgentLauncher(
    create_agent=create_agent,
    join_call=join_call,
    registry=registry,
)
runner = Runner(launcher)

# Prometheus metrics endpoint
async def metrics_endpoint() -> Response:
    _collect(launcher)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

runner.fast_api.add_api_route("/metrics", metrics_endpoint, methods=["GET"])

if __name__ == "__main__":
    runner.cli()
