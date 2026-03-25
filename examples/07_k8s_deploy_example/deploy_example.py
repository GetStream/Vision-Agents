import logging
import os
from typing import Any, Dict
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Response
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.agents.session_registry import (
    RedisSessionKVStore,
    SessionRegistry,
)
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

# Configure OpenTelemetry to export metrics via Prometheus
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# Active sessions is not covered by OTel MetricsCollector, track it separately
ACTIVE_SESSIONS = Gauge("ai_demo_active_sessions", "Number of active agent sessions")


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


# Launcher + Runner

redis_url = os.environ.get("REDIS_URL")
registry = None

if redis_url:
    store = RedisSessionKVStore(url=redis_url)
    registry = SessionRegistry(store=store)
    parsed = urlparse(redis_url)
    logger.info("Using Redis session registry: %s:%s", parsed.hostname, parsed.port)

launcher = AgentLauncher(
    create_agent=create_agent,
    join_call=join_call,
    registry=registry,
)
runner = Runner(launcher)


async def metrics_endpoint() -> Response:
    ACTIVE_SESSIONS.set(len(dict(launcher._sessions)))  # noqa: SLF001
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


runner.fast_api.add_api_route("/metrics", metrics_endpoint, methods=["GET"])

if __name__ == "__main__":
    runner.cli()
