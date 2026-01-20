"""
Prometheus Metrics Example for Stream Agents.

This example shows how to export real metrics from Stream Agents to Prometheus.
The metrics are automatically collected from LLM, STT, TTS, and other components
during a live video call.

Setup:
    1. Configure OpenTelemetry with Prometheus exporter BEFORE importing agents
    2. Create your agent with the desired plugins
    3. Attach MetricsCollector to opt-in to metrics collection
    4. Metrics are available at http://localhost:9464/metrics

Run with:
    cd examples/06_prometheus_metrics_example
    uv run python prometheus_metrics_example.py run --call-type default --call-id test-metrics

Then open http://localhost:9464/metrics to see real-time metrics as you talk to the agent.
"""

# =============================================================================
# IMPORTANT: Configure OpenTelemetry BEFORE importing vision_agents
# =============================================================================
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

# Start Prometheus HTTP server on port 9464
PROMETHEUS_PORT = 9464
start_http_server(PROMETHEUS_PORT)

# Configure OpenTelemetry to export to Prometheus
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# =============================================================================
# Now import vision_agents - metrics will be recorded automatically
# =============================================================================
import logging  # noqa: E402
from typing import Any, Dict  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from vision_agents.core import Agent, AgentLauncher, Runner, User  # noqa: E402
from vision_agents.core.observability import MetricsCollector  # noqa: E402
from vision_agents.core.utils.examples import get_weather_by_location  # noqa: E402
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_agent(**kwargs) -> Agent:
    """Create an agent with STT/LLM/TTS workflow."""
    llm = gemini.LLM("gemini-2.5-flash-lite")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Metrics Demo Agent", id="agent"),
        instructions=(
            "You're a voice AI assistant demonstrating metrics collection. "
            "Keep responses short and conversational. Be friendly and helpful."
        ),
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a call with metrics collection enabled."""
    # Attach MetricsCollector to record OpenTelemetry metrics
    _ = MetricsCollector(agent)

    logger.info("=" * 60)
    logger.info("Prometheus Metrics Agent")
    logger.info("=" * 60)
    logger.info(f"Metrics endpoint: http://localhost:{PROMETHEUS_PORT}/metrics")
    logger.info("")
    logger.info("Metrics being collected:")
    logger.info("  - llm_latency_ms, llm_time_to_first_token_ms")
    logger.info("  - llm_tokens_input, llm_tokens_output")
    logger.info("  - llm_tool_calls, llm_tool_latency_ms")
    logger.info("  - stt_latency_ms, stt_audio_duration_ms")
    logger.info("  - tts_latency_ms, tts_audio_duration_ms, tts_characters")
    logger.info("  - turn_duration_ms, turn_trailing_silence_ms")
    logger.info("=" * 60)

    call = await agent.create_call(call_type, call_id)

    with await agent.join(call):
        await agent.simple_response(
            "Hello! I'm demonstrating metrics collection. "
            "Ask me anything - try asking about the weather!"
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
