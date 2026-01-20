"""Golf Coach Example with Prometheus Metrics.

Run with:
    cd examples/02_golf_coach_example
    uv run python golf_coach_with_metrics.py run --call-type default --call-id test-metrics

Then open http://localhost:9464/metrics to see real-time metrics.
"""

# Configure OpenTelemetry BEFORE importing vision_agents
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

# Now import vision_agents - metrics will be recorded automatically
import logging  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from vision_agents.core import Agent, Runner, User  # noqa: E402
from vision_agents.core.agents import AgentLauncher  # noqa: E402
from vision_agents.core.observability import MetricsCollector  # noqa: E402
from vision_agents.plugins import getstream, openai, ultralytics  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI golf coach", id="golf-coach-agent"),
        instructions="Read @golf_coach.md",
        llm=openai.Realtime(fps=3),
        processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # Attach MetricsCollector to record OpenTelemetry metrics
    # Store reference to prevent garbage collection during call lifetime
    _metrics_collector = MetricsCollector(agent)

    logger.info("=" * 60)
    logger.info("Golf Coach with Realtime Metrics")
    logger.info("=" * 60)
    logger.info(f"Metrics endpoint: http://localhost:{PROMETHEUS_PORT}/metrics")
    logger.info("")
    logger.info("Realtime metrics being collected:")
    logger.info("  - realtime.sessions, realtime.session_duration.ms")
    logger.info("  - realtime.audio.input.bytes, realtime.audio.output.bytes")
    logger.info(
        "  - realtime.audio.input.duration.ms, realtime.audio.output.duration.ms"
    )
    logger.info("  - realtime.responses")
    logger.info("  - realtime.transcriptions.user, realtime.transcriptions.agent")
    logger.info("  - realtime.errors")
    logger.info("")
    logger.info("VLM/Video metrics:")
    logger.info("  - vlm.inferences, vlm.inference.latency.ms")
    logger.info("  - vlm.input_tokens, vlm.output_tokens")
    logger.info("  - video.frames_processed, video.detections")
    logger.info("=" * 60)

    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.llm.simple_response(
            text="Say hi. After the user does their golf swing offer helpful feedback."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
