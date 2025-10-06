import asyncio
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import gemini, getstream
from stream_agents.core.agents import Agent
from stream_agents.core.cli import start_dispatcher
from getstream import AsyncStream

load_dotenv()

async def start_agent() -> None:
    client = AsyncStream()

    with trace.get_tracer("agents").start_as_current_span("gemini_live_realtime"):
        agent_user = await client.create_user(name="My happy AI friend")

        agent = Agent(
            edge=getstream.Edge(),
            agent_user=agent_user,  # the user object for the agent (name, image etc)
            instructions="Read @voice-agent.md",
            llm=gemini.Realtime(),
            processors=[],  # processors can fetch extra data, check images/audio data or transform video
        )

        call = client.video.call("default", str(uuid4()))

        await agent.edge.open_demo(call)

        with await agent.join(call):
            await asyncio.sleep(5)
            await agent.llm.simple_response(text="Describe what you see and say hi")
            await agent.finish()  # run till the call ends


if __name__ == "__main__":
    from opentelemetry import metrics
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server

    resource = Resource.create(
        {
            "service.name": "agents",
        }
    )
    tp = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
    reader = PrometheusMetricReader()

    tp.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tp)

    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[reader])
    )
    start_http_server(port=9464)

    asyncio.run(start_dispatcher(start_agent))
