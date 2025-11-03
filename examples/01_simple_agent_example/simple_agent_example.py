import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import deepgram, getstream, gemini, vogent, elevenlabs
from vision_agents.core.profiling import Profiler

load_dotenv()


async def start_agent() -> None:
    llm = gemini.LLM("gemini-2.0-flash")
    # create an agent to run with Stream's edge, openAI llm
    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(
            name="My happy AI friend", id="agent"
        ),  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
        # llm with tts & stt. if you use a realtime (sts capable) llm the tts, stt and vad aren't needed
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        turn_detection=vogent.TurnDetection(),
        profiler=Profiler(),
        # vad=silero.VAD(),
        # realtime version (vad, tts and stt not needed)
        # llm=openai.Realtime()
    )

    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))

    # Have the agent join the call/room
    with await agent.join(call):
        # Example 1: standardized simple response
        # await agent.llm.simple_response("chat with the user about the weather.")
        # Example 2: use native openAI create response
        # await llm.create_response(input=[
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "input_text", "text": "Tell me a short poem about this image"},
        #             {"type": "input_image", "image_url": f"https://images.unsplash.com/photo-1757495361144-0c2bfba62b9e?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
        #         ],
        #     }
        # ],)

        # await agent.say("Hello, how are you?")
        # await asyncio.sleep(5)

        # Open the demo UI
        await agent.edge.open_demo(call)

        await agent.simple_response("tell me something interesting in a short sentence")

        # run till the call ends
        await agent.finish()


def setup_telemetry():
    import atexit
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource.create(
        {
            "service.name": "agents",
        }
    )
    tp = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)

    tp.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tp)

    def _flush_and_shutdown():
        tp.force_flush()
        tp.shutdown()

    atexit.register(_flush_and_shutdown)


if __name__ == "__main__":
    # setup_telemetry()
    asyncio.run(start_agent())
