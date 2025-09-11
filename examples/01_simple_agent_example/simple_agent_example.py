import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import elevenlabs, deepgram, anthropic
from stream_agents.core.agents import Agent
from stream_agents.core.edge import StreamEdge
from stream_agents.core.cli import start_dispatcher
from getstream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")



    # Create the agent
    agent = Agent(
        edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        # tts, llm, stt more. see the realtime example for sts
        #llm=openai.LLM(model="gpt-4o-mini"),
        llm=anthropic.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        # turn_detection=FalTurnDetection(api_key=os.getenv("FAL_KEY")),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    agent.edge.open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # Example 1: standardized simple response (aggregates delta/done)
        await agent.llm.simple_response(
            text="Please say verbatim: 'this is a test of the gemini realtime api.'."
        )

        # Example 2: provider-native passthrough for advanced control
        await agent.llm.native_send_realtime_input(
            text="Please say verbatim: 'this is a test using the gemini realtime api native input method.'."
        )

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
