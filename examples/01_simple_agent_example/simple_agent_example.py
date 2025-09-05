import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import elevenlabs, deepgram, openai, silero
from stream_agents.core import agents, edge, cli
from getstream import Stream
from getstream.plugins.common.events import EventType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")

    agent = agents.Agent(
        edge=edge.StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        vad=silero.VAD(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    @agent.on(EventType.CALL_MEMBER_ADDED)
    def my_handler(event):
        agent.logger.info(f"handled event {event}")

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    agent.edge.open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # Example 1: standardized simple response (aggregates delta/done)
        await agent.llm.simple_response("Please say verbatim: 'this is a test of the OpenAI realtime api.'.")

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(start_agent))
