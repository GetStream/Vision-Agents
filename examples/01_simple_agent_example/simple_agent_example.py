import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import elevenlabs, deepgram, openai, silero
from stream_agents.core import agents, edge, cli
from getstream import Stream
from stream_agents.core.events import EventType

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

    @agent.on(EventType.PARTICIPANT_JOINED)
    async def my_handler(participant):
        await asyncio.sleep(5)
        await agent.say(f"Hello, {participant.name}")
        agent.logger.info(f"handled event {participant}")

    @agent.on(EventType.CALL_MEMBER_ADDED)
    async def my_other_handler(participant):
        agent.logger.info(f"Call.* handled event {participant}")


    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI

    # Have the agent join the call/room
    with await agent.join(call):
        # Example 1: standardized simple response (aggregates delta/done)
        await asyncio.sleep(2)
        agent.edge.open_demo(call)
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(start_agent))
