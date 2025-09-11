import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import elevenlabs, deepgram, xai
from stream_agents.core import agents, edge, cli, utils
from getstream import Stream

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
        # tts, llm, stt more. see the realtime example for sts
        #llm=openai.LLM(model="gpt-4o-mini"),
        llm=xai.LLM(model="grok-4"),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    agent.edge.open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # Example 1: standardized simple response (aggregates delta/done)
        await agent.llm.simple_response("Introduce yourself as the mighty Grok capable of streaming AI responses")

        # Example 2: streaming response
        await agent.llm.create_response(
            input="You are a conversational Grok",
            stream=True
        )

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(start_agent))
