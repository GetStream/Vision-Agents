import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv
from getstream.plugins import DeepgramSTT, ElevenLabsTTS
from stream_agents.llm.llm import OpenAILLM
from stream_agents.turn_detection import FalTurnDetection
from stream_agents import Agent, Stream, StreamEdge, start_dispatcher, open_demo

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
        llm=OpenAILLM("gpt-4o"),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        turn_detection=FalTurnDetection(
            api_key=os.getenv("FAL_KEY")
        ),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # example of sending a system instruction. supports full openAI inputs
        await agent.create_response(
            "please say hi to the user and ask how their day is"
        )

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
