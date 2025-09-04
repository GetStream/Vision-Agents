import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv

from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from plugins.deepgram.stt import DeepgramSTT
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
        llm=OpenAILLM(),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        turn_detection=FalTurnDetection(api_key=os.getenv("FAL_KEY")),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # example of sending a system instruction. supports full openAI input
        # await agent.llm.simple_response( "please say hi to the user and ask how their day is")
        # Note how you can use native APIs. create response (openAI), create message (claude) and generate_content (gemini)
        img_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        await agent.llm.create_response(model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Tell me a short poem about this image"},
                        {"type": "input_image", "image_url": f"{img_url}"},
                    ],
                }
            ]
        )

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
