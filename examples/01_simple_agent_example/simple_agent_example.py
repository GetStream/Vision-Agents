import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.core.edge.types import User
from stream_agents.plugins import elevenlabs, deepgram, openai, silero, getstream
from stream_agents.core import agents, cli
from stream_agents.core.events import CallSessionParticipantJoinedEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

'''
TODO:
- show function calling
'''

async def start_agent() -> None:

    llm = openai.LLM(model="gpt-4o-mini")
    # create an agent to run with Stream's edge, openAI llm
    agent = agents.Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(name="My happy AI friend", id="agent"),  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
        # llm with tts & stt. if you use a realtime (sts capable) llm the tts, stt and vad aren't needed
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        vad=silero.VAD(),
        # realtime version (vad, tts and stt not needed)
        # llm=openai.Realtime()
    )

    await agent.create_user()

    @agent.subscribe
    async def my_handler(event: CallSessionParticipantJoinedEvent):
        await agent.say(f"Hello, {event.participant.user.name}")

    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))

    # Open the demo UI
    agent.edge.open_demo(call)

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

        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(start_agent))
