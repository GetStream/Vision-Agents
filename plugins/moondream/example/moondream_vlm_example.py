import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import deepgram, getstream, vogent, elevenlabs, moondream
from vision_agents.core.events import CallSessionParticipantJoinedEvent
import os

load_dotenv()

async def start_agent() -> None:
    llm = moondream.CloudVLM(
        api_key=os.getenv("MOONDREAM_API_KEY"),
        conf_threshold=0.3,
    )
    # create an agent to run with Stream's edge, openAI llm
    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(
            name="My happy AI friend", id="agent"
        ),
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        turn_detection=vogent.TurnDetection(),
    )

    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(2)
            await agent.simple_response("Describe what you currently see")


    # Have the agent join the call/room
    with await agent.join(call):
        # Open the demo UI
        await agent.edge.open_demo(call)
        # run till the call ends
        await agent.finish()

if __name__ == "__main__":
    # setup_telemetry()
    asyncio.run(start_agent())
