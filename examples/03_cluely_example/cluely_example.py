import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv
from getstream import AsyncStream
from getstream.models import UserRequest
from stream_agents.plugins import deepgram, elevenlabs, openai, ultralytics, getstream
from stream_agents.core import agents, cli

# Main feats:
# 1. API endpoints to create a sessions, end session
# 2. Generate 2 - 4 sentiments for new chat messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


async def main() -> None:
    """Create a simple agent and join a call."""
    call_id = str(uuid4())
    client = AsyncStream()
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    await client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    # TODO: LLM class
    agent = agents.Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user name etc for the agent
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        # tts, llm, stt more. see the realtime example for sts
        llm=openai.LLM(
            model="gpt-4o",
        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        # processors can fetch extra data, check images/audio data or transform video
        processors=[ultralytics.YOLOPoseProcessor()],
    )

    call = client.video.call("default", call_id)

    with await agent.join(call):
        await agent.edge.open_demo(call)
        await agent.llm.simple_response(text="Say hi. Explain what user shows with their hands")
        await agent.finish()  


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(main))
