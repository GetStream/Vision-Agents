import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import aws, getstream, cartesia, deepgram, smart_turn

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI"),
        instructions="Be nice to the user",
        llm=aws.LLM(model="qwen.qwen3-32b-v1:0"),
        tts=cartesia.TTS(),
        stt=deepgram.STT(),
        turn_detection=smart_turn.TurnDetection(buffer_in_seconds=2.0, confidence_threshold=0.5),
        # Enable turn detection with FAL/ Smart turn
    )
    await agent.create_user()

    call = agent.edge.client.video.call("default", str(uuid4()))
    await agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Say hi")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())
