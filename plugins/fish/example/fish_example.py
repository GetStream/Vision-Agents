import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import fish, getstream, smart_turn, gemini

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    """
    Example demonstrating Fish Audio TTS and STT integration with Vision Agents.
    
    This example creates an agent that uses:
    - Fish Audio for text-to-speech (TTS)
    - Fish Audio for speech-to-text (STT)
    - GetStream for edge/real-time communication
    - Smart Turn for turn detection
    
    Requirements:
    - FISH_API_KEY environment variable
    - STREAM_API_KEY and STREAM_API_SECRET environment variables
    """
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI"),
        instructions="You're a voice AI called Elon Musk. Short replies only no special characters. Read @elon.md to stay in character",
        tts=fish.TTS(),  # Uses Fish Audio for text-to-speech
        stt=fish.STT(),  # Uses Fish Audio for speech-to-text
        llm=gemini.LLM("gemini-2.0-flash"),
        turn_detection=smart_turn.TurnDetection(buffer_duration=2.0, confidence_threshold=0.5),
    )

    await agent.create_user()

    call = agent.edge.client.video.call("default", str(uuid4()))
    await agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Whats next for space?")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())

