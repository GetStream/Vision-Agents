import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import openrouter, getstream, elevenlabs, deepgram, smart_turn

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    """Example agent using OpenRouter LLM.
    
    This example demonstrates how to use the OpenRouter plugin with a Vision Agent.
    OpenRouter provides access to multiple LLM providers through a unified API.
    
    Set OPENROUTER_API_KEY environment variable before running.
    """
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="OpenRouter AI"),
        instructions="Be helpful and friendly to the user",
        llm=openrouter.LLM(
            model="openai/gpt-4o",  # Can also use other models like anthropic/claude-3-opus
        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        turn_detection=smart_turn.TurnDetection(
            buffer_duration=2.0, confidence_threshold=0.5
        )
    )
    await agent.create_user()

    call = agent.edge.client.video.call("default", str(uuid4()))
    await agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Hello! I'm powered by OpenRouter.")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())

