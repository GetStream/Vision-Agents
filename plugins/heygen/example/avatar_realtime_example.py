import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, heygen

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def start_avatar_agent() -> None:
    """Start a HeyGen avatar agent with Gemini Realtime LLM.
    
    This example demonstrates using a HeyGen avatar with a Realtime LLM.
    HeyGen provides the lip-synced avatar video based on text transcriptions,
    while Gemini Realtime provides the audio directly.
    """
    
    # Create agent with Gemini Realtime and HeyGen avatar
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Avatar AI Assistant"),
        instructions=(
            "You are a helpful AI assistant with a virtual avatar. "
            "Keep responses conversational and natural. "
            "Be friendly and engaging."
        ),
        llm=gemini.Realtime(
            model="gemini-2.5-flash-native-audio-preview-09-2025"
        ),
        processors=[
            heygen.AvatarPublisher(
                avatar_id="default",
                quality="high",
            )
        ],
    )
    
    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))
    
    # Join call first
    with await agent.join(call):
        # Open demo UI after joining
        await agent.edge.open_demo(call)
        
        # Start the conversation
        await agent.llm.simple_response(
            text="Hello! I'm your AI assistant. How can I help you today?"
        )
        
        # Keep running until the call ends
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_avatar_agent())

