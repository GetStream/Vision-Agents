import asyncio
import logging
import os
from uuid import uuid4

from dotenv import load_dotenv
from stream_agents.plugins import deepgram, elevenlabs
from stream_agents.core.processors.tavus_processor import TavusProcessor
from stream_agents.core.llm import OpenAILLM
from stream_agents.core.agents.agents import Agent
from stream_agents.core.edge.edge_transport import StreamEdge
from stream_agents.core.cli import start_dispatcher
from stream_agents.core.utils import open_demo
from stream_agents.core import logging_utils
from getstream import Stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logging_utils.initialize_logging_context()
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    # Get Tavus configuration from environment
    tavus_api_key = os.getenv("TAVUS_KEY")
    tavus_replica_id = os.getenv(
        "TAVUS_REPLICA_ID", "rfe12d8b9597"
    )  # Default from docs
    tavus_persona_id = os.getenv(
        "TAVUS_PERSONA_ID", "pdced222244b"
    )  # Default from docs

    if not tavus_api_key:
        raise ValueError("TAVUS_KEY environment variable is required")

    call_id = str(uuid4())
    token = logging_utils.set_call_context(call_id)

    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="Tavus AI Avatar Agent")

    # Create Tavus processor for AI avatar video/audio streaming
    tavus_processor = TavusProcessor(
        api_key=tavus_api_key,
        replica_id=tavus_replica_id,
        persona_id=tavus_persona_id,
        conversation_name="Stream Video Avatar Session",
        auto_create=True,  # Automatically create Tavus conversation
        auto_join=True,  # Automatically join Daily call
        audio_only=False,  # Full video avatar experience
        interval=0,  # Process every frame for real-time streaming
    )

    # Log the Tavus conversation details
    print("🎭 Tavus conversation created!")
    print(f"🔗 Conversation URL: {tavus_processor.conversation_url}")
    print(f"📺 Replica ID: {tavus_replica_id}")
    print(f"🤖 Persona ID: {tavus_persona_id}")

    # Create the agent with Tavus processor
    agent = Agent(
        edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're an AI avatar powered by Tavus technology. You can see and interact through video. Keep responses natural and conversational. You're streaming live video and audio, so be engaging and personable. Don't use special characters or formatting in speech.",
        # Enhanced LLM instructions for avatar interaction
        llm=OpenAILLM(
            model="gpt-4o",
        ),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        # Optional turn detection for better conversation flow
        # turn_detection=FalTurnDetection(),
        processors=[tavus_processor],  # Tavus processor provides AI avatar video/audio
    )

    # Create a call
    call = client.video.call("default", call_id)

    # Open the demo UI
    open_demo(call)

    print("🚀 Starting Tavus AI Avatar Agent...")
    print("💡 The agent will stream AI avatar video/audio from Tavus")
    print("🎥 Join the call to interact with your AI avatar!")

    try:
        # Have the agent join the call/room
        with await agent.join(call):
            await agent.finish()  # run till the call ends
    except Exception as e:
        print(f"❌ Error during agent execution: {e}")
    finally:
        # Clean up Tavus resources
        print("🧹 Cleaning up Tavus resources...")
        await tavus_processor.cleanup()
        print("✅ Cleanup completed!")
        logging_utils.clear_call_context(token)


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
