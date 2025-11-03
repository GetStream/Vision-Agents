import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, heygen, deepgram

load_dotenv()


async def start_avatar_agent() -> None:
    """Start an agent with HeyGen avatar using streaming LLM.
    
    This example demonstrates how to use HeyGen's avatar streaming
    with a regular streaming LLM. This approach has much lower latency
    than using Realtime LLMs because text goes directly to HeyGen
    without any transcription round-trip.
    
    HeyGen handles all TTS and lip-sync based on the LLM's text output.
    """
    
    # Create agent with HeyGen avatar and streaming LLM
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="AI Assistant with Avatar",
            id="agent"
        ),
        instructions=(
            "You're a friendly and helpful AI assistant. "
            "Keep your responses conversational and engaging. "
            "Don't use special characters or formatting."
        ),
        
        # Use regular streaming LLM (not Realtime) for lower latency
        llm=gemini.LLM("gemini-2.0-flash-exp"),
        
        # Add STT for speech input
        stt=deepgram.STT(),
        
        # Add HeyGen avatar as a video publisher
        # Note: mute_llm_audio is not needed since streaming LLM doesn't produce audio
        processors=[
            heygen.AvatarPublisher(
                avatar_id="default",  # Use your HeyGen avatar ID
                quality="high",       # Video quality: "low", "medium", "high"
                resolution=(1920, 1080),  # Output resolution
                mute_llm_audio=False,  # Not needed for streaming LLM
            )
        ]
    )
    
    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))
    
    # Join the call
    with await agent.join(call):
        # Open demo UI
        await agent.edge.open_demo(call)
        
        # Keep the call running
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_avatar_agent())

