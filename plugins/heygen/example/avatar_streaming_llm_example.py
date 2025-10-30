import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, heygen, deepgram

load_dotenv()


async def start_avatar_agent_streaming() -> None:
    """Start an agent with HeyGen avatar using streaming (non-Realtime) LLM.
    
    This example demonstrates how to use HeyGen's avatar streaming
    with a regular streaming LLM (gemini.LLM) + STT. HeyGen will handle
    both TTS and video generation based on the LLM's text output.
    
    This approach has lower latency than Realtime LLMs because:
    - Text is sent to HeyGen immediately as it's generated
    - No transcription round-trip (LLM → audio → transcription → HeyGen)
    - HeyGen handles TTS and lip-sync simultaneously
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
        
        # Use regular streaming LLM (not Realtime)
        llm=gemini.LLM("gemini-2.0-flash-exp"),
        
        # Add STT for speech input
        stt=deepgram.STT(),
        
        # Add HeyGen avatar as a video publisher
        # Note: mute_llm_audio is not needed here since gemini.LLM doesn't produce audio
        processors=[
            heygen.AvatarPublisher(
                avatar_id="default",  # Use your HeyGen avatar ID
                quality="high",       # Video quality: "low", "medium", "high"
                resolution=(1920, 1080),  # Output resolution
            )
        ]
    )
    
    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))
    
    # Join the call
    with await agent.join(call):
        # Set agent reference on avatar publisher for text event subscription
        avatar_publisher = agent.video_publishers[0]
        if hasattr(avatar_publisher, 'set_agent'):
            avatar_publisher.set_agent(agent)
        
        # Open demo UI
        await agent.edge.open_demo(call)
        
        # Keep the call running
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_avatar_agent_streaming())

