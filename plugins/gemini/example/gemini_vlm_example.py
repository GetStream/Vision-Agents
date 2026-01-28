"""
Gemini VLM Example

This example demonstrates how to use the Gemini VLM (Video Language Model) plugin
to create an AI agent that can see and understand video in real-time.

The agent can:
- Analyze video frames from the call participant
- Answer questions about what it sees
- Perform function calls based on visual context
- Support streaming responses with low latency

Usage:
    python gemini_vlm_example.py

Requirements:
    - GOOGLE_API_KEY environment variable set
    - STREAM_API_KEY environment variable set
    - Video-enabled call client (web, mobile, etc.)
"""

import logging
from typing import Any, Dict

from dotenv import load_dotenv
from google.genai import types

from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """
    Create an AI agent with Gemini VLM for video understanding.

    The agent uses:
    - Gemini VLM for video understanding and text generation
    - Deepgram for speech-to-text with eager turn detection
    - ElevenLabs for text-to-speech
    - Stream's edge network for low-latency video transport
    """

    # Initialize Gemini VLM with video processing capabilities
    llm = gemini.VLM(
        model="gemini-2.0-flash-exp",  # or "gemini-3-pro-preview" for advanced features
        fps=1,  # Process 1 frame per second
        frame_buffer_seconds=10,  # Keep last 10 seconds of video
        frame_format="jpeg",  # Use JPEG for smaller size
        frame_width=800,  # Resize frames to 800x600
        frame_height=600,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,  # High quality for video
        config=types.GenerateContentConfig(
            # Optional: Add built-in Gemini tools
            tools=[
                types.Tool(code_execution=types.ToolCodeExecution),  # Python code execution
                # types.Tool(google_search=types.ToolGoogleSearch),  # Web search
            ]
        ),
    )

    # Create the agent
    agent = Agent(
        edge=getstream.Edge(),  # Low latency edge network
        agent_user=User(name="Vision Assistant", id="agent"),
        instructions=(
            "You're a helpful AI assistant with vision capabilities. "
            "You can see the video feed from the user and answer questions about what you see. "
            "Keep your responses conversational and concise. "
            "When describing what you see, be specific but brief. "
            "Don't use special characters or formatting in your speech."
        ),
        processors=[],  # Add video processors here if needed
        llm=llm,
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Register custom functions that can be called by the LLM
    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        """Get weather information for the given location."""
        return await get_weather_by_location(location)

    @llm.register_function(
        description="Analyze and count objects visible in the current video frame"
    )
    async def count_objects(object_type: str) -> Dict[str, Any]:
        """
        Count specific objects in the video.
        The LLM can call this when the user asks about quantities.
        """
        return {
            "status": "success",
            "message": f"This would count {object_type} in the video frame using computer vision",
            "count": "N/A - implement with CV library",
        }

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Have the agent join a call and interact with participants.

    Args:
        agent: The agent instance to join the call.
        call_type: Type of call (e.g., "default").
        call_id: Unique identifier for the call.
    """
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call
    async with agent.join(call):
        # Send an initial greeting
        await agent.simple_response(
            "Greet the user and let them know you can see their video feed. "
            "Ask them what they'd like you to help with."
        )

        # Run until the call ends
        await agent.finish()


if __name__ == "__main__":
    # Run the agent with CLI interface
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
