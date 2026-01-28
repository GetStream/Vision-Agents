"""
Gemini VLM On-Demand Example

This example demonstrates the on-demand mode of Gemini VLM where video frames
are only captured and sent when the LLM explicitly requests them via function calls.

This is more efficient than continuous buffering when you only need vision occasionally.
For example, when the user asks "what do you see?" or "what color is my shirt?",
the LLM can call capture_frame() to analyze the current video.

Usage:
    python gemini_vlm_on_demand_example.py

Key Differences from Automatic Mode:
    - fps=0: Disables automatic frame buffering
    - enable_vision_tools=True: Registers capture_frame() and analyze_video() functions
    - LLM decides when to look at video based on conversation context
    - More token-efficient for use cases that don't need constant vision

Requirements:
    - GOOGLE_API_KEY environment variable set
    - STREAM_API_KEY environment variable set
    - Video-enabled call client (web, mobile, etc.)
"""

import logging
from typing import Any, Dict

from dotenv import load_dotenv

from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """
    Create an AI agent with Gemini VLM in on-demand mode.

    The agent uses:
    - Gemini VLM with fps=0 for on-demand frame capture
    - Vision tools enabled so LLM can call capture_frame() when needed
    - Deepgram for speech-to-text
    - ElevenLabs for text-to-speech
    """

    # Initialize Gemini VLM in ON-DEMAND mode
    llm = gemini.VLM(
        model="gemini-3-flash-preview",
        fps=0,  # 🔑 KEY: Set fps=0 for on-demand mode
        enable_vision_tools=True,  # 🔑 KEY: Enable capture_frame() and analyze_video()
        frame_format="jpeg",
        frame_width=800,
        frame_height=600,
    )

    # Create the agent
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Vision Assistant", id="agent"),
        instructions=(
            "You're a helpful AI assistant with vision capabilities. "
            "You can see the video when needed by calling capture_frame() or analyze_video(). "
            "IMPORTANT: Only use vision tools when the user asks about what they're showing, "
            "what you see, or needs visual analysis. Don't call vision tools for every message. "
            "When analyzing video, be specific and descriptive. "
            "Keep responses conversational and natural."
        ),
        processors=[],
        llm=llm,
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Register custom functions
    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        """Get weather information for the given location."""
        return await get_weather_by_location(location)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Have the agent join a call and interact with participants.

    The agent will use vision tools intelligently based on the conversation.

    Examples of interactions:
    - User: "Hello!" -> Agent responds without using vision
    - User: "What do you see?" -> Agent calls capture_frame() to analyze video
    - User: "What color is my shirt?" -> Agent calls analyze_video("shirt color")
    - User: "How many fingers am I holding up?" -> Agent uses vision to count

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
            "Greet the user warmly and let them know you can see their video "
            "when they need you to. Explain that you'll use vision intelligently "
            "only when they ask about what you see. Keep it brief and friendly."
        )

        # Run until the call ends
        await agent.finish()


if __name__ == "__main__":
    # Run the agent with CLI interface
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
