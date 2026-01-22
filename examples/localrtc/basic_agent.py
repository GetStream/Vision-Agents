"""Basic Local RTC Agent Example.

This example demonstrates the fundamental usage of Local RTC with Gemini Realtime.
It shows how to create an agent that uses local audio/video devices instead of
external RTC infrastructure like Stream or LiveKit.

Key Features:
- Local device access (microphone, camera, speakers)
- Gemini Realtime API for voice interaction
- Direct audio/video streaming without external services
- Simple agent lifecycle management

Use Cases:
- Development and testing without cloud dependencies
- Local prototyping of voice AI agents
- Privacy-focused applications that keep data local
- Desktop applications with direct device access
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, localrtc

logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Required: GOOGLE_API_KEY for Gemini
load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with Local RTC and Gemini Realtime.

    This function initializes an Agent that uses:
    - localrtc.Edge(): Manages local audio/video devices (microphone, camera, speakers)
    - gemini.Realtime(): Provides multimodal AI with voice interaction

    The agent can see video, hear audio, and respond with synthesized speech,
    all processed locally without external RTC infrastructure.

    Returns:
        Agent: Configured agent ready to join a call
    """
    # Create the Local RTC Edge transport
    # This handles all local device I/O: capturing from microphone/camera
    # and playing audio to speakers
    edge = localrtc.Edge(
        audio_device="default",  # Use system default microphone
        video_device=0,          # Use first available camera
        speaker_device="default", # Use system default speakers
        sample_rate=16000,       # 16kHz audio (standard for voice)
        channels=1,              # Mono audio
    )

    # Optional: List available devices before creating the agent
    # Uncomment to see what devices are available on your system:
    # devices = localrtc.Edge.list_devices()
    # print("Available audio inputs:", devices["audio_inputs"])
    # print("Available audio outputs:", devices["audio_outputs"])
    # print("Available video inputs:", devices["video_inputs"])

    # Create the agent with Local RTC and Gemini Realtime
    agent = Agent(
        edge=edge,  # Use Local RTC for device I/O
        agent_user=User(name="Local AI Assistant", id="agent"),
        instructions=(
            "You're a helpful voice AI assistant running locally. "
            "Keep your responses concise and conversational. "
            "You can see the user through the camera and hear them through the microphone."
        ),
        llm=gemini.Realtime(
            # Optional: Configure video frame rate (default is 1 fps)
            # fps=3,  # Send 3 video frames per second to Gemini
            # Optional: Configure other Gemini Realtime parameters
            # model="gemini-2.0-flash-exp",  # Specify model version
        ),
        processors=[],  # No additional processors in this basic example
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a call and start the agent.

    This function handles the agent's lifecycle:
    1. Creates a call/room for the agent to join
    2. Joins the call (starts device capture and streaming)
    3. Sends an initial message to greet the user
    4. Runs until the call ends

    Args:
        agent: The configured agent instance
        call_type: Type of call (e.g., "default")
        call_id: Unique identifier for this call
        **kwargs: Additional call configuration
    """
    # Create a call object
    # For Local RTC, this is a lightweight local construct (no external API calls)
    call = await agent.create_call(call_type, call_id)

    # Join the call using a context manager
    # This automatically:
    # - Starts capturing audio/video from local devices
    # - Begins streaming to Gemini Realtime
    # - Handles cleanup when the context exits
    async with agent.join(call):
        # Send an initial greeting to the user
        # This triggers Gemini to generate and speak a response
        await agent.simple_response("Say hello and introduce yourself briefly.")

        # Alternative: Use Gemini's native API for more control
        # await agent.llm.send_realtime_input(text="Hello! How can I help you today?")

        # Run the agent until the call ends
        # This keeps the agent active, processing audio/video and responding
        await agent.finish()


# Main entry point
if __name__ == "__main__":
    """Run the Local RTC agent.

    Usage:
        python basic_agent.py

    This will start the agent and open a local session.
    The agent will:
    1. Access your default microphone and camera
    2. Listen for your voice input
    3. Process video from your camera
    4. Respond through your speakers

    Requirements:
    - GOOGLE_API_KEY environment variable (in .env file)
    - Working microphone, camera, and speakers
    - Python packages: vision-agents-stream, vision-agents-plugin-gemini, vision-agents-plugin-localrtc

    To stop the agent, use Ctrl+C or close the application.
    """
    # The Runner and AgentLauncher handle the agent lifecycle
    # This provides a CLI interface for starting/stopping the agent
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
