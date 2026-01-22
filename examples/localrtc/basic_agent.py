"""Basic Local RTC Agent Example - Best Practices Demonstration.

This example demonstrates the recommended patterns for Vision Agents WebRTC integration.
It serves as the canonical reference for integrating external RTC systems with Vision Agents.

## API Best Practices Demonstrated

### 1. EdgeTransport Configuration
- Use explicit keyword arguments for all configuration
- Choose appropriate device identifiers (string names or integer indices)
- Configure audio parameters for optimal quality (16kHz mono for voice input)
- Let the framework handle audio format negotiation automatically

### 2. Standard Calling Conventions
- `edge.join(agent, room_id=..., room_type=...)` - Pass agent for audio negotiation
- `edge.publish_tracks(room, audio_track=..., video_track=...)` - Use explicit room parameter
- `edge.create_call(call_type, call_id)` - Create rooms with clear identifiers

### 3. Agent Lifecycle Management
- Use `async with agent.join(call)` for automatic cleanup
- Call `agent.create_call()` before joining
- Use `agent.simple_response()` for initial interaction
- Call `agent.finish()` to run until completion

### 4. Audio Format Negotiation
- Audio output format is automatically negotiated with LLM provider
- Input format is configured via EdgeTransport (e.g., 16kHz mono)
- Output format matches LLM requirements (e.g., 24kHz mono for Gemini)

## Key Features
- Local device access (microphone, camera, speakers)
- Gemini Realtime API for voice interaction
- Automatic audio format negotiation
- Direct audio/video streaming without external RTC services
- Production-ready error handling and cleanup

## Use Cases
- Development and testing without cloud dependencies
- Local prototyping of voice AI agents
- Privacy-focused applications that keep data local
- Desktop applications with direct device access
- Reference implementation for external RTC integration

## Public API Reference
This example uses the following public APIs:
- `localrtc.Edge` - EdgeTransport implementation for local devices
- `localrtc.Edge.list_devices()` - Device enumeration
- `Agent` - Core agent class from vision_agents.core
- `User` - User representation for participants
- `gemini.Realtime` - LLM provider with audio format requirements
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

    This function demonstrates best practices for Vision Agents WebRTC integration:
    - Uses recommended EdgeTransport API (LocalEdge)
    - Follows standard calling conventions with keyword arguments
    - Enables automatic audio format negotiation
    - Provides clear configuration with sensible defaults

    The agent uses:
    - localrtc.Edge(): Manages local audio/video devices (microphone, camera, speakers)
    - gemini.Realtime(): Provides multimodal AI with voice interaction

    The agent can see video, hear audio, and respond with synthesized speech,
    all processed locally without external RTC infrastructure.

    Returns:
        Agent: Configured agent ready to join a call

    Note:
        Audio format is automatically negotiated when the agent joins a call.
        The edge transport queries the LLM's requirements and configures output
        accordingly (e.g., 24kHz mono for Gemini).
    """
    # BEST PRACTICE: Create the Local RTC Edge transport with explicit parameters
    # This handles all local device I/O: capturing from microphone/camera
    # and playing audio to speakers
    edge = localrtc.Edge(
        audio_device="default",   # Use system default microphone
        video_device=0,           # Use first available camera (integer index)
        speaker_device="default", # Use system default speakers
        sample_rate=16000,        # Audio capture: 16kHz for voice input
        channels=1,               # Mono audio capture
        # Note: Output format (24kHz mono for Gemini) is automatically negotiated
        # custom_pipeline=None,   # Optional: Use GStreamer pipelines for advanced control
    )

    # BEST PRACTICE: Discover available devices before deployment
    # Uncomment to enumerate devices on your system:
    # devices = localrtc.Edge.list_devices()
    # print("Available audio inputs:", devices["audio_inputs"])
    # print("Available audio outputs:", devices["audio_outputs"])
    # print("Available video inputs:", devices["video_inputs"])
    # # Then use device index or name:
    # # edge = localrtc.Edge(audio_device=devices["audio_inputs"][0]["index"])

    # BEST PRACTICE: Create the agent with explicit configuration
    agent = Agent(
        edge=edge,  # EdgeTransport instance for device I/O
        agent_user=User(
            name="Local AI Assistant",
            id="agent",
            # image="https://example.com/avatar.png",  # Optional avatar
        ),
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

    This function demonstrates the recommended agent lifecycle pattern:
    1. Creates a call/room using the EdgeTransport API
    2. Joins the call (starts device capture and audio format negotiation)
    3. Sends an initial message to greet the user
    4. Runs until the call ends

    BEST PRACTICES DEMONSTRATED:
    - Uses agent.create_call() for standardized room creation
    - Uses async context manager for automatic cleanup
    - Enables audio format negotiation by passing agent to join()
    - Uses agent.simple_response() for initial interaction
    - Calls agent.finish() to run until completion

    Args:
        agent: The configured agent instance
        call_type: Type of call (e.g., "default", "audio", "video")
        call_id: Unique identifier for this call
        **kwargs: Additional call configuration

    Note:
        The join() call automatically negotiates audio format with the LLM
        provider (e.g., 24kHz mono for Gemini) when the agent is passed.
    """
    # BEST PRACTICE: Create a call using the agent's create_call method
    # For Local RTC, this creates a lightweight LocalRoom (no external API calls)
    # For Stream RTC, this creates a GetStream call in the cloud
    call = await agent.create_call(call_type, call_id)

    # BEST PRACTICE: Use async context manager for automatic cleanup
    # The agent.join(call) context manager:
    # - Negotiates audio format with the LLM provider
    # - Starts capturing audio/video from local devices
    # - Begins streaming to the LLM (e.g., Gemini Realtime)
    # - Automatically cleans up resources when the context exits
    async with agent.join(call):
        # BEST PRACTICE: Send an initial greeting to engage the user
        # This triggers the LLM to generate and speak a response
        await agent.simple_response("Say hello and introduce yourself briefly.")

        # Alternative approaches for initial interaction:
        # 1. Use LLM's native API for more control:
        #    await agent.llm.send_realtime_input(text="Hello! How can I help you today?")
        #
        # 2. Send text without voice response:
        #    await agent.send_text("Hello! I'm listening.")
        #
        # 3. Wait for user to speak first (no initial greeting):
        #    # Just call agent.finish() without sending a message

        # BEST PRACTICE: Run the agent until the call ends
        # This keeps the agent active, processing audio/video and responding
        # The agent will continue until:
        # - The user disconnects
        # - The context manager exits
        # - An error occurs
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
