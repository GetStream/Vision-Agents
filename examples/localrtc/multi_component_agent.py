"""Multi-Component Agent with Local RTC Example.

This example demonstrates how to build a sophisticated voice AI agent using Local RTC
with separate LLM, STT (Speech-to-Text), and TTS (Text-to-Speech) components.
Unlike the basic example which uses Gemini Realtime (an all-in-one solution),
this shows how to compose individual components for more flexibility and control.

Component Architecture:
=======================
The agent is built from three independent components that work together:

1. LLM (Language Model): openai.LLM()
   - Provides the intelligence and reasoning capabilities
   - Processes text input and generates text responses
   - Uses OpenAI's GPT models for natural language understanding
   - Can be swapped with other LLM providers (gemini.LLM(), etc.)

2. STT (Speech-to-Text): deepgram.STT()
   - Converts user's speech to text in real-time
   - Handles audio input from the microphone
   - Detects when user stops speaking (turn detection)
   - Sends transcribed text to the LLM for processing

3. TTS (Text-to-Speech): elevenlabs.TTS()
   - Converts LLM's text responses to natural speech
   - Generates audio output for the speakers
   - Provides high-quality voice synthesis
   - Can be configured with different voices and models

4. Edge Transport: localrtc.Edge()
   - Manages local device access (microphone, camera, speakers)
   - Handles audio/video streaming without external infrastructure
   - Configured with specific devices for precise control
   - Runs entirely locally without cloud dependencies

Data Flow:
==========
User speaks → Microphone → STT (Deepgram) → Text → LLM (OpenAI) → Text → TTS (ElevenLabs) → Speakers

Benefits of Multi-Component Architecture:
==========================================
- Flexibility: Choose best-in-class providers for each component
- Customization: Fine-tune each component independently
- Cost Control: Optimize costs by selecting appropriate tiers for each service
- Experimentation: Easy to swap components for testing and comparison
- Scalability: Components can be scaled independently based on needs

Use Cases:
==========
- Production applications requiring specific STT/TTS providers
- Cost-optimized solutions (e.g., cheaper STT with premium TTS)
- Multi-language support with specialized STT models
- Custom voice branding with specific TTS voices
- A/B testing different component combinations
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, localrtc, openai

logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Required:
# - OPENAI_API_KEY for OpenAI LLM
# - DEEPGRAM_API_KEY for Deepgram STT
# - ELEVENLABS_API_KEY for ElevenLabs TTS
load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with Local RTC and separate LLM, STT, and TTS components.

    This function demonstrates the multi-component architecture where each piece
    of the voice AI pipeline is independently configured:

    Component Configuration:
    ------------------------
    1. Local RTC Edge: Manages local audio/video devices
       - audio_device="default": System default microphone
       - speaker_device="default": System default speakers
       - video_device=0: First available camera (optional, for video support)
       - sample_rate=16000: 16kHz audio input (output auto-configured to 24kHz)
       - channels=1: Mono audio (sufficient for voice, reduces bandwidth)

    2. OpenAI LLM: Provides language understanding and generation
       - model="gpt-4o-mini": Fast, cost-effective model
       - Alternative: "gpt-4o" for more advanced reasoning
       - Processes text from STT and generates response text for TTS

    3. Deepgram STT: Converts speech to text
       - eager_turn_detection=True: Faster responses by detecting pauses quickly
       - Transcribes audio from microphone to text for LLM
       - Handles real-time streaming transcription

    4. ElevenLabs TTS: Converts text to speech
       - model_id="eleven_flash_v2_5": Fast, high-quality voice synthesis
       - Alternative: "eleven_turbo_v2_5" for even lower latency
       - Generates natural-sounding speech from LLM responses

    Device Selection:
    -----------------
    To list available devices before creating the agent:
        devices = localrtc.Edge.list_devices()
        print("Available audio inputs:", devices["audio_inputs"])
        print("Available audio outputs:", devices["audio_outputs"])
        print("Available video inputs:", devices["video_inputs"])

    Then configure Edge with specific device indices or names:
        edge = localrtc.Edge(
            audio_device=1,  # Use second microphone
            speaker_device="USB Audio Device",  # Use specific speaker
            video_device=0,  # Use first camera
        )

    Returns:
        Agent: Configured agent with multi-component architecture
    """
    # Create the Local RTC Edge transport
    # This manages all local device I/O: audio capture, video capture, and audio playback
    edge = localrtc.Edge(
        audio_device="default",  # Use system default microphone
        video_device=0,  # Use first available camera (optional for this voice-focused example)
        speaker_device="default",  # Use system default speakers
        sample_rate=16000,  # Input: 16kHz for voice (Output: auto 24kHz for compatibility)
        channels=1,  # Mono audio (sufficient for voice, more efficient than stereo)
    )

    # Optional: List and select specific devices
    # Uncomment to see available devices and configure Edge with specific choices:
    # devices = localrtc.Edge.list_devices()
    # print("Available audio inputs:")
    # for device in devices["audio_inputs"]:
    #     print(f"  {device['index']}: {device['name']}")
    # print("\nAvailable audio outputs:")
    # for device in devices["audio_outputs"]:
    #     print(f"  {device['index']}: {device['name']}")
    # print("\nAvailable video inputs:")
    # for device in devices["video_inputs"]:
    #     print(f"  {device['index']}: {device['name']}")

    # Create the Language Model (LLM)
    # This provides the intelligence and reasoning capabilities
    llm = openai.LLM(
        model="gpt-4o-mini"  # Fast, cost-effective model
        # Alternative models:
        # model="gpt-4o"  # More advanced reasoning, higher cost
        # model="gpt-4-turbo"  # Good balance of speed and capability
    )

    # Create the Speech-to-Text (STT) component
    # This converts user's speech from the microphone into text
    stt = deepgram.STT(
        eager_turn_detection=True  # Detect pauses quickly for faster responses
        # Note: eager_turn_detection trades accuracy for speed
        # Set to False for better accuracy at the cost of slight latency
    )

    # Create the Text-to-Speech (TTS) component
    # This converts the LLM's text responses into natural speech
    tts = elevenlabs.TTS(
        model_id="eleven_flash_v2_5"  # Fast, high-quality voice synthesis
        # Alternative models:
        # model_id="eleven_turbo_v2_5"  # Even lower latency
        # model_id="eleven_multilingual_v2"  # For multi-language support
        # You can also specify a voice_id parameter to use specific voices
    )

    # Create the agent with all components
    agent = Agent(
        edge=edge,  # Local RTC for device I/O
        agent_user=User(name="Local AI Assistant", id="agent"),
        instructions=(
            "You're a helpful voice AI assistant running locally with multiple components. "
            "Keep your responses concise and conversational. "
            "You can hear the user through the microphone and respond through the speakers. "
            "If a camera is available, you may also see the user."
        ),
        llm=llm,  # OpenAI for language processing
        stt=stt,  # Deepgram for speech-to-text
        tts=tts,  # ElevenLabs for text-to-speech
        processors=[],  # No additional processors in this example
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a call and start the agent.

    This function handles the agent's lifecycle:
    1. Creates a call/room for the agent to join
    2. Joins the call (starts device capture and streaming)
    3. Sends an initial message to greet the user
    4. Runs until the call ends

    The multi-component pipeline works as follows:
    - User speaks into microphone
    - Audio is captured by Local RTC Edge
    - STT (Deepgram) transcribes audio to text
    - LLM (OpenAI) processes text and generates response
    - TTS (ElevenLabs) converts response text to speech
    - Audio is played through speakers via Local RTC Edge

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
    # - Begins STT transcription pipeline
    # - Enables LLM processing
    # - Activates TTS synthesis
    # - Handles cleanup when the context exits
    async with agent.join(call):
        # Send an initial greeting to the user
        # This flows through the entire pipeline: LLM → TTS → Speakers
        await agent.simple_response("Say hello and introduce yourself briefly.")

        # Run the agent until the call ends
        # This keeps all components active:
        # - STT continuously transcribes user speech
        # - LLM processes transcriptions and generates responses
        # - TTS synthesizes responses to audio
        # - Edge manages all device I/O
        await agent.finish()


# Main entry point
if __name__ == "__main__":
    """Run the Local RTC multi-component agent.

    Usage:
        python multi_component_agent.py

    This will start the agent with separate LLM, STT, and TTS components.
    The agent will:
    1. Access your default microphone and speakers
    2. Listen for your voice input via Deepgram STT
    3. Process your speech with OpenAI LLM
    4. Respond through your speakers using ElevenLabs TTS

    Requirements:
    - Environment variables (in .env file):
        - OPENAI_API_KEY: OpenAI API key for LLM
        - DEEPGRAM_API_KEY: Deepgram API key for STT
        - ELEVENLABS_API_KEY: ElevenLabs API key for TTS
    - Hardware:
        - Working microphone
        - Working speakers
        - Optional: Camera for video support
    - Python packages:
        - vision-agents-stream
        - vision-agents-plugin-localrtc
        - vision-agents-plugin-openai
        - vision-agents-plugin-deepgram
        - vision-agents-plugin-elevenlabs

    Architecture Benefits:
    - Each component can be independently configured and optimized
    - Easy to swap providers (e.g., use gemini.LLM instead of openai.LLM)
    - Fine-grained control over latency, quality, and cost trade-offs
    - Supports mixing providers for best results (e.g., cheap STT + premium TTS)

    To stop the agent, use Ctrl+C or close the application.
    """
    # The Runner and AgentLauncher handle the agent lifecycle
    # This provides a CLI interface for starting/stopping the agent
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
