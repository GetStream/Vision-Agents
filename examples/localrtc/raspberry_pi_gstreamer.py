"""Raspberry Pi GStreamer Pipeline Example.

This example demonstrates how to use custom GStreamer pipelines with Local RTC
on Raspberry Pi, utilizing ALSA for audio and V4L2 for video capture.

Raspberry Pi Device Configuration:
===================================
The Raspberry Pi uses specific Linux device interfaces:
- ALSA (Advanced Linux Sound Architecture) for audio devices
- V4L2 (Video4Linux2) for camera/video devices

This example shows how to configure GStreamer pipelines to access these devices
directly, which is essential for embedded systems where device paths and names
may differ from desktop systems.

Key Features:
=============
- Custom GStreamer pipelines for Raspberry Pi hardware
- ALSA audio input/output configuration
- V4L2 video capture from Raspberry Pi Camera or USB webcams
- Device path discovery and configuration
- Optimized settings for embedded performance

Use Cases:
==========
- Raspberry Pi AI assistant projects
- Edge computing voice/video applications
- Embedded systems with custom hardware
- IoT devices with audio/video capabilities
- Projects requiring direct hardware control

Hardware Requirements:
======================
- Raspberry Pi (any model with camera/audio support)
- Raspberry Pi Camera Module or USB webcam
- USB microphone or audio HAT
- Speakers or audio output device
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


def find_raspberry_pi_devices() -> None:
    """Display instructions for finding device paths on Raspberry Pi.

    This function provides guidance on how to discover audio and video devices
    on a Raspberry Pi system using standard Linux command-line tools.

    Device Discovery Commands:
    ==========================

    1. List ALSA Audio Devices:
       $ arecord -L
       Lists all audio input devices with their ALSA identifiers.
       Common outputs:
       - hw:0,0 - First hardware device, first subdevice
       - hw:1,0 - Second hardware device (e.g., USB mic)
       - plughw:0,0 - Hardware device with automatic format conversion

    2. List ALSA Output Devices:
       $ aplay -L
       Lists all audio output devices with their ALSA identifiers.
       Common outputs:
       - hw:0,0 - Onboard audio (headphone jack)
       - hw:1,0 - USB audio device or HDMI audio

    3. List V4L2 Video Devices:
       $ v4l2-ctl --list-devices
       Lists all video capture devices with their device paths.
       Common outputs:
       - /dev/video0 - Raspberry Pi Camera Module
       - /dev/video1 - USB webcam

    4. Get Video Device Capabilities:
       $ v4l2-ctl -d /dev/video0 --all
       Shows detailed information about a specific video device including
       supported formats, resolutions, and frame rates.

    5. Test ALSA Audio Capture:
       $ arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 test.wav
       Records a test audio file to verify microphone is working.

    6. Test ALSA Audio Playback:
       $ aplay -D hw:0,0 test.wav
       Plays a test audio file to verify speaker is working.

    Example Device Paths:
    =====================
    - Raspberry Pi Camera: /dev/video0
    - USB Webcam: /dev/video1
    - USB Microphone: hw:1,0
    - Built-in Audio Out: hw:0,0
    - HDMI Audio: hw:2,0

    Notes:
    ======
    - Device paths may vary depending on your Raspberry Pi model and peripherals
    - Device numbers (hw:X,Y) can change if you add/remove USB devices
    - Always verify your device paths before running the agent
    - Use 'plughw:' prefix for automatic format conversion if needed
    """
    print("""
===========================================
Finding Device Paths on Raspberry Pi
===========================================

AUDIO INPUT (Microphone):
--------------------------
Run: arecord -L
Look for entries like:
  - hw:1,0 (USB microphone)
  - plughw:1,0 (USB mic with format conversion)

AUDIO OUTPUT (Speakers):
------------------------
Run: aplay -L
Look for entries like:
  - hw:0,0 (headphone jack)
  - hw:2,0 (HDMI audio)

VIDEO INPUT (Camera):
---------------------
Run: v4l2-ctl --list-devices
Look for entries like:
  - /dev/video0 (Raspberry Pi Camera)
  - /dev/video1 (USB webcam)

VERIFY DEVICE CAPABILITIES:
---------------------------
Run: v4l2-ctl -d /dev/video0 --all
Shows supported formats and resolutions

TEST AUDIO CAPTURE:
-------------------
Run: arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 test.wav
Records 16kHz mono audio (Ctrl+C to stop)

TEST AUDIO PLAYBACK:
--------------------
Run: aplay -D hw:0,0 test.wav
Plays the recorded test file

===========================================
    """)


async def create_agent(**kwargs) -> Agent:
    """Create an agent with custom GStreamer pipelines for Raspberry Pi.

    This agent is configured to use GStreamer pipelines that directly access
    Raspberry Pi hardware through ALSA and V4L2 interfaces. This approach
    provides maximum control and compatibility with embedded systems.

    GStreamer Pipeline Configuration:
    =================================

    Audio Source (ALSA):
    --------------------
    - alsasrc device=hw:1,0
      Captures audio from ALSA device hw:1,0 (typically USB microphone)
      - hw:1,0 format: hardware card 1, device 0
      - Change to your device from 'arecord -L'

    - audioconvert
      Converts audio format as needed for processing

    - audioresample
      Resamples audio to match the required sample rate (16kHz)

    Video Source (V4L2):
    --------------------
    - v4l2src device=/dev/video0
      Captures video from V4L2 device (Raspberry Pi Camera or USB webcam)
      - /dev/video0: First video device (usually Pi Camera)
      - /dev/video1: Second video device (usually USB webcam)
      - Find your device with 'v4l2-ctl --list-devices'

    - videoconvert
      Converts video format for processing

    Audio Sink (ALSA):
    ------------------
    - alsasink device=hw:0,0
      Outputs audio to ALSA device hw:0,0 (typically headphone jack)
      - hw:0,0: Onboard audio output
      - hw:2,0: HDMI audio output
      - Change to your device from 'aplay -L'

    Device Path Configuration:
    ==========================
    IMPORTANT: Update these paths based on your Raspberry Pi setup!

    1. Find your audio input device:
       $ arecord -L
       Example output: hw:1,0 (USB microphone)

    2. Find your audio output device:
       $ aplay -L
       Example output: hw:0,0 (headphone jack)

    3. Find your video device:
       $ v4l2-ctl --list-devices
       Example output: /dev/video0 (Raspberry Pi Camera)

    4. Update the custom_pipeline dictionary below with your device paths

    Performance Notes:
    ==================
    - Use lower video frame rates on older Pi models (fps=1 or fps=0.5)
    - Consider disabling video on Pi Zero for better performance
    - Monitor CPU usage and adjust settings accordingly
    - GStreamer provides hardware acceleration on supported Pi models

    Returns:
        Agent: Configured agent with Raspberry Pi GStreamer pipelines
    """
    # IMPORTANT: Update these device paths to match your Raspberry Pi configuration!
    # Use the commands documented above to find your specific device paths.

    # Custom GStreamer pipeline configuration for Raspberry Pi
    custom_pipeline = {
        # Audio input: ALSA source
        # Replace "hw:1,0" with your microphone device from 'arecord -L'
        # Common options:
        #   - hw:1,0 (USB microphone on card 1)
        #   - hw:0,0 (built-in microphone if available)
        #   - plughw:1,0 (USB mic with automatic format conversion)
        "audio_source": "alsasrc device=hw:1,0 ! audioconvert ! audioresample",

        # Video input: V4L2 source
        # Replace "/dev/video0" with your camera device from 'v4l2-ctl --list-devices'
        # Common options:
        #   - /dev/video0 (Raspberry Pi Camera Module)
        #   - /dev/video1 (USB webcam)
        #   - /dev/video2 (second USB webcam)
        "video_source": "v4l2src device=/dev/video0 ! videoconvert",

        # Audio output: ALSA sink
        # Replace "hw:0,0" with your speaker device from 'aplay -L'
        # Common options:
        #   - hw:0,0 (headphone jack / built-in audio)
        #   - hw:2,0 (HDMI audio output)
        #   - plughw:0,0 (built-in audio with format conversion)
        "audio_sink": "alsasink device=hw:0,0",
    }

    # Alternative pipeline configurations for different Raspberry Pi setups:

    # Example 1: USB microphone + HDMI audio output + USB webcam
    # custom_pipeline = {
    #     "audio_source": "alsasrc device=hw:1,0 ! audioconvert ! audioresample",
    #     "video_source": "v4l2src device=/dev/video1 ! videoconvert",
    #     "audio_sink": "alsasink device=hw:2,0",
    # }

    # Example 2: Built-in Pi Camera + USB audio HAT
    # custom_pipeline = {
    #     "audio_source": "alsasrc device=hw:1,0 ! audioconvert ! audioresample",
    #     "video_source": "v4l2src device=/dev/video0 ! videoconvert",
    #     "audio_sink": "alsasink device=hw:1,0",
    # }

    # Example 3: Using plughw for automatic format conversion (more compatible)
    # custom_pipeline = {
    #     "audio_source": "alsasrc device=plughw:1,0 ! audioconvert ! audioresample",
    #     "video_source": "v4l2src device=/dev/video0 ! videoconvert",
    #     "audio_sink": "alsasink device=plughw:0,0",
    # }

    # Create the Local RTC Edge transport with custom GStreamer pipelines
    edge = localrtc.Edge(
        sample_rate=16000,       # 16kHz audio (standard for voice)
        channels=1,              # Mono audio (sufficient for voice)
        custom_pipeline=custom_pipeline,  # Use our Raspberry Pi GStreamer configuration
    )

    # Note: When using custom_pipeline, the audio_device, video_device, and
    # speaker_device parameters are ignored. All device configuration happens
    # in the GStreamer pipeline strings above.

    # Optional: Display device discovery instructions
    # Uncomment the line below to see how to find device paths on your Pi:
    # find_raspberry_pi_devices()

    # Create the agent with Local RTC and Gemini Realtime
    agent = Agent(
        edge=edge,
        agent_user=User(name="Raspberry Pi AI Assistant", id="agent"),
        instructions=(
            "You're a helpful voice AI assistant running on a Raspberry Pi. "
            "Keep your responses concise and conversational. "
            "You can see the user through the camera and hear them through the microphone. "
            "You're optimized for embedded systems and edge computing."
        ),
        llm=gemini.Realtime(
            # Use lower frame rate on Raspberry Pi to reduce CPU load
            # fps=1 sends 1 video frame per second to Gemini
            # Lower this to 0.5 for older Pi models or disable video entirely
            fps=1,
        ),
        processors=[],  # No additional processors in this example
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a call and start the agent.

    This function handles the agent's lifecycle on Raspberry Pi:
    1. Creates a call/room for the agent to join
    2. Joins the call (starts GStreamer pipelines)
    3. Sends an initial message to greet the user
    4. Runs until the call ends

    The GStreamer pipelines will:
    - Start capturing audio from the ALSA microphone
    - Start capturing video from the V4L2 camera
    - Begin streaming to Gemini Realtime
    - Output audio responses to the ALSA speaker

    Args:
        agent: The configured agent instance
        call_type: Type of call (e.g., "default")
        call_id: Unique identifier for this call
        **kwargs: Additional call configuration

    Notes:
        - Ensure your device paths are correct before running
        - Monitor CPU usage on older Pi models
        - Use 'htop' to check system resources during operation
        - Press Ctrl+C to stop the agent
    """
    # Create a call object
    call = await agent.create_call(call_type, call_id)

    # Join the call using a context manager
    # This automatically:
    # - Initializes GStreamer pipelines
    # - Starts capturing audio/video from Raspberry Pi devices
    # - Begins streaming to Gemini Realtime
    # - Handles cleanup when the context exits
    async with agent.join(call):
        # Send an initial greeting to the user
        await agent.simple_response(
            "Say hello and mention that you're running on a Raspberry Pi. "
            "Keep it brief and friendly."
        )

        # Run the agent until the call ends
        # This keeps the agent active, processing audio/video and responding
        await agent.finish()


# Main entry point
if __name__ == "__main__":
    """Run the Raspberry Pi Local RTC agent with custom GStreamer pipelines.

    Usage:
        python raspberry_pi_gstreamer.py

    Prerequisites:
    ==============

    1. Install GStreamer and Python bindings:
       $ sudo apt-get update
       $ sudo apt-get install -y \\
           python3-gi \\
           gstreamer1.0-tools \\
           gstreamer1.0-plugins-base \\
           gstreamer1.0-plugins-good \\
           gstreamer1.0-plugins-bad \\
           gstreamer1.0-alsa \\
           gstreamer1.0-libav \\
           alsa-utils \\
           v4l-utils

    2. Configure your devices:
       $ arecord -L          # Find audio input device
       $ aplay -L            # Find audio output device
       $ v4l2-ctl --list-devices  # Find video device

    3. Update device paths in the create_agent() function above

    4. Set up environment variables in .env file:
       GOOGLE_API_KEY=your_google_api_key_here

    5. Install Python packages:
       $ pip install vision-agents-stream vision-agents-plugin-gemini vision-agents-plugin-localrtc

    Running the Agent:
    ==================
    $ python raspberry_pi_gstreamer.py

    The agent will:
    - Initialize GStreamer pipelines for your Raspberry Pi hardware
    - Access your ALSA audio devices (microphone and speaker)
    - Capture video from your V4L2 camera
    - Stream audio/video to Gemini Realtime
    - Play audio responses through your speaker

    Troubleshooting:
    ================

    Error: "No such device"
    - Check device paths with 'arecord -L', 'aplay -L', and 'v4l2-ctl --list-devices'
    - Update the device paths in custom_pipeline dictionary

    Error: "GStreamer is not available"
    - Install GStreamer: sudo apt-get install python3-gi gstreamer1.0-tools

    Error: High CPU usage
    - Lower video frame rate: fps=0.5 or fps=0.25 in gemini.Realtime()
    - Disable video: Remove video_source from custom_pipeline
    - Use hardware acceleration if available on your Pi model

    Error: No audio output
    - Test ALSA: aplay -D hw:0,0 /usr/share/sounds/alsa/Front_Center.wav
    - Check volume: alsamixer
    - Verify speaker device path

    Error: No audio input
    - Test ALSA: arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav
    - Check microphone levels: alsamixer (press F4)
    - Verify microphone device path

    Performance Tips:
    =================
    - Use fps=1 or lower for video on Raspberry Pi 3 and earlier
    - Consider audio-only mode for Pi Zero (remove video_source)
    - Monitor with: htop, watch -n 1 vcgencmd measure_temp
    - Overclock if needed (with proper cooling)

    To stop the agent, use Ctrl+C or close the application.
    """
    # Uncomment to see device discovery instructions before starting:
    # find_raspberry_pi_devices()

    # The Runner and AgentLauncher handle the agent lifecycle
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
