#!/usr/bin/env python3
"""
Simple Agent Example with Image Capture

This example demonstrates how to create and join a call using the new Agent class
from agents2.py with image capture capabilities. The agent will:

1. Join a Stream video call as an AI assistant
2. Respond to voice input using STT, LLM, and TTS
3. Capture video frames from participants every 3 seconds
4. Save captured frames as JPG images in the 'captured_frames' directory

This shows the core functionality of the Agent class including voice interaction
and video processing capabilities.
"""

import asyncio
import logging
import os
import sys
import webbrowser
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlencode

import openai
from getstream import Stream
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from getstream.video.rtc.utils import open_browser
from openai.helpers import LocalAudioPlayer
from stt import DeepgramSTT
from PIL import Image

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from getstream.models import UserRequest, User
from getstream.stream import Stream
from getstream.plugins.deepgram.stt import DeepgramSTT

from utils import open_demo

from models import OpenAILLM

# Import the new Agent class from agents2.py
from agents.agents2 import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ImageProcessor:
    """Protocol for image processors."""

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process a video frame image."""
        pass


class IntervalProcessor(ImageProcessor):
    """
    Base class for processors that run at regular intervals.
    
    This class provides interval-based processing functionality that can be
    inherited by other processors. Simply override the process_interval method
    to implement your custom processing logic.
    
    Example:
        class LoggingProcessor(IntervalProcessor):
            def __init__(self, interval: int = 5, receive_audio=True, receive_video=False):
                super().__init__(interval, receive_audio, receive_video)
            
            async def process_interval(self, image, user_id, metadata=None):
                logging.info(f"Processing image from {user_id}: {image.size}")
    """
    
    def __init__(self, interval: int = 3, receive_audio: bool = False, receive_video: bool = True):
        """
        Initialize the interval processor.
        
        Args:
            interval: Interval in seconds between processing
            receive_audio: Whether this processor should receive audio data
            receive_video: Whether this processor should receive video data
        """
        self.interval = interval
        self.receive_audio = receive_audio
        self.receive_video = receive_video
        self.last_process_time = 0
    
    async def should_process(self) -> bool:
        """Check if it's time to process based on the interval."""
        current_time = asyncio.get_event_loop().time()
        
        if current_time - self.last_process_time >= self.interval:
            self.last_process_time = current_time
            return True
        return False
    
    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process image if interval has elapsed."""
        if await self.should_process():
            await self.process_interval(image, user_id, metadata)
    
    async def process_interval(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Override this method to implement interval-based video processing."""
        pass

    async def process_audio(self, audio_data: bytes, user_id: str, metadata: dict = None) -> None:
        """Process audio data. Override this method to implement audio processing."""
        pass


class ImageCapture(IntervalProcessor):
    """Handles video frame capture and storage at regular intervals."""

    def __init__(self, output_dir: str = "captured_frames", interval: int = 3):
        """
        Initialize the image capture processor.

        Args:
            output_dir: Directory to save captured frames
            interval: Interval in seconds between captures
        """
        super().__init__(interval, receive_audio=False, receive_video=True)
        self.output_dir = Path(output_dir)
        self.frame_count = 0

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logging.info(f"📁 Saving captured frames to: {self.output_dir.absolute()}")

    async def process_interval(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process image by saving it as JPG at the specified interval."""
        await self.capture_frame(image, user_id)

    async def capture_frame(self, frame: Image.Image, user_id: str) -> str:
        """
        Capture and save a video frame as JPG.

        Args:
            frame: PIL Image to save
            user_id: ID of the user whose video is being captured

        Returns:
            Path to the saved image file
        """
        timestamp = int(asyncio.get_event_loop().time())
        filename = f"frame_{user_id}_{timestamp}_{self.frame_count:04d}.jpg"
        filepath = self.output_dir / filename

        # Save the frame as JPG
        frame.save(filepath, "JPEG", quality=90)

        self.frame_count += 1
        logging.info(
            f"📸 Captured frame {self.frame_count}: {filename} ({frame.width}x{frame.height})"
        )

        return str(filepath)


class ImageAnalyzer(IntervalProcessor):
    """Example processor that analyzes images at regular intervals."""
    
    def __init__(self, interval: int = 5):
        """
        Initialize the image analyzer.
        
        Args:
            interval: Interval in seconds between analyses
        """
        super().__init__(interval, receive_audio=False, receive_video=True)
        self.analysis_count = 0
    
    async def process_interval(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Analyze image properties at the specified interval."""
        self.analysis_count += 1
        
        # Simple image analysis
        width, height = image.size
        mode = image.mode
        
        logging.info(
            f"📊 Analysis #{self.analysis_count} for {user_id}: "
            f"{width}x{height} pixels, {mode} mode"
        )


class AudioLogger(IntervalProcessor):
    """Example processor that logs audio data at regular intervals."""
    
    def __init__(self, interval: int = 2):
        """
        Initialize the audio logger.
        
        Args:
            interval: Interval in seconds between audio logging
        """
        super().__init__(interval, receive_audio=True, receive_video=False)
        self.audio_count = 0
        self.last_audio_time = 0
    
    async def process_audio(self, audio_data: bytes, user_id: str, metadata: dict = None) -> None:
        """Log audio data information."""
        current_time = asyncio.get_event_loop().time()
        
        if current_time - self.last_audio_time >= self.interval:
            self.last_audio_time = current_time
            self.audio_count += 1
            
            logging.info(
                f"🎵 Audio #{self.audio_count} from {user_id}: "
                f"{len(audio_data)} bytes"
            )


async def main() -> None:
    """Create a simple agent and join a call."""
    
    load_dotenv()
    
    # Initialize Stream client
    client: Stream = Stream.from_env()

    # Create multiple interval processors
    image_capture = ImageCapture(output_dir="captured_frames", interval=3)
    image_analyzer = ImageAnalyzer(interval=5)  # Analyze every 5 seconds
    audio_logger = AudioLogger(interval=2)      # Log audio every 2 seconds
    
    # Create the agent with multiple processors
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    agent = Agent(
        llm=OpenAILLM(
            name="gpt-4o",
            instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful."
        ),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        agent_user=agent_user,
        processors=[image_capture, image_analyzer, audio_logger]  # Multiple interval processors
    )

    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))


    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        open_demo(client, call.id)
        await agent.join(call)
        
        # Keep the agent running
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")
        print()
        print("🎯 Agent is now active with the following features:")
        print("   • 🎤 Voice interaction (STT + LLM + TTS)")
        print("   • 📸 Image capture every 3 seconds")
        print("   • 📊 Image analysis every 5 seconds")
        print("   • 🎵 Audio logging every 2 seconds")
        print("   • 📁 Images saved to 'captured_frames' directory")
        print("   • Join the call from your browser to interact!")
        print()
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("👋 Shutting down agent...")
    except Exception as e:
        logging.error("❌ Error: %s", e)
    finally:
        # Clean up agent resources
        if 'agent' in locals():
            try:
                await agent.close()
                logging.info("✅ Agent cleanup completed")
            except Exception as e:
                logging.error(f"❌ Error during cleanup: {e}")
        
        # Display processing summary
        if 'image_capture' in locals() or 'image_analyzer' in locals() or 'audio_logger' in locals():
            print()
            print("📊 Processing Summary:")
            
            if 'image_capture' in locals():
                print(f"   • Total frames captured: {image_capture.frame_count}")
                print(f"   • Frames saved to: {image_capture.output_dir.absolute()}")
                
            if 'image_analyzer' in locals():
                print(f"   • Total analyses performed: {image_analyzer.analysis_count}")
                
            if 'audio_logger' in locals():
                print(f"   • Total audio logs: {audio_logger.audio_count}")

            if 'image_capture' in locals() and image_capture.frame_count > 0:
                print("   • Captured files:")
                for jpg_file in sorted(image_capture.output_dir.glob("*.jpg")):
                    print(f"     - {jpg_file.name}")


if __name__ == "__main__":
    asyncio.run(main())
