#!/usr/bin/env python3
"""
Video Capture Example for Stream Agents

This example demonstrates how to use the Agent class with built-in video processing
capabilities to capture video frames from a Stream video call and save them as JPG
images at configurable intervals.

Usage:
    python example_video_capture.py [--interval SECONDS] [--output DIR]

The example will:
1. Create a Stream video call
2. Join as an AI Agent with video processing enabled
3. Listen for video tracks from participants
4. Capture frames at the specified interval and save them as JPG files
5. Display the captured frames count and file names

Requirements:
    - STREAM_API_KEY and STREAM_SECRET environment variables
    - getstream-python package
    - Pillow (PIL) for image processing
    - aiortc for video frame handling
"""

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from getstream.stream import Stream
from utils import open_pronto

# Import Agent base class
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
from agents import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Protocol for image processors."""

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process a video frame image."""
        pass


class VideoFrameCapture(ImageProcessor):
    """Handles video frame capture and storage."""

    def __init__(self, output_dir: str = "captured_frames", capture_interval: int = 5):
        """
        Initialize the video frame capture.

        Args:
            output_dir: Directory to save captured frames
            capture_interval: Interval in seconds between captures
        """
        self.output_dir = Path(output_dir)
        self.capture_interval = capture_interval
        self.frame_count = 0
        self.last_capture_time = 0

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Saving captured frames to: {self.output_dir.absolute()}")

    async def should_capture_frame(self) -> bool:
        """Check if it's time to capture a new frame."""
        current_time = asyncio.get_event_loop().time()

        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            return True
        return False

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process image by saving it as JPG."""
        if await self.should_capture_frame():
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
        logger.info(
            f"📸 Captured frame {self.frame_count}: {filename} ({frame.width}x{frame.height})"
        )

        return str(filepath)


def create_user(client: Stream, user_id: str, name: str):
    """Create a user in Stream."""
    try:
        from getstream.models import UserRequest

        user_request = UserRequest(id=user_id, name=name)
        client.upsert_users(user_request)
        logger.info(f"👤 Created user: {name} ({user_id})")
    except Exception as e:
        logger.error(f"❌ Failed to create user {user_id}: {e}")


async def main(interval: int = 5, output_dir: str = "captured_frames"):
    """Main function to run the video capture example."""

    print("🎥 Stream Agents - Video Capture Example")
    print("=" * 50)
    print(
        f"This example captures video frames every {interval} seconds from a Stream video call."
    )
    print("Join the call from your browser to see your video being captured!")
    print()

    # Load environment variables
    load_dotenv()

    # Initialize Stream client
    try:
        client = Stream.from_env()
        logger.info("✅ Stream client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Stream client: {e}")
        logger.error(
            "Make sure STREAM_API_KEY and STREAM_SECRET are set in your .env file"
        )
        return

    # Create a unique call
    call_id = f"video-capture-{str(uuid4())[:8]}"
    call = client.video.call("default", call_id)
    logger.info(f"📞 Call ID: {call_id}")

    # Create user IDs
    participant_user_id = f"participant-{str(uuid4())[:8]}"

    # Create users
    create_user(client, participant_user_id, "Participant")

    # Create tokens
    participant_token = client.create_token(participant_user_id, expiration=3600)

    # Create the call
    try:
        call.get_or_create(data={"created_by_id": participant_user_id})
        logger.info("✅ Call created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create call: {e}")
        return

    # Initialize Agent with image processing capabilities
    frame_capture = VideoFrameCapture(output_dir=output_dir, capture_interval=interval)

    # Prepare image processors list
    image_processors = [frame_capture]

    agent = Agent(
        instructions="I am a video capture bot that processes video frames from participants.",
        image_interval=interval,  # Use the provided interval
        image_processors=image_processors,  # Use the frame capture processor
        target_user_id=None,  # Capture from any user
        name="Video Capture Bot",
    )

    # User creation callback
    def create_agent_user(bot_id: str, name: str):
        create_user(client, bot_id, name)

    try:
        # Open browser for participant to join
        open_pronto(client.api_key, participant_token, call_id)

        print()
        print("🎯 Ready to capture video frames!")
        print("   • Join the call from your browser")
        print("   • Turn on your camera")
        print(f"   • Frames will be captured every {interval} seconds")
        print("   • Press Ctrl+C to stop")
        print()

        # Join the call with the Agent
        await agent.join(call, user_creation_callback=create_agent_user)

    except KeyboardInterrupt:
        logger.info("⏹️  Video capture stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during video capture: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([agent.bot_id, participant_user_id])
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

        # Display summary
        print()
        print("📊 Capture Summary:")
        print(f"   • Total frames captured: {frame_capture.frame_count}")
        print(f"   • Frames saved to: {frame_capture.output_dir.absolute()}")

        if frame_capture.frame_count > 0:
            print("   • Captured files:")
            for jpg_file in sorted(frame_capture.output_dir.glob("*.jpg")):
                print(f"     - {jpg_file.name}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Video Capture Example for Stream Agents"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=5,
        help="Capture interval in seconds (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="captured_frames",
        help="Output directory for captured frames (default: captured_frames)",
    )

    args = parser.parse_args()

    # Run the example with parsed arguments
    asyncio.run(main(interval=args.interval, output_dir=args.output))
