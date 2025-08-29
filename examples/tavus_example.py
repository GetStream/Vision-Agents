#!/usr/bin/env python3
"""
Example usage of TavusProcessor with Daily integration.

This example demonstrates how to:
1. Create a Tavus conversation
2. Join the Daily call
3. Stream audio/video to Stream's platform
4. Handle cleanup properly

Prerequisites:
- TAVUS_KEY environment variable set
- Valid replica_id and persona_id from Tavus
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from stream_agents.processors.tavus_processor import TavusProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    PROCESSOR_AVAILABLE = False
    logger.error(f"TavusProcessor not available: {e}")

async def run_tavus_example():
    """Run the Tavus processor example."""
    if not PROCESSOR_AVAILABLE:
        logger.error("Cannot run example - TavusProcessor not available")
        return
    
    # Get configuration from environment
    api_key = os.getenv("TAVUS_KEY")
    replica_id = os.getenv("TAVUS_TEST_REPLICA_ID", "rfe12d8b9597")
    persona_id = os.getenv("TAVUS_TEST_PERSONA_ID", "pdced222244b")
    
    if not api_key:
        logger.error("TAVUS_KEY environment variable not set")
        return
    
    processor = None
    try:
        logger.info("🚀 Starting Tavus processor example...")
        
        # Create TavusProcessor with auto-create and auto-join
        processor = TavusProcessor(
            api_key=api_key,
            replica_id=replica_id,
            persona_id=persona_id,
            conversation_name="Example Conversation",
            auto_create=True,
            auto_join=True,
            interval=1  # Process every second
        )
        
        # Log the processor state
        state = processor.state()
        logger.info(f"📊 Processor state: {state}")
        
        # Get the conversation URL
        if processor.conversation_url:
            logger.info(f"🔗 Join the conversation at: {processor.conversation_url}")
        
        # Get the audio and video tracks for streaming
        audio_track = processor.create_audio_track()
        video_track = processor.create_video_track()
        
        logger.info(f"🎵 Audio track created: {type(audio_track)}")
        logger.info(f"🎥 Video track created: {type(video_track)}")
        
        # Simulate running for a short time
        logger.info("⏱️  Running for 30 seconds...")
        await asyncio.sleep(30)
        
        # Test getting some frames
        logger.info("🎬 Testing frame retrieval...")
        try:
            audio_frame = await audio_track.recv()
            logger.info(f"🎵 Got audio frame: {type(audio_frame)}")
        except Exception as e:
            logger.info(f"🎵 Audio frame error (expected): {e}")
        
        try:
            video_frame = await video_track.recv()
            logger.info(f"🎥 Got video frame: {video_frame.width}x{video_frame.height}")
        except Exception as e:
            logger.info(f"🎥 Video frame error: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error in example: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if processor:
            logger.info("🧹 Cleaning up processor...")
            await processor.cleanup()
        
        logger.info("✅ Example completed!")

async def run_audio_only_example():
    """Run an audio-only Tavus processor example."""
    if not PROCESSOR_AVAILABLE:
        return
    
    api_key = os.getenv("TAVUS_KEY")
    if not api_key:
        return
    
    logger.info("🎵 Starting audio-only Tavus processor example...")
    
    processor = None
    try:
        processor = TavusProcessor(
            api_key=api_key,
            replica_id=os.getenv("TAVUS_TEST_REPLICA_ID", "rfe12d8b9597"),
            persona_id=os.getenv("TAVUS_TEST_PERSONA_ID", "pdced222244b"),
            conversation_name="Audio Only Example",
            audio_only=True,
            auto_create=True,
            auto_join=True
        )
        
        state = processor.state()
        logger.info(f"📊 Audio-only processor state: {state}")
        
        # Only audio track should be available
        audio_track = processor.create_audio_track()
        logger.info(f"🎵 Audio track created: {type(audio_track)}")
        
        # Video track should still be created but not used
        video_track = processor.create_video_track()
        logger.info(f"🎥 Video track created (but unused): {type(video_track)}")
        
        await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"❌ Error in audio-only example: {e}")
    
    finally:
        if processor:
            await processor.cleanup()

async def main():
    """Main example function."""
    logger.info("🎭 Tavus Processor Examples")
    logger.info("=" * 50)
    
    # Run the main example
    await run_tavus_example()
    
    # Wait a bit between examples
    await asyncio.sleep(5)
    
    # Run the audio-only example
    await run_audio_only_example()
    
    logger.info("🎉 All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
