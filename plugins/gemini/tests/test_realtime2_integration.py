import asyncio
import os
import pytest
from dotenv import load_dotenv

from stream_agents.plugins.gemini.realtime2 import Realtime2
from tests.base_test import BaseTest

# Load environment variables
load_dotenv()


class TestRealtime2Integration(BaseTest):
    """Integration tests for Realtime2 connect flow"""

    @pytest.fixture
    async def realtime2(self):
        """Create and manage Realtime2 connection lifecycle"""
        realtime2 = Realtime2(
            model="gemini-2.5-flash-exp-native-audio-thinking-dialog",
        )
        try:
            yield realtime2
        finally:
            await realtime2.close()
    

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime2):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        await realtime2.simple_response("Hello, can you hear me?")

        # Wait for response
        await asyncio.sleep(3.0)
        assert len(events) > 0

    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime2, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        
        await realtime2.simple_response("Listen to the following story, what is Mia looking for?")
        await asyncio.sleep(10.0)
        await realtime2.send_audio_pcm(mia_audio_16khz)

        # Wait a moment to ensure processing
        await asyncio.sleep(10.0)
        assert len(events) > 0


    @pytest.mark.integration
    async def test_video_sending_flow(self, realtime2, bunny_video_track):
        """Test sending real video data and verify connection remains stable"""
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        await realtime2.simple_response("Describe what you see in this video please")
        await asyncio.sleep(10.0)
        # Start video sender with low FPS to avoid overwhelming the connection
        await realtime2._watch_video_track(bunny_video_track)
        
        # Let it run for a few seconds
        await asyncio.sleep(10.0)
        
        # Stop video sender
        await realtime2._stop_watching_video_track()
        assert len(events) > 0

    async def test_frame_to_png_bytes_with_bunny_video(self, bunny_video_track):
        """Test that _frame_to_png_bytes works with real bunny video frames"""
        # Get a frame from the bunny video track
        frame = await bunny_video_track.recv()
        png_bytes = Realtime2._frame_to_png_bytes(frame)
        
        # Verify we got PNG data
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        
        # Verify it's actually PNG data (PNG files start with specific bytes)
        assert png_bytes.startswith(b'\x89PNG\r\n\x1a\n')
        
        print(f"Successfully converted bunny video frame to PNG: {len(png_bytes)} bytes")

