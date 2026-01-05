import asyncio
import os

import dotenv
import pytest
from vision_agents.core.llm.events import (
    RealtimeAudioOutputEvent,
)
from vision_agents.plugins.qwen import Realtime

dotenv.load_dotenv()


@pytest.fixture()
async def llm():
    """Create and manage Realtime connection lifecycle."""
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY not set")
    realtime = Realtime(
        fps=1, vad_silence_duration_ms=0, vad_prefix_padding_ms=0, vad_threshold=0.1
    )
    yield realtime
    await realtime.close()


class TestQwen3Realtime:
    """Integration tests for Qwen3Realtime connect flow"""

    @pytest.mark.integration
    async def test_audio_sending_flow(self, llm, mia_audio_16khz, silence_1s_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []

        @llm.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        # Connect the llm
        await llm.connect()
        # Let it handle the connection events
        await asyncio.sleep(5.0)

        # Send 1s of silence first
        await llm.simple_audio_response(silence_1s_16khz)
        # Send audio
        await llm.simple_audio_response(mia_audio_16khz)
        # Send silence again
        await llm.simple_audio_response(silence_1s_16khz)

        # Let it run for a few sec
        await asyncio.sleep(10.0)

        # Verify that the model replied with audio
        assert len(events) > 0

    @pytest.mark.integration
    async def test_video_sending_flow(
        self,
        llm,
        bunny_video_track,
        describe_what_you_see_audio_16khz,
        silence_1s_16khz,
    ):
        """Test sending real video data and verify connection remains stable"""
        events = []

        @llm.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await llm.connect()
        # Let the model to handle all connection events
        await asyncio.sleep(5.0)

        # Send 1s of silence first
        await llm.simple_audio_response(silence_1s_16khz)
        # Start video sender with low FPS to avoid overwhelming the connection
        await llm.watch_video_track(bunny_video_track)
        # Send audio to the model (it does not support text inputs)
        await llm.simple_audio_response(describe_what_you_see_audio_16khz)
        # Send silence again
        await llm.simple_audio_response(silence_1s_16khz)
        # Let it run for a few seconds
        await asyncio.sleep(10.0)

        # Stop video sender
        await llm._stop_watching_video_track()
        # Verify that the model replied
        assert len(events) > 0
