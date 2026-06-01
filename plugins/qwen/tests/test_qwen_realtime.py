import asyncio
import os

import dotenv
import pytest
from vision_agents.core.llm.realtime import RealtimeAudioOutput
from vision_agents.plugins.qwen import Realtime

dotenv.load_dotenv()


@pytest.mark.skip()
@pytest.mark.integration
class TestQwen3RealtimeIntegration:
    """Integration tests for Qwen3Realtime connect flow"""

    @pytest.fixture
    async def llm(self):
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DASHSCOPE_API_KEY not set")
        rt = Realtime(
            fps=1,
            vad_silence_duration_ms=0,
            vad_prefix_padding_ms=0,
            vad_threshold=0.1,
        )
        try:
            await rt.connect()
            await asyncio.sleep(5.0)
            yield rt
        finally:
            await rt.close()

    async def test_audio_sending_flow(self, llm, mia_audio_16khz, silence_1s_16khz):
        """Test sending real audio data and verify connection remains stable"""
        # Send 1s of silence first
        await llm.simple_audio_response(silence_1s_16khz)
        # Send audio
        await llm.simple_audio_response(mia_audio_16khz)
        # Send silence again
        await llm.simple_audio_response(silence_1s_16khz)

        # Let it run for a few sec
        await asyncio.sleep(10.0)

        # Verify that the model replied with audio
        audio = [i for i in llm.output.peek() if isinstance(i, RealtimeAudioOutput)]
        assert len(audio) > 0

    async def test_video_sending_flow(
        self,
        llm,
        bunny_video_track,
        describe_what_you_see_audio_16khz,
        silence_1s_16khz,
    ):
        """Test sending real video data and verify connection remains stable"""
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
        await llm.stop_watching_video_track()
        # Verify that the model replied
        audio = [i for i in llm.output.peek() if isinstance(i, RealtimeAudioOutput)]
        assert len(audio) > 0
