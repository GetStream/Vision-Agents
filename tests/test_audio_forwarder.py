import asyncio

import numpy as np

from getstream.video.rtc.track_util import PcmData, AudioFormat
from vision_agents.core.utils.audio_forwarder import AudioForwarder
from tests.base_test import BaseTest


class TestAudioForwarder(BaseTest):
    async def test_audio_forwarder_basic(self, audio_track_48khz):
        """Test that AudioForwarder receives and forwards audio."""
        received_audio = []

        async def callback(pcm: PcmData):
            received_audio.append(pcm)

        forwarder = AudioForwarder(audio_track_48khz, callback)
        await forwarder.start()

        # Let it run for a bit to collect some audio
        await asyncio.sleep(0.5)

        await forwarder.stop()

        # Verify we received audio
        assert len(received_audio) > 0, "Should have received audio frames"

    async def test_audio_forwarder_resamples_to_16khz(self, audio_track_48khz):
        """Test that AudioForwarder resamples audio to 16kHz mono."""
        received_audio = []

        async def callback(pcm: PcmData):
            received_audio.append(pcm)

        forwarder = AudioForwarder(audio_track_48khz, callback)
        await forwarder.start()

        # Let it run for a bit
        await asyncio.sleep(0.3)

        await forwarder.stop()

        # Verify audio properties
        assert len(received_audio) > 0, "Should have received audio frames"

        for pcm in received_audio:
            assert pcm.sample_rate == 16000, "Audio should be resampled to 16kHz"
            assert pcm.channels == 1, "Audio should be mono"
            assert pcm.format == AudioFormat.S16, "Audio should be int16 format"

    async def test_audio_forwarder_stop_and_restart(self, audio_track_48khz):
        """Test that AudioForwarder can be stopped and doesn't error."""
        received_audio = []

        async def callback(pcm: PcmData):
            received_audio.append(pcm)

        forwarder = AudioForwarder(audio_track_48khz, callback)
        await forwarder.start()

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop it
        await forwarder.stop()

        initial_count = len(received_audio)
        assert initial_count > 0, "Should have received some audio before stopping"

        # Wait a bit more - shouldn't receive any more audio
        await asyncio.sleep(0.2)

        # Count should not have increased significantly (maybe 1-2 frames due to timing)
        assert len(received_audio) <= initial_count + 2, (
            "Should not receive audio after stopping"
        )

    async def test_audio_forwarder_callback_receives_valid_data(
        self, audio_track_48khz
    ):
        """Test that callback receives valid PcmData with actual audio samples."""
        received_audio = []

        async def callback(pcm: PcmData):
            received_audio.append(pcm)

        forwarder = AudioForwarder(audio_track_48khz, callback)
        await forwarder.start()

        await asyncio.sleep(0.3)

        await forwarder.stop()

        assert len(received_audio) > 0, "Should have received audio"

        # Check that we have actual audio data
        for pcm in received_audio:
            assert len(pcm.samples) > 0, "PcmData should contain samples"
            assert isinstance(pcm.samples, np.ndarray), "Samples should be numpy array"
            assert pcm.samples.dtype == np.int16, "Samples should be int16"
