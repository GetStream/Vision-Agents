import numpy as np
import pytest
from getstream.video.rtc import AudioFormat, PcmData
from vision_agents.core.vad.silero import SILERO_CHUNK, SileroVADSessionPool


@pytest.mark.integration
@pytest.mark.skip_blockbuster
class TestSileroVAD:
    """Integration tests for SileroVAD."""

    async def test_silence_not_detected_as_speech(self, tmp_path):
        """Test that silence returns low speech probability."""
        tmpdir = str(tmp_path)
        vad_pool = await SileroVADSessionPool.load(tmpdir)
        vad = vad_pool.session()

        # Create silence (all zeros)
        samples = np.zeros(SILERO_CHUNK * 2, dtype=np.float32)
        pcm = PcmData(
            samples=samples, sample_rate=16000, channels=1, format=AudioFormat.F32
        )

        score = vad.predict_speech(pcm)

        # Silence should have low speech probability
        assert score < 0.3

    async def test_mia_audio_detected_as_speech(self, tmp_path, mia_audio_16khz):
        """Test that real speech audio returns high speech probability."""
        tmpdir = str(tmp_path)
        vad_pool = await SileroVADSessionPool.load(tmpdir)
        vad = vad_pool.session()

        score = vad.predict_speech(mia_audio_16khz)

        # Real speech should have high speech probability
        assert score > 0.5
