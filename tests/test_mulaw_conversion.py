"""Tests for mulaw conversion utilities."""

import numpy as np
from getstream.video.rtc.track_util import PcmData, AudioFormat

# Import from the example's utils module using importlib since the folder name starts with a number
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "utils",
    os.path.join(os.path.dirname(__file__), "../examples/03_phone_and_rag_example/utils.py")
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

mulaw_to_pcm = utils.mulaw_to_pcm
pcm_to_mulaw = utils.pcm_to_mulaw
TWILIO_SAMPLE_RATE = utils.TWILIO_SAMPLE_RATE


class TestMulawConversion:
    def test_pcm_to_mulaw_and_back(self):
        """Test roundtrip conversion preserves audio structure."""
        # Use non-zero samples since mulaw doesn't preserve exact zero
        samples = np.array([100, 1000, -1000, 16000, -16000, 32000, -32000], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format=AudioFormat.S16,
        )
        
        mulaw_bytes = pcm_to_mulaw(pcm)
        recovered = mulaw_to_pcm(mulaw_bytes)
        
        assert recovered.sample_rate == TWILIO_SAMPLE_RATE
        assert recovered.channels == 1
        assert len(recovered.samples) == len(samples)
        
        # Mulaw is lossy, but signs should be preserved for non-zero values
        original_signs = np.sign(samples)
        recovered_signs = np.sign(recovered.samples)
        np.testing.assert_array_equal(original_signs, recovered_signs)

    def test_mulaw_to_pcm_output_format(self):
        """Test mulaw_to_pcm returns correct format."""
        # 0xFF is mulaw for silence (0)
        mulaw_bytes = bytes([0xFF, 0xFF, 0xFF, 0xFF])
        pcm = mulaw_to_pcm(mulaw_bytes)
        
        assert pcm.sample_rate == TWILIO_SAMPLE_RATE
        assert pcm.channels == 1
        assert pcm.format == AudioFormat.S16
        assert len(pcm.samples) == 4

    def test_pcm_to_mulaw_resamples(self):
        """Test that pcm_to_mulaw resamples non-8kHz audio."""
        samples = np.sin(np.linspace(0, 2 * np.pi, 160)) * 10000
        samples = samples.astype(np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.S16,
        )
        
        mulaw_bytes = pcm_to_mulaw(pcm)
        
        # Should have half the samples after resampling 16kHz -> 8kHz
        assert len(mulaw_bytes) == len(samples) // 2

    def test_mia_audio_roundtrip(self, mia_audio_16khz):
        """Test roundtrip with real speech audio."""
        mulaw_bytes = pcm_to_mulaw(mia_audio_16khz)
        recovered = mulaw_to_pcm(mulaw_bytes)
        
        assert recovered.sample_rate == TWILIO_SAMPLE_RATE
        assert recovered.channels == 1
        assert len(recovered.samples) > 0
        
        # Audio should have some energy (not all zeros)
        assert np.abs(recovered.samples).max() > 100

    def test_mia_audio_preserves_speech_characteristics(self, mia_audio_16khz):
        """Test that mulaw conversion preserves speech-like characteristics."""
        mulaw_bytes = pcm_to_mulaw(mia_audio_16khz)
        recovered = mulaw_to_pcm(mulaw_bytes)
        
        samples = recovered.samples
        assert samples.max() > 1000, "Audio should have positive peaks"
        assert samples.min() < -1000, "Audio should have negative peaks"
        
        # Mulaw has 256 quantization levels, so expect reasonable variation
        unique_values = len(np.unique(samples))
        assert unique_values > 50, "Audio should have variation"

