"""Comprehensive tests for mulaw/PCM audio conversion."""

import numpy as np
from getstream.video.rtc.track_util import PcmData, AudioFormat

from vision_agents.plugins.twilio.audio import (
    mulaw_to_pcm,
    pcm_to_mulaw,
    MULAW_DECODE_TABLE,
    TWILIO_SAMPLE_RATE,
)

# Known correct PCM values for each mulaw byte (ITU-T G.711 reference)
# These are the exact decode values that our MULAW_DECODE_TABLE should match
REFERENCE_DECODE_VALUES = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0,
]

# Known correct mulaw encoding for specific PCM values (verified against audioop)
REFERENCE_ENCODE_VALUES = {
    0: 0xFF,
    1: 0xFF,
    -1: 0x7E,
    100: 0xF2,
    -100: 0x72,
    1000: 0xCE,
    -1000: 0x4E,
    8000: 0xA0,
    -8000: 0x20,
    16000: 0x90,
    -16000: 0x10,
    32000: 0x80,
    -32000: 0x00,
    32767: 0x80,
    -32768: 0x00,
}


class TestMulawDecode:
    """Test mulaw to PCM decoding."""

    def test_decode_table_shape(self):
        """Verify decode table has 256 entries for all byte values."""
        assert len(MULAW_DECODE_TABLE) == 256
        assert MULAW_DECODE_TABLE.dtype == np.int16

    def test_decode_silence(self):
        """Mulaw 0xFF and 0x7F represent silence (near zero)."""
        # 0xFF = positive zero, 0x7F = negative zero in mulaw
        pcm = mulaw_to_pcm(bytes([0xFF]))
        assert pcm.samples[0] == 0

        pcm = mulaw_to_pcm(bytes([0x7F]))
        assert pcm.samples[0] == 0

    def test_decode_matches_reference(self):
        """Compare our decoder against ITU-T G.711 reference values."""
        for mulaw_val in range(256):
            mulaw_bytes = bytes([mulaw_val])
            our_pcm = mulaw_to_pcm(mulaw_bytes)
            our_sample = our_pcm.samples[0]
            ref_sample = REFERENCE_DECODE_VALUES[mulaw_val]

            assert our_sample == ref_sample, (
                f"Mismatch at mulaw {mulaw_val:#04x}: "
                f"ours={our_sample}, ref={ref_sample}"
            )


class TestMulawEncode:
    """Test PCM to mulaw encoding."""

    def test_encode_silence(self):
        """Zero PCM should encode to mulaw silence."""
        pcm = PcmData(
            samples=np.array([0], dtype=np.int16),
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format=AudioFormat.S16,
        )
        mulaw = pcm_to_mulaw(pcm)
        # 0xFF is positive silence
        assert mulaw[0] == 0xFF

    def test_encode_matches_reference(self):
        """Compare our encoder against ITU-T G.711 reference values."""
        mismatches = []
        for pcm_val, expected_mulaw in REFERENCE_ENCODE_VALUES.items():
            pcm = PcmData(
                samples=np.array([pcm_val], dtype=np.int16),
                sample_rate=TWILIO_SAMPLE_RATE,
                channels=1,
                format=AudioFormat.S16,
            )
            our_mulaw = pcm_to_mulaw(pcm)[0]

            if our_mulaw != expected_mulaw:
                mismatches.append(
                    f"PCM {pcm_val}: ours={our_mulaw:#04x}, expected={expected_mulaw:#04x}"
                )

        assert not mismatches, "Encoding mismatches:\n" + "\n".join(mismatches)

    def test_encode_symmetry(self):
        """Positive and negative values should encode symmetrically."""
        test_values = [100, 1000, 8000, 16000, 32000]

        for val in test_values:
            pos_pcm = PcmData(
                samples=np.array([val], dtype=np.int16),
                sample_rate=TWILIO_SAMPLE_RATE,
                channels=1,
                format=AudioFormat.S16,
            )
            neg_pcm = PcmData(
                samples=np.array([-val], dtype=np.int16),
                sample_rate=TWILIO_SAMPLE_RATE,
                channels=1,
                format=AudioFormat.S16,
            )

            pos_mulaw = pcm_to_mulaw(pos_pcm)[0]
            neg_mulaw = pcm_to_mulaw(neg_pcm)[0]

            # Sign bit is bit 7: positive has bit 7 set (after XOR), negative doesn't
            # The magnitude (lower 7 bits) should be the same
            assert (pos_mulaw & 0x7F) == (neg_mulaw & 0x7F), (
                f"Asymmetric encoding for ±{val}: "
                f"pos={pos_mulaw:#04x}, neg={neg_mulaw:#04x}"
            )


class TestRoundTrip:
    """Test round-trip conversion accuracy."""

    def test_roundtrip_preserves_waveform(self):
        """Verify PCM -> mulaw -> PCM round-trip maintains signal quality."""
        # Generate a test signal (440Hz sine wave)
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(TWILIO_SAMPLE_RATE * duration), dtype=np.float32)
        sine_wave = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

        original_pcm = PcmData(
            samples=sine_wave,
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format=AudioFormat.S16,
        )

        # Round-trip
        mulaw = pcm_to_mulaw(original_pcm)
        recovered_pcm = mulaw_to_pcm(mulaw)

        # Calculate error
        error = np.abs(original_pcm.samples.astype(np.int32) - recovered_pcm.samples.astype(np.int32))
        max_error = np.max(error)
        mean_error = np.mean(error)

        # Mulaw quantization introduces some error, but it should be bounded
        # Max quantization error for mulaw is about 2% of signal amplitude
        assert max_error < 2000, f"Max error too high: {max_error}"
        assert mean_error < 500, f"Mean error too high: {mean_error}"

    def test_roundtrip_known_values(self):
        """Verify round-trip produces expected decoded values."""
        # Known encode->decode pairs (verified against audioop)
        # PCM -> mulaw -> PCM (decoded value from MULAW_DECODE_TABLE)
        test_cases = [
            (0, 0),          # Silence stays at 0
            (32767, 32124),  # Max positive -> decode of 0x80
            (-32768, -32124),# Max negative -> decode of 0x00
            (16000, 15996),  # Mid-high positive
            (-16000, -15996),# Mid-high negative
            (1000, 988),     # Low positive
            (-1000, -988),   # Low negative
        ]

        for original, expected_recovered in test_cases:
            pcm = PcmData(
                samples=np.array([original], dtype=np.int16),
                sample_rate=TWILIO_SAMPLE_RATE,
                channels=1,
                format=AudioFormat.S16,
            )
            mulaw = pcm_to_mulaw(pcm)
            recovered = mulaw_to_pcm(mulaw).samples[0]

            assert recovered == expected_recovered, (
                f"Round-trip mismatch for PCM {original}: "
                f"got {recovered}, expected {expected_recovered}"
            )


class TestSignalQuality:
    """Test signal quality metrics."""

    def test_snr_acceptable(self):
        """Verify Signal-to-Noise Ratio is acceptable for voice."""
        # Generate voice-like signal (mix of frequencies)
        duration = 0.5
        samples_count = int(TWILIO_SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples_count, dtype=np.float32)

        # Mix of speech-like frequencies
        signal = (
            np.sin(2 * np.pi * 200 * t) * 8000
            + np.sin(2 * np.pi * 400 * t) * 6000
            + np.sin(2 * np.pi * 800 * t) * 4000
        ).astype(np.int16)

        original = PcmData(
            samples=signal,
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format=AudioFormat.S16,
        )

        # Round-trip
        mulaw = pcm_to_mulaw(original)
        recovered = mulaw_to_pcm(mulaw)

        # Calculate SNR
        signal_power = np.mean(original.samples.astype(np.float64) ** 2)
        noise = original.samples.astype(np.float64) - recovered.samples.astype(np.float64)
        noise_power = np.mean(noise**2)

        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

        # Mulaw should achieve at least 35dB SNR for typical voice signals
        assert snr_db > 35, f"SNR too low: {snr_db:.1f} dB"

    def test_no_clipping_distortion(self):
        """Ensure we don't introduce clipping at signal boundaries."""
        # Test values near the limits
        edge_values = np.array([-32768, -32767, 32766, 32767], dtype=np.int16)

        pcm = PcmData(
            samples=edge_values,
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format=AudioFormat.S16,
        )

        mulaw = pcm_to_mulaw(pcm)
        recovered = mulaw_to_pcm(mulaw)

        # Values should stay within int16 range after recovery
        assert np.all(recovered.samples >= -32768)
        assert np.all(recovered.samples <= 32767)

        # Sign should be preserved
        assert recovered.samples[0] < 0  # Was -32768
        assert recovered.samples[1] < 0  # Was -32767
        assert recovered.samples[2] > 0  # Was 32766
        assert recovered.samples[3] > 0  # Was 32767

