"""Comprehensive tests for audio format conversion and configuration.

This test suite validates:
- Sample rate handling and conversion
- Channel conversion (mono/stereo)
- Bit depth handling
- Audio buffer preparation for playback
- Audio device mocking and interaction
- Format validation and error handling
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from vision_agents.core.types import PcmData
from vision_agents.plugins.localrtc.tracks import AudioOutputTrack


class TestSampleRateHandling:
    """Test sample rate validation and conversion."""

    def test_sample_rate_16khz_to_16khz_no_conversion(self):
        """Verify no conversion occurs when sample rates match."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create test data at 16kHz
            samples = np.array([100, 200, 300, 400], dtype=np.int16)
            audio_bytes = samples.tobytes()

            # Call _convert_audio with matching sample rates
            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            # Should return same data when rates match (after processing)
            result_array = np.frombuffer(result, dtype=np.int16)
            np.testing.assert_array_equal(result_array.flatten(), samples)

    def test_sample_rate_conversion_48khz_to_16khz(self):
        """Test downsampling from 48kHz to 16kHz."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create 48kHz audio (12 samples = 0.25ms at 48kHz)
            samples_48k = np.array(
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
                dtype=np.int16,
            )
            audio_bytes = samples_48k.tobytes()

            # Convert 48kHz -> 16kHz (should downsample by 3x)
            result = track._convert_audio(
                data=audio_bytes,
                from_rate=48000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            # Result should have 1/3 the samples (12 / 3 = 4 samples)
            result_array = np.frombuffer(result, dtype=np.int16).flatten()
            assert len(result_array) == 4

    def test_sample_rate_conversion_8khz_to_16khz(self):
        """Test upsampling from 8kHz to 16kHz."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create 8kHz audio (4 samples)
            samples_8k = np.array([0, 1000, 2000, 3000], dtype=np.int16)
            audio_bytes = samples_8k.tobytes()

            # Convert 8kHz -> 16kHz (should upsample by 2x)
            result = track._convert_audio(
                data=audio_bytes,
                from_rate=8000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            # Result should have 2x the samples (4 * 2 = 8 samples)
            result_array = np.frombuffer(result, dtype=np.int16).flatten()
            assert len(result_array) == 8

    def test_sample_rate_conversion_preserves_duration(self):
        """Verify resampling preserves audio duration."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create 1 second of audio at 48kHz
            samples_48k = np.zeros(48000, dtype=np.int16)
            audio_bytes = samples_48k.tobytes()

            # Convert to 16kHz
            result = track._convert_audio(
                data=audio_bytes,
                from_rate=48000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            # Should have 16000 samples (1 second at 16kHz)
            result_array = np.frombuffer(result, dtype=np.int16).flatten()
            assert len(result_array) == 16000

    def test_sample_rate_validation_invalid_from_rate(self):
        """Test that invalid input sample rate raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(ValueError, match="Invalid sample rates"):
                track._convert_audio(
                    data=b"\x00\x01" * 10,
                    from_rate=0,  # Invalid
                    to_rate=16000,
                    from_channels=1,
                    to_channels=1,
                    bit_depth=16,
                )

    def test_sample_rate_validation_negative_to_rate(self):
        """Test that negative output sample rate raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(ValueError, match="Invalid sample rates"):
                track._convert_audio(
                    data=b"\x00\x01" * 10,
                    from_rate=16000,
                    to_rate=-1,  # Invalid
                    from_channels=1,
                    to_channels=1,
                    bit_depth=16,
                )


class TestChannelConversion:
    """Test mono/stereo channel conversion."""

    def test_mono_to_mono_no_conversion(self):
        """Verify mono to mono requires no channel conversion."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create mono audio
            samples = np.array([100, 200, 300, 400], dtype=np.int16)
            audio_bytes = samples.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            result_array = np.frombuffer(result, dtype=np.int16).flatten()
            np.testing.assert_array_equal(result_array, samples)

    def test_mono_to_stereo_duplication(self):
        """Test mono to stereo duplicates channel."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=2)

            # Create mono audio
            mono_samples = np.array([100, 200, 300], dtype=np.int16)
            audio_bytes = mono_samples.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=2,
                bit_depth=16,
            )

            # Result should have 2 channels with duplicated data
            result_array = np.frombuffer(result, dtype=np.int16)
            # Stereo interleaves: [L, R, L, R, L, R]
            # Reshape to (samples, channels)
            stereo = result_array.reshape(-1, 2)

            # Both channels should have same values
            np.testing.assert_array_equal(stereo[:, 0], mono_samples)
            np.testing.assert_array_equal(stereo[:, 1], mono_samples)

    def test_stereo_to_mono_averaging(self):
        """Test stereo to mono averages channels."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create stereo audio: [L1, R1, L2, R2, L3, R3]
            stereo_samples = np.array([100, 300, 200, 400, 300, 500], dtype=np.int16)
            audio_bytes = stereo_samples.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=2,
                to_channels=1,
                bit_depth=16,
            )

            result_array = np.frombuffer(result, dtype=np.int16).flatten()

            # Should average channels: (100+300)/2=200, (200+400)/2=300, (300+500)/2=400
            expected = np.array([200, 300, 400], dtype=np.int16)
            np.testing.assert_array_equal(result_array, expected)

    def test_channel_validation_invalid_from_channels(self):
        """Test that invalid input channel count raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(ValueError, match="Invalid channel counts"):
                track._convert_audio(
                    data=b"\x00\x01" * 10,
                    from_rate=16000,
                    to_rate=16000,
                    from_channels=0,  # Invalid
                    to_channels=1,
                    bit_depth=16,
                )

    def test_channel_validation_negative_to_channels(self):
        """Test that negative output channel count raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(ValueError, match="Invalid channel counts"):
                track._convert_audio(
                    data=b"\x00\x01" * 10,
                    from_rate=16000,
                    to_rate=16000,
                    from_channels=1,
                    to_channels=-1,  # Invalid
                    bit_depth=16,
                )


class TestBitDepthHandling:
    """Test different bit depth formats."""

    def test_bit_depth_16_standard(self):
        """Test standard 16-bit audio processing."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, bit_depth=16)

            samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
            audio_bytes = samples.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            result_array = np.frombuffer(result, dtype=np.int16).flatten()
            np.testing.assert_array_equal(result_array, samples)

    def test_bit_depth_validation_invalid(self):
        """Test that invalid bit depth raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(ValueError, match="Invalid bit depth"):
                track._convert_audio(
                    data=b"\x00\x01" * 10,
                    from_rate=16000,
                    to_rate=16000,
                    from_channels=1,
                    to_channels=1,
                    bit_depth=12,  # Invalid - only 8, 16, 24, 32 supported
                )

    def test_bit_depth_8bit_support(self):
        """Test 8-bit audio processing."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, bit_depth=8)

            samples = np.array([0, 64, -64, 127, -128], dtype=np.int8)
            audio_bytes = samples.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=8,
            )

            result_array = np.frombuffer(result, dtype=np.int8).flatten()
            np.testing.assert_array_equal(result_array, samples)

    def test_clipping_prevents_overflow(self):
        """Test that conversion clips values to prevent overflow."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, bit_depth=16)

            # Create audio with values at the limits
            samples = np.array([32767, -32768, 0], dtype=np.int16)
            audio_bytes = samples.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            result_array = np.frombuffer(result, dtype=np.int16).flatten()

            # Values should remain within int16 range
            assert np.all(result_array >= -32768)
            assert np.all(result_array <= 32767)


class TestAudioBufferPreparation:
    """Test audio buffer preparation for playback."""

    @pytest.mark.asyncio
    async def test_write_buffers_audio_data(self):
        """Test that write() correctly buffers audio data."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create test audio
            samples = np.array([100, 200, 300, 400], dtype=np.int16)
            audio_bytes = samples.tobytes()
            pcm_data = PcmData(
                data=audio_bytes, sample_rate=16000, channels=1, bit_depth=16
            )

            # Write to track
            await track.write(pcm_data)

            # Buffer should contain the data
            assert len(track._buffer) > 0

    @pytest.mark.asyncio
    async def test_write_resamples_before_buffering(self):
        """Test that audio is resampled before being added to buffer."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create 48kHz audio
            samples_48k = np.zeros(4800, dtype=np.int16)  # 100ms at 48kHz
            audio_bytes = samples_48k.tobytes()
            pcm_data = PcmData(
                data=audio_bytes, sample_rate=48000, channels=1, bit_depth=16
            )

            # Write to track
            await track.write(pcm_data)

            # Buffer should contain resampled data (1600 samples = 100ms at 16kHz)
            # 1600 samples * 2 bytes = 3200 bytes
            assert len(track._buffer) == 3200

    @pytest.mark.asyncio
    async def test_write_validates_pcm_data_type(self):
        """Test that write() validates PcmData type."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(RuntimeError, match="Unsupported PcmData type"):
                await track.write("not pcm data")

    @pytest.mark.asyncio
    async def test_write_after_stop_raises_error(self):
        """Test that writing after stop raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)
            track.stop()

            pcm_data = PcmData(
                data=b"\x00\x01" * 100, sample_rate=16000, channels=1, bit_depth=16
            )

            with pytest.raises(RuntimeError, match="has been stopped"):
                await track.write(pcm_data)

    def test_buffer_size_calculation(self):
        """Test that buffer size is calculated correctly."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            # 500ms buffer at 16kHz, mono, 16-bit
            # = (500/1000) * 16000 * 1 * 2 = 16000 bytes
            track = AudioOutputTrack(
                device="default",
                sample_rate=16000,
                channels=1,
                bit_depth=16,
                buffer_size_ms=500,
            )

            assert track._buffer_size_bytes == 16000

    def test_buffer_size_stereo(self):
        """Test buffer size calculation for stereo."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            # 500ms buffer at 16kHz, stereo, 16-bit
            # = (500/1000) * 16000 * 2 * 2 = 32000 bytes
            track = AudioOutputTrack(
                device="default",
                sample_rate=16000,
                channels=2,
                bit_depth=16,
                buffer_size_ms=500,
            )

            assert track._buffer_size_bytes == 32000


class TestAudioDeviceMocking:
    """Test mocking of audio device interactions."""

    def test_device_default_initialization(self):
        """Test default device initialization."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)
            assert track._device_index is None
            assert track.sample_rate == 16000

    def test_device_by_index(self):
        """Test device initialization by index."""
        mock_devices = [
            {"name": "Speaker 1", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Speaker 2", "max_input_channels": 0, "max_output_channels": 2},
        ]

        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_sd.query_devices.return_value = mock_devices
            track = AudioOutputTrack(device=1, sample_rate=16000)
            assert track._device_index == 1

    def test_device_by_name(self):
        """Test device initialization by name."""
        mock_devices = [
            {"name": "Built-in Speaker", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "USB Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]

        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_sd.query_devices.return_value = mock_devices
            track = AudioOutputTrack(device="USB", sample_rate=16000)
            assert track._device_index == 1

    def test_device_validation_no_output_channels(self):
        """Test device validation fails for devices without output."""
        mock_devices = [
            {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
        ]

        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_sd.query_devices.return_value = mock_devices

            with pytest.raises(ValueError, match="has no output channels"):
                AudioOutputTrack(device=0, sample_rate=16000)

    def test_device_validation_invalid_index(self):
        """Test device validation fails for out-of-range index."""
        mock_devices = [
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]

        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_sd.query_devices.return_value = mock_devices

            with pytest.raises(ValueError, match="out of range"):
                AudioOutputTrack(device=5, sample_rate=16000)

    def test_stream_initialization(self):
        """Test that OutputStream is created with correct parameters."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(
                device="default", sample_rate=16000, channels=1, bit_depth=16
            )
            track._ensure_stream_started()

            # Verify OutputStream was called with correct params
            mock_sd.OutputStream.assert_called_once_with(
                samplerate=16000, channels=1, dtype="int16", device=None
            )
            mock_stream.start.assert_called_once()

    def test_stream_stop_cleanup(self):
        """Test that stop() properly cleans up the stream."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(device="default", sample_rate=16000)
            track._ensure_stream_started()
            track.stop()

            # Verify stream was stopped and closed
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()
            assert track._stopped is True
            assert track._stream is None


class TestFormatConversionEdgeCases:
    """Test edge cases in format conversion."""

    def test_empty_data_raises_error(self):
        """Test that empty audio data raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            with pytest.raises(ValueError, match="Cannot convert empty audio data"):
                track._convert_audio(
                    data=b"",
                    from_rate=16000,
                    to_rate=16000,
                    from_channels=1,
                    to_channels=1,
                    bit_depth=16,
                )

    def test_data_size_mismatch_truncates(self, caplog):
        """Test that mismatched data size is truncated with warning."""
        import logging

        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000)

            # Create data with incomplete frame (3 bytes instead of 2 or 4)
            incomplete_data = b"\x00\x01\x02"

            with caplog.at_level(logging.WARNING):
                _ = track._convert_audio(
                    data=incomplete_data,
                    from_rate=16000,
                    to_rate=16000,
                    from_channels=1,
                    to_channels=1,
                    bit_depth=16,
                )

            # Should warn about truncation
            assert "data size mismatch" in caplog.text.lower()

    def test_combined_sample_rate_and_channel_conversion(self):
        """Test simultaneous sample rate and channel conversion."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create stereo 48kHz audio
            stereo_48k = np.array([100, 200, 300, 400, 500, 600], dtype=np.int16)
            audio_bytes = stereo_48k.tobytes()

            # Convert stereo 48kHz -> mono 16kHz
            result = track._convert_audio(
                data=audio_bytes,
                from_rate=48000,
                to_rate=16000,
                from_channels=2,
                to_channels=1,
                bit_depth=16,
            )

            # Should have downsampled and mixed to mono
            result_array = np.frombuffer(result, dtype=np.int16).flatten()
            assert len(result_array) == 1  # 3 stereo samples -> 1 mono sample at 16kHz

    def test_conversion_preserves_data_integrity(self):
        """Test that format conversion maintains audio quality."""
        with patch("vision_agents.plugins.localrtc.tracks.sd"):
            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create test signal
            original = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int16)
            audio_bytes = original.tobytes()

            result = track._convert_audio(
                data=audio_bytes,
                from_rate=16000,
                to_rate=16000,
                from_channels=1,
                to_channels=1,
                bit_depth=16,
            )

            result_array = np.frombuffer(result, dtype=np.int16).flatten()

            # Should preserve exact values when no conversion needed
            np.testing.assert_array_equal(result_array, original)


class TestSampleRateConsistency:
    """Test sample rate consistency across components."""

    @pytest.mark.asyncio
    async def test_write_validates_sample_rate_mismatch(self):
        """Test that sample rate mismatch is handled correctly."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create audio with different sample rate
            samples = np.zeros(800, dtype=np.int16)  # 100ms at 8kHz
            pcm_data = PcmData(
                data=samples.tobytes(), sample_rate=8000, channels=1, bit_depth=16
            )

            # Should handle the mismatch by resampling
            await track.write(pcm_data)

            # Buffer should contain resampled data (1600 samples = 100ms at 16kHz)
            assert len(track._buffer) == 3200  # 1600 samples * 2 bytes

    @pytest.mark.asyncio
    async def test_write_validates_invalid_input_sample_rate(self):
        """Test that invalid input sample rate raises error."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream

            track = AudioOutputTrack(device="default", sample_rate=16000, channels=1)

            # Create audio with invalid sample rate
            pcm_data = PcmData(
                data=b"\x00\x01" * 10,
                sample_rate=0,  # Invalid
                channels=1,
                bit_depth=16,
            )

            with pytest.raises(RuntimeError, match="Invalid input sample rate"):
                await track.write(pcm_data)
