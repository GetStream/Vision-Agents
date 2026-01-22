"""Tests for audio and video track implementations."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision_agents.core.types import PcmData
from vision_agents.plugins.localrtc.tracks import AudioInputTrack


def test_audio_input_track_init_default_device():
    """Test AudioInputTrack initialization with default device."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioInputTrack(device="default")
        assert track.sample_rate == 16000
        assert track.channels == 1
        assert track.bit_depth == 16
        assert track._device_index is None


def test_audio_input_track_init_custom_params():
    """Test AudioInputTrack initialization with custom parameters."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioInputTrack(
            device="default", sample_rate=48000, channels=2, bit_depth=24
        )
        assert track.sample_rate == 48000
        assert track.channels == 2
        assert track.bit_depth == 24


def test_audio_input_track_init_device_index():
    """Test AudioInputTrack initialization with device index."""
    mock_devices = [
        {"name": "Microphone 1", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        track = AudioInputTrack(device=0)
        assert track._device_index == 0


def test_audio_input_track_init_invalid_device_index():
    """Test AudioInputTrack initialization with invalid device index."""
    mock_devices = [
        {"name": "Microphone 1", "max_input_channels": 2, "max_output_channels": 0},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices

        with pytest.raises(ValueError, match="Device index 5 out of range"):
            AudioInputTrack(device=5)


def test_audio_input_track_init_device_no_input_channels():
    """Test AudioInputTrack initialization with device that has no input channels."""
    mock_devices = [
        {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices

        with pytest.raises(ValueError, match="has no input channels"):
            AudioInputTrack(device=0)


def test_audio_input_track_init_device_by_name():
    """Test AudioInputTrack initialization with device name."""
    mock_devices = [
        {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "External USB Mic", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        track = AudioInputTrack(device="USB")
        assert track._device_index == 1


def test_audio_input_track_init_device_name_not_found():
    """Test AudioInputTrack initialization with non-existent device name."""
    mock_devices = [
        {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices

        with pytest.raises(ValueError, match="not found"):
            AudioInputTrack(device="NonExistent")


def test_audio_input_track_init_no_sounddevice():
    """Test AudioInputTrack initialization when sounddevice is not available."""
    with patch("vision_agents.plugins.localrtc.tracks.sd", None):
        with pytest.raises(RuntimeError, match="sounddevice is not available"):
            AudioInputTrack()


def test_audio_input_track_init_invalid_device_type():
    """Test AudioInputTrack initialization with invalid device type."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        with pytest.raises(ValueError, match="Invalid device type"):
            AudioInputTrack(device=12.5)  # type: ignore


def test_audio_input_track_init_device_objects():
    """Test AudioInputTrack initialization with device objects instead of dicts."""
    mock_device1 = MagicMock()
    mock_device1.name = "Microphone Object"
    mock_device1.max_input_channels = 1
    mock_device1.max_output_channels = 0

    mock_device2 = MagicMock()
    mock_device2.name = "USB Mic Object"
    mock_device2.max_input_channels = 2
    mock_device2.max_output_channels = 0

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = [mock_device1, mock_device2]
        track = AudioInputTrack(device="USB")
        assert track._device_index == 1


def test_audio_input_track_capture():
    """Test AudioInputTrack.capture method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Create mock audio data
        sample_rate = 16000
        duration = 1.0
        channels = 1
        bit_depth = 16
        frames = int(duration * sample_rate)

        mock_audio = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )
        mock_sd.rec.return_value = mock_audio
        mock_sd.wait.return_value = None

        track = AudioInputTrack(device="default", sample_rate=sample_rate)

        # Capture audio
        start_time = time.time()
        pcm_data = track.capture(duration)
        end_time = time.time()

        # Verify PcmData
        assert isinstance(pcm_data, PcmData)
        assert pcm_data.sample_rate == sample_rate
        assert pcm_data.channels == channels
        assert pcm_data.bit_depth == bit_depth
        assert pcm_data.data == mock_audio.tobytes()
        assert pcm_data.timestamp is not None
        assert start_time <= pcm_data.timestamp <= end_time

        # Verify sounddevice calls
        mock_sd.rec.assert_called_once_with(
            frames=frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=None,
        )
        mock_sd.wait.assert_called_once()


def test_audio_input_track_capture_custom_params():
    """Test AudioInputTrack.capture with custom parameters."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        sample_rate = 48000
        duration = 2.0
        channels = 2
        bit_depth = 24
        frames = int(duration * sample_rate)

        mock_audio = np.random.randint(
            -8388608, 8388607, size=(frames, channels), dtype=np.int32
        )
        mock_sd.rec.return_value = mock_audio
        mock_sd.wait.return_value = None
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 2, "max_output_channels": 0}
        ]

        track = AudioInputTrack(
            device=0, sample_rate=sample_rate, channels=channels, bit_depth=bit_depth
        )
        pcm_data = track.capture(duration)

        assert pcm_data.sample_rate == sample_rate
        assert pcm_data.channels == channels
        assert pcm_data.bit_depth == bit_depth

        mock_sd.rec.assert_called_once_with(
            frames=frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="int24",
            device=0,
        )


def test_audio_input_track_capture_invalid_duration():
    """Test AudioInputTrack.capture with invalid duration."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioInputTrack(device="default")

        with pytest.raises(ValueError, match="Duration must be positive"):
            track.capture(0)

        with pytest.raises(ValueError, match="Duration must be positive"):
            track.capture(-1.0)


def test_audio_input_track_capture_error():
    """Test AudioInputTrack.capture handles errors gracefully."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.rec.side_effect = Exception("Device error")

        track = AudioInputTrack(device="default")

        with pytest.raises(RuntimeError, match="Failed to capture audio"):
            track.capture(1.0)


def test_audio_input_track_capture_error_with_device_index():
    """Test AudioInputTrack.capture error message includes device index."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]
        mock_sd.rec.side_effect = Exception("Device busy")

        track = AudioInputTrack(device=0)

        with pytest.raises(RuntimeError, match="device 0"):
            track.capture(1.0)


@pytest.mark.asyncio
async def test_audio_input_track_capture_async():
    """Test AudioInputTrack.capture_async method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        sample_rate = 16000
        duration = 0.5
        channels = 1
        frames = int(duration * sample_rate)

        mock_audio = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )
        mock_sd.rec.return_value = mock_audio
        mock_sd.wait.return_value = None

        track = AudioInputTrack(device="default")
        pcm_data = await track.capture_async(duration)

        assert isinstance(pcm_data, PcmData)
        assert pcm_data.sample_rate == sample_rate
        assert pcm_data.channels == channels
        assert pcm_data.data == mock_audio.tobytes()


def test_validate_device_index_with_device_objects():
    """Test _validate_device_index with device objects."""
    mock_device = MagicMock()
    mock_device.name = "Test Mic"
    mock_device.max_input_channels = 1
    mock_device.max_output_channels = 0

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = [mock_device]
        track = AudioInputTrack(device=0)
        assert track._device_index == 0


def test_validate_device_index_exception():
    """Test _validate_device_index handles exceptions."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.side_effect = Exception("Query failed")

        with pytest.raises(ValueError, match="Error validating device index"):
            AudioInputTrack(device=0)


def test_find_device_by_name_exception():
    """Test _find_device_by_name handles exceptions."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.side_effect = Exception("Query failed")

        with pytest.raises(ValueError, match="Error searching for device"):
            AudioInputTrack(device="TestMic")
