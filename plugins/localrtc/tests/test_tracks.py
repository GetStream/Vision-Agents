"""Tests for audio and video track implementations."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision_agents.core.types import PcmData
from vision_agents.plugins.localrtc.tracks import AudioInputTrack, AudioOutputTrack


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


# AudioOutputTrack tests


def test_audio_output_track_init_default_device():
    """Test AudioOutputTrack initialization with default device."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioOutputTrack(device="default")
        assert track.sample_rate == 16000
        assert track.channels == 1
        assert track.bit_depth == 16
        assert track._device_index is None


def test_audio_output_track_init_custom_params():
    """Test AudioOutputTrack initialization with custom parameters."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioOutputTrack(
            device="default", sample_rate=48000, channels=2, bit_depth=24
        )
        assert track.sample_rate == 48000
        assert track.channels == 2
        assert track.bit_depth == 24


def test_audio_output_track_init_device_index():
    """Test AudioOutputTrack initialization with device index."""
    mock_devices = [
        {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speaker 1", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        track = AudioOutputTrack(device=1)
        assert track._device_index == 1


def test_audio_output_track_init_invalid_device_index():
    """Test AudioOutputTrack initialization with invalid device index."""
    mock_devices = [
        {"name": "Speaker 1", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices

        with pytest.raises(ValueError, match="Device index 5 out of range"):
            AudioOutputTrack(device=5)


def test_audio_output_track_init_device_no_output_channels():
    """Test AudioOutputTrack initialization with device that has no output channels."""
    mock_devices = [
        {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices

        with pytest.raises(ValueError, match="has no output channels"):
            AudioOutputTrack(device=0)


def test_audio_output_track_init_device_by_name():
    """Test AudioOutputTrack initialization with device name."""
    mock_devices = [
        {"name": "Built-in Speaker", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "External USB Speaker", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        track = AudioOutputTrack(device="USB")
        assert track._device_index == 1


def test_audio_output_track_init_device_name_not_found():
    """Test AudioOutputTrack initialization with non-existent device name."""
    mock_devices = [
        {"name": "Built-in Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices

        with pytest.raises(ValueError, match="not found"):
            AudioOutputTrack(device="NonExistent")


def test_audio_output_track_init_no_sounddevice():
    """Test AudioOutputTrack initialization when sounddevice is not available."""
    with patch("vision_agents.plugins.localrtc.tracks.sd", None):
        with pytest.raises(RuntimeError, match="sounddevice is not available"):
            AudioOutputTrack()


def test_audio_output_track_init_invalid_device_type():
    """Test AudioOutputTrack initialization with invalid device type."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        with pytest.raises(ValueError, match="Invalid device type"):
            AudioOutputTrack(device=12.5)  # type: ignore


def test_audio_output_track_init_device_objects():
    """Test AudioOutputTrack initialization with device objects instead of dicts."""
    mock_device1 = MagicMock()
    mock_device1.name = "Speaker Object"
    mock_device1.max_input_channels = 0
    mock_device1.max_output_channels = 2

    mock_device2 = MagicMock()
    mock_device2.name = "USB Speaker Object"
    mock_device2.max_input_channels = 0
    mock_device2.max_output_channels = 2

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = [mock_device1, mock_device2]
        track = AudioOutputTrack(device="USB")
        assert track._device_index == 1


@pytest.mark.asyncio
async def test_audio_output_track_write():
    """Test AudioOutputTrack.write method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        sample_rate = 16000
        channels = 1
        bit_depth = 16

        track = AudioOutputTrack(
            device="default", sample_rate=sample_rate, channels=channels, bit_depth=bit_depth
        )

        # Create test audio data
        duration = 1.0
        frames = int(duration * sample_rate)
        audio_array = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )
        audio_bytes = audio_array.tobytes()

        pcm_data = PcmData(
            data=audio_bytes,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
        )

        # Mock play function
        mock_sd.play = MagicMock()

        # Write audio
        await track.write(pcm_data)

        # Verify play was called
        assert mock_sd.play.called


@pytest.mark.asyncio
async def test_audio_output_track_write_with_sample_rate_conversion():
    """Test AudioOutputTrack.write with sample rate conversion."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        track_rate = 16000
        data_rate = 48000
        channels = 1
        bit_depth = 16

        track = AudioOutputTrack(
            device="default", sample_rate=track_rate, channels=channels, bit_depth=bit_depth
        )

        # Create test audio data at different sample rate
        duration = 1.0
        frames = int(duration * data_rate)
        audio_array = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )
        audio_bytes = audio_array.tobytes()

        pcm_data = PcmData(
            data=audio_bytes,
            sample_rate=data_rate,
            channels=channels,
            bit_depth=bit_depth,
        )

        # Mock play function
        mock_sd.play = MagicMock()

        # Write audio with conversion
        await track.write(pcm_data)

        # Verify play was called
        assert mock_sd.play.called


@pytest.mark.asyncio
async def test_audio_output_track_write_after_stop():
    """Test AudioOutputTrack.write after stop raises error."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioOutputTrack(device="default")
        track.stop()

        pcm_data = PcmData(
            data=b"\x00\x01" * 100,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )

        with pytest.raises(RuntimeError, match="has been stopped"):
            await track.write(pcm_data)


@pytest.mark.asyncio
async def test_audio_output_track_write_error():
    """Test AudioOutputTrack.write handles errors gracefully."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.play = MagicMock(side_effect=Exception("Device error"))

        track = AudioOutputTrack(device="default")

        pcm_data = PcmData(
            data=b"\x00\x01" * 100,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )

        with pytest.raises(RuntimeError, match="Failed to play audio"):
            await track.write(pcm_data)


@pytest.mark.asyncio
async def test_audio_output_track_flush():
    """Test AudioOutputTrack.flush method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.wait = MagicMock()

        track = AudioOutputTrack(device="default")
        await track.flush()

        # Verify wait was called
        assert mock_sd.wait.called


@pytest.mark.asyncio
async def test_audio_output_track_flush_after_stop():
    """Test AudioOutputTrack.flush after stop does not raise error."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioOutputTrack(device="default")
        track.stop()

        # Should not raise error
        await track.flush()


@pytest.mark.asyncio
async def test_audio_output_track_flush_error():
    """Test AudioOutputTrack.flush handles errors gracefully."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.wait = MagicMock(side_effect=Exception("Wait error"))

        track = AudioOutputTrack(device="default")

        with pytest.raises(RuntimeError, match="Failed to flush audio output"):
            await track.flush()


def test_audio_output_track_stop():
    """Test AudioOutputTrack.stop method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.stop = MagicMock()

        track = AudioOutputTrack(device="default")
        track.stop()

        assert track._stopped is True
        assert mock_sd.stop.called


def test_audio_output_track_stop_multiple_times():
    """Test AudioOutputTrack.stop can be called multiple times safely."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.stop = MagicMock()

        track = AudioOutputTrack(device="default")
        track.stop()
        track.stop()  # Should not raise error

        assert track._stopped is True


def test_audio_output_track_stop_with_error():
    """Test AudioOutputTrack.stop handles errors gracefully."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.stop = MagicMock(side_effect=Exception("Stop error"))

        track = AudioOutputTrack(device="default")
        track.stop()  # Should not raise error

        assert track._stopped is True


def test_audio_output_track_validate_device_index_exception():
    """Test AudioOutputTrack _validate_device_index handles exceptions."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.side_effect = Exception("Query failed")

        with pytest.raises(ValueError, match="Error validating device index"):
            AudioOutputTrack(device=0)


def test_audio_output_track_find_device_by_name_exception():
    """Test AudioOutputTrack _find_device_by_name handles exceptions."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.side_effect = Exception("Query failed")

        with pytest.raises(ValueError, match="Error searching for device"):
            AudioOutputTrack(device="TestSpeaker")
