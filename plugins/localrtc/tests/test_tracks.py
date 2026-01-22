"""Tests for audio and video track implementations."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision_agents.core.types import PcmData
from vision_agents.plugins.localrtc.tracks import (
    AudioInputTrack,
    AudioOutputTrack,
    VideoInputTrack,
)


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
    """Test AudioInputTrack.capture method with persistent stream."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Create mock audio data
        sample_rate = 16000
        duration = 0.1  # Use short duration for test
        channels = 1
        bit_depth = 16
        frames = int(duration * sample_rate)

        mock_audio = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )

        # Create mock InputStream that stores callback and simulates audio
        stored_callback = None

        def mock_input_stream_init(**kwargs):
            nonlocal stored_callback
            stored_callback = kwargs.get("callback")
            mock_stream = MagicMock()
            mock_stream.start = MagicMock()
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()
            return mock_stream

        mock_sd.InputStream = mock_input_stream_init

        track = AudioInputTrack(device="default", sample_rate=sample_rate)

        # Start the track (this creates the stream)
        track.start()

        # Simulate the callback being called with audio data
        assert stored_callback is not None
        stored_callback(mock_audio, frames, None, None)

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

        # Clean up
        track.stop()


def test_audio_input_track_capture_custom_params():
    """Test AudioInputTrack.capture with custom parameters."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        sample_rate = 48000
        duration = 0.1  # Use short duration for test
        channels = 2
        bit_depth = 16  # Use 16-bit for simpler test
        frames = int(duration * sample_rate)

        mock_audio = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 2, "max_output_channels": 0}
        ]

        # Create mock InputStream that stores callback
        stored_callback = None

        def mock_input_stream_init(**kwargs):
            nonlocal stored_callback
            stored_callback = kwargs.get("callback")
            mock_stream = MagicMock()
            mock_stream.start = MagicMock()
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()
            return mock_stream

        mock_sd.InputStream = mock_input_stream_init

        track = AudioInputTrack(
            device=0, sample_rate=sample_rate, channels=channels, bit_depth=bit_depth
        )
        track.start()

        # Simulate callback with audio data
        assert stored_callback is not None
        stored_callback(mock_audio, frames, None, None)

        pcm_data = track.capture(duration)

        assert pcm_data.sample_rate == sample_rate
        assert pcm_data.channels == channels
        assert pcm_data.bit_depth == bit_depth

        track.stop()


def test_audio_input_track_capture_invalid_duration():
    """Test AudioInputTrack.capture with invalid duration."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioInputTrack(device="default")

        with pytest.raises(ValueError, match="Duration must be positive"):
            track.capture(0)

        with pytest.raises(ValueError, match="Duration must be positive"):
            track.capture(-1.0)


def test_audio_input_track_capture_error():
    """Test AudioInputTrack.capture handles timeout when no audio data arrives."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Create mock InputStream that doesn't produce any audio
        def mock_input_stream_init(**kwargs):
            mock_stream = MagicMock()
            mock_stream.start = MagicMock()
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()
            return mock_stream

        mock_sd.InputStream = mock_input_stream_init

        track = AudioInputTrack(device="default")

        # Should timeout since no audio data is being produced
        with pytest.raises(RuntimeError, match="Timeout waiting for audio data"):
            track.capture(0.1)  # Short duration to speed up test


def test_audio_input_track_capture_error_with_device_index():
    """Test AudioInputTrack.capture timeout message."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]

        # Create mock InputStream that doesn't produce any audio
        def mock_input_stream_init(**kwargs):
            mock_stream = MagicMock()
            mock_stream.start = MagicMock()
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()
            return mock_stream

        mock_sd.InputStream = mock_input_stream_init

        track = AudioInputTrack(device=0)

        # Should timeout since no audio data is being produced
        with pytest.raises(RuntimeError, match="Timeout waiting for audio data"):
            track.capture(0.1)


@pytest.mark.asyncio
async def test_audio_input_track_capture_async():
    """Test AudioInputTrack.capture_async method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        sample_rate = 16000
        duration = 0.1  # Short duration for test
        channels = 1
        frames = int(duration * sample_rate)

        mock_audio = np.random.randint(
            -32768, 32767, size=(frames, channels), dtype=np.int16
        )

        # Create mock InputStream that stores callback
        stored_callback = None

        def mock_input_stream_init(**kwargs):
            nonlocal stored_callback
            stored_callback = kwargs.get("callback")
            mock_stream = MagicMock()
            mock_stream.start = MagicMock()
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()
            return mock_stream

        mock_sd.InputStream = mock_input_stream_init

        track = AudioInputTrack(device="default")
        track.start()

        # Simulate callback with audio data
        assert stored_callback is not None
        stored_callback(mock_audio, frames, None, None)

        pcm_data = await track.capture_async(duration)

        assert isinstance(pcm_data, PcmData)
        track.stop()
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

        # Mock the OutputStream class
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

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

        # Write audio
        await track.write(pcm_data)

        # Verify OutputStream was created and started
        assert mock_sd.OutputStream.called
        assert mock_stream.start.called

        # Verify audio data was added to the buffer
        assert len(track._buffer) > 0


@pytest.mark.asyncio
async def test_audio_output_track_write_with_sample_rate_conversion():
    """Test AudioOutputTrack.write with sample rate conversion."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        track_rate = 16000
        data_rate = 48000
        channels = 1
        bit_depth = 16

        # Mock the OutputStream class
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

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

        # Write audio with conversion
        await track.write(pcm_data)

        # Verify OutputStream was created and started
        assert mock_sd.OutputStream.called
        assert mock_stream.start.called

        # Verify audio data was added to the buffer (should be resampled)
        # Original: 48000 Hz * 1s = 48000 samples
        # Resampled: 16000 Hz * 1s = 16000 samples
        # Each sample is 2 bytes (int16), so 16000 * 2 = 32000 bytes
        expected_bytes = int(track_rate * duration) * (bit_depth // 8)
        assert len(track._buffer) == expected_bytes


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
        # Mock OutputStream to raise an error when started
        mock_sd.OutputStream.side_effect = Exception("Device error")

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
        # Mock the OutputStream class
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        track = AudioOutputTrack(device="default")

        # Add some data to the buffer first
        track._buffer.extend(b"\x00\x01" * 100)

        # Start flush (it will wait for buffer to drain)
        # We need to drain the buffer to let flush complete
        track._buffer.clear()
        await track.flush()

        # Flush should complete without error when buffer is empty
        assert len(track._buffer) == 0


@pytest.mark.asyncio
async def test_audio_output_track_flush_after_stop():
    """Test AudioOutputTrack.flush after stop does not raise error."""
    with patch("vision_agents.plugins.localrtc.tracks.sd"):
        track = AudioOutputTrack(device="default")
        track.stop()

        # Should not raise error
        await track.flush()


@pytest.mark.asyncio
async def test_audio_output_track_flush_with_data():
    """Test AudioOutputTrack.flush waits for buffer to drain."""
    import asyncio

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Mock the OutputStream class
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        track = AudioOutputTrack(device="default")

        # Add some data to the buffer
        track._buffer.extend(b"\x00\x01" * 100)

        # Create a task to drain the buffer after a short delay
        async def drain_buffer():
            await asyncio.sleep(0.1)
            track._buffer.clear()

        # Start both flush and drain tasks
        drain_task = asyncio.create_task(drain_buffer())
        await track.flush()
        await drain_task

        # Buffer should be empty after flush
        assert len(track._buffer) == 0


def test_audio_output_track_stop():
    """Test AudioOutputTrack.stop method."""
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Mock the OutputStream class
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        track = AudioOutputTrack(device="default")

        # Start the stream first
        track._ensure_stream_started()

        # Now stop
        track.stop()

        assert track._stopped is True
        assert mock_stream.stop.called
        assert mock_stream.close.called
        assert track._stream is None


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


# VideoInputTrack tests


def test_video_input_track_init_default_device():
    """Test VideoInputTrack initialization with default device."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        track = VideoInputTrack(device=0)
        assert track.device == 0
        assert track.width == 640
        assert track.height == 480
        assert track.fps == 30


def test_video_input_track_init_custom_params():
    """Test VideoInputTrack initialization with custom parameters."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        track = VideoInputTrack(device=1, width=1920, height=1080, fps=60)
        assert track.device == 1
        assert track.width == 1920
        assert track.height == 1080
        assert track.fps == 60


def test_video_input_track_init_string_device():
    """Test VideoInputTrack initialization with device path."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        track = VideoInputTrack(device="/dev/video0")
        assert track.device == "/dev/video0"


def test_video_input_track_init_no_cv2():
    """Test VideoInputTrack initialization when cv2 is not available."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2", None):
        with pytest.raises(RuntimeError, match="opencv-python is not available"):
            VideoInputTrack()


def test_video_input_track_init_invalid_width():
    """Test VideoInputTrack initialization with invalid width."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        with pytest.raises(ValueError, match="Width must be positive"):
            VideoInputTrack(width=0)

        with pytest.raises(ValueError, match="Width must be positive"):
            VideoInputTrack(width=-100)


def test_video_input_track_init_invalid_height():
    """Test VideoInputTrack initialization with invalid height."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        with pytest.raises(ValueError, match="Height must be positive"):
            VideoInputTrack(height=0)

        with pytest.raises(ValueError, match="Height must be positive"):
            VideoInputTrack(height=-100)


def test_video_input_track_init_invalid_fps():
    """Test VideoInputTrack initialization with invalid fps."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        with pytest.raises(ValueError, match="FPS must be positive"):
            VideoInputTrack(fps=0)

        with pytest.raises(ValueError, match="FPS must be positive"):
            VideoInputTrack(fps=-30)


def test_video_input_track_open_device_success():
    """Test VideoInputTrack._open_device successfully opens device."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 480,
            mock_cv2.CAP_PROP_FPS: 30,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0)
        track._open_device()

        assert track._capture is not None
        mock_cv2.VideoCapture.assert_called_once_with(0)
        mock_capture.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_capture.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_HEIGHT, 480)
        mock_capture.set.assert_any_call(mock_cv2.CAP_PROP_FPS, 30)


def test_video_input_track_open_device_failed():
    """Test VideoInputTrack._open_device handles device open failure."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0)

        with pytest.raises(RuntimeError, match="Failed to open video device"):
            track._open_device()

        mock_capture.release.assert_called_once()


def test_video_input_track_open_device_exception():
    """Test VideoInputTrack._open_device handles exceptions."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_cv2.VideoCapture.side_effect = Exception("Device error")

        track = VideoInputTrack(device=0)

        with pytest.raises(RuntimeError, match="Error opening video device"):
            track._open_device()


def test_video_input_track_start_no_callback():
    """Test VideoInputTrack.start raises error when callback is None."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        track = VideoInputTrack(device=0)

        with pytest.raises(ValueError, match="Callback cannot be None"):
            track.start(callback=None)  # type: ignore


def test_video_input_track_start_already_running():
    """Test VideoInputTrack.start raises error when already running."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0)

        def callback(frame: np.ndarray, timestamp: float):
            pass

        track.start(callback=callback)

        with pytest.raises(RuntimeError, match="already running"):
            track.start(callback=callback)

        track.stop()


def test_video_input_track_start_and_stop():
    """Test VideoInputTrack.start and stop lifecycle."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0)

        def callback(frame: np.ndarray, timestamp: float):
            pass

        track.start(callback=callback)
        assert track._running is True
        assert track._capture is not None

        track.stop()
        assert track._running is False
        mock_capture.release.assert_called_once()


def test_video_input_track_stop_multiple_times():
    """Test VideoInputTrack.stop can be called multiple times safely."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0)

        def callback(frame: np.ndarray, timestamp: float):
            pass

        track.start(callback=callback)
        track.stop()
        track.stop()  # Should not raise error


def test_video_input_track_capture_frame():
    """Test VideoInputTrack.capture_frame captures a single frame."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640

        # Create mock frame
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_capture.read.return_value = (True, mock_frame)
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0, width=640, height=480)

        start_time = time.time()
        frame, timestamp = track.capture_frame()
        end_time = time.time()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert start_time <= timestamp <= end_time
        mock_capture.release.assert_called_once()


def test_video_input_track_capture_frame_failed():
    """Test VideoInputTrack.capture_frame handles read failure."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0)

        with pytest.raises(RuntimeError, match="Failed to read frame"):
            track.capture_frame()


def test_video_input_track_capture_loop():
    """Test VideoInputTrack._capture_loop delivers frames via callback."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640

        # Create mock frames
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_capture.read.return_value = (True, mock_frame)
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0, width=640, height=480, fps=30)

        # Track callback invocations
        frames_received = []

        def callback(frame: np.ndarray, timestamp: float):
            frames_received.append((frame, timestamp))

        track.start(callback=callback)

        # Let it run for a short time
        time.sleep(0.2)

        track.stop()

        # Should have received at least one frame
        assert len(frames_received) > 0

        # Verify frame structure
        frame, timestamp = frames_received[0]
        assert isinstance(frame, np.ndarray)
        assert isinstance(timestamp, float)


def test_video_input_track_capture_loop_read_error():
    """Test VideoInputTrack._capture_loop handles read errors gracefully."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_capture

        track = VideoInputTrack(device=0, width=640, height=480, fps=30)

        def callback(frame: np.ndarray, timestamp: float):
            pass

        # Redirect print to avoid test output noise
        with patch("builtins.print"):
            track.start(callback=callback)

            # Let it run briefly - should handle error without crashing
            time.sleep(0.1)

            track.stop()


def test_video_input_track_stop_when_not_running():
    """Test VideoInputTrack.stop when not running does nothing."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2"):
        track = VideoInputTrack(device=0)
        track.stop()  # Should not raise error
        assert track._running is False
