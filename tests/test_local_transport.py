"""Tests for LocalTransport audio/video I/O."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import queue

import numpy as np
import pytest

from getstream.video.rtc.track_util import AudioFormat, PcmData
from tests.base_test import BaseTest


# Mock sounddevice before importing local_transport
@pytest.fixture(autouse=True)
def mock_sounddevice():
    """Mock sounddevice module for CI compatibility."""
    mock_sd = MagicMock()
    mock_sd.CallbackFlags = MagicMock()

    # Mock InputStream
    mock_input_stream = MagicMock()
    mock_input_stream.start = MagicMock()
    mock_input_stream.stop = MagicMock()
    mock_input_stream.close = MagicMock()
    mock_sd.InputStream.return_value = mock_input_stream

    # Mock OutputStream
    mock_output_stream = MagicMock()
    mock_output_stream.start = MagicMock()
    mock_output_stream.stop = MagicMock()
    mock_output_stream.close = MagicMock()
    mock_sd.OutputStream.return_value = mock_output_stream

    # Mock query_devices
    mock_sd.query_devices.return_value = "Test Device List"
    mock_sd.default = MagicMock()
    mock_sd.default.device = [0, 1]

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        yield mock_sd


@pytest.fixture
def mock_av():
    """Mock av (PyAV) module for CI compatibility."""
    mock_av_module = MagicMock()

    # Mock VideoFrame
    mock_frame = MagicMock()
    mock_frame.pts = 0
    mock_frame.time_base = MagicMock()
    mock_frame.width = 640
    mock_frame.height = 480

    # Mock stream
    mock_stream = MagicMock()

    # Mock container
    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.demux.return_value = [[mock_frame]]

    mock_av_module.open.return_value = mock_container
    mock_av_module.VideoFrame.return_value = mock_frame
    mock_av_module.Fraction = MagicMock()

    return mock_av_module


@pytest.fixture
def mock_aiortc():
    """Mock aiortc module for CI compatibility."""
    mock_aiortc_module = MagicMock()

    # Mock VideoStreamTrack base class
    mock_video_stream_track = MagicMock()
    mock_aiortc_module.VideoStreamTrack = mock_video_stream_track

    return mock_aiortc_module


class TestLocalOutputAudioTrack(BaseTest):
    """Tests for LocalOutputAudioTrack class."""

    async def test_create_output_audio_track(self, mock_sounddevice):
        """Test creating an output audio track."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2)
        assert track._sample_rate == 48000
        assert track._channels == 2
        assert not track._stopped

    async def test_audio_track_start(self, mock_sounddevice):
        """Test starting the audio output stream."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2)
        track.start()

        assert track._running
        assert track._playback_thread is not None
        assert track._playback_thread.is_alive()

        # Clean up
        track.stop()

    async def test_audio_track_write(self, mock_sounddevice):
        """Test writing PCM data to the track."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2)
        track.start()

        # Create test PCM data
        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=48000,
            format=AudioFormat.S16,
            channels=2,
        )

        await track.write(pcm)

        # Verify data was queued
        assert not track._queue.empty()

    async def test_audio_track_stop(self, mock_sounddevice):
        """Test stopping the audio track."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2)
        track.start()
        track.stop()

        assert track._stopped
        assert not track._running

    async def test_audio_track_flush(self, mock_sounddevice):
        """Test flushing the audio track clears the queue."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2)
        track.start()

        # Add some data to the queue
        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=48000,
            format=AudioFormat.S16,
            channels=2,
        )
        await track.write(pcm)

        assert not track._queue.empty()

        await track.flush()

        assert track._queue.empty()

    async def test_audio_track_playback_thread_processes_queue(self, mock_sounddevice):
        """Test that the playback thread processes queued data."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2, blocksize=4)
        track.start()

        # Put data in the queue
        test_data = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int16)
        track._queue.put(test_data)

        # Give the playback thread time to process
        await asyncio.sleep(0.1)

        # Queue should be empty after processing
        assert track._queue.empty()

        track.stop()

    async def test_audio_track_queue_has_max_size(self, mock_sounddevice):
        """Test that the audio queue has a max size to prevent memory buildup."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        track = LocalOutputAudioTrack(sample_rate=48000, channels=2, blocksize=4)

        # Queue should have a max size
        assert track._queue.maxsize == 100

    async def test_audio_track_resampling(self, mock_sounddevice):
        """Test that PCM data is resampled when needed."""
        from vision_agents.core.edge.local_transport import LocalOutputAudioTrack

        # Create track with 48kHz stereo output
        track = LocalOutputAudioTrack(sample_rate=48000, channels=2)
        track.start()

        # Write 16kHz mono data
        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )

        await track.write(pcm)

        # Verify data was queued (resampling is handled by PcmData.resample)
        assert not track._queue.empty()


class TestLocalTransport(BaseTest):
    """Tests for LocalTransport class."""

    async def test_transport_initialization(self, mock_sounddevice):
        """Test LocalTransport initializes correctly."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport(
            sample_rate=48000,
            input_channels=1,
            output_channels=2,
        )

        assert transport._sample_rate == 48000
        assert transport._input_channels == 1
        assert transport._output_channels == 2
        assert not transport._running

    async def test_create_output_audio_track(self, mock_sounddevice):
        """Test creating an output audio track via transport."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport()
        track = transport.create_output_audio_track(framerate=48000, stereo=True)

        assert track is not None
        assert track._sample_rate == 48000
        assert track._channels == 2

    async def test_connect_starts_microphone(self, mock_sounddevice):
        """Test that connect() starts microphone capture."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport()

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.agent_user = MagicMock()

        connection = await transport.connect(mock_agent)

        assert transport._running
        assert connection is not None
        mock_sounddevice.InputStream.assert_called_once()
        mock_sounddevice.InputStream.return_value.start.assert_called_once()

        # Cleanup
        await transport.disconnect()

    async def test_disconnect_stops_audio(self, mock_sounddevice):
        """Test that disconnect() stops all audio."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport()

        mock_agent = MagicMock()
        await transport.connect(mock_agent)

        await transport.disconnect()

        assert not transport._running
        mock_sounddevice.InputStream.return_value.stop.assert_called_once()
        mock_sounddevice.InputStream.return_value.close.assert_called_once()

    async def test_publish_tracks_starts_output(self, mock_sounddevice):
        """Test that publish_tracks starts the audio output."""
        from vision_agents.core.edge.local_transport import (
            LocalTransport,
            LocalOutputAudioTrack,
        )

        transport = LocalTransport()
        track = transport.create_output_audio_track()

        await transport.publish_tracks(track, None)

        assert track._running
        assert track._playback_thread is not None

        # Clean up
        track.stop()

    async def test_register_user_is_noop(self, mock_sounddevice):
        """Test that register_user is a no-op."""
        from vision_agents.core.edge.local_transport import LocalTransport
        from vision_agents.core.edge.types import User

        transport = LocalTransport()
        user = User(id="test", name="Test User")

        # Should not raise
        await transport.register_user(user)

    async def test_subscribe_to_track_returns_none_for_unknown(self, mock_sounddevice):
        """Test that subscribe_to_track returns None for unknown track IDs."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport()
        result = transport.subscribe_to_track("some-track-id")

        assert result is None

    async def test_subscribe_to_track_returns_video_track(self, mock_sounddevice, mock_av):
        """Test that subscribe_to_track returns video track for local-video."""
        with patch.dict("sys.modules", {"av": mock_av}):
            with patch(
                "vision_agents.core.edge.local_transport.PYAV_AVAILABLE", True
            ):
                with patch(
                    "vision_agents.core.edge.local_transport.AIORTC_AVAILABLE", True
                ):
                    from vision_agents.core.edge.local_transport import LocalTransport

                    transport = LocalTransport(video_device="0")
                    transport.create_video_track()

                    result = transport.subscribe_to_track("local-video")

                    assert result is not None
                    assert result._device == "0"

    async def test_create_chat_channel_returns_none(self, mock_sounddevice):
        """Test that create_chat_channel returns None."""
        from vision_agents.core.edge.local_transport import LocalTransport
        from vision_agents.core.edge.types import User

        transport = LocalTransport()
        user = User(id="test", name="Test")

        result = await transport.create_chat_channel(None, user, "instructions")

        assert result is None


class TestLocalConnection(BaseTest):
    """Tests for LocalConnection class."""

    async def test_idle_since_returns_zero(self, mock_sounddevice):
        """Test that idle_since always returns 0."""
        from vision_agents.core.edge.local_transport import (
            LocalTransport,
            LocalConnection,
        )

        transport = LocalTransport()
        connection = LocalConnection(transport)

        assert connection.idle_since() == 0.0

    async def test_wait_for_participant_returns_immediately(self, mock_sounddevice):
        """Test that wait_for_participant returns immediately."""
        from vision_agents.core.edge.local_transport import (
            LocalTransport,
            LocalConnection,
        )

        transport = LocalTransport()
        connection = LocalConnection(transport)

        # Should not block
        await asyncio.wait_for(
            connection.wait_for_participant(timeout=10.0), timeout=1.0
        )

    async def test_connection_close(self, mock_sounddevice):
        """Test closing the connection stops audio."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport()
        mock_agent = MagicMock()
        connection = await transport.connect(mock_agent)

        await connection.close()

        assert not transport._running


class TestAudioReceivedEvent(BaseTest):
    """Tests for audio event emission."""

    async def test_microphone_emits_audio_event(self, mock_sounddevice):
        """Test that microphone input emits AudioReceivedEvent."""
        from vision_agents.core.edge.local_transport import LocalTransport
        from vision_agents.core.edge.events import AudioReceivedEvent

        transport = LocalTransport()

        received_events = []

        @transport.events.subscribe
        async def on_audio(event: AudioReceivedEvent):
            received_events.append(event)

        # Simulate microphone callback
        mock_data = np.array([[100], [200], [300], [400]], dtype=np.int16)
        await transport._microphone_callback_async(mock_data)

        # Give a moment for event to process
        await asyncio.sleep(0.01)

        assert len(received_events) == 1
        event = received_events[0]
        assert event.pcm_data is not None
        assert event.participant is not None
        assert event.participant.user_id == "local-user"


class TestListAudioDevices(BaseTest):
    """Tests for the list_audio_devices utility."""

    def test_list_audio_devices(self, mock_sounddevice, capsys):
        """Test listing audio devices."""
        from vision_agents.core.edge.local_transport import list_audio_devices

        list_audio_devices()

        captured = capsys.readouterr()
        assert "Available audio devices" in captured.out
        mock_sounddevice.query_devices.assert_called_once()


class TestLocalVideoTrack(BaseTest):
    """Tests for LocalVideoTrack class."""

    async def test_video_track_initialization(self, mock_sounddevice, mock_av):
        """Test creating a video track."""
        with patch.dict("sys.modules", {"av": mock_av}):
            with patch(
                "vision_agents.core.edge.local_transport.PYAV_AVAILABLE", True
            ):
                with patch(
                    "vision_agents.core.edge.local_transport.AIORTC_AVAILABLE", True
                ):
                    from vision_agents.core.edge.local_transport import (
                        LocalVideoTrack,
                    )

                    track = LocalVideoTrack(
                        device="0", width=640, height=480, fps=30
                    )

                    assert track._device == "0"
                    assert track._width == 640
                    assert track._height == 480
                    assert track._fps == 30
                    assert not track._started
                    assert not track._stopped

    async def test_video_track_stop(self, mock_sounddevice, mock_av):
        """Test stopping a video track."""
        with patch.dict("sys.modules", {"av": mock_av}):
            with patch(
                "vision_agents.core.edge.local_transport.PYAV_AVAILABLE", True
            ):
                with patch(
                    "vision_agents.core.edge.local_transport.AIORTC_AVAILABLE", True
                ):
                    from vision_agents.core.edge.local_transport import (
                        LocalVideoTrack,
                    )

                    track = LocalVideoTrack(device="0")
                    track._container = mock_av.open.return_value

                    track.stop()

                    assert track._stopped
                    assert track._container is None


class TestLocalTransportVideo(BaseTest):
    """Tests for LocalTransport video functionality."""

    async def test_transport_with_video_device(self, mock_sounddevice):
        """Test LocalTransport initializes with video device."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport(
            sample_rate=48000,
            video_device="0",
            video_width=1280,
            video_height=720,
            video_fps=15,
        )

        assert transport._video_device == "0"
        assert transport._video_width == 1280
        assert transport._video_height == 720
        assert transport._video_fps == 15

    async def test_transport_without_video_device(self, mock_sounddevice):
        """Test LocalTransport without video device returns None for video track."""
        from vision_agents.core.edge.local_transport import LocalTransport

        transport = LocalTransport()

        assert transport._video_device is None
        track = transport.create_video_track()
        assert track is None

    async def test_create_video_track_with_device(self, mock_sounddevice, mock_av):
        """Test creating a video track when device is configured."""
        with patch.dict("sys.modules", {"av": mock_av}):
            with patch(
                "vision_agents.core.edge.local_transport.PYAV_AVAILABLE", True
            ):
                with patch(
                    "vision_agents.core.edge.local_transport.AIORTC_AVAILABLE", True
                ):
                    from vision_agents.core.edge.local_transport import (
                        LocalTransport,
                    )

                    transport = LocalTransport(video_device="0")
                    track = transport.create_video_track()

                    assert track is not None
                    assert track._device == "0"


class TestCameraEnumeration(BaseTest):
    """Tests for camera enumeration functions."""

    def test_list_cameras_returns_list(self, mock_sounddevice):
        """Test that list_cameras returns a list."""
        with patch(
            "vision_agents.core.edge.local_transport.PYAV_AVAILABLE", True
        ):
            with patch("subprocess.run") as mock_run:
                # Mock ffmpeg output
                mock_run.return_value = MagicMock(
                    stderr="[AVFoundation video devices:]\n"
                    "[AVFoundation @ 0x1] [0] FaceTime HD Camera\n"
                    "[AVFoundation audio devices:]\n"
                )

                from vision_agents.core.edge.local_transport import list_cameras

                with patch("platform.system", return_value="Darwin"):
                    cameras = list_cameras()

                assert isinstance(cameras, list)

    def test_get_camera_input_format_darwin(self, mock_sounddevice):
        """Test camera input format for macOS."""
        from vision_agents.core.edge.local_transport import (
            _get_camera_input_format,
        )

        with patch("platform.system", return_value="Darwin"):
            assert _get_camera_input_format() == "avfoundation"

    def test_get_camera_input_format_linux(self, mock_sounddevice):
        """Test camera input format for Linux."""
        from vision_agents.core.edge.local_transport import (
            _get_camera_input_format,
        )

        with patch("platform.system", return_value="Linux"):
            assert _get_camera_input_format() == "v4l2"

    def test_get_camera_input_format_windows(self, mock_sounddevice):
        """Test camera input format for Windows."""
        from vision_agents.core.edge.local_transport import (
            _get_camera_input_format,
        )

        with patch("platform.system", return_value="Windows"):
            assert _get_camera_input_format() == "dshow"
