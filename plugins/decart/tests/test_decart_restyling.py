"""Tests for RestylingProcessor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import av
import pytest
from aiortc import MediaStreamTrack
from decart import DecartSDKError
import websockets

from vision_agents.plugins.decart import RestylingProcessor
from vision_agents.plugins.decart.decart_video_track import DecartVideoTrack


@pytest.fixture
def mock_video_track():
    """Mock video track."""
    track = MagicMock(spec=MediaStreamTrack)
    return track


@pytest.fixture
def sample_frame():
    """Test av.VideoFrame fixture."""
    from PIL import Image

    image = Image.new("RGB", (1280, 720), color="blue")
    return av.VideoFrame.from_image(image)


@pytest.fixture
def mock_decart_client():
    """Mock DecartClient with async close method."""
    with patch(
        "vision_agents.plugins.decart.decart_restyling_processor.DecartClient"
    ) as mock_client:
        mock_instance = MagicMock()
        mock_instance.close = AsyncMock()
        mock_instance.base_url = "https://api.decart.ai"
        mock_instance.api_key = "test_key"
        mock_client.return_value = mock_instance
        yield mock_client


class TestRestylingProcessor:
    """Tests for RestylingProcessor."""

    @pytest.mark.asyncio
    async def test_publish_video_track(self, mock_decart_client):
        """Test that publish_video_track returns DecartVideoTrack."""
        processor = RestylingProcessor(api_key="test_key")
        track = processor.publish_video_track()
        assert isinstance(track, DecartVideoTrack)
        await processor.close()

    @pytest.mark.asyncio
    async def test_process_video_triggers_connection(
        self, mock_video_track, mock_decart_client
    ):
        """Test that process_video triggers connection to Decart."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.connect = AsyncMock(return_value=mock_client_instance)
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            await processor.process_video(mock_video_track, None)

            assert processor._current_track == mock_video_track
            assert mock_realtime.connect.called
            await processor.close()

    @pytest.mark.asyncio
    async def test_process_video_prevents_duplicate_connections(
        self, mock_video_track, mock_decart_client
    ):
        """Test that process_video prevents duplicate connections."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.connect = AsyncMock(return_value=mock_client_instance)
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            processor._connecting = True

            await processor.process_video(mock_video_track, None)

            assert not mock_realtime.connect.called
            await processor.close()

    @pytest.mark.asyncio
    async def test_update_prompt_when_connected(self, mock_decart_client):
        """Test update_prompt updates prompt when connected."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")
            mock_client = AsyncMock()
            processor._realtime_client = mock_client
            processor._connected = True

            await processor.update_prompt("new style", enrich=False)

            mock_client.set_prompt.assert_called_once_with("new style", enrich=False)
            assert processor.initial_prompt == "new style"
            await processor.close()

    @pytest.mark.asyncio
    async def test_update_prompt_noop_when_disconnected(self, mock_decart_client):
        """Test update_prompt is no-op when disconnected."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(
                api_key="test_key", initial_prompt="original"
            )
            processor._realtime_client = None

            await processor.update_prompt("new style")

            assert processor.initial_prompt == "original"
            await processor.close()

    @pytest.mark.asyncio
    async def test_update_prompt_uses_default_enrich(self, mock_decart_client):
        """Test update_prompt uses default enrich value when not specified."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key", enrich=True)
            mock_client = AsyncMock()
            processor._realtime_client = mock_client

            await processor.update_prompt("new style")

            mock_client.set_prompt.assert_called_once_with("new style", enrich=True)
            await processor.close()

    @pytest.mark.asyncio
    async def test_set_mirror_when_connected(self, mock_decart_client):
        """Test set_mirror updates mirror mode when connected."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key", mirror=True)
            mock_client = AsyncMock()
            processor._realtime_client = mock_client

            await processor.set_mirror(False)

            mock_client.set_mirror.assert_called_once_with(False)
            assert processor.mirror is False
            await processor.close()

    @pytest.mark.asyncio
    async def test_set_mirror_noop_when_disconnected(self, mock_decart_client):
        """Test set_mirror is no-op when disconnected."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key", mirror=True)
            processor._realtime_client = None

            await processor.set_mirror(False)

            assert processor.mirror is True
            await processor.close()


class TestConnectionManagement:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_video_track, mock_decart_client):
        """Test connection lifecycle (connecting -> connected)."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.connect = AsyncMock(return_value=mock_client_instance)
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            assert not processor._connected
            assert not processor._connecting

            await processor._connect_to_decart(mock_video_track)

            assert processor._connected
            assert not processor._connecting
            assert processor._realtime_client is not None
            await processor.close()

    @pytest.mark.asyncio
    async def test_reconnection_on_connection_error(
        self, mock_video_track, mock_decart_client
    ):
        """Test reconnection on connection errors."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.connect = AsyncMock(return_value=mock_client_instance)
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            processor._current_track = mock_video_track

            error = DecartSDKError("connection timeout")
            processor._on_error(error)

            await asyncio.sleep(0.1)
            assert mock_realtime.connect.called
            await processor.close()

    @pytest.mark.asyncio
    async def test_reconnection_on_websocket_error(
        self, mock_video_track, mock_decart_client
    ):
        """Test reconnection on websocket connection errors."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.connect = AsyncMock(return_value=mock_client_instance)
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            processor._current_track = mock_video_track

            error = websockets.ConnectionClosedError(None, None)
            processor._on_error(error)

            await asyncio.sleep(0.1)
            assert mock_realtime.connect.called
            await processor.close()

    @pytest.mark.asyncio
    async def test_no_reconnection_on_non_connection_error(
        self, mock_video_track, mock_decart_client
    ):
        """Test no reconnection on non-connection errors."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")
            processor._current_track = mock_video_track

            error = DecartSDKError("invalid api key")
            processor._on_error(error)

            await asyncio.sleep(0.1)
            assert not processor._connected
            await processor.close()

    @pytest.mark.asyncio
    async def test_connection_change_updates_state(self, mock_decart_client):
        """Test that connection change events update state."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")

            processor._on_connection_change("connecting")
            assert processor._connected

            processor._on_connection_change("connected")
            assert processor._connected

            processor._on_connection_change("disconnected")
            assert not processor._connected

            processor._on_connection_change("error")
            assert not processor._connected
            await processor.close()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self, mock_video_track, mock_decart_client):
        """Test that disconnect cleans up properly."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.disconnect = AsyncMock()
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            await processor._connect_to_decart(mock_video_track)
            assert processor._connected

            await processor._disconnect_from_decart()

            assert not processor._connected
            assert processor._realtime_client is None
            mock_client_instance.disconnect.assert_called_once()
            await processor.close()

    @pytest.mark.asyncio
    async def test_processing_loop_reconnects(
        self, mock_video_track, mock_decart_client
    ):
        """Test that processing loop reconnects when connection is lost."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ) as mock_realtime:
            mock_client_instance = AsyncMock()
            mock_client_instance.connect = AsyncMock(return_value=mock_client_instance)
            mock_realtime.connect = AsyncMock(return_value=mock_client_instance)

            processor = RestylingProcessor(api_key="test_key")
            processor._current_track = mock_video_track
            processor._connected = False
            processor._connecting = False

            task = asyncio.create_task(processor._processing_loop())
            await asyncio.sleep(1.5)

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert mock_realtime.connect.called
            await processor.close()


class TestFrameHandling:
    """Tests for frame handling."""

    @pytest.mark.asyncio
    async def test_frames_received_from_decart_forwarded_to_track(
        self, sample_frame, mock_decart_client
    ):
        """Test that frames received from Decart are forwarded to video track."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")

            call_count = 0

            async def mock_recv():
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise asyncio.CancelledError()
                return sample_frame

            mock_transformed_stream = AsyncMock()
            mock_transformed_stream.recv = mock_recv

            task = asyncio.create_task(
                processor._receive_frames_from_decart(mock_transformed_stream)
            )
            await asyncio.sleep(0.1)
            processor._video_track.stop()

            try:
                await task
            except asyncio.CancelledError:
                pass

            assert processor._video_track.frame_queue.qsize() > 0
            await processor.close()

    @pytest.mark.asyncio
    async def test_frame_receiving_task_cancelled_on_close(
        self, sample_frame, mock_decart_client
    ):
        """Test that frame receiving task is cancelled on close."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")

            async def mock_recv():
                await asyncio.sleep(0.01)
                return sample_frame

            mock_transformed_stream = AsyncMock()
            mock_transformed_stream.recv = mock_recv

            task = asyncio.create_task(
                processor._receive_frames_from_decart(mock_transformed_stream)
            )
            await asyncio.sleep(0.05)

            await processor.close()
            await asyncio.sleep(0.1)

            assert task.done()
            await processor.close()

    @pytest.mark.asyncio
    async def test_on_remote_stream_starts_frame_receiving(
        self, sample_frame, mock_decart_client
    ):
        """Test that on_remote_stream starts frame receiving task."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")

            async def mock_recv():
                await asyncio.sleep(0.01)
                return sample_frame

            mock_transformed_stream = AsyncMock()
            mock_transformed_stream.recv = mock_recv

            processor._on_remote_stream(mock_transformed_stream)

            assert processor._frame_receiving_task is not None
            assert not processor._frame_receiving_task.done()

            await asyncio.sleep(0.1)
            processor._video_track.stop()
            await asyncio.sleep(0.1)

            await processor.close()

    @pytest.mark.asyncio
    async def test_on_remote_stream_cancels_previous_task(
        self, sample_frame, mock_decart_client
    ):
        """Test that on_remote_stream cancels previous frame receiving task."""
        with patch(
            "vision_agents.plugins.decart.decart_restyling_processor.RealtimeClient"
        ):
            processor = RestylingProcessor(api_key="test_key")

            async def mock_recv():
                await asyncio.sleep(0.01)
                return sample_frame

            mock_stream1 = AsyncMock()
            mock_stream1.recv = mock_recv

            mock_stream2 = AsyncMock()
            mock_stream2.recv = mock_recv

            processor._on_remote_stream(mock_stream1)
            task1 = processor._frame_receiving_task

            await asyncio.sleep(0.05)
            processor._on_remote_stream(mock_stream2)

            # Yield to allow cancellation to propagate
            await asyncio.sleep(0)
            assert task1.done()
            assert processor._frame_receiving_task != task1

            processor._video_track.stop()
            await asyncio.sleep(0.1)
            await processor.close()
