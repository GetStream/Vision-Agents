import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Any

import av
from aiortc import MediaStreamTrack

from stream_agents.plugins.gemini.realtime2 import Realtime2


class MockVideoFrame:
    """Mock video frame for testing"""
    def __init__(self, frame_id: int = 0):
        self.frame_id = frame_id
        self.timestamp = frame_id * 1000
    
    def to_ndarray(self, format: str = "rgb24"):
        """Mock to_ndarray method"""
        import numpy as np
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class MockVideoTrack(MediaStreamTrack):
    """Mock video track that produces frames"""
    def __init__(self, frames: List[MockVideoFrame] = None, delay: float = 0.01):
        super().__init__()
        self.frames = frames or []
        self.delay = delay
        self.current_frame = 0
        self._closed = False
    
    async def recv(self):
        if self._closed or self.current_frame >= len(self.frames):
            raise asyncio.CancelledError("Track closed")
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        frame = self.frames[self.current_frame]
        self.current_frame += 1
        return frame
    
    def stop(self):
        self._closed = True


class MockSession:
    """Mock Gemini session for testing"""
    def __init__(self):
        self.sent_media = []
        self.sent_audio = []
        self.sent_text = []
    
    async def send_realtime_input(self, *, media=None, audio=None, text=None):
        if media is not None:
            self.sent_media.append(media)
        if audio is not None:
            self.sent_audio.append(audio)
        if text is not None:
            self.sent_text.append(text)


class TestRealtime2VideoSender:
    """Test suite for Realtime2 video sender functionality"""
    
    @pytest.fixture
    def mock_frames(self):
        """Create mock video frames for testing"""
        return [MockVideoFrame(i) for i in range(5)]
    
    @pytest.fixture
    def mock_track(self, mock_frames):
        """Create mock video track"""
        return MockVideoTrack(frames=mock_frames, delay=0.001)
    
    @pytest.fixture
    def realtime2(self):
        """Create Realtime2 instance for testing"""
        with patch('stream_agents.plugins.gemini.realtime2.genai.Client'):
            return Realtime2(model="test-model")
    
    @pytest.fixture
    def mock_session(self):
        """Create mock session"""
        return MockSession()
    
    @pytest.mark.asyncio
    async def test_frame_to_png_bytes_with_pil(self, realtime2):
        """Test frame to PNG conversion with PIL available"""
        frame = MockVideoFrame()
        
        with patch('stream_agents.plugins.gemini.realtime2.Image') as mock_image:
            mock_img = MagicMock()
            mock_image.fromarray.return_value = mock_img
            mock_buf = MagicMock()
            mock_buf.getvalue.return_value = b"fake_png_data"
            
            with patch('io.BytesIO', return_value=mock_buf):
                result = realtime2._frame_to_png_bytes(frame)
                
                assert result == b"fake_png_data"
                mock_image.fromarray.assert_called_once()
                mock_img.save.assert_called_once_with(mock_buf, format="PNG")
    
    @pytest.mark.asyncio
    async def test_frame_to_png_bytes_without_pil(self, realtime2):
        """Test frame to PNG conversion without PIL"""
        frame = MockVideoFrame()
        
        with patch('stream_agents.plugins.gemini.realtime2.Image', None):
            result = realtime2._frame_to_png_bytes(frame)
            assert result == b""
    
    @pytest.mark.asyncio
    async def test_frame_to_png_bytes_with_to_image(self, realtime2):
        """Test frame to PNG conversion with frame.to_image method"""
        frame = MockVideoFrame()
        frame.to_image = MagicMock(return_value=MagicMock())
        
        with patch('stream_agents.plugins.gemini.realtime2.Image') as mock_image:
            mock_buf = MagicMock()
            mock_buf.getvalue.return_value = b"fake_png_data"
            
            with patch('io.BytesIO', return_value=mock_buf):
                result = realtime2._frame_to_png_bytes(frame)
                
                assert result == b"fake_png_data"
                frame.to_image.assert_called_once()
                mock_image.fromarray.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_video_sender_creates_forwarder(self, realtime2, mock_track):
        """Test that start_video_sender creates a VideoForwarder"""
        realtime2._session = MockSession()
        
        await realtime2.start_video_sender(mock_track, fps=2)
        
        assert realtime2._video_forwarder is not None
        assert realtime2._video_forwarder.input_track == mock_track
        assert realtime2._video_forwarder.fps == 2.0
    
    @pytest.mark.asyncio
    async def test_start_video_sender_stops_existing(self, realtime2, mock_track):
        """Test that start_video_sender stops existing forwarder"""
        realtime2._session = MockSession()
        
        # Start first forwarder
        await realtime2.start_video_sender(mock_track, fps=1)
        first_forwarder = realtime2._video_forwarder
        
        # Start second forwarder
        await realtime2.start_video_sender(mock_track, fps=2)
        
        assert realtime2._video_forwarder is not None
        assert realtime2._video_forwarder != first_forwarder
        assert realtime2._video_forwarder.fps == 2.0
    
    @pytest.mark.asyncio
    async def test_stop_video_sender(self, realtime2, mock_track):
        """Test stopping video sender"""
        realtime2._session = MockSession()
        
        await realtime2.start_video_sender(mock_track, fps=1)
        assert realtime2._video_forwarder is not None
        
        await realtime2.stop_video_sender()
        assert realtime2._video_forwarder is None
    
    @pytest.mark.asyncio
    async def test_send_video_frame_with_session(self, realtime2, mock_session):
        """Test sending video frame with active session"""
        realtime2._session = mock_session
        frame = MockVideoFrame()
        
        with patch.object(realtime2, '_frame_to_png_bytes', return_value=b"fake_png_data"):
            await realtime2._send_video_frame(frame)
            
            assert len(mock_session.sent_media) == 1
            blob = mock_session.sent_media[0]
            assert blob.data == b"fake_png_data"
            assert blob.mime_type == "image/png"
    
    @pytest.mark.asyncio
    async def test_send_video_frame_without_session(self, realtime2):
        """Test sending video frame without active session"""
        realtime2._session = None
        frame = MockVideoFrame()
        
        # Should not raise exception
        await realtime2._send_video_frame(frame)
    
    @pytest.mark.asyncio
    async def test_send_video_frame_empty_png(self, realtime2, mock_session):
        """Test sending video frame with empty PNG data"""
        realtime2._session = mock_session
        frame = MockVideoFrame()
        
        with patch.object(realtime2, '_frame_to_png_bytes', return_value=b""):
            await realtime2._send_video_frame(frame)
            
            # Should not send empty data
            assert len(mock_session.sent_media) == 0
    
    @pytest.mark.asyncio
    async def test_send_video_frame_error_handling(self, realtime2, mock_session):
        """Test error handling in send_video_frame"""
        realtime2._session = mock_session
        frame = MockVideoFrame()
        
        with patch.object(realtime2, '_frame_to_png_bytes', side_effect=Exception("Conversion error")):
            # Should not raise exception
            await realtime2._send_video_frame(frame)
            
            # Should not send anything due to error
            assert len(mock_session.sent_media) == 0
    
    @pytest.mark.asyncio
    async def test_video_sender_integration(self, realtime2, mock_track, mock_session):
        """Test full video sender integration"""
        realtime2._session = mock_session
        
        # Start video sender
        await realtime2.start_video_sender(mock_track, fps=10)
        
        try:
            # Let it run briefly to process frames
            await asyncio.sleep(0.1)
            
            # Should have sent some frames
            assert len(mock_session.sent_media) > 0
            
            # Verify all sent media are PNG blobs
            for blob in mock_session.sent_media:
                assert blob.mime_type == "image/png"
                assert isinstance(blob.data, bytes)
                assert len(blob.data) > 0
                
        finally:
            await realtime2.stop_video_sender()
    
    @pytest.mark.asyncio
    async def test_video_sender_fps_throttling(self, realtime2, mock_track, mock_session):
        """Test that video sender respects FPS throttling"""
        realtime2._session = mock_session
        
        # Start with low FPS
        await realtime2.start_video_sender(mock_track, fps=1)
        
        try:
            # Let it run for a bit
            await asyncio.sleep(0.5)
            
            # Should have sent frames, but not too many due to FPS limit
            sent_count = len(mock_session.sent_media)
            assert sent_count > 0
            assert sent_count <= 2  # Should be around 1 FPS
            
        finally:
            await realtime2.stop_video_sender()
    
    @pytest.mark.asyncio
    async def test_video_sender_with_track_errors(self, realtime2, mock_session):
        """Test video sender handles track errors gracefully"""
        # Create track that will error after a few frames
        error_track = MockVideoTrack(frames=[MockVideoFrame(i) for i in range(3)], delay=0.01)
        
        # Mock track to raise error after 2 frames
        original_recv = error_track.recv
        call_count = 0
        
        async def failing_recv():
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise Exception("Track error")
            return await original_recv()
        
        error_track.recv = failing_recv
        realtime2._session = mock_session
        
        # Start video sender
        await realtime2.start_video_sender(error_track, fps=10)
        
        try:
            # Let it run and handle the error
            await asyncio.sleep(0.1)
            
            # Should have sent some frames before error
            assert len(mock_session.sent_media) > 0
            
        finally:
            await realtime2.stop_video_sender()
