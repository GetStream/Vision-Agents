import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Any

import av
from aiortc import VideoStreamTrack

from stream_agents.plugins.gemini.queue import LatestNQueue
from stream_agents.plugins.gemini.realtime2 import VideoForwarder
from tests.base_test import BaseTest


class MockVideoFrame:
    """Mock video frame for testing"""
    def __init__(self, frame_id: int = 0):
        self.frame_id = frame_id
        self.timestamp = frame_id * 1000  # Simulate timestamp
    
    def to_ndarray(self, format: str = "rgb24"):
        """Mock to_ndarray method"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class MockVideoTrack(VideoStreamTrack):
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


class TestLatestNQueue:
    """Test suite for LatestNQueue"""
    
    @pytest.mark.asyncio
    async def test_basic_put_get(self):
        """Test basic put and get operations"""
        queue = LatestNQueue[int](maxlen=3)
        
        await queue.put_latest(1)
        await queue.put_latest(2)
        await queue.put_latest(3)
        
        assert await queue.get() == 1
        assert await queue.get() == 2
        assert await queue.get() == 3
    
    @pytest.mark.asyncio
    async def test_put_latest_discards_oldest(self):
        """Test that put_latest discards oldest items when full"""
        queue = LatestNQueue[int](maxlen=2)
        
        await queue.put_latest(1)
        await queue.put_latest(2)
        await queue.put_latest(3)  # Should discard 1
        
        assert await queue.get() == 2
        assert await queue.get() == 3
        
        # Queue should be empty now
        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()
    
    @pytest.mark.asyncio
    async def test_put_latest_nowait(self):
        """Test synchronous put_latest_nowait"""
        queue = LatestNQueue[int](maxlen=2)
        
        queue.put_latest_nowait(1)
        queue.put_latest_nowait(2)
        queue.put_latest_nowait(3)  # Should discard 1
        
        assert queue.get_nowait() == 2
        assert queue.get_nowait() == 3
    
    @pytest.mark.asyncio
    async def test_put_latest_nowait_discards_oldest(self):
        """Test that put_latest_nowait discards oldest when full"""
        queue = LatestNQueue[int](maxlen=3)
        
        # Fill queue
        queue.put_latest_nowait(1)
        queue.put_latest_nowait(2)
        queue.put_latest_nowait(3)
        
        # Add more items, should discard oldest
        queue.put_latest_nowait(4)  # Discards 1
        queue.put_latest_nowait(5)  # Discards 2
        
        # Should have 3, 4, 5
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        
        assert items == [3, 4, 5]
    
    @pytest.mark.asyncio
    async def test_queue_size_limits(self):
        """Test that queue respects size limits"""
        queue = LatestNQueue[int](maxlen=1)
        
        await queue.put_latest(1)
        assert queue.full()
        
        # Adding another should discard the first
        await queue.put_latest(2)
        assert queue.full()
        assert await queue.get() == 2
    
    @pytest.mark.asyncio
    async def test_generic_type_support(self):
        """Test that queue works with different types"""
        # Test with strings
        str_queue = LatestNQueue[str](maxlen=2)
        await str_queue.put_latest("a")
        await str_queue.put_latest("b")
        await str_queue.put_latest("c")  # Should discard "a"
        
        assert await str_queue.get() == "b"
        assert await str_queue.get() == "c"
        
        # Test with custom objects
        class TestObj:
            def __init__(self, value):
                self.value = value
        
        obj_queue = LatestNQueue[TestObj](maxlen=2)
        await obj_queue.put_latest(TestObj(1))
        await obj_queue.put_latest(TestObj(2))
        await obj_queue.put_latest(TestObj(3))  # Should discard first
        
        obj2 = await obj_queue.get()
        obj3 = await obj_queue.get()
        assert obj2.value == 2
        assert obj3.value == 3


class TestVideoForwarder(BaseTest):
    """Test suite for VideoForwarder using real video data"""
    
    @pytest.mark.asyncio
    async def test_video_forwarder_initialization(self, bunny_video_track):
        """Test VideoForwarder initialization"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=5, fps=30.0)
        
        assert forwarder.input_track == bunny_video_track
        assert forwarder.queue.maxsize == 5
        assert forwarder.fps == 30.0
        assert len(forwarder._tasks) == 0
        assert not forwarder._stopped.is_set()
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, bunny_video_track):
        """Test start and stop lifecycle"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3)
        
        # Start forwarder
        await forwarder.start()
        assert len(forwarder._tasks) == 1
        assert not forwarder._stopped.is_set()
        
        # Let it run briefly
        await asyncio.sleep(0.01)
        
        # Stop forwarder
        await forwarder.stop()
        assert len(forwarder._tasks) == 0
        assert forwarder._stopped.is_set()
    
    @pytest.mark.asyncio
    async def test_next_frame_pull_model(self, bunny_video_track):
        """Test next_frame pull model"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3)
        
        await forwarder.start()
        
        try:
            # Get first frame
            frame = await forwarder.next_frame(timeout=1.0)
            assert hasattr(frame, 'to_ndarray')  # Real video frame
            
            # Get a few more frames
            for _ in range(3):
                frame = await forwarder.next_frame(timeout=1.0)
                assert hasattr(frame, 'to_ndarray')  # Real video frame
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_next_frame_timeout(self):
        """Test next_frame timeout behavior"""
        # Create track with no frames
        empty_track = MockVideoTrack(frames=[])
        forwarder = VideoForwarder(empty_track, max_buffer=3)
        
        await forwarder.start()
        
        try:
            with pytest.raises(asyncio.TimeoutError):
                await forwarder.next_frame(timeout=0.1)
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_next_frame_coalesces_to_newest(self, bunny_video_track):
        """Test that next_frame coalesces backlog to newest frame"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=5)
        
        await forwarder.start()
        
        try:
            # Let multiple frames accumulate
            await asyncio.sleep(0.05)
            
            # Get frame - should be the newest available
            frame = await forwarder.next_frame(timeout=1.0)
            assert hasattr(frame, 'to_ndarray')  # Real video frame
            
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_callback_push_model(self, bunny_video_track):
        """Test callback-based push model"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        received_frames = []
        
        def on_frame(frame):
            received_frames.append(frame)
        
        await forwarder.start()
        
        try:
            # Start callback consumer
            await forwarder.start_event_consumer(on_frame)
            
            # Let it run and collect frames
            await asyncio.sleep(0.1)
            
            # Should have received some frames
            assert len(received_frames) > 0
            for frame in received_frames:
                assert hasattr(frame, 'to_ndarray')  # Real video frame
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_async_callback_push_model(self, bunny_video_track):
        """Test async callback-based push model"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        received_frames = []
        
        async def async_on_frame(frame):
            received_frames.append(frame)
            await asyncio.sleep(0.001)  # Simulate async work
        
        await forwarder.start()
        
        try:
            # Start async callback consumer
            await forwarder.start_event_consumer(async_on_frame)
            
            # Let it run and collect frames
            await asyncio.sleep(0.1)
            
            # Should have received some frames
            assert len(received_frames) > 0
            for frame in received_frames:
                assert hasattr(frame, 'to_ndarray')  # Real video frame
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_fps_throttling(self, bunny_video_track):
        """Test FPS throttling in callback mode"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=5.0)  # 5 FPS
        
        received_frames = []
        timestamps = []
        
        def on_frame(frame):
            received_frames.append(frame)
            timestamps.append(asyncio.get_event_loop().time())
        
        await forwarder.start()
        
        try:
            await forwarder.start_event_consumer(on_frame)
            
            # Let it run for a bit
            await asyncio.sleep(0.5)
            
            # Should have received frames
            assert len(received_frames) > 0
            
            # Check that frames are throttled (roughly 5 FPS)
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                # Should be roughly 1/5 = 0.2 seconds between frames
                assert avg_interval >= 0.15  # Allow some tolerance
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_consumers(self, bunny_video_track):
        """Test multiple callback consumers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3)
        
        received_frames_1 = []
        received_frames_2 = []
        
        def on_frame_1(frame):
            received_frames_1.append(frame)
        
        def on_frame_2(frame):
            received_frames_2.append(frame)
        
        await forwarder.start()
        
        try:
            # Start two consumers
            await forwarder.start_event_consumer(on_frame_1)
            await forwarder.start_event_consumer(on_frame_2)
            
            # Let them run
            await asyncio.sleep(0.1)
            
            # Both should have received frames
            assert len(received_frames_1) > 0
            assert len(received_frames_2) > 0
            
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_producer_handles_track_errors(self, bunny_video_track):
        """Test that producer handles track errors gracefully"""
        # Mock track to raise exception after a few frames
        call_count = 0
        original_recv = bunny_video_track.recv
        
        async def failing_recv():
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise Exception("Track error")
            return await original_recv()
        
        bunny_video_track.recv = failing_recv
        
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3)
        
        await forwarder.start()
        
        try:
            # Should still be able to get some frames before error
            frame = await forwarder.next_frame(timeout=1.0)
            assert hasattr(frame, 'to_ndarray')  # Real video frame
            
            # Let it run a bit more to trigger error
            await asyncio.sleep(0.1)
            
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_stop_drains_queue(self, bunny_video_track):
        """Test that stop drains the queue"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=5)
        
        await forwarder.start()
        
        try:
            # Let some frames accumulate
            await asyncio.sleep(0.05)
            
            # Stop should drain queue
            await forwarder.stop()
            
            # Queue should be empty after stop
            assert forwarder.queue.empty()
            
        except Exception:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_no_fps_limit(self, bunny_video_track):
        """Test behavior when fps is None (no limit)"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=None)
        
        received_frames = []
        timestamps = []
        
        def on_frame(frame):
            received_frames.append(frame)
            timestamps.append(asyncio.get_event_loop().time())
        
        await forwarder.start()
        
        try:
            await forwarder.start_event_consumer(on_frame)
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Should have received frames
            assert len(received_frames) > 0
            
            # With no FPS limit, frames should come as fast as possible
            # (limited by track delay and processing time)
            
        finally:
            await forwarder.stop()
    
    async def test_frame_count_at_10fps(self, bunny_video_track):
        """Test that VideoForwarder generates ~30 frames at 10fps from 3-second video"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=10, fps=10.0)
        
        received_frames = []
        timestamps = []
        
        def on_frame(frame):
            received_frames.append(frame)
            timestamps.append(asyncio.get_event_loop().time())
        
        await forwarder.start()
        
        try:
            await forwarder.start_event_consumer(on_frame)
            
            # Let it run for the full 3-second video duration
            await asyncio.sleep(3.5)  # Slightly longer to ensure we get all frames
            
            # Should have received approximately 30 frames (3 seconds * 10 fps)
            # Allow some tolerance for timing variations
            assert 25 <= len(received_frames) <= 35, f"Expected ~30 frames, got {len(received_frames)}"
            
            # Verify all frames are real video frames
            for frame in received_frames:
                assert hasattr(frame, 'to_ndarray')
            
            # Check that frames are roughly at 10fps intervals
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                # Should be roughly 1/10 = 0.1 seconds between frames
                assert 0.08 <= avg_interval <= 0.12, f"Expected ~0.1s intervals, got {avg_interval:.3f}s"
            
            print(f"Received {len(received_frames)} frames at 10fps from 3-second video")
            
        finally:
            await forwarder.stop()


class TestVideoForwarderIntegration(BaseTest):
    """Integration tests for VideoForwarder with real video data"""
    
    @pytest.mark.asyncio
    async def test_video_forwarder_with_output_track(self, bunny_video_track):
        """Test VideoForwarder with an output track scenario"""
        # Mock output track
        output_track = AsyncMock()
        
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        # Simulate writing to output track
        async def write_to_output(frame):
            await output_track.write(frame)
        
        await forwarder.start()
        
        try:
            # Start consumer that writes to output track
            await forwarder.start_event_consumer(write_to_output)
            
            # Let it run
            await asyncio.sleep(0.2)
            
            # Verify output track received frames
            assert output_track.write.call_count > 0
            
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_video_forwarder_with_callback_processing(self, bunny_video_track):
        """Test VideoForwarder with callback-based processing"""
        processed_frames = []
        
        async def process_frame(frame):
            # Simulate frame processing
            processed_data = frame.to_ndarray()
            processed_frames.append({
                'data_shape': processed_data.shape,
                'has_to_ndarray': hasattr(frame, 'to_ndarray')
            })
        
        forwarder = VideoForwarder(bunny_video_track, max_buffer=4, fps=20.0)
        
        await forwarder.start()
        
        try:
            await forwarder.start_event_consumer(process_frame)
            
            # Let processing run
            await asyncio.sleep(0.15)
            
            # Verify frames were processed
            assert len(processed_frames) > 0
            
            # Verify processing data
            for processed in processed_frames:
                assert 'data_shape' in processed
                assert 'has_to_ndarray' in processed
                assert processed['has_to_ndarray'] is True
                # Real video frames will have varying shapes
                assert len(processed['data_shape']) == 3  # height, width, channels
                
        finally:
            await forwarder.stop()
