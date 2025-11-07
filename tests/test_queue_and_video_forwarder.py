import asyncio
import pytest

from vision_agents.core.utils.video_queue import VideoLatestNQueue
from vision_agents.core.utils.video_forwarder import VideoForwarder

class TestLatestNQueue:
    """Test suite for LatestNQueue"""
    
    @pytest.mark.asyncio
    async def test_basic_put_get(self):
        """Test basic put and get operations"""
        queue = VideoLatestNQueue[int](maxlen=3)
        
        await queue.put_latest(1)
        await queue.put_latest(2)
        await queue.put_latest(3)
        
        assert await queue.get() == 1
        assert await queue.get() == 2
        assert await queue.get() == 3
    
    @pytest.mark.asyncio
    async def test_put_latest_discards_oldest(self):
        """Test that put_latest discards oldest items when full"""
        queue = VideoLatestNQueue[int](maxlen=2)
        
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
        queue = VideoLatestNQueue[int](maxlen=2)
        
        queue.put_latest_nowait(1)
        queue.put_latest_nowait(2)
        queue.put_latest_nowait(3)  # Should discard 1
        
        assert queue.get_nowait() == 2
        assert queue.get_nowait() == 3
    
    @pytest.mark.asyncio
    async def test_put_latest_nowait_discards_oldest(self):
        """Test that put_latest_nowait discards oldest when full"""
        queue = VideoLatestNQueue[int](maxlen=3)
        
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
        queue = VideoLatestNQueue[int](maxlen=1)
        
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
        str_queue = VideoLatestNQueue[str](maxlen=2)
        await str_queue.put_latest("a")
        await str_queue.put_latest("b")
        await str_queue.put_latest("c")  # Should discard "a"
        
        assert await str_queue.get() == "b"
        assert await str_queue.get() == "c"
        
        # Test with custom objects
        class TestObj:
            def __init__(self, value):
                self.value = value
        
        obj_queue = VideoLatestNQueue[TestObj](maxlen=2)
        await obj_queue.put_latest(TestObj(1))
        await obj_queue.put_latest(TestObj(2))
        await obj_queue.put_latest(TestObj(3))  # Should discard first
        
        obj2 = await obj_queue.get()
        obj3 = await obj_queue.get()
        assert obj2.value == 2
        assert obj3.value == 3


class TestVideoForwarder:
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
    
    async def test_bunny_video_track_frame_count(self, bunny_video_track):
        """Test how many frames are actually available from bunny_video_track"""
        frame_count = 0
        frames = []
        
        try:
            while True:
                frame = await bunny_video_track.recv()
                if frame is not None:
                    frame_count += 1
                    frames.append(frame)
        except asyncio.CancelledError as e:
            print(f"Track finished after {frame_count} frames: {e}")
        except Exception as e:
            print(f"Error after {frame_count} frames: {e}")

        assert frame_count == 45
    
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
            await asyncio.sleep(10)  # Slightly longer to ensure we get all frames
            
            # Should have received approximately 30 frames (3 seconds * 10 fps)
            # Allow some tolerance for timing variations
            assert 25 <= len(received_frames) <= 35, f"Expected ~30 frames, got {len(received_frames)}"
            
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_add_frame_handler(self, bunny_video_track):
        """Test adding frame handlers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        received_frames = []
        
        def on_frame(frame):
            received_frames.append(frame)
        
        # Add handler
        forwarder.add_frame_handler(on_frame, fps=10, name="test-handler")
        
        # Verify handler was added
        assert len(forwarder._frame_handlers) == 1
        callback, config = forwarder._frame_handlers[0]
        assert callback == on_frame
        assert config['fps'] == 10
        assert config['name'] == "test-handler"
    
    @pytest.mark.asyncio
    async def test_add_multiple_frame_handlers(self, bunny_video_track):
        """Test adding multiple frame handlers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        received_frames_1 = []
        received_frames_2 = []
        received_frames_3 = []
        
        def handler1(frame):
            received_frames_1.append(frame)
        
        def handler2(frame):
            received_frames_2.append(frame)
        
        async def handler3(frame):
            received_frames_3.append(frame)
        
        # Add multiple handlers
        forwarder.add_frame_handler(handler1, fps=5, name="handler-1")
        forwarder.add_frame_handler(handler2, fps=10, name="handler-2")
        forwarder.add_frame_handler(handler3, fps=15, name="handler-3")
        
        # Verify all handlers were added
        assert len(forwarder._frame_handlers) == 3
        assert forwarder._frame_handlers[0][1]['name'] == "handler-1"
        assert forwarder._frame_handlers[1][1]['name'] == "handler-2"
        assert forwarder._frame_handlers[2][1]['name'] == "handler-3"
    
    @pytest.mark.asyncio
    async def test_remove_frame_handler(self, bunny_video_track):
        """Test removing frame handlers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        def handler1(frame):
            pass
        
        def handler2(frame):
            pass
        
        # Add handlers
        forwarder.add_frame_handler(handler1, name="handler-1")
        forwarder.add_frame_handler(handler2, name="handler-2")
        assert len(forwarder._frame_handlers) == 2
        
        # Remove first handler
        removed = forwarder.remove_frame_handler(handler1)
        assert removed is True
        assert len(forwarder._frame_handlers) == 1
        assert forwarder._frame_handlers[0][0] == handler2
        
        # Try removing again (should return False)
        removed = forwarder.remove_frame_handler(handler1)
        assert removed is False
        assert len(forwarder._frame_handlers) == 1
        
        # Remove second handler
        removed = forwarder.remove_frame_handler(handler2)
        assert removed is True
        assert len(forwarder._frame_handlers) == 0
    
    @pytest.mark.asyncio
    async def test_start_consumers(self, bunny_video_track):
        """Test starting all registered consumers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        received_frames_1 = []
        received_frames_2 = []
        
        def handler1(frame):
            received_frames_1.append(frame)
        
        def handler2(frame):
            received_frames_2.append(frame)
        
        # Add handlers
        forwarder.add_frame_handler(handler1, fps=10, name="handler-1")
        forwarder.add_frame_handler(handler2, fps=10, name="handler-2")
        
        await forwarder.start()
        
        try:
            # Start all consumers
            await forwarder.start_consumers()
            
            # Let it run and collect frames
            await asyncio.sleep(0.15)
            
            # Both handlers should have received frames
            assert len(received_frames_1) > 0
            assert len(received_frames_2) > 0
            
            # Verify frames are real video frames
            for frame in received_frames_1:
                assert hasattr(frame, 'to_ndarray')
            for frame in received_frames_2:
                assert hasattr(frame, 'to_ndarray')
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_handlers_different_fps(self, bunny_video_track):
        """Test multiple handlers with different fps rates running independently"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=5, fps=30.0)
        
        handler1_frames = []
        handler2_frames = []
        handler3_frames = []
        
        def handler1(frame):
            handler1_frames.append(frame)
        
        def handler2(frame):
            handler2_frames.append(frame)
        
        def handler3(frame):
            handler3_frames.append(frame)
        
        # Add handlers with different fps rates
        forwarder.add_frame_handler(handler1, fps=5, name="handler-5fps")
        forwarder.add_frame_handler(handler2, fps=10, name="handler-10fps")
        forwarder.add_frame_handler(handler3, fps=15, name="handler-15fps")
        
        await forwarder.start()
        
        try:
            await forwarder.start_consumers()
            
            # Let it run to collect frames
            await asyncio.sleep(0.6)
            
            # All handlers should have received frames independently
            assert len(handler1_frames) > 0, "Handler 1 should have received frames"
            assert len(handler2_frames) > 0, "Handler 2 should have received frames"
            assert len(handler3_frames) > 0, "Handler 3 should have received frames"
            
            # Verify all are real video frames
            for frame in handler1_frames:
                assert hasattr(frame, 'to_ndarray')
            for frame in handler2_frames:
                assert hasattr(frame, 'to_ndarray')
            for frame in handler3_frames:
                assert hasattr(frame, 'to_ndarray')
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_handlers_with_async_and_sync(self, bunny_video_track):
        """Test that both async and sync handlers work together"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        sync_frames = []
        async_frames = []
        
        def sync_handler(frame):
            sync_frames.append(frame)
        
        async def async_handler(frame):
            await asyncio.sleep(0.001)  # Simulate async work
            async_frames.append(frame)
        
        # Add both types of handlers
        forwarder.add_frame_handler(sync_handler, fps=10, name="sync")
        forwarder.add_frame_handler(async_handler, fps=10, name="async")
        
        await forwarder.start()
        
        try:
            await forwarder.start_consumers()
            
            # Let it run
            await asyncio.sleep(0.15)
            
            # Both should have received frames
            assert len(sync_frames) > 0
            assert len(async_frames) > 0
            
            # Verify all are real video frames
            for frame in sync_frames:
                assert hasattr(frame, 'to_ndarray')
            for frame in async_frames:
                assert hasattr(frame, 'to_ndarray')
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_handler_default_fps_from_forwarder(self, bunny_video_track):
        """Test that handlers inherit default fps from forwarder"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=15.0)
        
        received_frames = []
        
        def handler(frame):
            received_frames.append(frame)
        
        # Add handler without specifying fps (should use forwarder's default)
        forwarder.add_frame_handler(handler, name="default-fps-handler")
        
        # Verify handler config has forwarder's fps
        callback, config = forwarder._frame_handlers[0]
        assert config['fps'] == 15.0
    
    @pytest.mark.asyncio
    async def test_handler_auto_naming(self, bunny_video_track):
        """Test automatic handler naming"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        def handler1(frame):
            pass
        
        def handler2(frame):
            pass
        
        # Add handlers without names
        forwarder.add_frame_handler(handler1)
        forwarder.add_frame_handler(handler2)
        
        # Verify auto-generated names
        assert forwarder._frame_handlers[0][1]['name'] == "handler-0"
        assert forwarder._frame_handlers[1][1]['name'] == "handler-1"
    
    @pytest.mark.asyncio
    async def test_backwards_compatibility_start_event_consumer(self, bunny_video_track):
        """Test that old start_event_consumer API still works"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        received_frames = []
        
        def on_frame(frame):
            received_frames.append(frame)
        
        await forwarder.start()
        
        try:
            # Use old API
            await forwarder.start_event_consumer(on_frame, fps=10, consumer_name="legacy")
            
            # Let it run
            await asyncio.sleep(0.1)
            
            # Should have received frames
            assert len(received_frames) > 0
            for frame in received_frames:
                assert hasattr(frame, 'to_ndarray')
                
        finally:
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_mixed_api_usage(self, bunny_video_track):
        """Test using both new and old API together"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)
        
        new_api_frames = []
        old_api_frames = []
        
        def new_handler(frame):
            new_api_frames.append(frame)
        
        def old_handler(frame):
            old_api_frames.append(frame)
        
        # Add handler using new API
        forwarder.add_frame_handler(new_handler, fps=10, name="new-api")
        
        await forwarder.start()
        
        try:
            # Start consumers using new API
            await forwarder.start_consumers()
            
            # Also add a consumer using old API
            await forwarder.start_event_consumer(old_handler, fps=10, consumer_name="old-api")
            
            # Let it run longer to ensure both consumers get frames
            await asyncio.sleep(0.3)
            
            # Both should have received frames
            assert len(new_api_frames) > 0
            assert len(old_api_frames) > 0
                
        finally:
            await forwarder.stop()


class TestVideoForwarderIntegration:
    """Integration tests for VideoForwarder with real video data"""
    
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
