import asyncio

import pytest
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_queue import VideoLatestNQueue


class TestLatestNQueue:
    """Test suite for LatestNQueue"""

    async def test_basic_put_get(self):
        """Test basic put and get operations"""
        queue = VideoLatestNQueue[int](maxlen=3)

        await queue.put_latest(1)
        await queue.put_latest(2)
        await queue.put_latest(3)

        assert await queue.get() == 1
        assert await queue.get() == 2
        assert await queue.get() == 3

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

    async def test_put_latest_nowait(self):
        """Test synchronous put_latest_nowait"""
        queue = VideoLatestNQueue[int](maxlen=2)

        queue.put_latest_nowait(1)
        queue.put_latest_nowait(2)
        queue.put_latest_nowait(3)  # Should discard 1

        assert queue.get_nowait() == 2
        assert queue.get_nowait() == 3

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

    async def test_queue_size_limits(self):
        """Test that queue respects size limits"""
        queue = VideoLatestNQueue[int](maxlen=1)

        await queue.put_latest(1)
        assert queue.full()

        # Adding another should discard the first
        await queue.put_latest(2)
        assert queue.full()
        assert await queue.get() == 2

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
    """Test suite for VideoForwarder"""

    async def test_multiple_handlers_with_different_fps(self, bunny_video_track):
        """Test forwarding to 2 handlers with different FPS rates"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=5, fps=30.0)

        handler1_frames = []
        handler1_timestamps = []
        handler2_frames = []
        handler2_timestamps = []

        def handler_5fps(frame):
            handler1_frames.append(frame)
            handler1_timestamps.append(asyncio.get_event_loop().time())

        def handler_10fps(frame):
            handler2_frames.append(frame)
            handler2_timestamps.append(asyncio.get_event_loop().time())

        try:
            forwarder.add_frame_handler(handler_5fps, fps=5, name="handler-5fps")
            forwarder.add_frame_handler(handler_10fps, fps=10, name="handler-10fps")

            # Run for 1 second
            await asyncio.sleep(1.0)

            # Verify both received frames
            assert len(handler1_frames) > 0
            assert len(handler2_frames) > 0

            # Verify FPS rates (5 fps handler should get ~5 frames, 10 fps should get ~10 frames)
            assert 3 <= len(handler1_frames) <= 7, (
                f"Expected ~5 frames for 5fps handler, got {len(handler1_frames)}"
            )
            assert 7 <= len(handler2_frames) <= 13, (
                f"Expected ~10 frames for 10fps handler, got {len(handler2_frames)}"
            )

            # Verify timing for 5fps handler (should be ~0.2s between frames)
            if len(handler1_timestamps) > 1:
                intervals = [
                    handler1_timestamps[i + 1] - handler1_timestamps[i]
                    for i in range(len(handler1_timestamps) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                assert 0.15 <= avg_interval <= 0.25, (
                    f"Expected ~0.2s between 5fps frames, got {avg_interval:.2f}s"
                )

            # Verify timing for 10fps handler (should be ~0.1s between frames)
            if len(handler2_timestamps) > 1:
                intervals = [
                    handler2_timestamps[i + 1] - handler2_timestamps[i]
                    for i in range(len(handler2_timestamps) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                assert 0.08 <= avg_interval <= 0.15, (
                    f"Expected ~0.1s between 10fps frames, got {avg_interval:.2f}s"
                )

        finally:
            await forwarder.stop()

    async def test_handler_fps_exceeds_forwarder_fps_raises_error(
        self, bunny_video_track
    ):
        """Test that adding handler with FPS > forwarder FPS raises ValueError"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)

        def handler(frame):
            pass

        # Try to add handler with higher FPS than forwarder
        try:
            with pytest.raises(
                ValueError,
                match="fps on handler.*cannot be greater than fps on forwarder",
            ):
                forwarder.add_frame_handler(handler, fps=15)
        finally:
            await forwarder.stop()

    async def test_stop_stops_handlers(self, bunny_video_track):
        """Test that stop() stops frame delivery to handlers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)

        received_frames = []

        def handler(frame):
            received_frames.append(frame)

        # Start receiving frames
        forwarder.add_frame_handler(handler, fps=10)
        await asyncio.sleep(0.1)

        frames_before_stop = len(received_frames)
        assert frames_before_stop > 0, "Should have received some frames before stop"

        # Stop forwarder
        await forwarder.stop()
        assert not forwarder.started

        # Wait a bit and verify no new frames are received
        await asyncio.sleep(0.1)
        frames_after_stop = len(received_frames)
        assert frames_after_stop == frames_before_stop, (
            "Should not receive frames after stop"
        )

    async def test_add_and_remove_handlers(self, bunny_video_track):
        """Test adding and removing frame handlers"""
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)

        def handler1(frame):
            pass

        def handler2(frame):
            pass

        # Add handlers
        forwarder.add_frame_handler(handler1, fps=5, name="handler-1")
        forwarder.add_frame_handler(handler2, fps=10, name="handler-2")
        assert len(forwarder.frame_handlers) == 2
        assert forwarder.started

        # Remove first handler
        removed = await forwarder.remove_frame_handler(handler1)
        assert removed is True
        assert len(forwarder.frame_handlers) == 1
        assert forwarder.frame_handlers[0].callback == handler2
        assert forwarder.started  # Should still be running

        # Try removing handler that doesn't exist
        removed = await forwarder.remove_frame_handler(handler1)
        assert removed is False
        assert len(forwarder.frame_handlers) == 1

        # Remove last handler (should auto-stop)
        removed = await forwarder.remove_frame_handler(handler2)
        assert removed is True
        assert len(forwarder.frame_handlers) == 0
        assert not forwarder.started  # Should have stopped automatically

    async def test_handler_raises_an_error_forwarder_keeps_working(
        self, bunny_video_track
    ):
        """
        Test that one failing frame handler doesn't take the forwarder down.
        """
        forwarder = VideoForwarder(bunny_video_track, max_buffer=3, fps=10.0)

        frames_to_succeed = 10
        frames_to_fail = 2
        frames_succeeded = 0
        frames_failed = 0
        finished = asyncio.Event()

        async def handler(_):
            nonlocal frames_failed, frames_succeeded

            if frames_failed < frames_to_fail:
                frames_failed += 1
                raise ValueError("Failed")

            frames_succeeded += 1

            if (
                frames_failed == frames_to_fail
                and frames_succeeded == frames_to_succeed
            ):
                finished.set()

        # Add handlers
        forwarder.add_frame_handler(handler, fps=5, name="handler-1")
        assert forwarder.started

        await finished.wait()
        assert frames_failed == frames_to_fail
        assert frames_succeeded == frames_to_succeed
