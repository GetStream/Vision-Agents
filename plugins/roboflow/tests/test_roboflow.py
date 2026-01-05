import asyncio
import io
import os
import pathlib
from unittest.mock import MagicMock

import aiofiles
import PIL.Image
import pytest
from av import VideoFrame
from rfdetr import RFDETRSegPreview
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.roboflow import (
    DetectionCompletedEvent,
    RoboflowCloudDetectionProcessor,
    RoboflowLocalDetectionProcessor,
)


@pytest.fixture()
async def cat_video_track(assets_dir) -> QueuedVideoTrack:
    qsize = 100
    track = QueuedVideoTrack(max_queue_size=qsize)
    async with aiofiles.open(pathlib.Path(assets_dir) / "cat.jpg", "rb") as f:
        data = io.BytesIO(await f.read())
        img = PIL.Image.open(data).convert("RGB")
        frame = VideoFrame.from_image(img)
        for _ in range(qsize):
            await track.add_frame(frame)
    return track


@pytest.fixture()
async def events_manager() -> EventManager:
    return EventManager()


@pytest.fixture()
def agent_mock(events_manager: EventManager) -> Agent:
    agent = MagicMock()
    agent.events = events_manager
    return agent


class TestRoboflowLocalDetectionProcessor:
    def test_init_prebuilt_model(self, assets_dir: pathlib.Path):
        RoboflowLocalDetectionProcessor(model_id="rfdetr-seg-preview")

    def test_init_custom_model(self):
        RoboflowLocalDetectionProcessor(model=RFDETRSegPreview())

    @pytest.mark.parametrize("annotate", [True, False])
    async def test_process_video_objects_detected(
        self, cat_video_track, agent_mock, events_manager, annotate: bool
    ):
        processor = RoboflowLocalDetectionProcessor(annotate=annotate)
        await processor.warmup()
        processor.attach_agent(agent_mock)

        # Use future to catch a detection event
        future = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        # Get the first frame from the input track to compare it with the output one later
        original_frame = await cat_video_track.recv()

        # Start the processor and wait for the event
        output_track = processor.publish_video_track()
        await processor.process_video(cat_video_track, "user_id")
        await asyncio.wait_for(future, 10)

        # Check the detection event
        detection = future.result()
        assert detection
        objects = detection.objects
        assert objects[0]["label"] == "cat"

        # Check the output track. The image size must be the same as the original one
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )

        if annotate:
            assert (original_frame.to_ndarray() != output_frame.to_ndarray()).any()
        else:
            assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        # Close the processor and check that the output track is stopped
        await processor.close()
        assert output_track.stopped

    async def test_process_video_nothing_detected_classes_set(
        self, cat_video_track, agent_mock, events_manager
    ):
        processor = RoboflowLocalDetectionProcessor(classes=["class-123"])
        await processor.warmup()
        processor.attach_agent(agent_mock)

        # Use future to catch a detection event
        future = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        # Get the first frame from the input track to compare it with the output one later
        original_frame = await cat_video_track.recv()

        # Start the processor and wait for the event
        output_track = processor.publish_video_track()
        await processor.process_video(cat_video_track, "user_id")

        # Expect a timeout because no event is emitted when nothing is detected
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, 5)

        # Check the output track. It must be the same as the original one
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        # Close the processor and check that the output track is stopped
        await processor.close()
        assert output_track.stopped

    async def test_process_video_nothing_detected(self, agent_mock, events_manager):
        processor = RoboflowLocalDetectionProcessor()
        await processor.warmup()
        processor.attach_agent(agent_mock)

        # Use future to catch a detection event
        future = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        # Use empty track, it should return blue screen on each recv()
        input_track = QueuedVideoTrack()
        original_frame = await input_track.recv()

        # Start the processor and wait for the event
        output_track = processor.publish_video_track()
        await processor.process_video(input_track, "user_id")

        # Expect a timeout because no event is emitted when nothing is detected
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, 5)

        # Check the output track. It must be the same as the original one
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        # Close the processor and check that the output track is stopped
        await processor.close()
        assert output_track.stopped


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ROBOFLOW_API_KEY"),
    reason="ROBOFLOW_API_KEY environment variable not set (required for cloud inference testing)",
)
@pytest.mark.skipif(
    not os.getenv("ROBOFLOW_API_URL"),
    reason="ROBOFLOW_API_URL environment variable not set (required for cloud inference testing)",
)
class TestRoboflowCloudDetectionProcessor:
    @pytest.mark.parametrize("annotate", [True, False])
    async def test_process_video_objects_detected(
        self, cat_video_track, agent_mock, events_manager, annotate: bool
    ):
        processor = RoboflowCloudDetectionProcessor(
            fps=1,
            annotate=annotate,
            classes=["cat"],
            model_id="yolo-nas-s-640",  # Use a general pre-trained model
        )
        processor.attach_agent(agent_mock)

        # Use future to catch a detection event
        future = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        # Get the first frame from the input track to compare it with the output one later
        original_frame = await cat_video_track.recv()

        # Start the processor and wait for the event
        output_track = processor.publish_video_track()
        await processor.process_video(cat_video_track, "user_id")
        await asyncio.wait_for(future, 10)

        # Check the detection event
        detection = future.result()
        assert detection
        objects = detection.objects
        assert objects[0]["label"] == "cat"

        # Check the output track. The image size must be the same as the original one
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )

        if annotate:
            assert (original_frame.to_ndarray() != output_frame.to_ndarray()).any()
        else:
            assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        # Close the processor and check that the output track is stopped
        await processor.close()
        assert output_track.stopped

    async def test_process_video_nothing_detected(self, agent_mock, events_manager):
        processor = RoboflowCloudDetectionProcessor(model_id="yolo-nas-s-640", fps=1)
        processor.attach_agent(agent_mock)

        # Use future to catch a detection event
        future = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        # Use empty track, it should return blue screen on each recv()
        input_track = QueuedVideoTrack()
        original_frame = await input_track.recv()

        # Start the processor and wait for the event
        output_track = processor.publish_video_track()
        await processor.process_video(input_track, "user_id")

        # Expect a timeout because no event is emitted when nothing is detected
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, 5)

        # Check the output track. It must be the same as the original one
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        # Close the processor and check that the output track is stopped
        await processor.close()
        assert output_track.stopped

    async def test_process_video_nothing_detected_classes_set(
        self, agent_mock, cat_video_track, events_manager
    ):
        processor = RoboflowCloudDetectionProcessor(
            model_id="yolo-nas-s-640", classes=["class-123"], fps=1
        )
        processor.attach_agent(agent_mock)

        # Use future to catch a detection event
        future = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        input_track = cat_video_track
        original_frame = await input_track.recv()

        # Start the processor and wait for the event
        output_track = processor.publish_video_track()
        await processor.process_video(input_track, "user_id")

        # Expect a timeout because no event is emitted when nothing is detected
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, 5)

        # Check the output track. It must be the same as the original one
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        # Close the processor and check that the output track is stopped
        await processor.close()
        assert output_track.stopped
