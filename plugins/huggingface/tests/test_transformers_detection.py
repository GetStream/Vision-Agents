"""Tests for TransformersDetectionProcessor - local object detection."""

import asyncio
import io
import os
import pathlib
from unittest.mock import MagicMock

import aiofiles
import numpy as np
import PIL.Image
import pytest
import torch
from av import VideoFrame
from conftest import skip_blockbuster
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.huggingface import DetectionCompletedEvent
from vision_agents.plugins.huggingface.transformers_detection import (
    DetectionResources,
    TransformersDetectionProcessor,
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


class TestDetectClassFiltering:
    """Test class filtering logic in _detect() directly.

    Integration tests can only prove class *exclusion* (via timeout when nothing
    matches). This test uses mocked resources to prove that inclusion filtering
    returns exactly the expected classes when multiple object types are present.
    """

    def test_classes_filter_keeps_only_matching_labels(self):
        model = MagicMock()
        model.config.id2label = {0: "cat", 1: "dog", 2: "person"}

        image_processor = MagicMock()
        input_tensor = torch.zeros(1, 3, 224, 224)
        image_processor.return_value = {"pixel_values": input_tensor}
        image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.95, 0.90, 0.85]),
                "labels": torch.tensor([0, 1, 2]),
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 100.0, 200.0],
                        [50.0, 60.0, 150.0, 250.0],
                        [200.0, 100.0, 400.0, 300.0],
                    ]
                ),
            }
        ]

        resources = DetectionResources(
            model=model,
            image_processor=image_processor,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        processor = TransformersDetectionProcessor(classes=["cat", "person"])
        processor.on_warmed_up(resources)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        objects = processor._detect(image)

        labels = [o["label"] for o in objects]
        assert labels == ["cat", "person"]
        assert objects[0]["confidence"] == 0.95
        assert objects[1]["confidence"] == 0.85


MODEL_ID = os.getenv("TRANSFORMERS_TEST_DETECTION_MODEL", "PekingU/rtdetr_v2_r18vd")


@pytest.mark.integration
@skip_blockbuster
class TestTransformersDetectionProcessor:
    @pytest.mark.parametrize("annotate", [True, False])
    async def test_process_video_objects_detected(
        self, cat_video_track, agent_mock, events_manager, annotate: bool
    ):
        processor = TransformersDetectionProcessor(
            model=MODEL_ID, annotate=annotate, fps=10
        )
        await processor.warmup()
        processor.attach_agent(agent_mock)

        future: asyncio.Future[DetectionCompletedEvent] = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        original_frame = await cat_video_track.recv()

        output_track = processor.publish_video_track()
        await processor.process_video(cat_video_track, "user_id")
        await asyncio.wait_for(future, 30)

        detection = future.result()
        assert detection.objects
        assert detection.inference_time_ms is not None
        assert detection.inference_time_ms > 0
        assert detection.image_width > 0
        assert detection.image_height > 0
        assert detection.detection_count == len(detection.objects)
        # RT-DETRv2 is a COCO model — cat.jpg should yield "cat"
        labels = [o["label"] for o in detection.objects]
        assert "cat" in labels
        for obj in detection.objects:
            assert 0 < obj["confidence"] <= 1.0

        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )

        if annotate:
            assert (original_frame.to_ndarray() != output_frame.to_ndarray()).any()
        else:
            assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        await processor.close()
        assert output_track.stopped

    async def test_class_filter_excludes_detections(
        self, cat_video_track, agent_mock, events_manager
    ):
        """When classes filter is set to something not in the image, no events fire."""
        processor = TransformersDetectionProcessor(
            model=MODEL_ID, classes=["nonexistent-class-xyz"], fps=10
        )
        await processor.warmup()
        processor.attach_agent(agent_mock)

        future: asyncio.Future[DetectionCompletedEvent] = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        original_frame = await cat_video_track.recv()

        output_track = processor.publish_video_track()
        await processor.process_video(cat_video_track, "user_id")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, 5)

        # Frame should pass through unmodified
        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        await processor.close()
        assert output_track.stopped

    async def test_process_video_nothing_detected(self, agent_mock, events_manager):
        """Empty input (blue screen) should produce no detection events."""
        processor = TransformersDetectionProcessor(model=MODEL_ID, fps=10)
        await processor.warmup()
        processor.attach_agent(agent_mock)

        future: asyncio.Future[DetectionCompletedEvent] = asyncio.Future()

        @events_manager.subscribe
        async def on_event(event: DetectionCompletedEvent):
            future.set_result(event)

        # Use empty track — returns blue screen on each recv()
        input_track = QueuedVideoTrack()
        original_frame = await input_track.recv()

        output_track = processor.publish_video_track()
        await processor.process_video(input_track, "user_id")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, 5)

        output_frame = await output_track.recv()
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        await processor.close()
        assert output_track.stopped
