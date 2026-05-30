"""Tests for TransformersSegmentationProcessor - local segmentation."""

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
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.huggingface import SegmentationCompletedEvent
from vision_agents.plugins.huggingface.events import DetectedObject
from vision_agents.plugins.huggingface.transformers_segmentation import (
    SegmentationResources,
    TransformersSegmentationProcessor,
)

from conftest import skip_blockbuster


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


FAKE_DETECTIONS: list[DetectedObject] = [
    DetectedObject(label="cat", confidence=0.95, x1=10, y1=20, x2=100, y2=200),
    DetectedObject(label="dog", confidence=0.90, x1=150, y1=60, x2=300, y2=250),
]


class TestSegmentWithMockedModel:
    """Test _segment() with mocked SAM model and processor."""

    def _make_resources(
        self, num_objects: int = 2, img_h: int = 480, img_w: int = 640
    ) -> SegmentationResources:
        model = MagicMock()
        pred_masks = torch.randn(1, num_objects, 1, 256, 256)
        iou_scores = torch.tensor([[[0.85], [0.92]]])[:, :num_objects, :]
        model.return_value = MagicMock(pred_masks=pred_masks, iou_scores=iou_scores)

        processor = MagicMock()
        processor.return_value = {
            "pixel_values": torch.zeros(1, 3, 1024, 1024),
            "original_sizes": torch.tensor([[img_h, img_w]]),
            "reshaped_input_sizes": torch.tensor([[1024, 1024]]),
            "input_boxes": torch.zeros(1, num_objects, 4),
        }
        binary_masks = torch.ones(num_objects, 1, img_h, img_w, dtype=torch.bool)
        processor.post_process_masks.return_value = [binary_masks]

        return SegmentationResources(
            model=model,
            processor=processor,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_segment_returns_objects_and_masks(self):
        resources = self._make_resources(num_objects=2)
        processor = TransformersSegmentationProcessor()
        processor.on_warmed_up(resources)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        objects, masks = processor._segment(image, FAKE_DETECTIONS)

        assert len(objects) == 2
        assert objects[0]["label"] == "cat"
        assert objects[1]["label"] == "dog"
        assert objects[0]["confidence"] == 0.85
        assert objects[1]["confidence"] == 0.92
        assert objects[0]["mask_area"] == 480 * 640
        assert masks.shape == (2, 480, 640)

    def test_segment_passes_correct_boxes_to_processor(self):
        resources = self._make_resources(num_objects=2)
        processor = TransformersSegmentationProcessor()
        processor.on_warmed_up(resources)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        processor._segment(image, FAKE_DETECTIONS)

        call_kwargs = resources.processor.call_args
        input_boxes = call_kwargs.kwargs.get(
            "input_boxes", call_kwargs.args[0] if call_kwargs.args else None
        )
        assert input_boxes == [[[10, 20, 100, 200], [150, 60, 300, 250]]]

    def test_segment_extracts_sizes_before_device_move(self):
        resources = self._make_resources(num_objects=1)
        processor = TransformersSegmentationProcessor()
        processor.on_warmed_up(resources)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        processor._segment(image, FAKE_DETECTIONS[:1])

        post_process_call = resources.processor.post_process_masks.call_args
        original_sizes = post_process_call.args[1]
        assert isinstance(original_sizes, torch.Tensor)
        assert original_sizes.device == torch.device("cpu")

    def test_segment_casts_float_inputs_to_model_dtype(self):
        resources = self._make_resources(num_objects=1)
        resources.dtype = torch.float16
        processor = TransformersSegmentationProcessor()
        processor.on_warmed_up(resources)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        processor._segment(image, FAKE_DETECTIONS[:1])

        pixel_values = resources.model.call_args.kwargs["pixel_values"]
        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.dtype == torch.float16

    def test_segment_empty_detections(self):
        resources = self._make_resources(num_objects=0)
        processor = TransformersSegmentationProcessor()
        processor.on_warmed_up(resources)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        objects, masks = processor._segment(image, [])

        assert objects == []
        resources.model.assert_not_called()

    def test_set_detections_updates_latest(self):
        processor = TransformersSegmentationProcessor()
        assert processor._latest_detections == []

        processor.set_detections(FAKE_DETECTIONS)
        assert len(processor._latest_detections) == 2
        assert processor._latest_detections[0]["label"] == "cat"


class TestSegmentationAnnotation:
    """Test that annotation draws masks on the image."""

    def test_annotate_changes_image(self):
        processor = TransformersSegmentationProcessor(mask_opacity=0.5)

        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        objects = [
            FAKE_DETECTIONS[0],
        ]
        masks = np.ones((1, 100, 100), dtype=bool)

        annotated = processor._annotate(image, objects, masks)

        assert annotated.shape == image.shape
        assert (annotated != image).any()


SEGMENTATION_MODEL_ID = os.getenv(
    "TRANSFORMERS_TEST_SEGMENTATION_MODEL", "facebook/sam2-hiera-tiny"
)
DETECTION_MODEL_ID = os.getenv(
    "TRANSFORMERS_TEST_DETECTION_MODEL", "PekingU/rtdetr_v2_r18vd"
)


@pytest.mark.integration
@skip_blockbuster
class TestTransformersSegmentationProcessor:
    @pytest.mark.parametrize("annotate", [True, False])
    async def test_process_video_with_detection_chaining(
        self, cat_video_track, agent_mock, events_manager, annotate: bool
    ):
        """Run detection + segmentation on cat.jpg, verify masks are produced."""
        from vision_agents.plugins.huggingface.transformers_detection import (
            TransformersDetectionProcessor,
        )

        detection = TransformersDetectionProcessor(
            model=DETECTION_MODEL_ID, fps=10, annotate=False
        )
        segmentation = TransformersSegmentationProcessor(
            model=SEGMENTATION_MODEL_ID, fps=10, annotate=annotate
        )
        await detection.warmup()
        await segmentation.warmup()
        detection.attach_agent(agent_mock)
        segmentation.attach_agent(agent_mock)

        seg_future: asyncio.Future[SegmentationCompletedEvent] = asyncio.Future()

        @events_manager.subscribe
        async def on_seg_event(event: SegmentationCompletedEvent):
            if not seg_future.done():
                seg_future.set_result(event)

        original_frame = await cat_video_track.recv()
        assert isinstance(original_frame, VideoFrame)

        seg_output_track = segmentation.publish_video_track()
        detection.publish_video_track()
        await detection.process_video(cat_video_track, "user_id")
        await segmentation.process_video(cat_video_track, "user_id")

        result = await asyncio.wait_for(seg_future, 60)

        assert result.objects
        assert result.masks
        assert result.inference_time_ms is not None
        assert result.inference_time_ms > 0
        assert result.image_width > 0
        assert result.image_height > 0
        assert result.detection_count == len(result.objects)

        labels = [o["label"] for o in result.objects]
        assert "cat" in labels

        for obj in result.objects:
            assert obj["mask_area"] > 0
            assert 0 < obj["confidence"] <= 1.0

        for mask in result.masks:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == bool
            assert mask.shape == (result.image_height, result.image_width)

        output_frame = await seg_output_track.recv()
        assert isinstance(output_frame, VideoFrame)
        assert (original_frame.width, original_frame.height) == (
            output_frame.width,
            output_frame.height,
        )

        if annotate:
            assert (original_frame.to_ndarray() != output_frame.to_ndarray()).any()
        else:
            assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        await segmentation.close()
        await detection.close()
        assert seg_output_track.stopped

    async def test_no_detections_passes_through(self, agent_mock, events_manager):
        """Without detection events, frames pass through unmodified."""
        segmentation = TransformersSegmentationProcessor(
            model=SEGMENTATION_MODEL_ID, fps=10
        )
        await segmentation.warmup()
        segmentation.attach_agent(agent_mock)

        seg_future: asyncio.Future[SegmentationCompletedEvent] = asyncio.Future()

        @events_manager.subscribe
        async def on_seg_event(event: SegmentationCompletedEvent):
            if not seg_future.done():
                seg_future.set_result(event)

        input_track = QueuedVideoTrack()
        original_frame = await input_track.recv()
        assert isinstance(original_frame, VideoFrame)

        output_track = segmentation.publish_video_track()
        await segmentation.process_video(input_track, "user_id")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(seg_future, 5)

        output_frame = await output_track.recv()
        assert isinstance(output_frame, VideoFrame)
        assert (original_frame.to_ndarray() == output_frame.to_ndarray()).all()

        await segmentation.close()
        assert output_track.stopped
