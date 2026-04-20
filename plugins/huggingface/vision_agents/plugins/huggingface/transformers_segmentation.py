"""
TransformersSegmentationProcessor - Local segmentation via HuggingFace Transformers.

Runs prompt-based segmentation models (SAM2, SAM-HQ, EdgeTAM, etc.) on video
frames using bounding boxes from an upstream detection processor as prompts.

Example:
    from vision_agents.plugins import huggingface

    detection = huggingface.TransformersDetectionProcessor(
        model="PekingU/rtdetr_v2_r101vd",
    )
    segmentation = huggingface.TransformersSegmentationProcessor(
        model="facebook/sam2.1-hiera-tiny",
    )

    agent = Agent(processors=[detection, segmentation], ...)

    @agent.events.subscribe
    async def on_segmentation(event: huggingface.SegmentationCompletedEvent):
        for obj in event.objects:
            print(f"Segmented {obj['label']} ({obj['mask_area']} px)")
"""

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import av
import numpy as np
import supervision as sv
import torch
from PIL import Image
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

from .annotation import annotate_image
from .events import (
    DetectedObject,
    DetectionCompletedEvent,
    SegmentedObject,
    SegmentationCompletedEvent,
)
from .transformers_llm import (
    DeviceType,
    TorchDtypeType,
    _model_load_lock,
    resolve_torch_dtype,
)

if TYPE_CHECKING:
    import aiortc
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class SegmentationResources:
    """Container for a loaded segmentation model, processor, and device."""

    def __init__(
        self,
        model: "PreTrainedModel",
        processor: Any,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype


class TransformersSegmentationProcessor(
    VideoProcessorPublisher, Warmable[SegmentationResources]
):
    """Local segmentation using HuggingFace Transformers.

    Runs models like SAM2, SAM-HQ, or EdgeTAM directly on your hardware for
    real-time segmentation of video frames. Chains with any detection processor
    that emits ``DetectionCompletedEvent`` — detection bounding boxes are used
    as box prompts for the segmentation model.

    Args:
        model: HuggingFace model ID. Default ``"facebook/sam2.1-hiera-tiny"``.
            Larger variants: ``"facebook/sam2.1-hiera-small"``,
            ``"facebook/sam2.1-hiera-base-plus"``, ``"facebook/sam2.1-hiera-large"``.
        fps: Frame processing rate. Default 5.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        torch_dtype: ``"auto"``, ``"float16"``, ``"bfloat16"``, or ``"float32"``.
            ``"auto"`` picks float16 on MPS, bfloat16 on CUDA, float32 on CPU.
        annotate: Draw masks and bounding boxes on the output video. Default ``True``.
        mask_opacity: Mask overlay opacity (0–1). Default 0.35.
        mask_threshold: Binarization threshold for masks. Default 0.0.
    """

    name = "transformers_segmentation"

    def __init__(
        self,
        model: str = "facebook/sam2.1-hiera-tiny",
        fps: int = 5,
        device: DeviceType = "auto",
        torch_dtype: TorchDtypeType = "auto",
        annotate: bool = True,
        mask_opacity: float = 0.35,
        mask_threshold: float = 0.0,
    ):
        if not 0 <= mask_opacity <= 1.0:
            raise ValueError("mask_opacity must be between 0 and 1.")

        self.model_id = model
        self.fps = fps
        self.annotate = annotate

        self._device_config = device
        self._torch_dtype_config = torch_dtype
        self._mask_opacity = mask_opacity
        self._mask_threshold = mask_threshold

        self._resources: Optional[SegmentationResources] = None
        self._events: Optional[EventManager] = None

        self._closed = False
        self._last_log_time: float = 0.0
        self._latest_detections: list[DetectedObject] = []
        self._video_forwarder: Optional[VideoForwarder] = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transformers_segmentation"
        )
        self._video_track: QueuedVideoTrack = QueuedVideoTrack(
            fps=self.fps,
            max_queue_size=self.fps,
        )

    async def on_warmup(self) -> SegmentationResources:
        logger.info(f"Loading segmentation model: {self.model_id}")
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info(f"Segmentation model loaded on device: {resources.device}")
        return resources

    def on_warmed_up(self, resource: SegmentationResources) -> None:
        self._resources = resource

    def _resolve_device(self) -> torch.device:
        if self._device_config == "cuda":
            return torch.device("cuda")
        if self._device_config == "mps":
            return torch.device("mps")
        if self._device_config == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    def _load_model_sync(self) -> SegmentationResources:
        from transformers import AutoModelForMaskGeneration, AutoProcessor

        device = self._resolve_device()
        dtype = resolve_torch_dtype(self._torch_dtype_config)
        with _model_load_lock:
            model = AutoModelForMaskGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype
            )
            model = model.to(device)

        model.eval()

        processor = AutoProcessor.from_pretrained(self.model_id)

        first_param = next(model.parameters())
        return SegmentationResources(
            model=model,
            processor=processor,
            device=first_param.device,
            dtype=first_param.dtype,
        )

    async def process_video(
        self,
        track: "aiortc.VideoStreamTrack",
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._video_forwarder is not None:
            logger.info("Stopping ongoing segmentation processing for new video track")
            await self._video_forwarder.remove_frame_handler(self._process_frame)

        logger.info(f"Starting Transformers segmentation at {self.fps} FPS")
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                track,
                max_buffer=self.fps,
                fps=self.fps,
                name="transformers_segmentation_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._process_frame,
            fps=float(self.fps),
            name="transformers_segmentation",
        )

    def publish_video_track(self) -> QueuedVideoTrack:
        return self._video_track

    async def stop_processing(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._process_frame)
            self._video_forwarder = None
            logger.info("Stopped Transformers segmentation processing")

    async def close(self) -> None:
        await self.stop_processing()
        self._closed = True
        self._executor.shutdown(wait=False)
        self._video_track.stop()
        self.unload()
        logger.info("Transformers segmentation processor closed")

    @property
    def events(self) -> EventManager:
        if self._events is None:
            raise ValueError("Agent is not attached to the processor yet")
        return self._events

    def attach_agent(self, agent: Agent) -> None:
        self._events = agent.events
        self._events.register(DetectionCompletedEvent, SegmentationCompletedEvent)

        @self._events.subscribe
        async def on_detection(event: DetectionCompletedEvent) -> None:
            self._latest_detections = list(event.objects)

    def set_detections(self, detections: list[DetectedObject]) -> None:
        """Manually provide detections for cross-plugin chaining.

        Use this when the upstream detection processor is not from the
        HuggingFace plugin. Wire it from your own event handler::

            @agent.events.subscribe
            async def on_roboflow_detection(event: roboflow.DetectionCompletedEvent):
                segmentation_processor.set_detections([
                    DetectedObject(
                        label=o["label"], confidence=0.0,
                        x1=o["x1"], y1=o["y1"], x2=o["x2"], y2=o["y2"],
                    )
                    for o in event.objects
                ])
        """
        self._latest_detections = list(detections)

    async def _process_frame(self, frame: av.VideoFrame) -> None:
        if self._closed or self._resources is None:
            return

        detections = self._latest_detections
        if not detections:
            await self._video_track.add_frame(frame)
            return

        image = frame.to_ndarray(format="rgb24")
        start_time = time.perf_counter()

        try:
            segmented_objects, masks = await self._run_inference(image, detections)
        except (RuntimeError, ValueError, OSError):
            logger.exception("Frame segmentation failed")
            await self._video_track.add_frame(frame)
            return

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        if not segmented_objects:
            await self._video_track.add_frame(frame)
            return

        if self.annotate:
            annotated = await asyncio.to_thread(
                self._annotate, image, segmented_objects, masks
            )
            annotated_frame = av.VideoFrame.from_ndarray(annotated)
            annotated_frame.pts = frame.pts
            annotated_frame.time_base = frame.time_base
            await self._video_track.add_frame(annotated_frame)
        else:
            await self._video_track.add_frame(frame)

        now = time.perf_counter()
        if now - self._last_log_time >= 5.0:
            labels = [o["label"] for o in segmented_objects]
            logger.info(
                f"Segmented {len(segmented_objects)} objects ({', '.join(labels)}) "
                f"in {inference_time_ms:.0f}ms"
            )
            self._last_log_time = now

        img_height, img_width = image.shape[:2]
        self.events.send(
            SegmentationCompletedEvent(
                plugin_name=self.name,
                objects=segmented_objects,
                masks=[m for m in masks],
                image_width=img_width,
                image_height=img_height,
                inference_time_ms=inference_time_ms,
                model_id=self.model_id,
            )
        )

    async def _run_inference(
        self,
        image: np.ndarray,
        detections: list[DetectedObject],
    ) -> tuple[list[SegmentedObject], np.ndarray]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._segment, image, detections
        )

    def _segment(
        self,
        image: np.ndarray,
        detections: list[DetectedObject],
    ) -> tuple[list[SegmentedObject], np.ndarray]:
        """Run segmentation on a single frame (called in thread pool)."""
        resources = self._resources
        if resources is None or not detections:
            return [], np.empty(0)

        pil_image = Image.fromarray(image)
        boxes = [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections]

        inputs = resources.processor(
            images=pil_image, input_boxes=[boxes], return_tensors="pt"
        )

        original_sizes = inputs.pop("original_sizes")
        reshaped_input_sizes = inputs.pop("reshaped_input_sizes")

        inputs = {
            k: (
                v.to(device=resources.device, dtype=resources.dtype)
                if v.is_floating_point()
                else v.to(device=resources.device)
            )
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = resources.model(**inputs, multimask_output=False)

        masks_tensor = resources.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            binarize=True,
            mask_threshold=self._mask_threshold,
        )

        per_object_masks = masks_tensor[0].squeeze(1).numpy()
        iou_scores = outputs.iou_scores[0].squeeze(-1).cpu()

        objects: list[SegmentedObject] = []
        for i, detection in enumerate(detections):
            mask = per_object_masks[i]
            objects.append(
                SegmentedObject(
                    label=detection["label"],
                    confidence=round(iou_scores[i].item(), 4),
                    x1=detection["x1"],
                    y1=detection["y1"],
                    x2=detection["x2"],
                    y2=detection["y2"],
                    mask_area=int(mask.sum()),
                )
            )

        return objects, per_object_masks

    def _annotate(
        self,
        image: np.ndarray,
        objects: list[SegmentedObject],
        masks: np.ndarray,
    ) -> np.ndarray:
        """Annotate image with masks, bounding boxes, and labels."""
        xyxy = np.array([[o["x1"], o["y1"], o["x2"], o["y2"]] for o in objects])
        classes = {i: o["label"] for i, o in enumerate(objects)}
        class_ids = np.arange(len(objects))
        detections = sv.Detections(xyxy=xyxy, mask=masks, class_id=class_ids)

        return annotate_image(
            image, detections, classes, mask_opacity=self._mask_opacity
        )

    def unload(self) -> None:
        """Release model from memory."""
        self._resources = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
