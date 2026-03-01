"""
TransformersDetectionProcessor - Local object detection via HuggingFace Transformers.

Runs detection models (e.g. RT-DETRv2) directly on your hardware.

Example:
    from vision_agents.plugins import huggingface

    processor = huggingface.TransformersDetectionProcessor(
        model="PekingU/rtdetr_v2_r101vd",
    )

    agent = Agent(processors=[processor], ...)

    @agent.events.subscribe
    async def on_detection(event: huggingface.DetectionCompletedEvent):
        for obj in event.objects:
            print(f"Detected {obj['label']}")
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import av
import numpy as np
import torch
from PIL import Image

from vision_agents.core import Agent
from vision_agents.core.events import DetectedObject, EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

from .events import DetectionCompletedEvent
from .transformers_llm import DeviceType

if TYPE_CHECKING:
    import aiortc
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class DetectionResources:
    """Container for a loaded detection model, image processor, and device."""

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: Any,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.dtype = dtype


class TransformersDetectionProcessor(
    VideoProcessorPublisher, Warmable[DetectionResources]
):
    """Local object detection using HuggingFace Transformers.

    Runs models like RT-DETRv2 directly on your hardware for real-time
    object detection on video frames. Emits ``DetectionCompletedEvent``
    for each processed frame.

    Args:
        model: HuggingFace model ID (e.g. ``"PekingU/rtdetr_v2_r101vd"``).
        conf_threshold: Confidence threshold for detections (0–1). Default 0.5.
        fps: Frame processing rate. Default 10.
        classes: Optional list of class names to detect (e.g. ``["person"]``).
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        annotate: Draw bounding boxes on the output video. Default ``True``.
    """

    name = "transformers_detection"

    def __init__(
        self,
        model: str = "PekingU/rtdetr_v2_r101vd",
        conf_threshold: float = 0.5,
        fps: int = 10,
        classes: Optional[list[str]] = None,
        device: DeviceType = "auto",
        annotate: bool = True,
    ):
        if not 0 <= conf_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.model_id = model
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.annotate = annotate

        self._device_config = device
        self._classes = classes or []

        self._resources: Optional[DetectionResources] = None
        self._events: Optional[EventManager] = None

        self._closed = False
        self._last_log_time: float = 0.0
        self._video_forwarder: Optional[VideoForwarder] = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transformers_detection"
        )
        self._video_track: QueuedVideoTrack = QueuedVideoTrack(
            fps=self.fps,
            max_queue_size=self.fps,
        )

    async def on_warmup(self) -> DetectionResources:
        logger.info(f"Loading detection model: {self.model_id}")
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info(f"Detection model loaded on device: {resources.device}")
        return resources

    def on_warmed_up(self, resource: DetectionResources) -> None:
        self._resources = resource

    def _load_model_sync(self) -> DetectionResources:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        load_kwargs: dict[str, Any] = {}

        if self._device_config == "auto":
            load_kwargs["device_map"] = "auto"
        elif self._device_config == "cuda":
            load_kwargs["device_map"] = {"": "cuda"}

        model = AutoModelForObjectDetection.from_pretrained(
            self.model_id, **load_kwargs
        )

        if self._device_config == "mps":
            model = model.to(torch.device("mps"))

        model.eval()

        image_processor = AutoImageProcessor.from_pretrained(self.model_id)

        first_param = next(model.parameters())
        return DetectionResources(
            model=model,
            image_processor=image_processor,
            device=first_param.device,
            dtype=first_param.dtype,
        )

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._video_forwarder is not None:
            logger.info("Stopping ongoing detection processing for new video track")
            await self._video_forwarder.remove_frame_handler(self._process_frame)

        logger.info(f"Starting Transformers detection at {self.fps} FPS")
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                track,
                max_buffer=self.fps,
                fps=self.fps,
                name="transformers_detection_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._process_frame,
            fps=float(self.fps),
            name="transformers_detection",
        )

    def publish_video_track(self) -> QueuedVideoTrack:
        return self._video_track

    async def stop_processing(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._process_frame)
            self._video_forwarder = None
            logger.info("Stopped Transformers detection processing")

    async def close(self) -> None:
        await self.stop_processing()
        self._closed = True
        self._executor.shutdown(wait=False)
        self._video_track.stop()
        self.unload()
        logger.info("Transformers detection processor closed")

    @property
    def events(self) -> EventManager:
        if self._events is None:
            raise ValueError("Agent is not attached to the processor yet")
        return self._events

    def attach_agent(self, agent: Agent) -> None:
        self._events = agent.events
        self._events.register(DetectionCompletedEvent)

    async def _process_frame(self, frame: av.VideoFrame) -> None:
        if self._closed or self._resources is None:
            return

        image = frame.to_ndarray(format="rgb24")
        start_time = time.perf_counter()

        try:
            detected_objects = await self._run_inference(image)
        except Exception:
            logger.exception("Frame detection failed")
            await self._video_track.add_frame(frame)
            return

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        if not detected_objects:
            await self._video_track.add_frame(frame)
            return

        if self.annotate:
            annotated = await asyncio.to_thread(self._annotate, image, detected_objects)
            annotated_frame = av.VideoFrame.from_ndarray(annotated)
            annotated_frame.pts = frame.pts
            annotated_frame.time_base = frame.time_base
            await self._video_track.add_frame(annotated_frame)
        else:
            await self._video_track.add_frame(frame)

        now = time.perf_counter()
        if now - self._last_log_time >= 5.0:
            labels = [o["label"] for o in detected_objects]
            logger.info(
                f"Detected {len(detected_objects)} objects ({', '.join(labels)}) "
                f"in {inference_time_ms:.0f}ms"
            )
            self._last_log_time = now

        img_height, img_width = image.shape[:2]
        self.events.send(
            DetectionCompletedEvent(
                plugin_name=self.name,
                objects=detected_objects,
                image_width=img_width,
                image_height=img_height,
                inference_time_ms=inference_time_ms,
                model_id=self.model_id,
            )
        )

    async def _run_inference(self, image: np.ndarray) -> list[DetectedObject]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._detect, image)

    def _detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Run detection on a single frame (called in thread pool)."""
        resources = self._resources
        if resources is None:
            return []

        pil_image = Image.fromarray(image)
        inputs = resources.image_processor(images=pil_image, return_tensors="pt")
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
            outputs = resources.model(**inputs)

        h, w = image.shape[:2]
        results = resources.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(h, w)]),
            threshold=self.conf_threshold,
        )[0]

        id2label: dict[int, str] = resources.model.config.id2label or {}
        objects: list[DetectedObject] = []

        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label = id2label.get(label_id.item(), str(label_id.item()))

            if self._classes and label not in self._classes:
                continue

            x1, y1, x2, y2 = box.tolist()
            objects.append(
                DetectedObject(
                    label=label,
                    confidence=round(score.item(), 4),
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
            )

        return objects

    def _annotate(
        self,
        image: np.ndarray,
        objects: list[DetectedObject],
    ) -> np.ndarray:
        """Annotate image with bounding boxes and labels."""
        import supervision as sv

        from vision_agents.core.utils.annotation import annotate_image

        xyxy = np.array([[o["x1"], o["y1"], o["x2"], o["y2"]] for o in objects])
        classes = {i: o["label"] for i, o in enumerate(objects)}
        class_ids = np.arange(len(objects))
        detections = sv.Detections(xyxy=xyxy, class_id=class_ids)

        return annotate_image(image, detections, classes)

    def unload(self) -> None:
        """Release model from memory."""
        self._resources = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
