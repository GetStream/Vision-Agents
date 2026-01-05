import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional, Type, cast

import aiortc
import av
import numpy as np
import supervision as sv
from rfdetr.detr import (
    RFDETR,
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegPreview,
    RFDETRSmall,
)
from supervision import Detections
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

from .events import DetectedObject, DetectionCompletedEvent
from .utils import annotate_image

logger = logging.getLogger(__name__)


RFDETRModelID = Literal[
    "rfdetr-base",
    "rfdetr-large",
    "rfdetr-nano",
    "rfdetr-small",
    "rfdetr-medium",
    "rfdetr-seg-preview",
]

_RFDETR_MODELS: dict[str, Type[RFDETR]] = {
    "rfdetr-base": RFDETRBase,
    "rfdetr-large": RFDETRLarge,
    "rfdetr-nano": RFDETRNano,
    "rfdetr-small": RFDETRSmall,
    "rfdetr-medium": RFDETRMedium,
    "rfdetr-seg-preview": RFDETRSegPreview,
}


class RoboflowLocalDetectionProcessor(VideoProcessorPublisher, Warmable[RFDETR]):
    """
    A VideoProcessor for real-time object detection with Roboflow's RF-DETR models.
    This processor downloads pre-trained models from Roboflow and runs them locally.
    Use it to detect and label objects on the video frames and react on them.
    On each detection, the Processor emits `DetectionCompletedEvent` with the data about detected objects.

    Example usage:

        ```
        from vision_agents.core import Agent
        from vision_agents.plugins import roboflow

        processor = roboflow.RoboflowLocalDetectionProcessor(...)

        agent = Agent(processors=[processor], ...)

        @agent.events.subscribe
        async def on_detection_completed(event: roboflow.DetectionCompletedEvent):
            # React on detected objects here
            ...

        ```

    Args:
        model_id: identifier of the model to be used.
            Available models are: "rfdetr-base", "rfdetr-large", "rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-seg-preview".
            Default - "rfdetr-seg-preview".
        conf_threshold: Confidence threshold for detections (0 - 1.0). Default - 0.5.
        fps: Frame processing rate. Default - 10.
        classes: optional list of class names to be detected.
            Example: ["person", "sports ball"]
            Verify that the classes a supported by the given model.
            Default - None (all classes are detected).
        model: optional instance of `RFDETRModel` to be used for detections.
            Use it provide a model of choosing with custom parameters.
        annotate: if True, annotate the detected objects with boxes and labels.
            Default - True.
        dim_background_factor: how much to dim the background around detected objects from 0 to 1.0.
            Effective only when annotate=True.
            Default - 0.0 (no dimming).
        annotate_text_scale: annotation text scale. Default - 0.75.
        annotate_text_padding: annotation text padding. Default - 1.
        annotate_box_thickness: annotation box thickness. Default - 2.
        annotate_text_position: annotation text position. Default - `sv.Position.TOP_CENTER`.
    """

    name = "roboflow_local"

    def __init__(
        self,
        model_id: Optional[RFDETRModelID] = "rfdetr-seg-preview",
        conf_threshold: float = 0.5,
        fps: int = 10,
        classes: Optional[list[str]] = None,
        model: Optional[RFDETR] = None,
        annotate: bool = True,
        dim_background_factor: float = 0.0,
        annotate_text_scale: float = 0.75,
        annotate_text_padding: int = 1,
        annotate_box_thickness: int = 2,
        annotate_text_position: sv.Position = sv.Position.TOP_CENTER,
    ):
        if not 0 <= conf_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.conf_threshold = conf_threshold

        self._model: Optional[RFDETR] = None
        self._model_id: Optional[RFDETRModelID] = None

        if model is not None:
            self._model = model
            self._model_id = model.size
        elif model_id:
            if model_id not in _RFDETR_MODELS:
                raise ValueError(
                    f'Unknown model_id "{model_id}"; available models: {", ".join(_RFDETR_MODELS.keys())}'
                )
            self._model_id = model_id
        else:
            raise ValueError("Either model_id or model must be provided")

        self.fps = fps
        self.dim_background_factor = max(0.0, dim_background_factor)
        self.annotate = annotate

        self._events: Optional[EventManager] = None

        # Limit object detection to certain classes only.
        self._classes = classes or []

        self._closed = False
        self._video_forwarder: Optional[VideoForwarder] = None

        # Thread pool for async inference
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="roboflow_processor"
        )
        # Video track for publishing
        self._video_track: QueuedVideoTrack = QueuedVideoTrack(
            fps=self.fps,
            max_queue_size=self.fps,  # Buffer 1s of the video
        )
        self._annotate_text_scale = annotate_text_scale
        self._annotate_text_padding = annotate_text_padding
        self._annotate_box_thickness = annotate_box_thickness
        self._annotate_text_position = annotate_text_position

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Process incoming video track with Roboflow detection.
        """

        if self._video_forwarder is not None:
            logger.info(
                "🎥 Stopping the ongoing Roboflow video processing because the new video track is published"
            )
            await self._video_forwarder.remove_frame_handler(self._process_frame)

        logger.info(f"🎥 Starting Roboflow video processing at {self.fps} FPS")
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                track,
                max_buffer=self.fps,  # 1 second
                fps=self.fps,
                name="roboflow_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._process_frame, fps=float(self.fps), name="roboflow_processor"
        )

    def publish_video_track(self) -> QueuedVideoTrack:
        """Return the video track for publishing processed frames."""
        return self._video_track

    async def close(self):
        """Clean up resources."""
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._process_frame)
        self._closed = True
        self._executor.shutdown(wait=False)
        self._video_track.stop()
        logger.info("🎥 Roboflow Processor closed")

    @property
    def events(self) -> EventManager:
        if self._events is None:
            raise ValueError("Agent is not attached to the processor yet")
        return self._events

    async def on_warmup(self) -> RFDETR:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._load_model)

    def on_warmed_up(self, resource: RFDETR) -> None:
        self._model = resource

    def attach_agent(self, agent: Agent):
        self._events = agent.events
        self._events.register(DetectionCompletedEvent)

    def _load_model(self) -> RFDETR:
        """
        Load a public model from Roboflow Universe.

        Format: workspace/project or workspace/project/version
        """
        if self._model_id is None:
            raise ValueError("Model id is not set")

        logger.info(f"📦 Loading Roboflow model {self._model_id}")
        with warnings.catch_warnings():
            # Suppress warnings from the insides of RF-DETR models
            warnings.filterwarnings("ignore")
            model = _RFDETR_MODELS[self._model_id]()
            try:
                model.optimize_for_inference()
            except RuntimeError:
                # Workaround for a bug in 1.3.0 https://github.com/roboflow/rf-detr/issues/383.
                # Models other than rfdetr-seg-preview fail with "compile=True"
                model.optimize_for_inference(compile=False)

        logger.info(f"✅ Loaded Roboflow model {self._model_id}")
        return model

    async def _process_frame(self, frame: av.VideoFrame) -> None:
        """Process frame, run detection, annotate, and publish."""
        if self._closed:
            return None

        if self._model is None:
            raise RuntimeError("The Roboflow model is not loaded")

        image = frame.to_ndarray(format="rgb24")
        try:
            # Run inference
            detections = await self._run_inference(image)
        except Exception:
            logger.exception("❌ Frame processing failed")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)
            return None

        if detections.class_id is None or not detections.class_id.size:
            # The inference wasn't able to complete or nothing was detected
            await self._video_track.add_frame(frame)
            return None

        if self.annotate:
            # Annotate frame with detections
            annotated_image = annotate_image(
                image,
                detections,
                classes=self._model.class_names,
                dim_factor=self.dim_background_factor,
                text_scale=self._annotate_text_scale,
                text_position=self._annotate_text_position,
                text_padding=self._annotate_text_padding,
                box_thickness=self._annotate_box_thickness,
            )
            # Convert back to av.VideoFrame
            annotated_frame = av.VideoFrame.from_ndarray(annotated_image)
            annotated_frame.pts = frame.pts
            annotated_frame.time_base = frame.time_base
            await self._video_track.add_frame(annotated_frame)
        else:
            # Forward original frame
            await self._video_track.add_frame(frame)

        # Publish the event with detected data
        img_height, img_width = image.shape[0:2]

        detected_objects = [
            DetectedObject(
                label=self._model.class_names[class_id], x1=x1, y1=y1, x2=x2, y2=y2
            )
            for class_id, (x1, y1, x2, y2) in zip(
                detections.class_id, detections.xyxy.astype(float)
            )
        ]

        self.events.send(
            DetectionCompletedEvent(
                objects=detected_objects,
                raw_detections=detections,
                image_width=img_width,
                image_height=img_height,
            )
        )
        return None

    async def _run_inference(self, image: np.ndarray) -> Detections:
        """Run Roboflow inference on frame."""
        loop = asyncio.get_running_loop()
        model = cast(RFDETR, self._model)

        # Run inference in thread pool (Roboflow SDK is synchronous)
        def detect(img: np.ndarray) -> Detections:
            detected = model.predict(img, confidence=self.conf_threshold)
            detected_obj = detected[0] if isinstance(detected, list) else detected
            if detected_obj.class_id is None:
                return sv.Detections.empty()

            # Filter only classes we want to detect
            if self._classes:
                classes_ids = [
                    k for k, v in model.class_names.items() if v in self._classes
                ]
                detected_class_ids = (
                    detected_obj.class_id if detected_obj.class_id is not None else []
                )
                detected_obj = cast(
                    Detections,
                    detected_obj[np.isin(detected_class_ids, classes_ids)],
                )

            if detected_obj.class_id is not None and detected_obj.class_id.size:
                # Return detected classes
                return detected_obj
            else:
                # Return empty Detections object if there are no detected classes
                return sv.Detections.empty()

        return await loop.run_in_executor(self._executor, detect, image)
