import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import aiortc
import av
import numpy as np
import supervision as sv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.roboflow.events import (
    DetectedObject,
    DetectionCompletedEvent,
)
from vision_agents.plugins.roboflow.utils import annotate_image

logger = logging.getLogger(__name__)


class RoboflowCloudDetectionProcessor(VideoProcessorPublisher):
    """
    A VideoProcessor for real-time object detection with Roboflow's models.
    This processor uses models from Roboflow Universe, and calls Roboflow's serverless API for inference.

    Use it to detect and label objects on the video frames and react on them.
    On each detection, the Processor emits `DetectionCompletedEvent` with the data about detected objects.

    Example usage:

        ```
        from vision_agents.core import Agent
        from vision_agents.plugins import roboflow

        processor = roboflow.RoboflowCloudDetectionProcessor(...)

        agent = Agent(processors=[processor], ...)

        @agent.events.subscribe
        async def on_detection_completed(event: roboflow.DetectionCompletedEvent):
            # React on detected objects here
            ...

        ```
    Real-time object detection using Roboflow Universe public models.

    Loads models from Roboflow Universe. If version is not specified, uses latest.

    Find models at: https://universe.roboflow.com
    From URL like: universe.roboflow.com/workspace/project
    Use model_id: "workspace/project"

    Args:
        model_id: Universe model id. Example: "football-players-detection-3zvbc/20".
        api_key: Roboflow API key. If not provided, will use ROBOFLOW_API_KEY env variable.
        api_url: Roboflow API url. If not provided, will use ROBOFLOW_API_URL env variable.
        conf_threshold: Confidence threshold for detections (0 - 1.0). Default - 0.5.
        fps: Frame processing rate. Default - 5.
        classes: optional list of class names to be detected.
            Example: ["person", "sports ball"]
            Verify that the classes a supported by the given model.
            Default - None (all classes are detected).
        client: optional custom instance of `inference_sdk.InferenceHTTPClient`.
        annotate: if True, annotate the detected objects with boxes and labels.
            Default - True.
        dim_background_factor: how much to dim the background around detected objects from 0 to 1.0.
            Effective only when annotate=True.
            Default - 0.0 (no dimming).
        annotate_text_scale: annotation text scale. Default - 0.75.
        annotate_text_padding: annotation text padding. Default - 1.
        annotate_box_thickness: annotation box thickness. Default - 2.
        annotate_text_position: annotation text position. Default - `sv.Position.TOP_CENTER`.

    Examples:
        Example usage:

        ```
        from vision_agents.core import Agent
        from vision_agents.plugins import roboflow

        processor = roboflow.RoboflowCloudDetectionProcessor(...)

        agent = Agent(processors=[processor], ...)

        @agent.events.subscribe
        async def on_detection_completed(event: roboflow.DetectionCompletedEvent):
            # React on detected objects here
            ...

    """

    name = "roboflow_cloud"

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        conf_threshold: float = 0.5,
        fps: int = 5,
        classes: Optional[list[str]] = None,
        client: Optional[InferenceHTTPClient] = None,
        annotate: bool = True,
        dim_background_factor: float = 0.0,
        annotate_text_scale: float = 0.75,
        annotate_text_padding: int = 1,
        annotate_box_thickness: int = 2,
        annotate_text_position: sv.Position = sv.Position.TOP_CENTER,
    ):
        if not model_id:
            raise ValueError("model_id is required")

        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        api_url = api_url or os.getenv("ROBOFLOW_API_URL")

        if client is not None:
            self._client = client
        elif not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY required. Get it from https://app.roboflow.com → Settings → API"
            )
        elif not api_url:
            raise ValueError("ROBOFLOW_API_URL is required")
        else:
            self._client = InferenceHTTPClient(
                api_url=api_url,
                api_key=api_key,
            )

        if not 0 <= conf_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.conf_threshold = conf_threshold
        self.model_id = model_id
        self.fps = fps
        self.dim_background_factor = max(0.0, dim_background_factor)
        self.annotate = annotate
        self._annotate_text_scale = annotate_text_scale
        self._annotate_text_padding = annotate_text_padding
        self._annotate_box_thickness = annotate_box_thickness
        self._annotate_text_position = annotate_text_position

        self._events: Optional[EventManager] = None
        self._client.configure(
            InferenceConfiguration(confidence_threshold=conf_threshold)
        )

        # Limit object detection to certain classes only.
        self._classes = classes

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

        logger.info("🔍 Roboflow Processor initialized")

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ):
        """Process incoming video track with Roboflow detection."""
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
                incoming_track,
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

    def attach_agent(self, agent: Agent):
        self._events = agent.events
        self._events.register(DetectionCompletedEvent)

    async def _process_frame(self, frame: av.VideoFrame):
        """Process frame, run detection, annotate, and publish."""
        if self._closed:
            return

        image = frame.to_ndarray(format="rgb24")
        try:
            # Run inference
            detections, classes = await self._run_inference(image)
        except Exception:
            logger.exception("❌ Frame processing failed")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)
            return

        if detections.class_id is None or not detections.class_id.size:
            # Nothing detected, pass original frame and exit early
            await self._video_track.add_frame(frame)
            return

        if self.annotate:
            # Annotate frame with detections
            annotated_image = annotate_image(
                image,
                detections,
                classes,
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
            # Send the annotated frame to the output video track
            await self._video_track.add_frame(annotated_frame)
        else:
            # Pass original frame downstream
            await self._video_track.add_frame(frame)

        # Publish the event with detected data
        img_height, img_width = image.shape[0:2]
        detected_objects = [
            DetectedObject(label=classes[class_id], x1=x1, y1=y1, x2=x2, y2=y2)
            for class_id, (x1, y1, x2, y2) in zip(
                detections.class_id, detections.xyxy.astype(float)
            )
        ]

        self.events.send(
            DetectionCompletedEvent(
                raw_detections=detections,
                objects=detected_objects,
                image_width=img_width,
                image_height=img_height,
            )
        )

    async def _run_inference(
        self, image: np.ndarray
    ) -> tuple[sv.Detections, dict[int, str]]:
        """Run Roboflow cloud inference on frame."""

        detected = await self._client.infer_async(image, self.model_id)
        logger.debug(f"Roboflow cloud inference complete in {detected['time']}")
        detected_obj = detected[0] if isinstance(detected, list) else detected
        detections = detected_obj.get("predictions", [])
        # Build a mapping of classes ids to name for labelling
        class_ids_to_labels: dict[int, str] = {}

        if not detections:
            # Exit early if nothing is detected
            return sv.Detections.empty(), class_ids_to_labels

        # Convert the inference result to `sv.Detections` format
        x1_list, y1_list, x2_list, y2_list, confidences, class_ids = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for detection in detections:
            class_id = detection["class_id"]
            class_name = detection["class"]
            # Filter only classes we want to detect
            if self._classes and class_name not in self._classes:
                continue
            class_ids.append(class_id)
            class_ids_to_labels[class_id] = class_name

            x1 = int(detection["x"] - detection["width"] / 2)
            y1 = int(detection["y"] - detection["height"] / 2)
            x2 = int(detection["x"] + detection["width"] / 2)
            y2 = int(detection["y"] + detection["height"] / 2)

            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
            confidences.append(detection["confidence"])

        if class_ids:
            detections_obj = sv.Detections(
                xyxy=np.array(list(zip(x1_list, y1_list, x2_list, y2_list))),
                confidence=np.array(confidences),
                class_id=np.array(class_ids),
            )
        else:
            detections_obj = sv.Detections.empty()
        return detections_obj, class_ids_to_labels
