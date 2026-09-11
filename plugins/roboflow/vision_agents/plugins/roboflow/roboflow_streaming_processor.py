"""Stream video through Roboflow Serverless Video Streaming (WebRTC)."""

import json
import time

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from typing import Optional, Protocol

import aiortc
import av
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
from vision_agents.core import Agent
from vision_agents.core.events import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.processors.observations import ObservationBuffer
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

from .annotation import annotate_image
from .events import DetectedObject, DetectionCompletedEvent
from .predictions import detection_state, detections_from_predictions

try:
    from inference_sdk.webrtc import ManualSource, StreamConfig
    from inference_sdk.webrtc.model_workflows import build_model_workflow
except ImportError:
    ManualSource = None
    StreamConfig = None
    build_model_workflow = None

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://serverless.roboflow.com"


class _FrameSource(Protocol):
    def send(self, frame: np.ndarray) -> None: ...


class _WebRTCSession(Protocol):
    def run(self) -> None: ...

    def close(self) -> None: ...

    def on_data(
        self, name: str
    ) -> Callable[[Callable[..., None]], Callable[..., None]]: ...

    def on_error(self, handler: Callable[..., None]) -> Callable[..., None]: ...


class RoboflowStreamingProcessor(VideoProcessorPublisher):
    """Run a Roboflow model or Workflow over a live WebRTC stream.

    Call frames go to Roboflow Serverless (or a self-hosted Inference Server)
    through ``ManualSource`` and predictions come back on the data channel. Every
    frame is republished with the latest boxes drawn on, each detection emits
    ``DetectionCompletedEvent``, and ``state()`` holds the latest labels for the
    LLM's ``get_video_state`` tool.

    Args:
        model_id: A serverless model such as ``rfdetr-nano``, or a Universe model.
        workflow_id: A saved Workflow instead of ``model_id``. It must expose a
            ``predictions`` output.
        workspace: The Workflow's workspace. Required with ``workflow_id``.
        api_key: Roboflow API key. Defaults to ``ROBOFLOW_API_KEY``.
        api_url: Defaults to Roboflow Serverless.
        requested_region: ``us``, ``eu`` or ``ap``.
        fps: Frames sent per second.
        classes: Keep only these labels.
        task_type: The model's task, so ``model_id`` needs no lookup.
        frame_source: ``annotated`` or ``raw``, which ``latest_frame`` serves
            when the tool does not say.
    """

    name = "roboflow_streaming"

    def __init__(
        self,
        model_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: str = _DEFAULT_API_URL,
        requested_region: Optional[str] = None,
        fps: int = 5,
        classes: Optional[list[str]] = None,
        task_type: str = "object-detection",
        frame_source: str = "annotated",
    ):
        super().__init__()
        if model_id and workflow_id:
            raise ValueError("pass model_id or workflow_id, not both")
        if not model_id and not workflow_id:
            raise ValueError("model_id or workflow_id is required")
        if workflow_id and not workspace:
            raise ValueError("workspace is required when streaming a workflow by id")

        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY required. Get it from https://app.roboflow.com → Settings → API"
            )

        self.model_id = model_id
        self.workflow_id = workflow_id
        self.workspace = workspace
        self.fps = fps
        self.requested_region = requested_region
        self.task_type = task_type
        self.frame_source = frame_source
        self._classes = classes
        self._api_key = api_key
        self._api_url = api_url
        self._label = model_id or workflow_id or ""

        self._events: Optional[EventManager] = None
        self._closed = False
        self._video_forwarder: Optional[VideoForwarder] = None
        self._source: Optional[_FrameSource] = None
        self._session: Optional[_WebRTCSession] = None
        self._session_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.observation_buffer: ObservationBuffer = ObservationBuffer()
        self._source_name = self.name
        self._processed_at_ms = 0
        self._latest_state: dict[str, object] = detection_state([])
        self._latest_detections: sv.Detections = sv.Detections.empty()
        self._latest_classes: dict[int, str] = {}
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._video_track: QueuedVideoTrack = QueuedVideoTrack(
            fps=self.fps,
            max_queue_size=self.fps,
        )

        logger.info("🔍 Roboflow streaming processor initialized")

    async def start(self) -> None:
        """Open the Roboflow session now, so detections are ready before the first frame."""
        if ManualSource is None:
            raise ImportError(
                "RoboflowStreamingProcessor requires inference-sdk[webrtc]>=1.3.5. "
                "Install it with: uv add 'vision-agents-plugins-roboflow[webrtc]'"
            )
        if self._session is not None:
            return
        self._loop = asyncio.get_running_loop()
        source = ManualSource()
        client = InferenceHTTPClient(api_url=self._api_url, api_key=self._api_key)
        config = StreamConfig(
            stream_output=[],
            data_output=["predictions"],
            realtime_processing=True,
            declared_fps=self.fps,
            requested_region=self.requested_region,
        )
        if self.model_id:
            session = client.webrtc.stream(
                source=source,
                config=config,
                workflow=build_model_workflow(self.model_id, self.task_type),
            )
        else:
            session = client.webrtc.stream(
                source=source,
                config=config,
                workflow=self.workflow_id,
                workspace=self.workspace,
            )
        self._bind_session_handlers(session)
        self._source = source
        self._session = session
        self._session_thread = threading.Thread(
            target=session.run, daemon=True, name="roboflow_webrtc"
        )
        self._session_thread.start()
        logger.info(
            "Roboflow streaming session started for %s at %s FPS", self._label, self.fps
        )

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ):
        """Send call frames to Roboflow and republish them with boxes."""
        self._source_name = f"{self.name}/{participant_id or 'unknown'}"
        self.observation_buffer.clear()
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_frame)
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                incoming_track,
                max_buffer=self.fps,
                fps=self.fps,
                name="roboflow_streaming_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._send_frame, fps=float(self.fps), name="roboflow_streaming"
        )

    def publish_video_track(self) -> QueuedVideoTrack:
        return self._video_track

    async def stop_processing(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_frame)
            self._video_forwarder = None
            logger.info("Roboflow streaming stopped (participant left)")

    async def close(self):
        await self.stop_processing()
        self._closed = True
        await self._stop_session()
        self.observation_buffer.clear()
        self._video_track.stop()
        logger.info("Roboflow streaming processor closed")

    def state(self) -> dict[str, object]:
        return self._latest_state

    @property
    def events(self) -> EventManager:
        if self._events is None:
            raise ValueError("Agent is not attached to the processor yet")
        return self._events

    def attach_agent(self, agent: Agent):
        self._events = agent.events
        self._events.register(DetectionCompletedEvent)

    async def _send_frame(self, frame: av.VideoFrame):
        if self._closed:
            return
        image = frame.to_ndarray(format="bgr24")
        self._latest_raw = image.copy()
        self.observation_buffer.append(
            self._source_name,
            av.VideoFrame.from_ndarray(image.copy(), format="bgr24"),
            processor_result=json.dumps(self._latest_state),
            processed_at_ms=self._processed_at_ms,
            aligned=False,
        )
        if self._source is not None:
            try:
                self._source.send(image)
            except RuntimeError:
                # ManualSource refuses frames until the peer connection is up.
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Roboflow streaming still connecting; frame not sent")
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        detections = self._latest_detections
        if detections.class_id is not None and detections.class_id.size:
            rgb = annotate_image(
                rgb,
                detections,
                self._latest_classes,
                box_thickness=4,
                text_scale=1.0,
            )
        await self._video_track.add_frame(
            av.VideoFrame.from_ndarray(rgb, format="rgb24")
        )
        self._latest_annotated = rgb.copy()

    def latest_frame(self, annotated: bool = True) -> Optional[av.VideoFrame]:
        """The most recent frame, annotated or raw depending on ``annotated``."""
        array = self._latest_annotated if annotated else self._latest_raw
        if array is None:
            return None
        fmt = "rgb24" if annotated else "bgr24"
        return av.VideoFrame.from_ndarray(array, format=fmt)

    def _bind_session_handlers(self, session: _WebRTCSession) -> None:
        @session.on_data("predictions")
        def on_data(predictions: object, metadata: object = None) -> None:
            self._on_predictions(predictions)

        @session.on_error
        def on_error(errors: object, metadata: object = None) -> None:
            logger.error("Roboflow streaming error: %s", errors)

    def _on_predictions(self, payload: object) -> None:
        """Runs on the WebRTC thread; the event is handed back to the loop."""
        if self._closed:
            return
        detections, classes, objects = detections_from_predictions(
            payload, classes=self._classes
        )
        self._latest_detections = detections
        self._latest_classes = classes
        confidences = (
            [float(c) for c in detections.confidence]
            if detections.confidence is not None
            else None
        )
        self._processed_at_ms = int(time.time() * 1000)
        self._latest_state = {
            **detection_state(objects, confidences),
            "processed_at_ms": self._processed_at_ms,
            "frame_alignment": "unavailable",
        }
        if objects and self._loop is not None:
            self._loop.call_soon_threadsafe(self._emit_detection, detections, objects)

    def _emit_detection(
        self, detections: sv.Detections, objects: list[DetectedObject]
    ) -> None:
        self.events.send(
            DetectionCompletedEvent(
                plugin_name=self.name,
                raw_detections=detections,
                objects=objects,
                model_id=self._label,
            )
        )
        self.metrics.on_video_detection(
            provider=self.name, model=self._label, detection_count=len(objects)
        )

    async def _stop_session(self) -> None:
        session = self._session
        self._session = None
        self._source = None
        if session is not None:
            await asyncio.to_thread(session.close)
        thread = self._session_thread
        self._session_thread = None
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 5)
