"""Object detection on the call's video, asked of the acceleration router."""

import asyncio
import base64
import io
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import aiortc
import av
import httpx
import numpy as np
from PIL import Image, ImageDraw
from vision_agents.core.events import EventManager, PluginBaseEvent
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.processors.observations import ObservationBuffer
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

from ._backend import Backend
from ._errors import RouterError
from ._generated.api.default import detect
from ._generated.models import (
    DetectedObject,
    DetectionOptions,
    DetectionOptionsOverwrites,
    DetectionRequest,
    DetectionResult,
    DetectionStatus,
)
from ._generated.types import Unset

if TYPE_CHECKING:
    from vision_agents.core import Agent

logger = logging.getLogger(__name__)

# x_min, y_min, x_max, y_max as fractions of the image.
Box = tuple[float, float, float, float]


@dataclass
class Track:
    """One thing followed from one detection to the next.

    Attributes:
        id: Stable for as long as the thing stays in view.
        label: What the detector called it.
        box: Where it was last seen, as fractions of the image.
        confidence: The detector's score for the last sighting.
        first_seen_ms: When it appeared, in Unix milliseconds.
        last_seen_ms: When it was last seen, in Unix milliseconds.
        embedding: The last vector the detector returned for it, when asked for one.
        identity: Who ``identify`` said it is. Empty when nobody was recognised.
        misses: Detections in a row that did not find it.
    """

    id: int
    label: str
    box: Box
    confidence: float
    first_seen_ms: int
    last_seen_ms: int
    embedding: Optional[list[float]] = None
    identity: str = ""
    misses: int = 0

    def as_dict(self) -> dict[str, object]:
        """The track as the model reads it, without the embedding."""
        described: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "box": [round(edge, 3) for edge in self.box],
            "seen_for_s": round((self.last_seen_ms - self.first_seen_ms) / 1000, 1),
        }
        if self.identity:
            described["identity"] = self.identity
        return described


@dataclass
class ObjectAppearedEvent(PluginBaseEvent):
    """Something came into view.

    Attributes:
        track: What appeared and where.
        source: The video source its frames are retained under, for ``get_video_frames``.
    """

    type: str = field(default="plugin.stream.object_appeared", init=False)
    track: Optional[Track] = None
    source: str = ""


@dataclass
class ObjectDisappearedEvent(PluginBaseEvent):
    """Something left the view, or stopped being found in it.

    Attributes:
        track: What went, with when it was last seen.
        source: The video source its frames are retained under, for ``get_video_frames``.
    """

    type: str = field(default="plugin.stream.object_disappeared", init=False)
    track: Optional[Track] = None
    source: str = ""


def iou(a: Box, b: Box) -> float:
    """How much two boxes overlap: their intersection over their union."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


class Tracks:
    """Follows boxes across detections by how much they overlap.

    A box of the same label overlapping a track by at least ``min_iou`` is that track
    again; the best overlaps are paired first. A track not found ``misses`` detections in
    a row has gone, so one frame the detector misses does not make it disappear.
    """

    def __init__(self, min_iou: float = 0.3, misses: int = 2):
        if not 0 < min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
        if misses < 0:
            raise ValueError("misses cannot be negative")
        self._min_iou = min_iou
        self._misses = misses
        self._live: dict[int, Track] = {}
        self._next_id = 1

    @property
    def live(self) -> list[Track]:
        """The tracks in view, oldest first."""
        return list(self._live.values())

    def update(
        self, found: list[DetectedObject], now_ms: int
    ) -> tuple[list[Track], list[Track]]:
        """Fold one detection into the tracks.

        Args:
            found: What the detector returned for one frame.
            now_ms: When that frame was captured, in Unix milliseconds.

        Returns:
            The tracks that appeared and the tracks that went.
        """
        boxes = [_box(detected) for detected in found]
        pairs = sorted(
            (
                (iou(track.box, box), track.id, index)
                for track in self._live.values()
                for index, box in enumerate(boxes)
                if track.label.lower() == found[index].label.lower()
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        for overlap, track_id, index in pairs:
            if overlap < self._min_iou:
                break
            if track_id in matched_tracks or index in matched_boxes:
                continue
            matched_tracks.add(track_id)
            matched_boxes.add(index)
            track = self._live[track_id]
            track.box = boxes[index]
            track.confidence = found[index].confidence
            track.last_seen_ms = now_ms
            track.misses = 0
            embedding = _embedding(found[index])
            if embedding is not None:
                track.embedding = embedding

        gone = []
        for track in list(self._live.values()):
            if track.id in matched_tracks:
                continue
            track.misses += 1
            if track.misses > self._misses:
                gone.append(self._live.pop(track.id))

        appeared = []
        for index, detected in enumerate(found):
            if index in matched_boxes:
                continue
            track = Track(
                id=self._next_id,
                label=detected.label,
                box=boxes[index],
                confidence=detected.confidence,
                first_seen_ms=now_ms,
                last_seen_ms=now_ms,
                embedding=_embedding(detected),
            )
            self._next_id += 1
            self._live[track.id] = track
            appeared.append(track)
        return appeared, gone

    def clear(self) -> None:
        """Forget every track without reporting any of them gone."""
        self._live.clear()


class ChangeGate:
    """Says whether a frame differs enough from the last one let through to ask about.

    Frames are compared as small grayscale thumbnails: the share of pixels that moved by
    more than ``pixel_delta`` grey levels has to reach ``threshold``. Sensor noise moves
    every pixel a little and a package moves a few pixels a lot, which is what counting
    pixels rather than averaging them tells apart. A frame is let through every
    ``refresh_interval`` seconds anyway, so a detection the scene settled on is checked.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        pixel_delta: int = 24,
        refresh_interval: float = 5.0,
        size: tuple[int, int] = (64, 36),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self._threshold = threshold
        self._pixel_delta = pixel_delta
        self._refresh_interval = refresh_interval
        self._size = size
        self._last: Optional[np.ndarray] = None
        self._last_at = 0.0

    def changed(self, rgb: np.ndarray, now: float) -> bool:
        """Whether ``rgb``, seen at monotonic time ``now``, should be detected on."""
        thumbnail = np.asarray(
            Image.fromarray(rgb).convert("L").resize(self._size), dtype=np.int16
        )
        if self._last is not None and now - self._last_at < self._refresh_interval:
            moved = np.abs(thumbnail - self._last) > self._pixel_delta
            if float(np.mean(moved)) < self._threshold:
                return False
        self._last = thumbnail
        self._last_at = now
        return True

    def reset(self) -> None:
        """Let the next frame through."""
        self._last = None


class DetectionProcessor(VideoProcessorPublisher):
    """Detect objects in the call's video through the acceleration router.

    Frames that barely changed are never sent. The rest go to ``POST /v1/detection`` one
    at a time, scaled to ``width``, and what comes back is followed across frames:
    ``ObjectAppearedEvent`` and ``ObjectDisappearedEvent`` say when something came or
    went, ``state()`` is what is in view for ``get_video_state``, and the published
    track has the boxes drawn on. Frames are retained in ``observation_buffer`` under
    ``<name>/<participant>`` for ``get_video_frames``.

    Example:
        ```
        objects = stream.DetectionProcessor(
            target="detection-open", classes=["person", "package"]
        )
        agent = Agent(config="security_camera", processors=[objects])

        @agent.events.subscribe
        async def on_appeared(event: stream.ObjectAppearedEvent): ...
        ```
    """

    name = "detection"

    def __init__(
        self,
        target: str = "detection-fast",
        classes: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        min_area: float = 0.0,
        embeddings: bool = False,
        overwrites: Optional[dict[str, dict[str, object]]] = None,
        identify: Optional[Callable[[list[float]], str]] = None,
        name: str = "detection",
        fps: int = 5,
        width: int = 640,
        change_threshold: float = 0.01,
        refresh_interval: float = 5.0,
        min_iou: float = 0.3,
        misses: int = 2,
        frame_source: str = "annotated",
        backend: Optional[Backend] = None,
    ):
        """Say what to look for and how often.

        Args:
            target: A ``provider/model`` or a shortcut such as ``detection-open``.
            classes: The labels to keep. An open-vocabulary model looks for these words.
            min_confidence: Drop boxes scored below this.
            min_area: Drop boxes covering less than this share of the image, which is
                what a detector finding specks looks like.
            embeddings: Ask for a vector on every box, for ``identify``.
            overwrites: Settings for one provider the options have no word for, keyed by
                provider, such as a Roboflow Workflow.
            identify: Names whoever an embedding belongs to, or returns an empty string.
                Asked for every track with an embedding until it answers.
            name: The processor's name in ``get_video_state`` and its video source.
            fps: Frames read from the call per second.
            width: Frames wider than this are scaled down before they are sent.
            change_threshold: The share of pixels that has to change for a frame to be
                sent.
            refresh_interval: Seconds after which a frame is sent even if unchanged.
            min_iou: The overlap that makes a box the same track as before.
            misses: Detections in a row a track may go unfound before it is gone.
            frame_source: ``annotated`` or ``raw``, which ``latest_frame`` serves.
            backend: The router to ask. Defaults to the one the environment names.
        """
        super().__init__()
        if fps < 1:
            raise ValueError("fps must be at least 1")
        if width < 32:
            raise ValueError("width must be at least 32 pixels")
        if not name:
            raise ValueError("a processor needs a name")
        self.name = name
        self.fps = fps
        self.frame_source = frame_source
        self.backend = backend or Backend()
        self.observation_buffer: ObservationBuffer = ObservationBuffer()
        self._width = width
        self._min_area = min_area
        self._identify = identify
        self._options = DetectionOptions(target=target)
        if classes:
            self._options.classes = classes
        if min_confidence is not None:
            self._options.min_confidence = min_confidence
        if embeddings:
            self._options.embeddings = True
        if overwrites:
            self._options.overwrites = DetectionOptionsOverwrites.from_dict(overwrites)

        self._gate = ChangeGate(
            threshold=change_threshold, refresh_interval=refresh_interval
        )
        self._tracks = Tracks(min_iou=min_iou, misses=misses)
        self._events: Optional[EventManager] = None
        self._forwarder: Optional[VideoForwarder] = None
        self._detecting: Optional[asyncio.Task[list[Track]]] = None
        self._closed = False
        self._source = f"{name}/unknown"
        self._buffered_at_ms = 0
        self._processed_at_ms = 0
        self._state: dict[str, object] = {"objects": [], "counts": {}}
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._video_track = QueuedVideoTrack(fps=fps, max_queue_size=fps)

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Start watching a new video track, forgetting what the last one showed."""
        await self.stop_processing()
        self._source = f"{self.name}/{participant_id or 'unknown'}"
        self.observation_buffer.clear()
        self._tracks.clear()
        self._gate.reset()
        self._forwarder = shared_forwarder or VideoForwarder(
            track, max_buffer=self.fps, fps=self.fps, name=f"{self.name}_forwarder"
        )
        self._forwarder.add_frame_handler(
            self._on_frame, fps=float(self.fps), name=self.name
        )

    async def stop_processing(self) -> None:
        """Stop reading frames and abandon the detection in flight."""
        if self._forwarder is not None:
            await self._forwarder.remove_frame_handler(self._on_frame)
            self._forwarder = None
        if self._detecting is not None:
            self._detecting.cancel()
            self._detecting = None

    async def close(self) -> None:
        """Stop for good."""
        await self.stop_processing()
        self._closed = True
        self.observation_buffer.clear()
        self._video_track.stop()

    @property
    def tracks(self) -> list[Track]:
        """What is in view, oldest first."""
        return self._tracks.live

    @property
    def source(self) -> str:
        """The video source this processor's frames are retained under."""
        return self._source

    def attach_agent(self, agent: "Agent") -> None:
        self._events = agent.events
        self._events.register(ObjectAppearedEvent, ObjectDisappearedEvent)

    def publish_video_track(self) -> QueuedVideoTrack:
        return self._video_track

    def state(self) -> dict[str, object]:
        return self._state

    def latest_frame(self, annotated: bool = True) -> Optional[av.VideoFrame]:
        """The most recent frame, with boxes drawn on or as it came."""
        array = self._latest_annotated if annotated else self._latest_raw
        if array is None:
            return None
        return av.VideoFrame.from_ndarray(array, format="rgb24")

    async def detect(
        self, frame: av.VideoFrame, captured_at_ms: Optional[int] = None
    ) -> list[Track]:
        """Ask the router about one frame and fold the answer into the tracks.

        A detection that failed at the provider leaves the tracks as they were, since
        finding nothing is not the same as there being nothing.

        Args:
            frame: The frame to look at.
            captured_at_ms: When it was captured. Defaults to now.

        Returns:
            What is in view afterwards.
        """
        return await self._detect(
            frame.to_ndarray(format="rgb24"),
            captured_at_ms or int(time.time() * 1000),
        )

    async def _detect(self, rgb: np.ndarray, captured_at_ms: int) -> list[Track]:
        scaled = self._scaled(rgb)
        image = await asyncio.to_thread(_jpeg, scaled)
        started = time.perf_counter()
        result = await detect.asyncio(
            client=self.backend.client(),
            body=DetectionRequest(
                image=base64.b64encode(image).decode(), options=self._options
            ),
        )
        if not isinstance(result, DetectionResult):
            raise RouterError("the router did not answer with a detection")
        if result.status == DetectionStatus.FAILED:
            logger.warning(
                "%s found nothing: %s (%s)", self.name, result.error, result.error_code
            )
            return self.tracks

        found = [
            detected
            for detected in result.detections
            if _area(_box(detected)) >= self._min_area
        ]
        appeared, gone = self._tracks.update(found, captured_at_ms)
        if self._identify is not None:
            for track in self._tracks.live:
                if not track.identity and track.embedding:
                    track.identity = self._identify(track.embedding)

        live = self._tracks.live
        model = "" if isinstance(result.model, Unset) else result.model
        self._processed_at_ms = captured_at_ms
        self._state = {
            "objects": [track.as_dict() for track in live],
            "counts": dict(Counter(track.label for track in live)),
            "processed_at_ms": captured_at_ms,
            "provider": "" if isinstance(result.provider, Unset) else result.provider,
            "model": model,
        }
        self.observation_buffer.append(
            self._source,
            av.VideoFrame.from_image(scaled),
            captured_at_ms=captured_at_ms,
            processor_result=json.dumps(self._state),
            processed_at_ms=captured_at_ms,
            aligned=True,
        )
        self._buffered_at_ms = captured_at_ms
        self.metrics.on_video_detection(
            provider=self.name,
            model=model or None,
            detection_count=len(found),
            inference_time_ms=(time.perf_counter() - started) * 1000,
        )
        if self._events is not None:
            for track in appeared:
                self._events.send(
                    ObjectAppearedEvent(
                        plugin_name=self.name, track=track, source=self._source
                    )
                )
            for track in gone:
                self._events.send(
                    ObjectDisappearedEvent(
                        plugin_name=self.name, track=track, source=self._source
                    )
                )
        return live

    async def _on_frame(self, frame: av.VideoFrame) -> None:
        if self._closed:
            return
        # Read once, on the loop. The frame is shared with every other processor, and a
        # VideoFrame caches its converter, so converting one from two threads at once
        # frees the converter twice. Everything below works on this array instead.
        rgb = frame.to_ndarray(format="rgb24")
        now_ms = int(time.time() * 1000)
        self._latest_raw = rgb
        # A track that was just missed is looked for again at once rather than when the
        # scene next changes, so something taken away is gone in seconds.
        unsettled = any(track.misses for track in self._tracks.live)
        if self._detecting is None and (
            self._gate.changed(rgb, time.monotonic()) or unsettled
        ):
            self._detecting = asyncio.create_task(
                self._detect_in_background(rgb, now_ms)
            )
        elif now_ms - self._buffered_at_ms >= 1000:
            # One frame a second at least, so a capture always has a recent one to pin.
            self.observation_buffer.append(
                self._source,
                av.VideoFrame.from_image(self._scaled(rgb)),
                captured_at_ms=now_ms,
                processor_result=json.dumps(self._state),
                processed_at_ms=self._processed_at_ms,
            )
            self._buffered_at_ms = now_ms
        annotated = self._annotate(rgb)
        self._latest_annotated = annotated
        published = av.VideoFrame.from_ndarray(annotated, format="rgb24")
        published.pts = frame.pts
        published.time_base = frame.time_base
        await self._video_track.add_frame(published)

    async def _detect_in_background(
        self, rgb: np.ndarray, captured_at_ms: int
    ) -> list[Track]:
        try:
            return await self._detect(rgb, captured_at_ms)
        except (RouterError, httpx.HTTPError):
            logger.exception("%s could not detect", self.name)
            return self.tracks
        finally:
            self._detecting = None

    def _scaled(self, rgb: np.ndarray) -> Image.Image:
        image = Image.fromarray(rgb)
        if image.width > self._width:
            height = max(2, round(image.height * self._width / image.width))
            image = image.resize((self._width, height))
        return image

    def _annotate(self, rgb: np.ndarray) -> np.ndarray:
        live = self._tracks.live
        if not live:
            return rgb
        image = Image.fromarray(rgb).copy()
        draw = ImageDraw.Draw(image)
        for track in live:
            box = (
                track.box[0] * image.width,
                track.box[1] * image.height,
                track.box[2] * image.width,
                track.box[3] * image.height,
            )
            draw.rectangle(box, outline=(0, 220, 120), width=3)
            draw.text(
                (box[0] + 4, box[1] + 2),
                track.identity or track.label,
                fill=(0, 220, 120),
            )
        return np.asarray(image)


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _box(detected: DetectedObject) -> Box:
    box = detected.box
    return (box.x_min, box.y_min, box.x_max, box.y_max)


def _area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _embedding(detected: DetectedObject) -> Optional[list[float]]:
    if isinstance(detected.embedding, Unset):
        return None
    return list(detected.embedding)
