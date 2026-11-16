"""Bounded visual evidence, timestamped at local frame reception (Unix milliseconds)."""

import json
import time
from collections import deque
from dataclasses import dataclass

import av

from vision_agents.core.llm import ImageContent
from vision_agents.core.llm.llm import jpeg_bytes


@dataclass(frozen=True)
class Observation:
    """A retained source frame. Producers must not mutate frames after publishing."""

    source: str
    frame_id: str
    captured_at_ms: int
    frame: av.VideoFrame
    processor_result: str = ""
    processed_at_ms: int = 0
    aligned: bool = False

    def content(self) -> list[dict[str, object]]:
        """Encode evidence off the media/control loop."""
        metadata = {
            "source": self.source,
            "frame_id": self.frame_id,
            "captured_at_ms": self.captured_at_ms,
            "clock": "receiver_unix_ms",
            "processed_at_ms": self.processed_at_ms,
            "processor_result": self.processor_result,
            "processor_result_aligned": self.aligned,
        }
        return [
            {"type": "text", "text": json.dumps(metadata)},
            ImageContent(data=jpeg_bytes(self.frame)).as_content_part(),
        ]


class ObservationBuffer:
    """Keep recent frames within both a count and a memory budget."""

    def __init__(self, max_frames: int = 64, max_bytes: int = 32 << 20):
        if max_frames < 1 or max_bytes < 1:
            raise ValueError("observation limits must be positive")
        self._frames: deque[tuple[Observation, int]] = deque()
        self._max_frames = max_frames
        self._max_bytes = max_bytes
        self._bytes = 0
        self._sequence = 0

    @property
    def sources(self) -> set[str]:
        cutoff = int(time.time() * 1000) - 5000
        return {
            item.source for item, _ in self._frames if item.captured_at_ms >= cutoff
        }

    def append(
        self,
        source: str,
        frame: av.VideoFrame,
        *,
        captured_at_ms: int | None = None,
        processor_result: str = "",
        processed_at_ms: int = 0,
        aligned: bool = False,
    ) -> None:
        """Retain a frame without encoding it; evict oldest entries first."""
        size = sum(plane.buffer_size for plane in frame.planes)
        if size > self._max_bytes:
            return
        self._sequence += 1
        observation = Observation(
            source,
            f"{source}:{self._sequence}",
            captured_at_ms if captured_at_ms is not None else int(time.time() * 1000),
            frame,
            processor_result,
            processed_at_ms,
            aligned,
        )
        self._frames.append((observation, size))
        self._bytes += size
        while len(self._frames) > self._max_frames or self._bytes > self._max_bytes:
            _, removed = self._frames.popleft()
            self._bytes -= removed

    def select(
        self, source: str, at_ms: int, limit: int = 1
    ) -> tuple[Observation, ...]:
        """Pin evidence at or before task acceptance; never substitute newer frames."""
        if not 1 <= limit <= 8:
            raise ValueError("request between 1 and 8 frames")
        frames = [
            item
            for item, _ in self._frames
            if item.source == source and item.captured_at_ms <= at_ms
        ]
        if not frames or at_ms - frames[-1].captured_at_ms > 5000:
            raise ValueError(
                f"no recent frame for {source}; ask the user to show it again"
            )
        if len(frames) < limit:
            raise ValueError(
                f"only {len(frames)} retained frames for {source}; requested {limit}"
            )
        return tuple(frames[-limit:])

    def remove(self, source: str) -> None:
        self._frames = deque(
            (item, size) for item, size in self._frames if item.source != source
        )
        self._bytes = sum(size for _, size in self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0
