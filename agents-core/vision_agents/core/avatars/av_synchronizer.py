import asyncio
import collections
import fractions
import logging
import time

import av
import av.frame
from aiortc.mediastreams import MediaStreamError
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.inference import AudioOutputChunk, AudioOutputStream
from vision_agents.core.utils.video_track import (
    QueuedVideoTrack,
    VideoTrackClosedError,
)
from vision_agents.core.utils.video_utils import ensure_even_dimensions, resize_frame

__all__ = ["AVSynchronizer"]

logger = logging.getLogger(__name__)

# aiortc hardcodes 30fps via its module-level VIDEO_PTIME; _SyncedVideoTrack
# overrides next_timestamp() to honor its configured fps using these constants.
_VIDEO_CLOCK_RATE = 90000
_VIDEO_TIME_BASE = fractions.Fraction(1, _VIDEO_CLOCK_RATE)


class AVSynchronizer:
    """A utility class to synchronize avatar video and audio output for WebRTC publishing.

    Creates paired audio and video tracks where video frames are delayed
    to match the audio buffer depth, keeping lip-sync accurate.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        max_queue_size: int = 300,
    ) -> None:
        self._audio_output = AudioOutputStream()
        self._video_output = _SyncedVideoTrack(
            audio_output_stream=self._audio_output,
            width=width,
            height=height,
            fps=fps,
            max_queue_size=max(1, max_queue_size),
        )

    @property
    def video_output(self) -> QueuedVideoTrack:
        return self._video_output

    @property
    def audio_output(self) -> AudioOutputStream:
        return self._audio_output

    async def write_video(self, frame: av.VideoFrame) -> None:
        """Queue a video frame, delayed by the current audio buffer depth."""
        await self._video_output.add_frame(frame)

    async def write_audio(self, pcm: PcmData) -> None:
        """Write audio PCM data to the audio track."""
        await self._audio_output.send(AudioOutputChunk(data=pcm))

    async def flush(self) -> None:
        """Discard all pending video frames and flush buffered audio."""
        # video track already flushes audio too
        await self._video_output.flush()

    def close(self):
        self._audio_output.close()


class _SyncedVideoTrack(QueuedVideoTrack):
    """QueuedVideoTrack that delays frames to stay in sync with an audio buffer.

    Frames are stamped with a release time based on the companion audio
    track's buffer depth.
    ``recv`` holds each frame until its release time,
    repeating the last delivered frame in the meantime.
    """

    def __init__(
        self, audio_output_stream: AudioOutputStream, max_queue_size: int, **kwargs: int
    ) -> None:
        super().__init__(**kwargs)
        self._audio_output_stream = audio_output_stream
        self._pending: collections.deque[tuple[float, av.VideoFrame]] = (
            collections.deque(maxlen=max_queue_size)
        )

    async def add_frame(self, frame: av.VideoFrame) -> None:
        """Queue a frame, delayed by the current audio buffer depth."""
        if self._stopped:
            return
        if frame.width != self.width or frame.height != self.height:
            frame = await asyncio.to_thread(
                resize_frame, frame, self.width, self.height
            )
        else:
            frame = ensure_even_dimensions(frame)
        release_at = (
            asyncio.get_running_loop().time() + self._audio_output_stream.buffered
        )
        self._pending.append((release_at, frame))

    async def recv(self) -> av.frame.Frame:
        """Return the next frame, releasing it only once its delay has elapsed.

        Pacing is enforced by ``next_timestamp()``, which sleeps to maintain
        the frame rate.
        """
        if self._stopped:
            raise VideoTrackClosedError("Track stopped")

        if self._pending:
            release_at, frame = self._pending[0]
            if asyncio.get_running_loop().time() >= release_at:
                self._pending.popleft()
                self.last_frame = frame

        pts, time_base = await self.next_timestamp()
        result = self.last_frame
        result.pts = pts
        result.time_base = time_base
        return result

    async def flush(self) -> None:
        """Discard all pending frames and flush buffered audio."""
        self._pending.clear()
        await self._audio_output_stream.flush()

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Pace frames at ``self.fps`` instead of aiortc's hardcoded 30fps."""
        if self.readyState != "live":
            raise MediaStreamError
        ptime = 1.0 / self.fps
        if hasattr(self, "_timestamp"):
            self._timestamp += int(ptime * _VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / _VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, _VIDEO_TIME_BASE
