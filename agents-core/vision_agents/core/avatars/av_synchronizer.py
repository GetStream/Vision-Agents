import asyncio
import collections
import logging

import av
import av.frame
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.inference import AudioOutputChunk, AudioOutputStream
from vision_agents.core.utils.video_track import (
    QueuedVideoTrack,
    VideoTrackClosedError,
)
from vision_agents.core.utils.video_utils import ensure_even_dimensions

__all__ = ["AVSynchronizer"]

logger = logging.getLogger(__name__)


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
            max_queue_size=max_queue_size,
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
        await self._video_output.flush()
        await self._audio_output.flush()

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
