import asyncio
import collections
import logging

import av
import av.frame
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.utils.video_track import (
    QueuedVideoTrack,
    VideoTrackClosedError,
)
from vision_agents.core.utils.video_utils import ensure_even_dimensions

__all__ = ["AVSynchronizer"]

logger = logging.getLogger(__name__)


class _BufferTrackingAudioTrack(AudioStreamTrack):
    """AudioStreamTrack that exposes the amount of buffered audio in seconds.

    Reads ``len(self._buffer)`` directly — always accurate regardless of
    silence emission, flush, or write/recv timing.
    """

    @property
    def buffered(self) -> float:
        """Return the amount of seconds of audio pending in the buffer."""
        return len(self._buffer) / (
            self.sample_rate * self.channels * self._bytes_per_sample
        )


class _SyncedVideoTrack(QueuedVideoTrack):
    """QueuedVideoTrack that delays frames to stay in sync with an audio buffer.

    Frames are stamped with a release time based on the companion audio
    track's buffer depth.
    ``recv`` holds each frame until its release time,
    repeating the last delivered frame in the meantime.
    """

    def __init__(
        self, audio_track: _BufferTrackingAudioTrack, max_queue_size: int, **kwargs: int
    ) -> None:
        super().__init__(**kwargs)
        self._audio_track = audio_track
        self._pending: collections.deque[tuple[float, av.VideoFrame]] = (
            collections.deque(maxlen=max_queue_size)
        )

    async def add_frame(self, frame: av.VideoFrame) -> None:
        """Queue a frame, delayed by the current audio buffer depth."""
        if self._stopped:
            return
        frame = ensure_even_dimensions(frame)
        release_at = asyncio.get_running_loop().time() + self._audio_track.buffered
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
        await self._audio_track.flush()


class AVSynchronizer:
    """Synchronizes avatar video and audio output for WebRTC publishing.

    Creates paired audio and video tracks where video frames are delayed
    to match the audio buffer depth, keeping lip-sync accurate.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        max_queue_size: int = 300,
        sample_rate: int = 48000,
        channels: int = 2,
    ) -> None:
        self._audio_track = _BufferTrackingAudioTrack(
            sample_rate=sample_rate, channels=channels, format="s16"
        )
        self._video_track = _SyncedVideoTrack(
            audio_track=self._audio_track,
            width=width,
            height=height,
            fps=fps,
            max_queue_size=max_queue_size,
        )

    @property
    def video_track(self) -> QueuedVideoTrack:
        return self._video_track

    @property
    def audio_track(self) -> AudioStreamTrack:
        return self._audio_track

    async def write_video(self, frame: av.VideoFrame) -> None:
        """Queue a video frame, delayed by the current audio buffer depth."""
        await self._video_track.add_frame(frame)

    async def write_audio(self, pcm: PcmData) -> None:
        """Write audio PCM data to the audio track."""
        await self._audio_track.write(pcm=pcm)

    async def flush(self) -> None:
        """Discard all pending video frames and flush buffered audio."""
        await self._video_track.flush()
