import asyncio
import fractions
import time

import aiortc
import av
from getstream.video.rtc import AudioStreamTrack

__all__ = ["AvatarInputTrack"]


class AvatarInputTrack(AudioStreamTrack):
    """
    An input audio track for the LemonSlice avatar.

    Key differences from the base AudioStreamTrack:
    - Tracks the last produced "pts" and exposes it as public API.
    - Returns all available data on "recv()" instead of real-time pacing
      to reduce avatar latency.
    """

    _timestamp: int | None
    _start: float | None
    _last_frame_time: float | None

    async def pts(self) -> int:
        async with self._buffer_lock:
            # next-to-emit PTS + samples still queued = wire PTS of the last buffered sample
            ts = self._timestamp or 0
            return ts + len(self._buffer) // self._bytes_per_sample

    async def recv(self) -> av.AudioFrame:
        """Drain buffered audio without pacing; pace only when emitting silence."""
        if self.readyState != "live":
            raise aiortc.mediastreams.MediaStreamError

        samples_per_frame = int(aiortc.mediastreams.AUDIO_PTIME * self.sample_rate)

        if self._timestamp is None:
            self._start = time.time()
            timestamp = 0
        else:
            timestamp = self._timestamp + samples_per_frame
        self._timestamp = timestamp

        async with self._buffer_lock:
            if len(self._buffer) >= self._bytes_per_frame:
                audio_bytes = bytes(self._buffer[: self._bytes_per_frame])
                del self._buffer[: self._bytes_per_frame]
                has_data = True
            elif len(self._buffer) > 0:
                audio_bytes = bytes(self._buffer)
                audio_bytes += bytes(self._bytes_per_frame - len(audio_bytes))
                self._buffer.clear()
                has_data = True
            else:
                audio_bytes = bytes(self._bytes_per_frame)
                has_data = False

        if not has_data:
            # Per-frame sleep keeps silence at the frame cadence even when PTS runs ahead of wall clock after a burst.
            await asyncio.sleep(aiortc.mediastreams.AUDIO_PTIME)

        self._last_frame_time = time.time()

        layout = "stereo" if self.channels == 2 else "mono"
        av_format = "flt" if self.format == "f32" else "s16"
        frame = av.AudioFrame(
            format=av_format, layout=layout, samples=samples_per_frame
        )
        frame.planes[0].update(audio_bytes)
        frame.pts = timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame
